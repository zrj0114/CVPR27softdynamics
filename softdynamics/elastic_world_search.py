"""Material search using independent observed-state one-step predictions.

Original MPM validation is a separate, explicitly counted downstream action.
Every transition starts from a fixed current observed-state estimate.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .elastic_data import ElasticSpace
from .elastic_downstream import bounds, tensor_parameters
from .io import read_json, write_json, sha256
from .joint_world import load_joint
from .joint_world import context_batch
from .joint_acceleration import normalized_weights
from .elastic_step_observation import ObservedStepEpisode
from .teacher_contact import require_selected_architecture,contract


def parameter_weighted_gradient(losses, coordinates, weights):
    """Combine each parameter's derivatives using its own fixed episode weights.

    This vector need not be the derivative of a single scalar objective.
    Candidate selection is explicitly measured by the equal-episode mean loss.
    """
    if losses.ndim != 1 or weights.shape != (len(losses), coordinates.numel()):
        raise ValueError('Expected one loss per episode and one weight column per parameter')
    weights = normalized_weights(weights)
    derivatives = [torch.autograd.grad(losses, coordinates, grad_outputs=weights[:, j],
        retain_graph=j+1 < coordinates.numel())[0][j] for j in range(coordinates.numel())]
    return torch.stack(derivatives)


def optimize(objective, config, device='cpu', starts=5, steps=160, seed=9101, callback=None, weights=None):
    if starts < 1 or steps < 1:
        raise ValueError('Positive starts and steps required')
    lo, hi = [torch.tensor(x, device=device, dtype=torch.float32) for x in bounds(config)]
    if weights is not None:
        weights = normalized_weights(torch.as_tensor(weights, dtype=lo.dtype, device=device)).clone()
        if weights.shape[1] != 2:
            raise ValueError('Expected independent E and nu confidence columns')
    initial = (-lo/(hi-lo)).clamp(0,1)
    candidates = [initial]
    if starts > 1:
        candidates.extend(torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(starts-1).to(device))
    best_loss, best_q = float('inf'), None
    history = []
    evaluations = 0
    started = time.monotonic()
    for index, value in enumerate(candidates):
        unit = value.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([unit], lr=.04)
        local_best = float('inf')
        local_q = None
        for step in range(steps+1):
            optimizer.zero_grad(set_to_none=True)
            q = lo+(hi-lo)*unit
            losses = objective(q)
            if weights is not None and losses.shape != (len(weights),):
                raise ValueError('Confidence weighting requires a vector of episode losses')
            loss = losses.mean()
            evaluations += 1
            scalar = float(loss.detach())
            finite = bool(torch.isfinite(losses).all())
            if finite and scalar < local_best:
                local_best, local_q = scalar, q.detach().cpu().tolist()
            if finite and scalar < best_loss:
                best_loss, best_q = scalar, q.detach().cpu().tolist()
            if not finite:
                break
            if step == steps:
                break  # The final updated point has also been evaluated.
            if weights is None:
                loss.backward()
            else:
                unit.grad = parameter_weighted_gradient(losses, unit, weights)
            if unit.grad is None or not torch.isfinite(unit.grad).all():
                break
            torch.nn.utils.clip_grad_norm_([unit], 5.)
            optimizer.param_groups[0]['lr'] = .004+.036*.5*(1+np.cos(np.pi*step/steps))
            optimizer.step()
            with torch.no_grad():
                unit.clamp_(0,1)
        row = dict(start=index, steps_evaluated=step+1, best_q=local_q, best_loss=local_best)
        history.append(row)
        if callback:
            callback(dict(starts_completed=index+1, starts=starts, best_q=best_q,
                best_model_loss=best_loss, model_evaluations=evaluations, elapsed_s=time.monotonic()-started))
    if best_q is None:
        raise FloatingPointError('All neural searches failed; no candidate can be exported')
    return dict(q=best_q, parameters=ElasticSpace(config).parameters(best_q), model_loss=best_loss,
        history=history, model_evaluations=evaluations, optimizer_steps_per_start=steps,
        original_mpm_calls=0, original_mpm_validated=False, elapsed_s=time.monotonic()-started,
        selection='Lowest equal-episode mean loss across all finite evaluated points',
        gradient_weighting='independent_parameter_columns' if weights is not None else 'uniform',
        episode_parameter_weights=None if weights is None else weights.cpu().tolist(),
        weights_frozen_during_search=True)


def fit(checkpoint, downstream_run, obj, output, device='cuda:0', starts=5, steps=160, weighting='confidence'):
    if weighting not in ('confidence', 'uniform'):
        raise ValueError('Unknown interaction weighting')
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing neural material selection')
    torch.set_num_threads(2)
    spec = read_json(Path(downstream_run)/'protocol.json')
    physical_plan = read_json(Path(downstream_run)/'physics/plan.json')
    item = next(o for o in physical_plan['objects'] if o['object']==obj)
    joint, bundle = load_joint(checkpoint, device)
    require_selected_architecture(bundle['architecture'])
    if bundle['architecture']['material_dim'] != 2:
        raise ValueError('Expected a two-parameter world model')
    if bundle['protocol'].get('prediction_feedback', True) or bundle['protocol'].get('recurrent_horizon') != 1:
        raise ValueError('Expected an independently trained one-step checkpoint')
    if obj in bundle['protocol']['training_objects']:
        raise ValueError('Training object used as a test object')
    joint.eval().requires_grad_(False)
    episodes = [ObservedStepEpisode(spec['data_root'], obj, e, device) for e in item['support']]
    config = spec['elastic_config']
    weights = None
    confidence_record = None
    if weighting == 'confidence':
        if tuple(joint.confidence_parameters) != ('E_pa', 'nu'):
            raise ValueError('Checkpoint confidence columns must be ordered E_pa, nu')
        reference_q = torch.zeros(2, device=device)
        reference_theta = tensor_parameters(reference_q, config)
        with torch.no_grad():
            contexts = context_batch([e.confidence_context for e in episodes], device)
            raw = joint.confidence(contexts, reference_theta[None].expand(len(episodes), -1))
            weights = normalized_weights(raw)
        confidence_record = dict(parameters=['E_pa', 'nu'], reference_q=reference_q.cpu().tolist(),
            reference_parameters=reference_theta.cpu().tolist(), raw=raw.cpu().tolist(),
            normalized=weights.cpu().tolist(),
            context='Measured support interaction only; same representation as confidence training',
            frozen='Once at the common reference material, shared by all starts and iterations')
    def objective(q):
        theta = tensor_parameters(q, config)
        # The same E/nu is shared by every independent transition and episode.
        return torch.stack([e.loss(e.predict(joint.world,theta,gradients=True)) for e in episodes])
    def progress(row):
        write_json(output.with_suffix('.progress.json'), row)
        print('NEURAL_SEARCH', row, flush=True)
    result = optimize(objective, config, device, starts, steps, callback=progress, weights=weights)
    result.update(object=obj, support=item['support'], checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), checkpoint_step=bundle['step'],
        network_weights_frozen=True, query_observations_used=False, physical_state_anchors_used=False,
        observed_state_conditioning=True,
        full_trajectory_backpropagation=False, prediction_feedback=False, recurrent_horizon=1,
        objective='Phase-balanced one-step observed material-track increment Huber loss',
        weighting=weighting, confidence=confidence_record,
        selection_metric='Equal-episode mean; weighting changes parameter updates, not reported fit quality',
        state_estimation=[e.state_estimation for e in episodes], created_at=time.time())
    result['contact_contract']=contract()
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--downstream-run', type=Path, required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--starts', type=int, default=5)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--weighting', choices=('confidence', 'uniform'), default='confidence')
    args = parser.parse_args()
    fit(args.checkpoint,args.downstream_run,args.object,args.output,args.device,args.starts,args.steps,args.weighting)


if __name__ == '__main__':
    main()
