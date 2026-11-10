"""Fixed validation objects: rollout, signed response and material recovery.

Small-budget inverse fits are diagnostics on validation data, not claims of
real-data recovery or original-MPM validation. Test objects never enter here.
"""
from collections import defaultdict
import numpy as np
import torch

from .coupled_world_data import temporal_batch
from .coupled_world_model import forecast, response_summary
from .elastic_world_validation import summarize_windows


def selections(replay, horizon=16):
    groups = defaultdict(list)
    for identity, t in replay.trajectories.items():
        groups[t['task']['object'], t['task']['episode']].append(identity)
    result = []
    for (obj, episode), ids in sorted(groups.items()):
        identity = sorted(ids)[episode % len(ids)]
        t = replay.trajectories[identity]
        for phase in range(3):
            choices = [i for i in range(t['minimum_start'], len(t['state'])-horizon)
                       if np.all(t['phase'][i+1:i+horizon+1] == phase)]
            if choices:
                result.append((identity, choices[len(choices)//2], phase))
    if not result:
        raise ValueError('No fixed validation windows')
    return result


def inverse_diagnostic(model, replay, device, steps=12):
    """One fixed non-reference material per validation object, two windows.

Observations before each short forecast estimate its state; no target future
positions are supplied to the recurrent rollout. All parameter paths inside
the prediction window remain differentiable.
"""
    results = []
    cfg = replay.plan['config']
    lo = torch.tensor([np.log(cfg['E_range_pa'][0]/cfg['reference_E_pa']),
                       (cfg['nu_range'][0]-cfg['reference_nu'])/cfg['nu_scale']], device=device)
    hi = torch.tensor([np.log(cfg['E_range_pa'][1]/cfg['reference_E_pa']),
                       (cfg['nu_range'][1]-cfg['reference_nu'])/cfg['nu_scale']], device=device)
    original = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    try:
        for obj in replay.object_names():
            eligible = [k for k, t in replay.trajectories.items()
                        if t['task']['object'] == obj and t['task']['variant'] == 3]
            if not eligible:
                continue
            identity = sorted(eligible)[0]
            t = replay.trajectories[identity]
            choices = []
            for phase in (1, 2):
                starts = np.flatnonzero(t['phase'][:len(t['state'])-8] == phase)
                starts = starts[starts >= max(3, t['minimum_start'])]
                if len(starts):
                    choices.append((identity, int(starts[len(starts)//2])))
            if not choices:
                continue
            batch, geo = temporal_batch(replay, choices, 8, device)
            unit = ((torch.zeros_like(lo)-lo)/(hi-lo)).detach().requires_grad_(True)
            optimizer = torch.optim.Adam([unit], lr=.06)
            best_loss, best_q = float('inf'), None
            for iteration in range(steps+1):
                with torch.enable_grad():
                    optimizer.zero_grad(set_to_none=True)
                    q = lo+(hi-lo)*unit
                    trial = dict(batch, theta=model.theta(q).float()[None].expand(len(choices), -1))
                    predicted = forecast(model.world, trial, geo, gradients=True)
                    residual = (predicted[..., :3]-batch['states'][:, 1:, :, :3])*geo['scale_m'][:, None, None, None]/.003
                    loss = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual))
                    value = float(loss.detach())
                    if np.isfinite(value) and value < best_loss:
                        best_loss, best_q = value, q.detach().clone()
                    if iteration == steps or not np.isfinite(value):
                        break
                    loss.backward()
                    if unit.grad is None or not torch.isfinite(unit.grad).all():
                        break
                    torch.nn.utils.clip_grad_norm_([unit], 5.)
                    optimizer.step()
                    with torch.no_grad():
                        unit.clamp_(0, 1)
            if best_q is None:
                results.append(dict(object=obj, normalized_parameter_error=1., failed=True))
                continue
            truth = torch.tensor(t['task']['q'], device=device)
            error = float(((best_q-truth).abs()/(hi-lo)).mean())
            theta = model.theta(best_q).detach().cpu().tolist()
            results.append(dict(object=obj, id=identity, normalized_parameter_error=error,
                E_relative_error=abs(theta[0]/float(t['theta'][0])-1),
                nu_absolute_error=abs(theta[1]-float(t['theta'][1])),
                estimated_parameters=theta, true_parameters=t['theta'].tolist(), best_loss=best_loss))
    finally:
        for parameter, required in zip(model.parameters(), original):
            parameter.requires_grad_(required)
    return dict(mean_normalized_parameter_error=float(np.mean([r['normalized_parameter_error'] for r in results])),
                records=results, optimizer_steps=steps, starts=1, original_mpm_validated=False)


@torch.no_grad()
def evaluate(model, replay, auxiliary, pairs, device, inverse_steps=12):
    model.eval()
    windows = selections(replay)
    rows = []
    for identity, start, phase in windows:
        batch, geo = temporal_batch(replay, [(identity, start)], 16, device)
        predicted = forecast(model.world, batch, geo, gradients=False)
        actual = batch['states'][:, 1:, :, :3]
        scale = geo['scale_m'][0]*1000
        errors = torch.linalg.vector_norm(predicted[..., :3]-actual, dim=-1)*scale
        static = torch.linalg.vector_norm(batch['states'][:, :1, :, :3]-actual, dim=-1)*scale
        elapsed = (batch['times'][:, 1:]-batch['times'][:, :1])/.03333333333333333
        cv = batch['states'][:, :1, :, :3]+batch['states'][:, :1, :, 3:6]*elapsed[:, :, None, None]
        cv_error = torch.linalg.vector_norm(cv-actual, dim=-1)*scale
        rows.append(dict(object=replay.trajectories[identity]['task']['object'], id=identity,
            start=start, phase=phase, position_mm=float(errors.mean()), single_step_mm=float(errors[:, :1].mean()),
            static_mm=float(static.mean()), constant_velocity_mm=float(cv_error.mean())))
    dynamics = summarize_windows(rows)
    dynamics['single_step_mm'] = float(np.mean([r['single_step_mm'] for r in rows]))
    dynamics['constant_velocity_mm'] = float(np.mean([r['constant_velocity_mm'] for r in rows]))
    scores, correlations, response_errors = [], [], []
    tau = model.tau.cpu().numpy()
    for obj, ids in sorted(auxiliary.by_object.items()):
        selected = [ids[i] for i in np.unique(np.linspace(0, len(ids)-1, min(24, len(ids))).astype(int))]
        for index, identity in enumerate(selected):
            b = auxiliary.batch([identity], tau, device)
            score, rho, _ = model.confidence(b['contexts'], b['q'])
            scores.append(float((score-b['score']).abs()[b['valid']].mean()))
            if b['correlation_valid'][0]:
                correlations.append(float((rho-b['rho']).abs().mean()))
            if index == 0:
                J = model.local_response(b['contexts'][0], b['q'][0], b['radii'][0])
                target = b['J'][0]
                error = (J-target).square().sum(0)/target.square().sum(0).clamp_min(model.tau.square())
                response_errors.append(float(error[b['valid'][0]].mean()))
    pair_rows = []
    for obj, choices in sorted(pairs.by_object.items()):
        a, b = choices[0]
        end = len(replay.trajectories[a]['state'])-1
        batch, geo = pairs.batch((a, b, end), device)
        predicted = forecast(model.world, batch, geo, gradients=False)
        # Full-history response is measured over interaction and release.
        phase = batch['phase'] > 0
        target = batch['states'][:, 1:, :, :3]
        actual = (target[1]-target[0])[phase]*geo['scale_m'][0]
        estimated = (predicted[1, ..., :3]-predicted[0, ..., :3])[phase]*geo['scale_m'][0]
        floor = (.003*batch['parameter_distance'].clamp_min(.1)).square()
        response = float((estimated-actual).square().mean()/actual.square().mean().clamp_min(floor))
        pair_rows.append(dict(object=obj, relative_response_mse=response, horizon=end))
    recovery = inverse_diagnostic(model, replay, device, inverse_steps) if inverse_steps else None
    pair_error = float(np.mean([r['relative_response_mse'] for r in pair_rows]))
    response_error = float(np.mean(response_errors))
    # Fixed scales and coefficients, not tuned using test objects.
    score = dynamics['mean_mm']/3 + .1*pair_error + .1*response_error + .1*(
        recovery['mean_normalized_parameter_error'] if recovery else 0.)
    return dict(score=score, dynamics=dynamics, sensitivity_mae=float(np.mean(scores)),
        correlation_mae=float(np.mean(correlations)) if correlations else None,
        local_response_relative_mse=response_error, paired_response_relative_mse=pair_error,
        paired_records=pair_rows, recovery=recovery, windows=rows)
