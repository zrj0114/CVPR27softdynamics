"""Continuous, model-assisted CMA proposals with full-physics fitness.

The surrogate preselects a population; only complete support simulations enter
CMA.tell. Four confidence columns remain four separate parameter objectives.
Window resets limit surrogate error propagation without dropping scored frames.
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cma_search import BlockCMA, weighted_loss
from .io import read_json, sha256, write_json
from .parameters import PARAMETERS
from .shared_world_inverse import RealEpisode
from .shared_world_material_windows import WindowedRealEpisode
from .shared_world_model import physical_parameters, recurrent_channels
from .shared_world_training import load_model


class PreselectedBlockCMA(BlockCMA):
    """Checkpoint both the large sample pool and its selected physical population."""
    def __init__(self, *args, surrogate_population=16, **kwargs):
        super().__init__(*args, **kwargs)
        if surrogate_population < self.population:
            raise ValueError('Surrogate pool must contain the physical population')
        self.surrogate_population = surrogate_population
        self.surrogate_candidates = 0

    def ask(self):
        p = super().ask()
        if p is None or 'pool_q' in p:
            return p
        extra = self.es.ask(number=self.surrogate_population-self.population) if self.surrogate_population > self.population else []
        x = p['x'] + extra
        q = np.tile(self.anchor, (len(x), 1))
        q[:, self.block % 4] = np.clip(np.asarray(x)[:, 0], 0., 1.)*.05-.025
        p.update(pool_x=x, pool_q=q, preselected=False)
        return p

    def preselect(self, support_losses, provenance=None):
        if self.pending is None or self.pending.get('preselected'):
            raise ValueError('An unselected pending pool is required')
        p, j = self.pending, self.block % 4
        losses = np.asarray(support_losses, float)
        if losses.shape != (self.surrogate_population, len(self.confidence)):
            raise ValueError('Expected one surrogate support vector per pool candidate')
        costs = np.array([weighted_loss(row, self.confidence[:, j]) for row in losses])
        # Invalid predictions do not remove a physical evaluation from the
        # budget. Stable ties/all-invalid pools retain the original CMA samples.
        indices = np.argsort(costs, kind='stable')[:self.population]
        p.update(x=[p['pool_x'][i] for i in indices], q=p['pool_q'][indices].copy(),
                 preselected=True, selected_indices=indices, surrogate_costs_mm=costs,
                 surrogate_support_losses_mm=losses, preselection_provenance=provenance)
        self.surrogate_candidates += self.surrogate_population
        return p

    def tell(self, candidate_losses, incumbent_losses):
        if self.pending is None or not self.pending.get('preselected'):
            raise ValueError('Surrogate preselection must precede physical tell')
        p = self.pending
        metadata = {k: p[k] for k in ('pool_q', 'selected_indices', 'surrogate_costs_mm',
                    'surrogate_support_losses_mm', 'preselection_provenance')}
        row = super().tell(candidate_losses, incumbent_losses)
        row.update(**metadata, surrogate_candidate_vectors=self.surrogate_candidates,
                   surrogate_episode_evaluations=self.surrogate_candidates*len(self.confidence),
                   cma_fitness_source='Full original physical-engine support rollouts')
        return row


def batched_window_reference_rollout(model, reference, pads, theta, geometry, times,
                                     horizon=32, batch_windows=8, predictions=None):
    """Cover every frame with bounded recursion and identical baseline batching.

    Reference predictions are detached nominal model outputs, not learned
    response/Jacobian labels. Window start states come only from fixed support
    simulation. Different windows never share a recurrent state.
    """
    if (reference.shape[0] != 1 or reference.shape[1] != pads.shape[1]
            or times.shape != reference.shape[:2] or horizon < 1 or batch_windows < 1):
        raise ValueError('Invalid window reference inputs')
    length, channels = reference.shape[1], recurrent_channels(model)
    starts = list(range(0, length-1, horizon))
    if predictions is not None and len(predictions) != (len(starts)+batch_windows-1)//batch_windows:
        raise ValueError('Wrong nominal window batching')
    result = reference[0].detach().clone()
    nominal = []
    for group, offset in enumerate(range(0, len(starts), batch_windows)):
        indices = torch.tensor(starts[offset:offset+batch_windows], device=reference.device)
        size = len(indices)
        geo = {k: v.expand(size, *v.shape[1:]) for k, v in geometry.items()}
        material = theta.expand(size, -1)
        state = reference[0, indices].detach()
        values = []
        for step in range(min(horizon, length-1-int(indices.min()))):
            frame = (indices+step).clamp_max(length-2)
            valid = indices+step < length-1
            args = (pads[0, frame], pads[0, frame+1], material, geo, times[0, frame+1]-times[0, frame])
            if predictions is None:
                # Use exactly the same batch dimensions as candidate prediction.
                raw = model(reference[0, frame].detach(), *args)
                values.append(raw[..., :channels].detach())
            else:
                raw = model(state, *args)
                dynamic = reference[0, frame+1, :, :channels].detach() + (
                    raw[..., :channels]-predictions[group][step].detach())
                state = torch.cat((dynamic, raw[..., channels:]), -1)
                result = result.index_copy(0, frame[valid]+1, state[valid])
        if predictions is None:
            nominal.append(torch.stack(values))
    return nominal if predictions is None else result


class CMARealEpisode(WindowedRealEpisode):
    def __init__(self, *args, batch_windows=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_windows = batch_windows

    @torch.no_grad()
    def set_physical_anchor(self, model, theta, physical_root, q):
        if not model.direct_displacement:
            raise ValueError('CMA references require observed-velocity dynamics')
        self.reference_predictions = batched_window_reference_rollout(model, self.window_starts,
            self.pads, theta[None], self.geometry, self.times, self.window_horizon, self.batch_windows)
        reproduced = self.predict(model, theta, gradients=False)
        channels = recurrent_channels(model)
        if not torch.equal(reproduced[..., :channels], self.window_starts[0, ..., :channels]):
            raise ValueError('Nominal windows do not reproduce physical recurrent states exactly')
        RealEpisode.set_physical_anchor(self, model, theta, physical_root, q)

    def predict(self, model, theta, gradients=False):
        if not hasattr(self, 'reference_predictions'):
            raise ValueError('Initialize nominal window predictions first')
        return batched_window_reference_rollout(model, self.window_starts, self.pads, theta[None],
            self.geometry, self.times, self.window_horizon, self.batch_windows, self.reference_predictions)


@torch.no_grad()
def evaluate_pool(request_path, output, device='cuda:0'):
    """Support observations only; return surrogate scores, never physical fitness."""
    request_path, output = Path(request_path), Path(output)
    if output.exists():
        record = read_json(output)
        if record['request_sha256'] != sha256(request_path):
            raise ValueError('Changed completed surrogate request')
        return record
    r = read_json(request_path)
    if sha256(r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Changed frozen world-model checkpoint')
    if sha256(Path(r['data_root'])/'plan.json') != r['data_plan_sha256']:
        raise ValueError('Changed world data plan')
    if sha256(r['benchmark_plan']) != r['protocol_sha256'] or r['query_observations_used']:
        raise ValueError('Changed benchmark protocol or invalid query access')
    physical = read_json(Path(r['physical_reference'])/'plan.json')
    if sha256(Path(r['physical_reference'])/'plan.json') != r['physical_reference_sha256']:
        raise ValueError('Changed fixed physical reference')
    item = next(o for o in physical['objects'] if o['object'] == r['object'])
    if r['support'] != item['support'] or set(r['support']) & set(item['query']):
        raise ValueError('Surrogate request includes query observations or changed support')
    q = np.asarray(r['pool_q'], float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all() or (abs(q) > .025).any():
        raise ValueError('Invalid continuous surrogate pool')
    torch.set_num_threads(2)
    model, bundle = load_model(r['checkpoint'], device)
    if r['object'] in bundle['training_objects']:
        raise ValueError('World model was trained on a downstream evaluation object')
    model.requires_grad_(False); model.eval()
    base = torch.tensor([physical['config']['baseline'][k] for k in PARAMETERS], device=device, dtype=torch.float32)
    scales = torch.tensor(physical['config']['coordinate_scales'], device=device, dtype=torch.float32)
    nominal_theta = physical_parameters(torch.tensor(physical['incumbent_q'], device=device, dtype=torch.float32), base, scales)
    losses = np.full((len(q), len(r['support'])), np.inf)
    anchors, failures = [], []
    started = time.monotonic()
    for column, episode in enumerate(r['support']):
        e = CMARealEpisode(r['data_root'], r['object'], episode, device, r['state_reference'],
                           r['physical_reference'], r['window_horizon'], batch_windows=r['batch_windows'])
        e.set_physical_anchor(model, nominal_theta, r['physical_reference'], physical['incumbent_q'])
        nominal_loss = float(e.loss(e.predict(model, nominal_theta)))
        from .recovery_data import trajectory_folder
        source_score = trajectory_folder(r['physical_reference'], r['object'], episode, physical['incumbent_q'])/'scores.json'
        baseline_score = None
        if source_score.exists():
            baseline_score = read_json(source_score)['conditions']['real']['objective_mm']
            if not np.isfinite(nominal_loss) or abs(nominal_loss-baseline_score) > .001:
                raise ValueError('Surrogate baseline observation score differs from full physics')
        anchors.append(dict(episode=episode, dense_sha256=e.anchor['complete_sha256'],
            state_sha256=e.state_anchor_record['trajectory_sha256'],
            nominal_loss_mm=nominal_loss, physical_reference_loss_mm=baseline_score,
            physical_seconds=e.anchor['physical_seconds'], state_capture_seconds=e.state_anchor_record['elapsed_s']))
        for candidate, coordinates in enumerate(q):
            theta = physical_parameters(torch.tensor(coordinates, dtype=torch.float32, device=device), base, scales)
            states = e.predict(model, theta)
            if torch.isfinite(states).all():
                value = float(e.loss(states))
                if np.isfinite(value):
                    losses[candidate, column] = value
            if not np.isfinite(losses[candidate, column]):
                failures.append(dict(candidate=candidate, episode=episode, reason='nonfinite_surrogate'))
            write_json(output.with_suffix('.progress.json'), dict(pid=os.getpid(),
                episode=episode, candidate=candidate+1, pool=len(q), completed_support=column,
                support=len(r['support']), elapsed_s=time.monotonic()-started))
        print('SURROGATE_SUPPORT', r['object'], episode, 'pool', len(q), 'seconds', time.monotonic()-started, flush=True)
        del e
    result = dict(schema='softdynamics.world-cma-pool.v1', request_sha256=sha256(request_path),
        object=r['object'], support=r['support'], pool_q=q, support_losses_mm=losses, failures=failures,
        elapsed_s=time.monotonic()-started, checkpoint_sha256=r['checkpoint_sha256'],
        query_observations_used=False, world_weights_frozen=True, recorded_actions_fixed=True,
        scored_frames='All original observation frames; no temporal subsampling',
        max_recurrent_horizon=r['window_horizon'], anchor_provenance=anchors,
        physical_fitness=False, response_supervision=False)
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    evaluate_pool(a.request, a.output, a.device)


if __name__ == '__main__':
    main()
