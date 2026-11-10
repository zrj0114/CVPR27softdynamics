"""Compare world checkpoints on one immutable set of physical trajectories.

The same window coordinates, targets and baseline velocities are used even when
checkpoints have different state conventions. Newly collected trajectories cannot
silently change an in-progress comparison.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .io import read_json, sha256, write_json
from .shared_world_model import DT_REF, rollout
from .shared_world_training import Replay, load_model, prediction_summary


def freeze(root, output, split='validation', windows=8, full=True):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Evaluation manifest is immutable')
    replay = Replay(root, [split]); replay.refresh()
    trajectories = {}; cases = []
    for identity, t in sorted(replay.trajectories.items()):
        trajectories[identity] = dict(object=t['task']['object'],
            trajectory_sha256=t['record']['trajectory_sha256'], frames=len(t['state']))
        for horizon in replay.plan['config']['evaluation_horizons']:
            if len(t['state']) <= horizon:
                continue
            for start in np.unique(np.linspace(0, len(t['state'])-horizon-1, windows).astype(int)):
                cases.append(dict(id=identity, start=int(start), horizon=horizon, steps=horizon))
        if full and t['task']['variant'] == 0:
            cases.append(dict(id=identity, start=0, horizon='full', steps=len(t['state'])-1))
    if not cases:
        raise ValueError('No valid evaluation trajectories')
    manifest = dict(schema='softdynamics.shared-world-evaluation.v1',
        root=str(root.resolve()), data_plan_sha256=sha256(root/'plan.json'), split=split,
        trajectories=trajectories, cases=cases, failures_at_freeze=replay.failures,
        created_at=time.time(), selection='All valid trajectories available at freeze time')
    write_json(output, manifest)
    return manifest


def validate_manifest(replay, manifest):
    if sha256(replay.root/'plan.json') != manifest['data_plan_sha256']:
        raise ValueError('Changed data plan')
    for identity, expected in manifest['trajectories'].items():
        if identity not in replay.trajectories:
            raise ValueError('Missing frozen evaluation trajectory: '+identity)
        t = replay.trajectories[identity]
        if t['record']['trajectory_sha256'] != expected['trajectory_sha256'] or len(t['state']) != expected['frames']:
            raise ValueError('Changed frozen evaluation trajectory: '+identity)


@torch.no_grad()
def evaluate(manifest_path, checkpoint_path, output, device='cuda:0'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'evaluation.json').exists():
        raise ValueError('Evaluation is immutable')
    torch.set_num_threads(2)
    manifest = read_json(manifest_path)
    model, bundle = load_model(checkpoint_path, device)
    if bundle['data_plan_sha256'] != manifest['data_plan_sha256']:
        raise ValueError('Checkpoint and evaluation use different data plans')
    # Keep instantaneous velocities for a baseline common to all checkpoints.
    replay = Replay(manifest['root'], [manifest['split']]); replay.refresh()
    validate_manifest(replay, manifest)
    objects = sorted({v['object'] for v in manifest['trajectories'].values()})
    if set(objects) & set(bundle['training_objects']):
        raise ValueError('Held-out object leakage')
    observed = {}
    for identity in manifest['trajectories']:
        t = replay.trajectories[identity]
        velocity = np.zeros_like(t['state'][..., 3:6])
        velocity[1:] = np.diff(t['state'][..., :3], axis=0)/(np.diff(t['times'])[:, None, None]/DT_REF)
        if 'initial_observed_velocity_m_s' in t:
            velocity[0] = t['initial_observed_velocity_m_s']*DT_REF/float(t['scale_m'])
        observed[identity] = velocity
    rows = []; started = time.monotonic()
    for index, case in enumerate(manifest['cases']):
        identity, start, horizon = case['id'], case['start'], case['steps']
        t = replay.trajectories[identity]
        b, g = replay.batch([(identity, start)], horizon, device)
        initial = b['states'][:, 0].clone()
        instantaneous = initial[..., 3:6].clone()
        backward = torch.as_tensor(observed[identity][start][None], device=device)
        if model.direct_displacement:
            initial[..., 3:6] = backward
        prediction = rollout(model, initial, b['pads'], b['theta'], g, b['times'])
        truth = b['states'][:, 1:, :, :3]
        x = initial[:, None, :, :3]
        elapsed = b['times'][:, 1:, None, None]/DT_REF
        methods = [('world', prediction[..., :3]), ('persistence', x.expand_as(truth)),
            ('constant_velocity_instantaneous', x+instantaneous[:, None]*elapsed),
            ('constant_velocity_observed', x+backward[:, None]*elapsed)]
        for name, positions in methods:
            error = torch.linalg.vector_norm(positions-truth, dim=-1)*g['scale_m'][:, None, None]*1000
            rows.append(dict(id=identity, object=t['task']['object'], variant=t['task']['variant'],
                horizon=case['horizon'], start=start, steps=horizon, method=name,
                mean_mm=float(error.mean()), endpoint_mm=float(error[:, -1].mean()),
                finite=bool(torch.isfinite(error).all())))
        if case['horizon'] == 'full':
            np.savez_compressed(output/(identity.replace('/', '__')+'.npz'),
                prediction=prediction.cpu().numpy(), origin_m=t['origin_m'], scale_m=t['scale_m'])
        write_json(output/'progress.json', dict(completed=index+1, total=len(manifest['cases']),
            last_case=case, elapsed_s=time.monotonic()-started))
    result = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        checkpoint_step=bundle['step'], manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
        split=manifest['split'], objects=objects, rows=rows, summary=prediction_summary(rows),
        world_input_velocity='observed_backward_difference' if model.direct_displacement else 'MPM_instantaneous',
        common_baselines=True, elapsed_s=time.monotonic()-started,
        training_objects=bundle['training_objects'], failures_at_freeze=manifest['failures_at_freeze'])
    write_json(output/'evaluation.json', result)
    return result


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('freeze'); a.add_argument('--root', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--split', choices=['validation', 'test'], default='validation')
    a.add_argument('--windows', type=int, default=8); a.add_argument('--no-full', action='store_true')
    a = sub.add_parser('evaluate'); a.add_argument('--manifest', type=Path, required=True)
    a.add_argument('--checkpoint', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.mode == 'freeze':
        freeze(a.root, a.output, a.split, a.windows, not a.no_full)
    else:
        evaluate(a.manifest, a.checkpoint, a.output, a.device)


if __name__ == '__main__':
    main()
