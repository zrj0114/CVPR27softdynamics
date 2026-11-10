"""Test a frozen learned action direction in the actual simulator.

This is downstream validation, never a response/Jacobian training target. All
step sizes are declared before running; failures and worsening steps are kept.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .io import load_npz, read_json, sha256, write_json
from .shared_world_control import PhysicalControl, bounded_pose_sequence, pose_rollout, goal_cost, anchored_positions
from .shared_world_training import load_model


@torch.no_grad()
def audit(root, source, output, device='cuda:0', scales=(0., .015625, .03125, .0625, .125, .25, .5, 1.)):
    root, source, output = Path(root), Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        raise ValueError('Action audit is frozen')
    case = read_json(source/'case.json')
    checkpoint = case['checkpoint']
    if sha256(checkpoint) != case['checkpoint_sha256']:
        raise ValueError('Frozen world checkpoint changed')
    model, bundle = load_model(checkpoint, device)
    model.requires_grad_(False); torch.set_num_threads(2)
    physical = PhysicalControl(root, case['object'], case['episode'], case['horizon'], device)
    if physical.snapshot_sha256 != case['snapshot_sha256']:
        raise ValueError('Physical initial snapshot differs from original control comparison')
    if physical.parameters != case['parameters']:
        raise ValueError('Physical material differs from original control comparison')
    trajectories = {}
    for name in ['hold', 'reference']:
        record = read_json(source/name/'complete.json')
        if not record['valid'] or record['snapshot_sha256'] != physical.snapshot_sha256:
            raise ValueError('Invalid physical reference')
        if sha256(source/name/'trajectory.npz') != record['trajectory_sha256']:
            raise ValueError('Changed physical reference payload')
        trajectories[name] = load_npz(source/name/'trajectory.npz')
    initial, pads0, theta, geometry, times = physical.tensors(device, model.direct_displacement)
    goal = torch.tensor((trajectories['reference']['state'][-1, :, :3]-physical.origin)/physical.scale, device=device)[None]
    anchor = torch.tensor((trajectories['hold']['state'][..., :3]-physical.origin)/physical.scale, device=device)[None]
    raw = torch.tensor(load_npz(source/'world_prediction.npz')['raw'], device=device)
    baseline_pads = bounded_pose_sequence(pads0, raw*0, case['horizon'], geometry['scale_m'])[0]
    baseline = pose_rollout(model, initial, baseline_pads, theta, geometry, times)[..., :3]
    protocol = dict(source=str(source), source_case_sha256=sha256(source/'case.json'),
        source_actions_sha256=sha256(source/'world_prediction.npz'), checkpoint_sha256=sha256(checkpoint),
        object=case['object'], episode=case['episode'], snapshot_sha256=physical.snapshot_sha256,
        scales=list(scales), scale_definition='Multiply total translation, rotation and aperture offsets by alpha',
        response_training_labels=False, task='Same terminal surface goal as the source control benchmark',
        prefix_seconds=physical.prefix_s, created_at=time.time())
    write_json(output/'protocol.json', protocol)
    rows = []; started = time.monotonic()
    for scale in scales:
        pads, _ = bounded_pose_sequence(pads0, raw, case['horizon'], geometry['scale_m'],
            translation_m=.012*scale, rotation_rad=.08*scale, aperture_m=.006*scale)
        prediction = pose_rollout(model, initial, pads, theta, geometry, times)
        corrected = anchored_positions(prediction[..., :3], baseline, anchor)
        predicted_mse, predicted_mm = goal_cost(corrected[:, -1], goal, geometry)
        states, record = physical.rollout(pads[0].cpu().numpy(), output/('scale_'+str(scale)))
        mse = mm = None
        if record['valid']:
            actual = torch.tensor((states[-1, :, :3]-physical.origin)/physical.scale, device=device)[None]
            actual_mse, actual_mm = goal_cost(actual, goal, geometry)
            mse, mm = float(actual_mse), float(actual_mm)
        rows.append(dict(scale=scale, physical_valid=record['valid'], physical_mse=mse, physical_mm=mm,
            predicted_mse=float(predicted_mse), predicted_mm=float(predicted_mm),
            physical_seconds=record['elapsed_s'], error=record['error']))
        write_json(output/'progress.json', dict(rows=rows, elapsed_s=time.monotonic()-started))
    write_json(output/'result.json', dict(**protocol, rows=rows, elapsed_s=time.monotonic()-started))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    audit(a.root, a.source, a.output, a.device)


if __name__ == '__main__':
    main()
