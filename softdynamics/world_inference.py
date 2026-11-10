"""Export an initial-state/action query and run the learned asset without EMPM."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .io import load_npz, read_json, sha256, write_json
from .world_model import normalize_state, physical_state, rollout
from .world_training import load_checkpoint


def export_query(root, task_id, output):
    root, output = Path(root), Path(output)
    plan = read_json(root/'plan.json')
    folder = root/'trajectories'/task_id
    record = read_json(folder/'complete.json')
    if not record['valid'] or record['diagnostic_truncation']:
        raise ValueError('Query requires a valid complete teacher trajectory')
    if sha256(folder/'trajectory.npz') != record['payload_sha256']:
        raise ValueError('Teacher payload changed')
    fit_identity = sha256(root/'first_contribution.json') if 'handoff' in plan else ''
    if fit_identity and record.get('first_contribution_sha256') != fit_identity:
        raise ValueError('Teacher material does not match first-contribution selection')
    data = load_npz(folder/'trajectory.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, initial_state=data['state'][0], pads=data['pads'],
                        eligibility=data['eligibility'], times=data['times'],
                        origin_m=data['origin_m'], scale_m=data['scale_m'],
                        nodes_sha256=plan['nodes_sha256'], asset_sha256=record['source_hashes']['asset'],
                        first_contribution_sha256=fit_identity)


def predict(checkpoint, query, output, device='cuda:0'):
    checkpoint, output = Path(checkpoint), Path(output)
    manifest = read_json(checkpoint.parent/'manifest.json')
    data = load_npz(query)
    for identity in ('nodes_sha256', 'asset_sha256'):
        if str(data[identity]) != manifest[identity]:
            raise ValueError(f'Wrong fixed asset query: {identity}')
    fit_identity = manifest.get('first_contribution_sha256')
    if fit_identity and str(data.get('first_contribution_sha256', '')) != fit_identity:
        raise ValueError('Wrong first-contribution material selection for this asset')
    model, blob = load_checkpoint(checkpoint, device)
    if blob['manifest_sha256'] != sha256(checkpoint.parent/'manifest.json'):
        raise ValueError('Checkpoint manifest changed')
    N = len(model.canonical); T = len(data['times'])
    if data['initial_state'].shape != (N, 24) or data['pads'].shape != (T, 4, 13) or T < 2:
        raise ValueError('Expected one physical initial state and at least two controller endpoints')
    if data['eligibility'].shape != (N, 4) or (np.diff(data['times']) <= 0).any():
        raise ValueError('Invalid eligibility or time sequence')
    if any(not np.isfinite(data[k]).all() for k in ('initial_state', 'pads', 'times', 'origin_m', 'scale_m')):
        raise ValueError('Non-finite query')
    scale = float(data['scale_m']); origin = data['origin_m']
    if scale <= 0:
        raise ValueError('Non-positive object scale')
    initial = normalize_state(data['initial_state'], origin, scale)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)[None]
    torch.set_num_threads(1)
    with torch.no_grad():
        predicted = rollout(model, tensor(initial), tensor(data['pads']), tensor(data['eligibility']),
                            tensor(data['times']), tensor((-.01-origin[1])/scale))[0].cpu().numpy()
    physical = physical_state(predicted, origin, scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predicted_state=physical, times=data['times'][1:])
    write_json(output.with_suffix('.json'),dict(valid=bool(np.isfinite(physical).all()),
               checkpoint_sha256=sha256(checkpoint), query_sha256=sha256(query),
               first_contribution_sha256=fit_identity,
               frames=T-1, nodes=N, future_teacher_states_in_query=False,
               simulator_called=False, tactile_input=False))
    return physical


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['export-query', 'predict'])
    p.add_argument('--root', type=Path); p.add_argument('--task')
    p.add_argument('--checkpoint', type=Path); p.add_argument('--query', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.mode == 'export-query': export_query(a.root, a.task, a.output)
    else: predict(a.checkpoint, a.query, a.output, a.device)


if __name__ == '__main__': main()
