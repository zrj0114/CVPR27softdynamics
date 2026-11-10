"""Collect a complete same-policy label using the validated spatial refinement."""
import argparse
from pathlib import Path
from .io import read_json, write_json, sha256
from .spatial_grid import runtime

def collect(root, identity):
    root = Path(root)
    from .elastic_data import collect as original
    task = next(t for t in read_json(root / 'plan.json')['tasks'] if t['id'] == identity)
    policy = task['numerical_retry']
    if task['dtmax'] != policy['original_dtmax'] * policy['timestep_factor']:
        raise ValueError('Timestep differs from the recorded numerical policy')
    if read_json(root / 'acceleration.json').get('transfer_equivalence'):
        raise ValueError('Spatial refinement must retain its actual source provenance')
    with runtime(policy):
        result = original(root, identity)
    result['numerical_retry'] = dict(policy, wrapper_sha256=sha256(Path(__file__)))
    write_json(root / 'trajectories' / identity / 'complete.json', result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--task', required=True); a = p.parse_args()
    if not collect(a.root, a.task)['valid']: raise SystemExit(2)
