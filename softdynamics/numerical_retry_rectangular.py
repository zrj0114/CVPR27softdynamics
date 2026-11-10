"""Collect complete episode-consistent labels on the validated rectangular grid."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from .io import read_json, write_json, sha256
from .rectangular_grid import runtime


def collect(root, identity):
    from . import elastic_data, shared_world_observation
    root = Path(root)
    task = next(t for t in read_json(root/'plan.json')['tasks'] if t['id']==identity)
    policy = task['numerical_retry']
    if task['dtmax'] != policy['original_dtmax'] * policy['timestep_factor']:
        raise ValueError('Numerical timestep differs from the recorded episode policy')
    if read_json(root/'acceleration.json').get('transfer_equivalence'):
        raise ValueError('Rectangular execution must record its actual transfer source without an old-grid alias')
    original = shared_world_observation.padded_simulation_grid
    @contextmanager
    def fixed_allocation(padding, lower_cells=None):
        if padding or lower_cells is not None:
            raise ValueError('Implicit coordinate-shifting retries are disabled for rectangular storage')
        yield
    shared_world_observation.padded_simulation_grid = fixed_allocation
    try:
        with runtime(policy['grid_shape'], policy['grid_index_offset_cells']):
            result = elastic_data.collect(root, identity)
    finally:
        shared_world_observation.padded_simulation_grid = original
    result['numerical_retry'] = dict(policy, wrapper_sha256=sha256(Path(__file__)))
    write_json(root/'trajectories'/identity/'complete.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',required=True)
    a=p.parse_args()
    if not collect(a.root,a.task)['valid']:
        raise SystemExit(2)
