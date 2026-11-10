"""Isolated numerical retries with explicit, episode-wide grid/timestep settings."""
import argparse
from contextlib import contextmanager
from pathlib import Path

from .io import read_json,write_json,sha256


@contextmanager
def fixed_upper_grid(size):
    """Extend positive grid bounds while preserving the initial world lattice."""
    from .adapters import pad_empm
    if size is None:
        yield
        return
    if not isinstance(size,int) or size%16 or not 64<=size<=384:
        raise ValueError('Expected a supported whole-cell cubic allocation')
    original=pad_empm.simulation_domain
    def extended(*args,**kwargs):
        base,shift=original(*args,**kwargs)
        if size<base:
            raise ValueError('Numerical retry cannot shrink the original domain')
        return size,shift
    pad_empm.simulation_domain=extended
    try:
        yield
    finally:
        pad_empm.simulation_domain=original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--task',required=True)
    a=parser.parse_args()
    plan=read_json(a.root/'plan.json')
    task=next(t for t in plan['tasks'] if t['id']==a.task)
    policy=task.get('numerical_retry',{})
    expected=float(policy.get('original_dtmax',task['dtmax']))*float(policy.get('timestep_factor',1.))
    if expected!=task['dtmax'] or task['dtmax']<=0:
        raise ValueError('Task timestep disagrees with the explicit retry policy')
    from .elastic_data import collect
    with fixed_upper_grid(policy.get('grid_size')):
        result=collect(a.root,a.task)
    result['numerical_retry']=dict(policy,execution_adapter_sha256=sha256(Path(__file__)))
    write_json(a.root/'trajectories'/a.task/'complete.json',result)
    if not result['valid']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
