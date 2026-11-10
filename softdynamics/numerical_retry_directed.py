"""Explicit static grid allocation with directed whole-cell boundary padding."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .io import read_json,write_json,sha256


@contextmanager
def fixed_grid(size,lower_cells):
    from .adapters import pad_empm
    if not isinstance(size,int) or size%16 or not 64<=size<=384:
        raise ValueError('Unsupported fixed grid size')
    lower=np.asarray(lower_cells)
    if lower.shape!=(3,) or not np.all(lower==np.floor(lower)) or np.any(lower<0):
        raise ValueError('Expected three nonnegative whole-cell offsets')
    original=pad_empm.simulation_domain
    def expanded(initial,poses,valid,dx,*args,**kwargs):
        base,shift=original(initial,poses,valid,dx,*args,**kwargs)
        if size<base or np.any(lower>size-base):
            raise ValueError('Directed retry must contain the entire original grid')
        return size,shift+(lower*dx).astype(shift.dtype)
    pad_empm.simulation_domain=expanded
    try:
        yield
    finally:
        pad_empm.simulation_domain=original


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--task',required=True)
    a=parser.parse_args();plan=read_json(a.root/'plan.json');task=next(t for t in plan['tasks'] if t['id']==a.task)
    policy=task['numerical_retry'];source_hash=sha256(Path(__file__))
    if task['dtmax']!=policy['original_dtmax']*policy['timestep_factor']:
        raise ValueError('Timestep disagrees with numerical policy')
    from .elastic_data import collect
    with fixed_grid(policy['grid_size'],policy['lower_padding_cells']):
        result=collect(a.root,a.task)
    result['numerical_retry']=dict(policy,execution_adapter_sha256=source_hash)
    write_json(a.root/'trajectories'/a.task/'complete.json',result)
    if not result['valid']:raise SystemExit(2)


if __name__=='__main__':main()
