"""Audited fixed-state finite differences from the current four-parameter teacher.

The derivative conditions on the full intermediate simulator state. It does
not differentiate the state history and is not a full-rollout total derivative.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from .data import fps
from .io import read_json, write_json as write, json_value as clean, sha256 as sha, load_npz
from .labels import audit_columns, central_column
from .parameters import ParameterSpace

def select_windows(data, meta, phase_weights, per_phase=2, horizon=1):
    """Choose endpoints from observed metadata only; keep every chosen window."""
    if 'teacher_windows' in meta:
        windows = meta['teacher_windows']
        if not windows or any(w['start'] < 0 or w['end']-w['start'] != horizon or
                              int(data['frames'][w['index']]) != w['end'] for w in windows):
            raise ValueError('Invalid explicit teacher endpoints')
        for phase, weight in phase_weights.items():
            if not np.isclose(sum(w['weight'] for w in windows if w['phase']==phase), weight):
                raise ValueError('Explicit teacher windows change phase weights')
        return sorted(windows, key=lambda w:w['end'])
    rows = []
    for name, weight in phase_weights.items():
        eligible = []
        for i, frame in enumerate(data['frames']):
            phase = 'before' if frame < meta['clip_start'] else ('interaction' if frame <= meta['clip_end'] else 'release')
            if phase == name and frame >= horizon:
                eligible.append((i, int(frame)))
        if not eligible:
            raise ValueError(f'No eligible {name} window')
        ids = np.unique(np.linspace(0, len(eligible)-1, min(per_phase, len(eligible))).astype(int))
        for k in ids:
            i, frame = eligible[k]
            rows.append(dict(index=i, start=frame-horizon, end=frame, phase=name, weight=weight/len(ids)))
    return sorted(rows, key=lambda r: r['end'])


def generate(runtime, data, meta, space, spec, phase_weights, output, nominal_observer=None):
    """Start with a freshly initialized truth runtime and preserve all windows."""
    if runtime.frame != 0:
        raise ValueError('Teacher runtime must start at the original initial state')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    parameters, coordinate = space.parameters, space.coordinate
    windows = select_windows(data, meta, phase_weights, spec['windows_per_phase'], 1)
    eligible = np.flatnonzero(data['track_visible'][0] & np.isfinite(data['tracks'][0]).all(1))
    if not len(eligible):
        raise ValueError('No visible points in initial frame')
    ids = eligible[fps(data['tracks'][0, eligible], spec['points'])]
    binding, weights = data['binding'][ids], data['weights'][ids]
    starts = {w['start'] for w in windows}
    snapshots = {}
    if nominal_observer is not None:
        nominal_observer(runtime)
    if 0 in starts:
        snapshots[0] = runtime.snapshot()
    for frame in range(1, len(runtime.times)):
        check = runtime.advance(frame)
        if not check['valid']:
            raise RuntimeError(f'Invalid nominal rollout at frame {frame}: {check}')
        if nominal_observer is not None:
            nominal_observer(runtime)
        if frame in starts:
            snapshots[frame] = runtime.snapshot()
    def vector(w):
        p=runtime.positions(data['saved_ids'])
        x=data['tracks'][0,ids]+np.sum((p[binding]-data['initial'][binding])*weights[...,None],axis=1)
        visible=data['track_visible'][w['index'],ids]
        assert visible.any()
        return x[visible].ravel()*1000./math.sqrt(visible.sum())
    records=[]
    for qi,q in enumerate(spec['workpoints']):
        work=parameters(q);coarse=[];fine=[];physical=[];base_vectors=[];window_info=[]
        for wi,w in enumerate(windows):
            snap=snapshots[w['start']];digest=runtime.digest(snap)
            runtime.restore(snap,work);check=runtime.advance(w['end'])
            if not check['valid']:
                raise RuntimeError(f'Invalid nominal query {qi} at frame {w["end"]}: {check}')
            y0=vector(w);nominal=runtime.state_hashes();base_vectors.append(y0)
            columns=[];ok=np.ones(4,bool)
            for radius in spec['radii']:
                js=[]
                for j in range(4):
                    responses=[];offsets=[]
                    for sign in [-1,1]:
                        v=np.array(q);v[j]+=sign*radius;p=parameters(v)
                        runtime.restore(snap,p);check=runtime.advance(w['end']);ok[j]&=check['valid']
                        responses.append(vector(w) if check['valid'] else np.full_like(y0,np.nan))
                        offsets.append(abs(coordinate(p)[j]-coordinate(work)[j]))
                    col,_=central_column(y0,responses[1],responses[0],offsets[1],offsets[0]);js.append(col)
                columns.append(np.stack(js,axis=1))
            runtime.restore(snap,work);check=runtime.advance(w['end'])
            if not check['valid']:
                raise RuntimeError(f'Invalid nominal repeat {qi} at frame {w["end"]}: {check}')
            assert nominal==runtime.state_hashes() and digest==runtime.digest(snap),'Branch state contamination'
            coarse.append(columns[0]);fine.append(columns[1]);physical.append(ok)
            window_info.append(dict(**w,snapshot_digest=digest,nominal_hashes=nominal,zero_repeat_identical=True))
        a=np.concatenate([np.sqrt(w['weight'])*x for w,x in zip(windows,coarse)])
        b=np.concatenate([np.sqrt(w['weight'])*x for w,x in zip(windows,fine)])
        valid=np.stack(physical).all(0);audit=audit_columns(a,b,valid,spec['relative_column_tolerance'])
        failed=np.flatnonzero(~audit['valid']).tolist()
        # Prespecified finer-step audit for failed axes, retaining every window.
        refined=[]
        if failed:
            a2=a.copy();b2=b.copy()
            for j in failed:
                cols=[];ok=True
                for radius in spec['refinement_radii']:
                    pieces=[]
                    for wi,w in enumerate(windows):
                        responses=[];offsets=[]
                        for sign in [-1,1]:
                            v=np.array(q);v[j]+=sign*radius;p=parameters(v)
                            runtime.restore(snapshots[w['start']],p);check=runtime.advance(w['end']);ok &= check['valid']
                            responses.append(vector(w) if check['valid'] else np.full_like(base_vectors[wi],np.nan))
                            offsets.append(abs(coordinate(p)[j]-coordinate(work)[j]))
                        col,_=central_column(base_vectors[wi],responses[1],responses[0],offsets[1],offsets[0]);pieces.append(np.sqrt(w['weight'])*col)
                    cols.append(np.concatenate(pieces))
                a2[:,j]=cols[0];b2[:,j]=cols[1];valid[j]=ok;refined.append(j)
            audit2=audit_columns(a2,b2,valid,spec['relative_column_tolerance'])
            for j in failed:a[:,j]=a2[:,j];b[:,j]=b2[:,j]
            audit=audit2
        h=b.T@b;norm=np.sqrt(np.diag(h));correlation=h/np.maximum(norm[:,None]*norm[None,:],1e-20)
        path=out/f'workpoint_{qi}.npz'
        np.savez_compressed(path,J_coarse=a,J_fine=b,S=audit['S'],valid=audit['valid'],correlation=correlation,
            starts=[w['start'] for w in windows],ends=[w['end'] for w in windows],q=q,point_ids=data['point_ids'][ids])
        row=dict(workpoint=qi,q=q,**audit,correlation=correlation,
            refined_axes=refined,path=path.name,sha256=sha(path),windows=window_info)
        records.append(row);write(out/'progress.json',clean(dict(workpoints=records,completed=len(records),total=len(spec['workpoints']))))
        print('TEACHER_POINT',qi,audit['S'],audit['valid'],flush=True)
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--truth', default='center')
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    space = ParameterSpace.from_config(config)
    truth = next(t for t in config['truths'] if t['name'] == args.truth)
    from .adapters.empm import BranchRuntime
    runtime = BranchRuntime(args.episode, space.parameters(truth['q']), args.padding,
                            .0001, args.device, args.runtime, args.workspace)
    records = generate(runtime, load_npz(args.observation), read_json(args.metadata),
                       space, config['teacher'], config['fitting']['phase_weights'], args.output)
    write(args.output / 'manifest.json', dict(episode=args.episode, truth=args.truth,
          records=records, source_hashes=runtime.source_hashes,
          config_sha256=sha(args.config), observation_sha256=sha(args.observation)))


if __name__ == '__main__':
    main()
