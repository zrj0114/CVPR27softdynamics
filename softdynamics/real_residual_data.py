"""Pin measured observations and fixed bindings; never manufacture GT nodes.

The NPZ packet contract also accepts the initial-only real comparison export.
MPM reference trajectories in its receipt are not used as real-data targets.
Object splits come from the frozen base checkpoint, not user-supplied labels.
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch

from .io import load_npz, read_json, write_json, sha256


ARRAYS = ('state0', 'x0', 'origin', 'scale', 'canonical', 'neighbors', 'surface_mask',
          'pads', 'times', 'P', 'track0', 'tracks', 'visible', 'surface0',
          'surface_binding', 'surface_weights', 'reference', 'reference_valid', 'phase')


def validate_packet(data):
    missing = set(ARRAYS)-set(data)
    if missing:
        raise ValueError('Missing real-observation arrays: '+str(sorted(missing)))
    n, t, p, s = len(data['x0']), len(data['times']), len(data['track0']), len(data['surface0'])
    shapes = dict(state0=(n,24), x0=(n,3), origin=(3,), scale=(), times=(t,), canonical=(n,3), surface_mask=(n,),
        pads=(t,4,13), P=(p,n), track0=(p,3), tracks=(t,p,3), visible=(t,p),
        surface0=(s,3), phase=(t,))
    # Binding stencil width is an export choice, not a learned correspondence.
    shapes['surface_weights'] = data['surface_binding'].shape
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError('Incorrect shape for '+key)
    if n < 2 or t < 2 or p < 1 or s < 1 or data['surface_binding'].ndim != 2 or data['surface_binding'].shape[0] != s:
        raise ValueError('Empty geometry/timeline or invalid surface bindings')
    if data['reference'].ndim != 3 or data['reference'].shape[0] != t or data['reference'].shape[-1] != 3 or data['reference_valid'].shape != data['reference'].shape[:2]:
        raise ValueError('Incorrect observed surface layout')
    if data['neighbors'].ndim != 2 or data['neighbors'].shape[0] != n or data['neighbors'].shape[1] < 1:
        raise ValueError('Incorrect graph layout')
    for key in ('neighbors', 'surface_binding'):
        a = data[key]
        if a.size == 0 or a.dtype.kind not in 'iu' or a.min() < 0 or a.max() >= n:
            raise ValueError('Invalid node indices: '+key)
    scale = float(data['scale'])
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(data['times']).all() or not (np.diff(data['times']) > 0).all():
        raise ValueError('Invalid physical scale or timestamps')
    for key in ('visible', 'reference_valid', 'surface_mask'):
        if data[key].dtype != bool:
            raise ValueError('Boolean mask required: '+key)
    for key in ('state0','x0','origin','canonical','pads','P','track0','surface0','surface_weights'):
        if not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite causal input/binding: '+key)
    active = data['pads'][...,12]
    if not np.isin(active,[0.,1.]).all():
        raise ValueError('Pad existence flag must be binary')
    rotations = data['pads'][... ,3:12].reshape(t,4,3,3)[active.astype(bool)]
    if len(rotations) and (not np.allclose(rotations.swapaxes(-1,-2)@rotations,np.eye(3),atol=1e-3)
                           or not np.allclose(np.linalg.det(rotations),1.,atol=1e-3)):
        raise ValueError('Active pad rotation is not a rigid transform')
    if not np.isfinite(data['tracks'][data['visible']]).all() or not np.isfinite(data['reference'][data['reference_valid']]).all():
        raise ValueError('Nonfinite visible GT')
    for key in ('P', 'surface_weights'):
        if (data[key] < 0).any() or not np.allclose(data[key].sum(-1), 1., atol=1e-5):
            raise ValueError('Fixed convex bindings required: '+key)
    if not np.isin(data['phase'], [0,1,2]).all() or not data['visible'][1:].any():
        raise ValueError('Missing scored GT or invalid phases')
    np.testing.assert_allclose(data['state0'][:,:3]*scale+data['origin'], data['x0'], atol=2e-6)
    return data


def prepare(observations, base_checkpoint, output, fit_episodes=(0,1,2), query_episodes=tuple(range(3,10))):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new directory; frozen base/data are never overwritten')
    if (not fit_episodes or set(fit_episodes) & set(query_episodes) or
            len(set(fit_episodes)) != len(fit_episodes) or len(set(query_episodes)) != len(query_episodes)):
        raise ValueError('Separate nonempty support and query episode sets required')
    observations, base_checkpoint = Path(observations).resolve(), Path(base_checkpoint).resolve()
    source_hash, base_hash = sha256(observations), sha256(base_checkpoint)
    bundle = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
    if bundle.get('schema') != 'softdynamics.coupled-temporal-checkpoint.v1':
        raise ValueError('Prepare from an original coupled MPM checkpoint')
    protocol = bundle['protocol']
    if set(protocol['training_objects']) & set(protocol['validation_objects']):
        raise ValueError('Overlapping frozen object splits')
    allowed = {o:split for split,key in [('train','training_objects'),('validation','validation_objects')]
               for o in protocol[key]}
    excluded = set(protocol.get('test_objects_excluded', []))
    index = read_json(observations)
    records = []
    for row in index['episodes']:
        obj, episode = row['object'], int(row['episode'])
        if Path(obj).name != obj or obj in ('.','..') or episode < 0:
            raise ValueError('Invalid observation identity')
        if obj in excluded or obj not in allowed:
            raise ValueError('Object outside frozen fitting/validation splits: '+obj)
        role = 'fit' if episode in fit_episodes else 'query' if episode in query_episodes else None
        if role is None:
            continue
        path = Path(row.get('data', observations.parent/'episodes'/obj/f'episode_{episode}'/'data.npz'))
        if not path.is_absolute():
            path = observations.parent/path
        digest = sha256(path)
        if row.get('data_sha256', digest) != digest:
            raise ValueError('Observation packet changed: '+str(path))
        validate_packet(load_npz(path))
        records.append(dict(object=obj, episode=episode, split=allowed[obj], role=role,
            source=str(path.resolve()), sha256=digest,
            path=f'episodes/{obj}/episode_{episode}/data.npz'))
    identities = [(r['object'],r['episode']) for r in records]
    if len(set(identities)) != len(identities) or not records:
        raise ValueError('Duplicate or missing episodes')
    for obj in {r['object'] for r in records}:
        if {r['episode'] for r in records if r['object']==obj and r['role']=='fit'} != set(fit_episodes):
            raise ValueError('Incomplete material-support set: '+obj)
    if not any(r['split']=='train' and r['role']=='fit' for r in records):
        raise ValueError('No real training support interactions')
    output.mkdir(parents=True)
    shutil.copyfile(base_checkpoint, output/'base.pt')
    for row in records:
        dest = output/row['path']; dest.parent.mkdir(parents=True)
        shutil.copyfile(row['source'], dest)
        if sha256(dest) != row['sha256']:
            raise ValueError('Observation changed during copy')
    if sha256(output/'base.pt') != base_hash or sha256(observations) != source_hash:
        raise ValueError('Base or index changed during preparation')
    manifest = dict(schema='softdynamics.real-residual-data.v1', base_checkpoint='base.pt',
        base_sha256=base_hash, base_step=bundle['step'], observations=str(observations),
        observations_sha256=source_hash, fit_episodes=list(fit_episodes), query_episodes=list(query_episodes),
        records=records, targets='Measured tracks/surfaces with visibility; fixed initial bindings',
        teacher_trajectories_used_as_real_targets=False, sensitivity_semantics='Uncorrected MPM/base response',
        query_used_for_material_fit=False, query_used_for_residual_training=False)
    write_json(output/'manifest.json', manifest)
    return manifest


class RealReplay:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root/'manifest.json')
        if self.manifest['schema'] != 'softdynamics.real-residual-data.v1' or sha256(self.root/'base.pt') != self.manifest['base_sha256']:
            raise ValueError('Changed frozen base/data schema')
        self.data = {}
        for row in self.manifest['records']:
            path = self.root/row['path']
            if sha256(path) != row['sha256']:
                raise ValueError('Changed real GT packet')
            self.data[(row['object'],row['episode'])] = validate_packet(load_npz(path))

    def records(self, split=None, role=None, obj=None):
        return [r for r in self.manifest['records'] if
                (split is None or r['split']==split) and (role is None or r['role']==role) and
                (obj is None or r['object']==obj)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit-episodes', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--query-episodes', type=int, nargs='+', default=list(range(3,10)))
    a = p.parse_args()
    result = prepare(a.observations,a.base_checkpoint,a.output,a.fit_episodes,a.query_episodes)
    print('Pinned',len(result['records']),'real episodes; base step',result['base_step'])


if __name__ == '__main__':
    main()
