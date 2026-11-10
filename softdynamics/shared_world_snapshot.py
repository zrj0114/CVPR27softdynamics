"""Exact persistent-state cache for repeated elastic FCR control prefixes.

Only original recorded-control prefixes are shared. Material values, time step,
domain, input payloads and physical source code participate in the cache key.
Goals, policy weights and the subsequent horizon do not affect that prefix.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .adapters.empm import BranchRuntime,FIELDS
from .io import load_npz,read_json,sha256,write_json


def prefix_identity_key(identity):
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_prefix_snapshot(snapshot,previous_positions,identity):
    if snapshot['frame']!=identity['start_frame'] or not np.isfinite(snapshot['time']):
        raise ValueError('Cached physical prefix has the wrong frame or time')
    if set(snapshot['arrays'])!=set(FIELDS):raise ValueError('Incomplete persistent physical state')
    n=identity['particles']
    matrices={'particle_C','particle_F_trial','particle_F','particle_stress'}
    for key,values in snapshot['arrays'].items():
        shape=(n,3) if key in {'particle_x','particle_v'} else (n,3,3) if key in matrices else (6*n,) if key=='particle_cov' else (n,)
        dtype=np.dtype('int32' if key=='particle_selection' else 'float32')
        if values.shape!=shape or values.dtype!=dtype or not np.isfinite(values).all():
            raise ValueError('Invalid cached particle field: '+key)
    if identity['start_frame']==0:
        if previous_positions is not None:raise ValueError('Initial frame cannot have a preceding observation')
    elif (previous_positions is None or previous_positions.shape!=(identity['sampled_nodes'],3)
          or not np.isfinite(previous_positions).all()):
        raise ValueError('Cached prefix lacks its past-only velocity observation')


def physical_prefix(cache_root,identity,build):
    """Return a validated full snapshot and preceding sampled positions.

    ``build`` executes the unchanged original prefix on a cache miss. A corrupt
    completed entry is rejected instead of silently being replaced.
    """
    folder=Path(cache_root)/prefix_identity_key(identity);folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with (folder/'prefix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reused=(folder/'complete.json').exists()
        if reused:
            record=read_json(folder/'complete.json')
            if record['identity']!=identity:raise ValueError('Physical prefix cache identity changed')
            if sha256(folder/'snapshot.npz')!=record['payload_sha256']:
                raise ValueError('Physical prefix cache payload changed')
            payload=load_npz(folder/'snapshot.npz')
            if set(payload)!=set(FIELDS)|{'time','frame','previous_positions'}:
                raise ValueError('Physical prefix payload has unexpected fields')
            snapshot=dict(arrays={key:payload[key] for key in FIELDS},
                time=float(payload['time']),frame=int(payload['frame']))
            previous=payload['previous_positions'] if len(payload['previous_positions']) else None
            if BranchRuntime.digest(snapshot)!=record['snapshot_sha256']:
                raise ValueError('Physical prefix state digest changed')
            validate_prefix_snapshot(snapshot,previous,identity)
        else:
            build_started=time.monotonic();snapshot,previous=build()
            build_seconds=time.monotonic()-build_started
            validate_prefix_snapshot(snapshot,previous,identity)
            payload={**snapshot['arrays'],'time':np.float64(snapshot['time']),
                'frame':np.int64(snapshot['frame']),
                'previous_positions':previous if previous is not None else np.empty((0,3),np.float32)}
            np.savez_compressed(folder/'snapshot.tmp.npz',**payload)
            (folder/'snapshot.tmp.npz').replace(folder/'snapshot.npz')
            record=dict(schema='softdynamics.physical-prefix.v1',identity=identity,
                payload_sha256=sha256(folder/'snapshot.npz'),snapshot_sha256=BranchRuntime.digest(snapshot),
                prefix_build_seconds=build_seconds,created_at=time.time())
            write_json(folder/'complete.json',record)
        info=dict(folder=str(folder),identity_sha256=prefix_identity_key(identity),reused=reused,
            snapshot_sha256=record['snapshot_sha256'],payload_sha256=record['payload_sha256'],
            original_prefix_build_seconds=record['prefix_build_seconds'],
            cache_setup_seconds=time.monotonic()-started,
            physical_prefix_frames_replayed=0 if reused else identity['start_frame'])
        return snapshot,previous,info
