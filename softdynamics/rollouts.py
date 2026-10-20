"""Full EMPM rollouts with exact scoring and a source/input-keyed local cache."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

from .io import load_npz, read_json, relative_path, sha256, write_json
from .objective import score

FLAGS = {'E_pa': '--young-modulus', 'nu': '--poisson-ratio',
         'ground_friction': '--ground-friction', 'velocity_decay_rate_s_inv': '--velocity-decay-rate',
         'gripper_static_friction': '--contact-friction', 'gripper_sliding_friction': '--sliding-friction'}


class CachedRollouts:
    def __init__(self, manifest, workspace, cache, config, space, devices=(0,), runtime=None):
        self.manifest = Path(manifest).resolve()
        self.rows = read_json(self.manifest)
        self.workspace = Path(workspace).resolve()
        self.cache = Path(cache).resolve()
        self.config, self.space = config, space
        self.devices = list(devices)
        self.runtime = Path(runtime).resolve() if runtime else Path(__file__).resolve().parents[1]/'runtime'
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError('Provide distinct GPU IDs')
        if not self.rows or len({r['episode'] for r in self.rows}) != len(self.rows):
            raise ValueError('Provide unique episode IDs')
        # Changing source, asset, initial states or actions invalidates the cache.
        files = sorted(self.runtime.rglob('*.py')) + [self.workspace/'asset/asset.json']
        asset = read_json(self.workspace/'asset/asset.json')
        files += [self.workspace/'asset'/name for name in sorted(asset['payload_sha256'])]
        for row in self.rows:
            e = row['episode']
            files += [self.workspace/f'initial_states/episode_{e}.npz',
                      self.workspace/f'queries/episode_{e}/actions.npz',
                      self.workspace/f'queries/episode_{e}/episode.json',
                      relative_path(self.manifest, row['observation']),
                      relative_path(self.manifest, row['metadata'])]
        digest = hashlib.sha256()
        for path in files:
            digest.update(sha256(path).encode())
        digest.update(json.dumps(dict(rows=self.rows, fitting=config['fitting']), sort_keys=True).encode())
        self.fingerprint = digest.hexdigest()

    def one(self, parameters, row, device):
        key = hashlib.sha256(json.dumps(dict(parameters=parameters,
              source=self.fingerprint, device=device), sort_keys=True).encode()).hexdigest()[:24]
        root = self.cache/key
        result_path = root/f'episode_{row["episode"]}.json'
        if result_path.exists():
            result = read_json(result_path)
            return result['objective_mm'] if result['valid'] else np.inf
        root.mkdir(parents=True, exist_ok=True)
        folder = root/f'episode_{row["episode"]}'
        if not folder.exists():
            command = [sys.executable, '-u', str(self.runtime/'run_episode.py'),
                       '--episode', str(row['episode']), '--device', 'cuda:0',
                       '--transfer-mode', 'deterministic', '--compact-output',
                       '--grid-padding-cells', str(row.get('padding', 0)),
                       '--output-root', str(root)]
            for key, flag in FLAGS.items():
                command.extend([flag, str(parameters[key])])
            with (root/f'episode_{row["episode"]}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(device),
                         'SOFTDYNAMICS_WORKSPACE': str(self.workspace), 'OMP_NUM_THREADS': '1',
                         'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, check=False)
        status_path = folder/'status.json'
        if not status_path.exists():
            raise RuntimeError(f'Incomplete rollout preserved for inspection: {folder}')
        status = read_json(status_path)
        if status['status'] != 'completed':
            if not status.get('error', '').startswith('Frame '):
                raise RuntimeError(f'Runtime failure, not a physical rejection: {status}')
            write_json(result_path, dict(valid=False, failure=status))
            return np.inf
        validation = read_json(folder/'validation.json')
        if not all(r['finite'] and r['positiveJ'] and r['particlesAtDomainClamp'] == 0
                   for r in validation['perFrame']):
            write_json(result_path, dict(valid=False, reason='Physical validation failed'))
            return np.inf
        trajectory = load_npz(folder/'trajectory.npz')
        data = load_npz(relative_path(self.manifest, row['observation']))
        meta = read_json(relative_path(self.manifest, row['metadata']))
        np.testing.assert_array_equal(trajectory['saved_particle_indices'], data['saved_ids'])
        np.testing.assert_array_equal(trajectory['positions'][0], data['initial'])
        positions = trajectory['positions'][data['frames']]
        surface = positions[:, data['surface_ids']]
        tracks = np.empty_like(data['tracks'], dtype=np.float32)
        for start in range(0, len(tracks), 24):
            motion = positions[start:start+24, data['binding']] - data['initial'][data['binding']][None]
            tracks[start:start+24] = data['tracks'][0][None] + (motion*data['weights'][None, :, :, None]).sum(2)
        observed_surface = [data['observations'][i, :n] for i, n in enumerate(data['counts'])]
        obs = dict(surface=observed_surface, tracks=data['tracks'], visible=data['track_visible'],
                   meta={**meta, 'frames': data['frames']}, trees=[cKDTree(x) for x in observed_surface])
        result = score(dict(surface=surface, tracks=tracks), obs, self.config['fitting'])
        write_json(result_path, dict(valid=True, **result, parameters=parameters,
                   source_fingerprint=self.fingerprint))
        return result['objective_mm']

    def __call__(self, q):
        parameters = self.space.parameters(q)
        output = np.empty(len(self.rows))
        # Fixed episode/device assignment avoids duplicate jobs on a GPU and
        # keeps hardware consistent between compared parameter candidates.
        def worker(slot, device):
            for i in range(slot, len(self.rows), len(self.devices)):
                output[i] = self.one(parameters, self.rows[i], device)
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            tasks = [executor.submit(worker, i, device) for i, device in enumerate(self.devices)]
            for task in tasks:
                task.result()
        return output
