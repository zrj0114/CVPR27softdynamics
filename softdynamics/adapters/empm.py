"""Checkpoint/restore adapter around the unchanged deterministic FCR runtime.

Only active particle state is persistent. Grid/contact accumulators are scratch
cleared by every step; we also clear them explicitly before every branch.
Elastic FCR has no evolving model-side plastic history. Other materials refused.
"""
import hashlib
import importlib.util
import math
import sys
import numpy as np
from pathlib import Path
from ..io import read_json as read, sha256 as sha

FIELDS = ('particle_x', 'particle_v', 'particle_C', 'particle_F_trial', 'particle_F',
          'particle_stress', 'particle_cov', 'particle_vol', 'particle_mass',
          'particle_density', 'particle_selection')
DYNAMICS = ('particle_x', 'particle_v', 'particle_C', 'particle_F_trial')


class BranchRuntime:
    def __init__(self, episode, parameters, padding, dtmax, device='cuda:0', runtime=None, workspace=None):
        RUNTIME = Path(runtime).resolve() if runtime else Path(__file__).resolve().parents[2]/'runtime'
        workspace = Path(workspace).resolve() if workspace else RUNTIME
        sys.path.insert(0, str(RUNTIME))
        spec = importlib.util.spec_from_file_location('short_branch_upstream', RUNTIME/'run_episode.py')
        self.up = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.up)
        up = self.up
        up.ROOT = workspace
        self.prepared, _, self.poses, self.valid, self.times, refs, self.meta = up.load_episode(episode)
        asset_path = workspace/'asset'; asset = read(asset_path/'asset.json')
        for name, digest in asset['payload_sha256'].items():
            if sha(asset_path/name) != digest: raise ValueError('Asset payload mismatch')
        initial_path = workspace/f'initial_states/episode_{episode}.npz'
        if read(initial_path.with_suffix('.json'))['state_sha256'] != sha(initial_path):
            raise ValueError('Initial state hash mismatch')
        with np.load(asset_path/'body.npz') as z: body = dict(z)
        with np.load(initial_path) as z: initial = dict(z)
        if str(initial['asset_sha256']) != sha(asset_path/'asset.json'): raise ValueError('Wrong shared asset')
        np.testing.assert_array_equal(initial['material_ids'], body['material_ids'])
        xyz = initial['positions_world_m']; F = initial['deformation_gradient']
        if not (np.isfinite(xyz).all() and np.isfinite(F).all() and (np.linalg.det(F)>0).all()):
            raise ValueError('Invalid initial state')
        patches = [up.material_patch(xyz, self.poses[f, g]) for g, f in enumerate(refs)]
        up.wp.config.kernel_cache_dir = str(workspace/'.warp-cache'); up.wp.init()
        self.solver, self.model, self.states, self.contacts, self.shift = up.initialize(
            xyz, patches, self.times, device, padding, parameters['gripper_static_friction'],
            parameters['gripper_sliding_friction'], parameters['E_pa'], body['rest_volumes_m3'],
            body['masses_kg'], F, parameters['nu'], parameters['ground_friction'])
        if self.model.material != 0: raise ValueError('Snapshot contract supports elastic FCR only')
        if not (self.states[0].particle_selection.numpy() == 0).all():
            raise ValueError('Snapshot contract requires all particles active')
        self.transfer = up.DeterministicTransfer(self.model, device)
        self.current = 0; self.frame = 0; self.dtmax = dtmax; self.device = device
        self.parameters = dict(parameters); self.n = len(xyz)
        self.source_hashes = dict(asset=sha(asset_path/'asset.json'), initial=sha(initial_path),
                                  actions=sha(self.prepared/'actions.npz'))

    def set_parameters(self, parameters):
        self.parameters = dict(parameters)
        self.solver.set_E_nu(self.model, parameters['E_pa'], parameters['nu'], device=self.device)
        self.solver.prepare_mu_lam(self.model, self.states[self.current], device=self.device)
        for c in self.contacts:
            c.friction = parameters['ground_friction']
            c.contact_friction = parameters['gripper_static_friction']
            c.sliding_friction = parameters['gripper_sliding_friction']

    def snapshot(self):
        arrays = {name: getattr(self.states[self.current], name).numpy().copy() for name in FIELDS}
        return dict(arrays=arrays, time=float(self.solver.time), frame=self.frame)

    @staticmethod
    def digest(snapshot):
        h = hashlib.sha256()
        for name in FIELDS: h.update(name.encode()); h.update(snapshot['arrays'][name].tobytes())
        h.update(np.float64(snapshot['time']).tobytes()); h.update(str(snapshot['frame']).encode())
        return h.hexdigest()

    def restore(self, snapshot, parameters):
        for state in self.states:
            for name, values in snapshot['arrays'].items(): getattr(state, name).assign(values)
            for name in ('grid_m', 'grid_v_in', 'grid_v_out'): getattr(state, name).zero_()
        for c in self.contacts:
            for name in ('mass', 'right_mass', 'momentum', 'weight', 'correction'): getattr(c, name).zero_()
        self.transfer.index.invalid.zero_()
        self.current = 0; self.frame = snapshot['frame']; self.solver.time = snapshot['time']
        self.set_parameters(parameters)

    def advance(self, end):
        if end <= self.frame: raise ValueError('Branch must advance in time')
        for f in range(self.frame+1, end+1):
            interval = self.times[f]-self.times[f-1]
            steps = math.ceil(interval/self.dtmax); dt = interval/steps
            self.model.grid_v_damping_scale = math.exp(-self.parameters['velocity_decay_rate_s_inv']*dt)
            for g, c in enumerate(self.contacts):
                enabled, transforms, linear, angular = self.up.interval_motion(
                    self.poses[f-1,g], self.poses[f,g], self.valid[f-1,g], self.valid[f,g], interval, self.shift)
                c.enabled.assign(enabled); c.transforms.assign(transforms)
                c.linear.assign(linear); c.angular.assign(angular); c.frame_time = float(self.times[f-1])
            for _ in range(steps):
                self.transfer.step(self.solver, self.model, self.states[self.current], self.states[1-self.current], dt)
                self.current = 1-self.current
            self.transfer.check(); self.frame = f
        return self.check()

    def positions(self, ids):
        return (self.states[self.current].particle_x.numpy()[ids]-self.shift).astype(np.float32)

    def state_hashes(self):
        return {name: hashlib.sha256(getattr(self.states[self.current], name).numpy().tobytes()).hexdigest() for name in DYNAMICS}

    def check(self):
        state = self.states[self.current]
        x = state.particle_x.numpy(); F = state.particle_F_trial.numpy()
        finite = all(np.isfinite(getattr(state, k).numpy()).all() for k in DYNAMICS)
        det = np.linalg.det(F); margin = float(np.minimum(x, self.model.grid_lim-x).min())
        valid = bool(finite and det.min()>0 and margin>2.01*self.model.dx)
        return dict(valid=valid, finite=bool(finite), min_J=float(det.min()), max_J=float(det.max()), domain_margin_m=margin)
