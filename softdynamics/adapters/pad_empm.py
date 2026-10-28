"""FCR/MPM teacher with measured independent pads and an action-bounded domain."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from .empm import BranchRuntime
from ..io import load_npz, read_json, sha256


def simulation_domain(initial, poses, valid, dx, ground=-.01, operation=None, margin=.075):
    """Use only initial geometry and known actions, never future object tracks."""
    domain_valid = valid.copy()
    if operation is not None:
        domain_valid[:operation[0]] = False
        domain_valid[operation[1]+1:] = False
    centers = poses[..., :3, 3][domain_valid]
    points = np.concatenate([initial, centers])
    low, high = points.min(0)-margin, points.max(0)+margin
    low[1] = min(low[1], ground-.035)
    high[1] = max(high[1], initial[:, 1].max()+.12)
    size = max(64, int(np.ceil(np.max(high-low)/dx/16))*16)
    if size > 384:
        raise ValueError(f'Action-bounded cubic domain exceeds 384 cells: {size}')
    return size, -low


class PadRuntime(BranchRuntime):
    def __init__(self, episode, parameters, workspace, runtime, dtmax=.0002, device='cuda:0', domain_margin=.075):
        runtime, workspace = Path(runtime), Path(workspace)
        sys.path.insert(0, str(runtime))
        sys.path.insert(0, str(runtime/'gs-layer'))
        spec = importlib.util.spec_from_file_location('pad_upstream', runtime/'run_episode.py')
        self.up = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.up)
        up = self.up
        from pad_contact_kernels import accumulate_pad_contact
        from material_contact_kernels import material_grid_boundary
        self.prepared = workspace/'queries'/f'episode_{episode}'
        self.meta = read_json(self.prepared/'episode.json')
        a = load_npz(self.prepared/'actions.npz')
        if sha256(self.prepared/'actions.npz') != self.meta['source_sha256']['actions']:
            raise ValueError('Actions changed after preparation')
        self.poses, self.valid = a['poses'], a['valid']
        if not np.array_equal(np.isfinite(self.poses).all(axis=(-1,-2)), self.valid):
            raise ValueError('Pad validity disagrees with finite poses')
        self.times = (a['source_timestamps_us']-a['source_timestamps_us'][0])*1e-6
        if not (np.diff(self.times)>0).all():
            raise ValueError('Nonmonotonic action timestamps')
        asset_path = workspace/'asset'
        asset = read_json(asset_path/'asset.json')
        for name, digest in asset['payload_sha256'].items():
            if sha256(asset_path/name) != digest:
                raise ValueError('Asset payload changed')
        body = load_npz(asset_path/'body.npz')
        initial_path = workspace/'initial_states'/f'episode_{episode}.npz'
        initial = load_npz(initial_path)
        if sha256(initial_path) != read_json(initial_path.with_suffix('.json'))['state_sha256']:
            raise ValueError('Initial state changed')
        if str(initial['asset_sha256']) != sha256(asset_path/'asset.json'):
            raise ValueError('Initial state belongs to a different asset')
        np.testing.assert_array_equal(initial['material_ids'], body['material_ids'])
        xyz, F = initial['positions_world_m'], initial['deformation_gradient']
        dx = .4/128
        grid, shift = simulation_domain(xyz, self.poses, self.valid, dx,
                                        operation=(self.meta['clip_start'],self.meta['clip_end']),margin=domain_margin)
        up.wp.config.kernel_cache_dir = str(workspace.parents[1]/'warp-cache')
        up.wp.init()
        patches = [(np.full(len(xyz),-1,np.int32), np.zeros_like(xyz), np.zeros((2,3)))
                   for _ in range(self.poses.shape[1])]
        self.solver, self.model, self.states, self.contacts, self.shift = up.initialize(
            xyz, patches, self.times, device, 0, parameters['gripper_static_friction'],
            parameters['gripper_sliding_friction'], parameters['E_pa'], body['rest_volumes_m3'],
            body['masses_kg'], F, parameters['nu'], parameters['ground_friction'],
            grid_spacing_m=dx, grid_size=grid, world_shift=shift)
        self.solver.pre_p2g_operations = []
        self.solver.impulse_params = []
        self.solver.grid_postprocess = []
        self.solver.collider_params = []
        self.solver.modify_bc = []
        for c in self.contacts:
            c.footprint_y, c.footprint_z = .032, .0125
            c.radius = 2*dx
            self.solver.grid_postprocess.append(accumulate_pad_contact)
            self.solver.collider_params.append(c)
            self.solver.modify_bc.append(None)
        self.solver.grid_postprocess.append(material_grid_boundary)
        self.solver.collider_params.append(self.contacts[0])
        self.solver.modify_bc.append(None)
        self.transfer = up.DeterministicTransfer(self.model, device)
        self.current, self.frame, self.device = 0, 0, device
        self.dtmax, self.parameters, self.n = dtmax, dict(parameters), len(xyz)
        self.source_hashes = dict(asset=sha256(asset_path/'asset.json'), initial=sha256(initial_path),
                                 actions=sha256(self.prepared/'actions.npz'),
                                 pad_kernel=sha256(runtime/'gs-layer/pad_contact_kernels.py'),
                                 adapter=sha256(Path(__file__)),
                                 initializer=sha256(runtime/'run_episode.py'),
                                 transfer=sha256(runtime/'gs-layer/deterministic_transfer.py'))
        self.domain = dict(grid_size=grid, dx_m=dx, shift=shift.tolist(),margin_m=domain_margin)

    def check(self):
        check=super().check()
        if not check['valid']:
            x=self.states[self.current].particle_x.numpy()
            check['axis_domain_margins_m']=np.minimum(x,self.model.grid_lim-x).min(0).tolist()
            if check['finite'] and check['min_J']>0 and check['domain_margin_m']<=2.01*self.model.dx:
                check['failure_kind']='domain_boundary'
        return check
