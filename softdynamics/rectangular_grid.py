"""Expand grid storage using integer offsets, retaining physical FP32 coordinates."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

from .io import sha256
from .adapters.empm import DYNAMICS
from .shared_world_observation import GPUObservation


def allocation(shape, offset, original_size):
    shape, offset = np.asarray(shape), np.asarray(offset)
    if (shape.shape != (3,) or offset.shape != (3,) or
            shape.dtype.kind not in 'iu' or offset.dtype.kind not in 'iu' or
            np.any(shape < 16) or np.any(shape % 16) or np.any(offset < 0) or
            np.any(offset + original_size > shape) or np.prod(shape.astype(np.int64)) > 128_000_000):
        raise ValueError('Rectangular storage must retain the original grid and fit the explicit allocation budget')
    return tuple(map(int, shape)), tuple(map(int, offset))


def install(sim, shape, offset):
    import warp as wp
    from rectangular_transfer import DeterministicTransfer
    shape, offset = allocation(shape, offset, sim.model.grid_dim_x)
    for state in sim.states:
        state.grid_m = wp.zeros(shape, dtype=float, device=sim.device)
        state.grid_v_in = wp.zeros(shape, dtype=wp.vec3, device=sim.device)
        state.grid_v_out = wp.zeros(shape, dtype=wp.vec3, device=sim.device)
    momentum = wp.zeros(shape, dtype=wp.vec3, device=sim.device)
    weight = wp.zeros(shape, dtype=float, device=sim.device)
    for contact in sim.contacts:
        contact.mass = wp.zeros(shape, dtype=float, device=sim.device)
        contact.right_mass = wp.zeros(shape, dtype=float, device=sim.device)
        contact.momentum, contact.weight = momentum, weight
    sim.model.grid_dim_x, sim.model.grid_dim_y, sim.model.grid_dim_z = shape
    sim.model.grid_lim = max(shape) * sim.model.dx
    sim.transfer = DeterministicTransfer(sim.model, sim.device, offset)
    sim.rectangular_lower = (-np.asarray(offset) * sim.model.dx).astype(np.float32)
    sim.rectangular_upper = ((np.asarray(shape) - offset) * sim.model.dx).astype(np.float32)
    sim.domain.update(grid_size=max(shape), grid_shape=list(shape), grid_index_offset_cells=list(offset),
                      coordinate_policy='unchanged_particle_positions_and_pad_transforms_integer_storage_offset')
    source = Path(__file__).resolve().parents[1]/'runtime/gs-layer'
    sim.source_hashes['base_transfer'] = sim.source_hashes['transfer']
    sim.source_hashes['transfer'] = sha256(source/'rectangular_transfer.py')
    sim.source_hashes['rectangular_g2p'] = sha256(source/'rectangular_g2p.py')
    sim.source_hashes['rectangular_active_grid'] = sha256(source/'rectangular_active_grid.py')
    sim.source_hashes['rectangular_adapter'] = sha256(Path(__file__))

    def cpu_check():
        state = sim.states[sim.current]
        fields = {name: getattr(state, name).numpy() for name in DYNAMICS}
        x = fields['particle_x']
        finite = all(np.isfinite(value).all() for value in fields.values())
        det = np.linalg.det(fields['particle_F_trial'])
        margin = float(np.minimum(x-sim.rectangular_lower, sim.rectangular_upper-x).min())
        return dict(valid=bool(finite and det.min()>0 and margin>2.01*sim.model.dx), finite=bool(finite),
                    min_J=float(det.min()), max_J=float(det.max()), domain_margin_m=margin)
    sim.check = cpu_check


class RectangularObservation(GPUObservation):
    @torch.inference_mode()
    def check(self):
        sim = self.sim
        fields = self.views[sim.current]
        x, F = fields['particle_x'], fields['particle_F_trial']
        lower = torch.as_tensor(sim.rectangular_lower, device=x.device)
        upper = torch.as_tensor(sim.rectangular_upper, device=x.device)
        finite = torch.stack([torch.isfinite(value).all() for value in fields.values()]).all()
        det = torch.linalg.det(F.double()).float()
        margins = torch.minimum(x-lower, upper-x)
        axis = margins.amin(0)
        values = torch.stack((finite.float(), det.amin(), det.amax(), axis.amin())).cpu().numpy()
        finite = bool(values[0]); min_j, max_j, margin = map(float, values[1:])
        valid = finite and min_j>0 and margin>2.01*sim.model.dx
        result = dict(valid=valid, finite=finite, min_J=min_j, max_J=max_j, domain_margin_m=margin)
        if not valid:
            result.update(axis_domain_margins_m=axis.cpu().tolist(),
                          lower_domain_margins_m=(x-lower).amin(0).cpu().tolist(),
                          upper_domain_margins_m=(upper-x).amin(0).cpu().tolist(),
                          max_particle_speed_m_s=float(fields['particle_v'].norm(dim=-1).amax().cpu()),
                          particles_at_boundary=int((margins.amin(-1)<=2.01*sim.model.dx).sum().cpu()))
            if finite and min_j>0 and margin<=2.01*sim.model.dx:
                result['failure_kind'] = 'domain_boundary'
        return result


@contextmanager
def runtime(shape, offset):
    from .adapters import pad_empm
    from . import shared_world_observation as observation
    original_runtime, original_observer = pad_empm.PadRuntime, observation.GPUObservation
    class RectangularRuntime(original_runtime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install(self, shape, offset)
    pad_empm.PadRuntime = RectangularRuntime
    observation.GPUObservation = RectangularObservation
    try:
        yield
    finally:
        pad_empm.PadRuntime, observation.GPUObservation = original_runtime, original_observer
