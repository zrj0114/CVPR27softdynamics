"""Refine the grid lattice while retaining physical particles and collider slabs."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .io import load_npz, sha256


def settings(policy):
    factor = float(policy['grid_spacing_factor'])
    if factor != .5:
        raise ValueError('Only the tested half-spacing numerical refinement is supported')
    from .teacher_contact import GRID_DX_M, SLAB_CELLS
    physical_slab = SLAB_CELLS * GRID_DX_M
    if float(policy['physical_contact_slab_m']) != physical_slab:
        raise ValueError('Spatial refinement must preserve the selected physical pad slab')
    return factor, physical_slab


def refine_domain(function, factor, old_dx):
    def refined(initial, poses, valid, dx, ground=-.01, operation=None, margin=.075):
        if dx != old_dx * factor:
            raise ValueError('Runtime grid spacing differs from the numerical policy')
        size, shift = function(initial, poses, valid, old_dx, ground, operation, margin)
        new_size = size * 2
        if new_size ** 3 > 128000000:
            raise ValueError('Refined cubic grid exceeds the verified memory budget')
        return new_size, shift
    return refined


@contextmanager
def runtime(policy):
    from .adapters import pad_empm
    factor, physical_slab = settings(policy)
    original = pad_empm.PadRuntime
    original_domain = pad_empm.simulation_domain
    old_dx, old_cells = pad_empm.GRID_DX_M, pad_empm.SLAB_CELLS

    class SpatialRuntime(original):
        def __init__(self, episode, parameters, workspace, *args, **kwargs):
            super().__init__(episode, parameters, workspace, *args, **kwargs)
            initial = load_npz(Path(workspace) / 'initial_states' / f'episode_{episode}.npz')
            expected_size, expected_shift = original_domain(initial['positions_world_m'], self.poses, self.valid, old_dx,
                                                           margin=self.domain['margin_m'])
            np.testing.assert_array_equal(self.shift, expected_shift)
            np.testing.assert_array_equal(self.states[0].particle_x.numpy(), (initial['positions_world_m'] + expected_shift).astype(np.float32))
            np.testing.assert_array_equal(self.states[0].particle_F_trial.numpy(), initial['deformation_gradient'].astype(np.float32))
            if not all(np.isclose(c.radius, physical_slab, atol=1e-9, rtol=0) for c in self.contacts):
                raise ValueError('Spatial refinement changed a physical collider slab')
            self.domain.update(numerical_spatial_refinement_factor=factor, original_grid_spacing_m=old_dx,
                               original_grid_size=expected_size, physical_pad_contact_slab_m=physical_slab,
                               physical_particles_and_initial_state_preserved=True)
            self.source_hashes['spatial_refinement_adapter'] = sha256(Path(__file__))

    pad_empm.GRID_DX_M = old_dx * factor
    pad_empm.SLAB_CELLS = physical_slab / pad_empm.GRID_DX_M
    pad_empm.simulation_domain = refine_domain(original_domain, factor, old_dx)
    pad_empm.PadRuntime = SpatialRuntime
    try:
        yield
    finally:
        pad_empm.PadRuntime = original
        pad_empm.simulation_domain = original_domain
        pad_empm.GRID_DX_M, pad_empm.SLAB_CELLS = old_dx, old_cells
