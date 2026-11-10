import numpy as np
import pytest
from softdynamics.adapters import pad_empm
from softdynamics.spatial_grid import runtime, refine_domain, settings

POLICY = {'grid_spacing_factor': .5, 'physical_contact_slab_m': .00625}

def test_refinement_preserves_shift_and_physical_extent():
    x = np.array([[.1, -.01, .1], [.2, .08, .3]])
    poses = np.tile(np.eye(4), (1, 1, 2, 1, 1)); poses[..., :3, 3] = [.1, .1, .15]
    valid = np.ones((1, 1, 2), dtype=bool)
    size, shift = pad_empm.simulation_domain(x, poses, valid, .003125)
    fine_size, fine_shift = refine_domain(pad_empm.simulation_domain, .5, .003125)(x, poses, valid, .0015625)
    np.testing.assert_array_equal(shift, fine_shift)
    assert size * .003125 == fine_size * .0015625

def test_rejects_collider_geometry_changes():
    with pytest.raises(ValueError): settings(dict(POLICY, physical_contact_slab_m=.003125))

def test_rejects_an_unbounded_allocation():
    f = refine_domain(lambda *a: (368, np.zeros(3)), .5, .003125)
    with pytest.raises(ValueError): f(None, None, None, .0015625)

def test_runtime_restores_all_lattice_settings_after_exception():
    before = (pad_empm.PadRuntime, pad_empm.simulation_domain, pad_empm.GRID_DX_M, pad_empm.SLAB_CELLS)
    with pytest.raises(RuntimeError):
        with runtime(POLICY):
            assert pad_empm.GRID_DX_M * pad_empm.SLAB_CELLS == .00625
            raise RuntimeError('probe interrupted')
    assert before == (pad_empm.PadRuntime, pad_empm.simulation_domain, pad_empm.GRID_DX_M, pad_empm.SLAB_CELLS)
