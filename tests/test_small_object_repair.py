import hashlib
import io
import tarfile

import numpy as np
import pytest

from softdynamics.numerical_retry import fixed_upper_grid
from softdynamics.numerical_retry_directed import fixed_grid
from softdynamics.teacher50_geometry import carve_consensus


def test_expanding_upper_bounds_preserves_lattice_and_restores_domain_function(monkeypatch):
    from softdynamics.adapters import pad_empm
    shift=np.array([.047718732213939616,.08624999858438968,.16799181906893604])
    def domain(*args,**kwargs):return 192,shift.copy()
    monkeypatch.setattr(pad_empm,'simulation_domain',domain)
    with fixed_upper_grid(384):
        size,actual=pad_empm.simulation_domain(None,None,None,.003125)
        assert size==384
        assert actual.tobytes()==shift.tobytes()
    assert pad_empm.simulation_domain is domain
    with pytest.raises(ValueError,match='shrink'):
        with fixed_upper_grid(128):pad_empm.simulation_domain(None,None,None,.003125)
    assert pad_empm.simulation_domain is domain


def source_frame(tmp_path):
    buf=io.BytesIO()
    np.savez(buf,pts=np.ones((8,3),np.float32),visibility_matrix=np.ones((8,5),bool))
    data=buf.getvalue()
    with tarfile.open(tmp_path/'pcd_clean.tar','w') as archive:
        info=tarfile.TarInfo('pcd_clean/000000.npz');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    return hashlib.sha256(data).hexdigest()


def test_directed_padding_contains_original_domain_and_keeps_grid_spacing(monkeypatch):
    from softdynamics.adapters import pad_empm
    shift=np.array([.047718732213939616,.08624999858438968,.16799181906893604])
    dx=.003125
    def domain(*args,**kwargs):return 192,shift.copy()
    monkeypatch.setattr(pad_empm,'simulation_domain',domain)
    with fixed_grid(384,[64,64,160]):
        size,new_shift=pad_empm.simulation_domain(None,None,None,dx)
        np.testing.assert_allclose((new_shift-shift)/dx,[64,64,160],rtol=0,atol=1e-13)
        assert np.all(-new_shift<=-shift)
        assert np.all(size*dx-new_shift>=192*dx-shift)
    assert pad_empm.simulation_domain is domain
    with pytest.raises(ValueError,match='entire original grid'):
        with fixed_grid(384,[0,0,193]):pad_empm.simulation_domain(None,None,None,dx)
    assert pad_empm.simulation_domain is domain


def test_object_annotation_cannot_be_applied_to_another_recorded_frame(tmp_path):
    source_frame(tmp_path)
    np.savez(tmp_path/'initial_object_point_ids.npz',point_ids=np.arange(8),
             first_frame_sha256='incorrect',reason='fixture')
    with pytest.raises(ValueError,match='different source frame'):
        carve_consensus(tmp_path,{},tmp_path/'volume.npz')


@pytest.mark.parametrize('ids',[np.array([0,0]),np.array([-1,2]),np.array([0,8]),np.array([0.,2.])])
def test_invalid_object_point_id_annotation_is_rejected(tmp_path,ids):
    digest=source_frame(tmp_path)
    np.savez(tmp_path/'initial_object_point_ids.npz',point_ids=ids,
             first_frame_sha256=digest,reason='fixture')
    with pytest.raises(ValueError,match='Invalid initial object point annotation'):
        carve_consensus(tmp_path,{},tmp_path/'volume.npz')
