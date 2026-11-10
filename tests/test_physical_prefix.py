import copy
import numpy as np
import pytest

from softdynamics.adapters.empm import FIELDS,BranchRuntime
from softdynamics.shared_world_snapshot import physical_prefix,prefix_identity_key


def example():
    identity=dict(start_frame=5,particles=2,sampled_nodes=1,dtmax=.000119314,
        parameters={'E_pa':3500.,'nu':.27},source_sha256='original',domain={'dx':.003125})
    arrays={}
    for i,key in enumerate(FIELDS):
        shape=(2,3) if key in {'particle_x','particle_v'} else (2,3,3) if key in {
            'particle_C','particle_F_trial','particle_F','particle_stress'} else (12,) if key=='particle_cov' else (2,)
        arrays[key]=np.full(shape,i,dtype=np.int32 if key=='particle_selection' else np.float32)
    snapshot=dict(arrays=arrays,time=.15,frame=5)
    previous=np.array([[.01,.02,.03]],np.float32)
    return identity,snapshot,previous


def test_prefix_cache_preserves_every_field_and_past_observation(tmp_path):
    identity,snapshot,previous=example();calls=[]
    def build():calls.append(True);return snapshot,previous
    a,pa,cold=physical_prefix(tmp_path,identity,build)
    b,pb,warm=physical_prefix(tmp_path,identity,build)
    assert len(calls)==1 and not cold['reused'] and warm['reused']
    assert warm['physical_prefix_frames_replayed']==0 and cold['physical_prefix_frames_replayed']==5
    assert BranchRuntime.digest(a)==BranchRuntime.digest(b)==BranchRuntime.digest(snapshot)
    assert b['arrays']['particle_cov'].shape==(12,)
    for key in FIELDS:np.testing.assert_array_equal(b['arrays'][key],snapshot['arrays'][key])
    np.testing.assert_array_equal(pa,previous);np.testing.assert_array_equal(pb,previous)


@pytest.mark.parametrize('changed',['dtmax','parameters','source_sha256','domain'])
def test_prefix_cache_never_aliases_different_physics(tmp_path,changed):
    identity,snapshot,previous=example();other=copy.deepcopy(identity)
    other[changed]={'dtmax':.000149422,'parameters':{'E_pa':3510.,'nu':.27},
        'source_sha256':'modified','domain':{'dx':.0032}}[changed]
    assert prefix_identity_key(identity)!=prefix_identity_key(other)
    calls=[]
    def build():calls.append(True);return snapshot,previous
    physical_prefix(tmp_path,identity,build);physical_prefix(tmp_path,other,build)
    assert len(calls)==2


def test_corrupt_completed_prefix_is_rejected_without_rebuilding(tmp_path):
    identity,snapshot,previous=example();calls=[]
    def build():calls.append(True);return snapshot,previous
    physical_prefix(tmp_path,identity,build)
    path=tmp_path/prefix_identity_key(identity)/'snapshot.npz';path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload changed'):physical_prefix(tmp_path,identity,build)
    assert len(calls)==1


def test_prefix_cache_rejects_missing_previous_positions(tmp_path):
    identity,snapshot,_=example()
    with pytest.raises(ValueError,match='past-only'):physical_prefix(tmp_path,identity,lambda:(snapshot,None))
