import numpy as np
import pytest
import torch

from softdynamics.joint_acceleration import normalized_weights,weighted_parameter_gradient
from softdynamics.joint_world import interaction_context
from test_joint_world import example


def test_confidence_is_detached_and_each_parameter_normalized():
    values=torch.tensor([[0.,.1,.8,.3],[0.,.9,.2,.7]],requires_grad=True)
    w=normalized_weights(values)
    assert not w.requires_grad
    torch.testing.assert_close(w.sum(0),torch.ones(4))
    torch.testing.assert_close(w[:,0],torch.tensor([.5,.5]))
    q=torch.ones(4,requires_grad=True)
    losses=[q.sum(),(q*torch.tensor([2.,3.,4.,5.])).sum()]
    g=weighted_parameter_gradient(losses,q,values)
    torch.testing.assert_close(g,torch.tensor([1.5,2.8,1.6,3.8]))
    assert values.grad is None


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):normalized_weights(torch.ones(2,5))
    with pytest.raises(ValueError):normalized_weights(torch.full((2,4),float('nan')))


def test_pad_matrix_translation_rotation_are_preserved():
    d=example()
    from scipy.spatial.transform import Rotation
    rot=Rotation.from_rotvec([.2,-.4,.3]).as_matrix().astype('f')
    mat=np.column_stack((rot,np.array([.4,-.2,.7],dtype='f')))
    d['actions'].reshape(30,4,13)[:,0,:12]=mat.reshape(12)
    c=interaction_context(d,dict(clip_start=10,clip_end=20))
    np.testing.assert_array_equal(c['pads0'][0,0,:3],mat[:,3])
    np.testing.assert_array_equal(c['pads0'][0,0,3:12].reshape(3,3),rot)


def test_physics_completion_is_published_after_all_outputs(tmp_path,monkeypatch):
    from softdynamics import recovery_data as rd
    from softdynamics import shared_world_observation as observation
    from softdynamics.joint_acceleration import collect_physics
    from softdynamics.io import write_json,read_json
    from softdynamics.recovery_data import trajectory_folder
    data=tmp_path/'data';nodes=data/'nodes.npz';data.mkdir()
    np.savez(nodes,material_indices=np.array([0]))
    write_json(tmp_path/'protocol.json',dict(data_root=str(data)))
    write_json(data/'plan.json',dict(objects=[dict(object='object',nodes=str(nodes))]))
    write_json(tmp_path/'physics/plan.json',dict(objects=[dict(object='object',support=[0],query=[1],
        episodes=[dict(episode=0),dict(episode=1)])]))
    folder=trajectory_folder(tmp_path/'physics','object',1,[0.]*4)
    class FakeRuntime:
        pass
    class FakeObserver:
        def __init__(self,*args):pass
        def install_validation(self):pass
    def simulate(*args):
        record=dict(valid=True,elapsed_s=0.)
        rd.write_json(folder/'complete.json',record)
        assert not (folder/'complete.json').exists()
        return record
    def score(*args):
        assert rd.read_json(folder/'complete.json')['valid']
        assert not (folder/'complete.json').exists()
        rd.write_json(folder/'scores.json',dict(valid=True))
    monkeypatch.setattr(rd,'PadRuntime',FakeRuntime)
    monkeypatch.setattr(rd,'simulate',simulate)
    monkeypatch.setattr(rd,'score_candidate',score)
    monkeypatch.setattr(observation,'GPUObservation',FakeObserver)
    original_read,original_write=rd.read_json,rd.write_json
    collect_physics(tmp_path,'object',1,[0.]*4,'cpu',capture=False)
    assert (folder/'complete.json').exists() and (folder/'scores.json').exists()
    assert (folder/'execution.json').exists() and not (folder/'complete.pending.json').exists()
    assert rd.read_json is original_read and rd.write_json is original_write
