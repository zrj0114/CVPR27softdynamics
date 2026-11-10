import numpy as np
import torch

from softdynamics.joint_world import JointWorld, interaction_context, context_batch
from softdynamics.shared_world_model import confidence_gradient


def example():
    rng=np.random.default_rng(9);T,N=30,128
    initial=rng.normal(size=(N,3)).astype('f')*.1
    x=initial[None]+np.arange(T,dtype='f')[:,None,None]*.001
    actions=np.zeros((T,4,13),np.float32)
    pose=np.column_stack((np.eye(3,dtype='f'),np.zeros(3,dtype='f')))
    actions[:,:2,:12]=pose.reshape(12)
    actions[:,:2,12]=1
    actions[:,1,3]=.2
    return dict(positions=x,valid=np.ones((T,N),bool),actions=actions.reshape(T,52),
                times=np.arange(T,dtype='f')/30,scale_m=np.float32(.2),origin_m=np.zeros(3,dtype='f'))


def test_joint_heads_share_encoder_and_are_independent():
    torch.set_num_threads(1);torch.manual_seed(4)
    m=JointWorld(dict(hidden=16,layers=1,heads=2,kinematic_only=True,direct_displacement=True,contact_geometry=True))
    c=context_batch([interaction_context(example(),dict(clip_start=10,clip_end=20))],'cpu')
    theta=torch.tensor([[3500.,.27,.8,.4]])
    prediction=m.confidence(c,theta)
    assert prediction.shape==(1,4) and torch.all((prediction>0)&(prediction<1))
    assert not torch.allclose(prediction.sum(1),torch.ones(1))
    prediction[0,0].backward()
    assert m.world.state_encoder[0].weight.grad.abs().sum()>0
    assert m.world.pad_encoder[0].weight.grad.abs().sum()>0
    assert m.confidence_heads[1][-1][-1].weight.grad.abs().sum()==0


def test_dynamics_remains_differentiable_with_frozen_weights():
    m=JointWorld(dict(hidden=16,layers=1,heads=2,kinematic_only=True,direct_displacement=True,contact_geometry=True))
    c=context_batch([interaction_context(example(),dict(clip_start=10,clip_end=20))],'cpu')
    m.requires_grad_(False)
    theta=torch.tensor([[3500.,.27,.8,.4]],requires_grad=True)
    pads=c['pads1'][:,0].clone().requires_grad_(True)
    y=m.world(c['states'][:,0],c['pads0'][:,0],pads,theta,
              {k:v[:,0] for k,v in c['geometry'].items()},c['dt'][:,0])
    g=torch.autograd.grad(y[...,:3].square().sum(),(theta,pads))
    assert all(torch.isfinite(t).all() and t.abs().sum()>0 for t in g)
    assert all(p.grad is None for p in m.parameters())


def test_four_columns_do_not_collapse_to_scalar_weight():
    q=torch.zeros(4,requires_grad=True)
    losses=[(q*torch.tensor([1.,2.,3.,4.])).sum(),(q*torch.tensor([9.,8.,7.,6.])).sum()]
    weights=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    assert torch.equal(confidence_gradient(losses,q,weights),torch.tensor([1.,8.,3.,6.]))


def test_context_uses_past_velocity_and_preserves_missing_masks():
    d=example();d['valid'][:]=False
    c=interaction_context(d,dict(clip_start=10,clip_end=20))
    assert not c['valid'].any()
    assert np.all(c['states'][...,3:6]==0)
    np.testing.assert_array_equal(c['pads0'][:,1,0],np.float32(.2))
    np.testing.assert_array_equal(c['pads0'][0,1,3:12].reshape(3,3),np.eye(3))
