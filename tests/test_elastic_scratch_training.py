import numpy as np
import pytest
import torch

from softdynamics.elastic_data import ElasticSpace
from softdynamics.elastic_train import TrainingJoint, atomic_checkpoint, horizon_for_step, clip_gradients, joint_gradients
from softdynamics.joint_world import interaction_context, context_batch
from test_joint_world import example


def test_only_two_variables_and_fixed_contact():
    config=dict(baseline=dict(E_pa=3500.,nu=.27,ground_friction=.85,
        velocity_decay_rate_s_inv=.36,gripper_static_friction=2.),
        reference_E_pa=3500.,reference_nu=.27,nu_scale=.2)
    space=ElasticSpace(config)
    for q in ([-1.,-.8],[0.,0.],[1.,.7]):
        p=space.parameters(q)
        np.testing.assert_allclose(space.coordinate(p),q,atol=1e-12)
        assert p['ground_friction']==.85 and p['velocity_decay_rate_s_inv']==.36
    with pytest.raises(ValueError):space.parameters([0.,0.,0.,0.])


def test_joint_training_changes_both_branches_and_retains_material_gradients():
    torch.set_num_threads(1);torch.manual_seed(3)
    m=TrainingJoint(dict(hidden=16,layers=1,heads=2,kinematic_only=True,
        direct_displacement=True,contact_geometry=True,material_dim=2),('E_pa','nu'))
    c=context_batch([interaction_context(example(),dict(clip_start=10,clip_end=20))],'cpu')
    theta=torch.tensor([[3500.,.27]])
    geo={k:v[:,0] for k,v in c['geometry'].items()}
    states=torch.stack((c['states'][:,0],c['states'][:,0]+.001),1)
    batch=dict(states=states,pads=torch.stack((c['pads0'][:,0],c['pads1'][:,0]),1),
               theta=theta,times=torch.tensor([[0.,1/30]]))
    dynamic,conf=m(batch,geo,(c,theta,torch.tensor([[.2,.8]]),torch.ones(1,2,dtype=torch.bool)))
    (dynamic+conf).backward()
    assert m.world.material_encoder[0].in_features==2
    assert m.world.output[-1][-1].weight.grad.abs().sum()>0
    assert all(h[-1][-1].weight.grad.abs().sum()>0 for h in m.confidence_heads)
    assert m.world.state_encoder[0].weight.grad.abs().sum()>0
    m.requires_grad_(False);theta=theta.clone().requires_grad_(True)
    y=m.world(batch['states'][:,0],batch['pads'][:,0],batch['pads'][:,1],theta,geo,torch.tensor([1/30]))
    grad=torch.autograd.grad(y[...,:3].square().sum(),theta)[0]
    assert grad.shape==(1,2) and torch.isfinite(grad).all() and torch.all(grad.abs()>0)


def test_exact_40000_curriculum_and_two_checkpoints(tmp_path):
    horizons=[horizon_for_step(i) for i in range(40000)]
    assert {h:horizons.count(h) for h in set(horizons)}=={1:2000,8:8000,16:15000,32:15000}
    atomic_checkpoint(tmp_path/'latest.pt',{'step':100})
    atomic_checkpoint(tmp_path/'latest.pt',{'step':200})
    atomic_checkpoint(tmp_path/'best.pt',{'step':100})
    assert sorted(p.name for p in tmp_path.iterdir())==['best.pt','latest.pt']
    assert torch.load(tmp_path/'latest.pt',weights_only=True)['step']==200
    with pytest.raises(ValueError):atomic_checkpoint(tmp_path/'epoch_200.pt',{})


def test_gradient_norm_overflow_preserves_direction():
    parameter=torch.nn.Parameter(torch.zeros(2))
    parameter.grad=torch.tensor([3e20,4e20])
    assert not torch.isfinite(torch.linalg.vector_norm(parameter.grad))
    norm=clip_gradients([parameter])
    assert torch.isfinite(norm)
    torch.testing.assert_close(parameter.grad,torch.tensor([.6,.8]))


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_invalid_gradient_values_are_never_accepted(invalid):
    parameter=torch.nn.Parameter(torch.zeros(2));parameter.grad=torch.tensor([1.,invalid])
    with pytest.raises(RuntimeError,match='non-finite'):clip_gradients([parameter])
    assert parameter.grad[0]==1.


def test_normal_gradient_clipping_matches_torch():
    a=torch.nn.Parameter(torch.zeros(2));b=torch.nn.Parameter(torch.zeros(2))
    a.grad=torch.tensor([3.,4.]);b.grad=a.grad.clone()
    expected=torch.nn.utils.clip_grad_norm_([a],1.,error_if_nonfinite=True)
    actual=clip_gradients([b])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_backward_overflow_retries_same_loss_before_optimizer_step():
    class LargeDerivative(torch.nn.Module):
        def __init__(self):
            super().__init__();self.value=torch.nn.Parameter(torch.zeros(()));self.calls=0
        def forward(self,*args):
            self.calls+=1
            return (self.value*1e20)*1e20,self.value.square()
    model=LargeDerivative();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    dynamic,conf,norm,scale=joint_gradients(model,model,optimizer,None,None,None)
    assert model.calls==2 and scale==2.**-16
    assert model.value==0 and dynamic==0 and conf==0
    assert torch.isfinite(norm) and norm>1e39
    torch.testing.assert_close(model.value.grad,torch.ones(()))
    optimizer.step()
    torch.testing.assert_close(model.value,torch.tensor(-.1))
