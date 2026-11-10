import pytest
import torch

from softdynamics.shared_world_inverse import phase_objective
from softdynamics.shared_world_objective_audit import phase_coefficients
from softdynamics.shared_world_material_windows import step_reference_rollout


def test_local_contributions_preserve_full_episode_phase_weighting():
    frames=torch.tensor([0,2,3,5,7,8,11,12,15])
    weights=dict(before=.1,interaction=.45,release=.45)
    costs=torch.tensor([2.,4.,8.,16.,32.,64.,128.,256.,512.],dtype=torch.float64,requires_grad=True)
    coefficients=phase_coefficients(frames,3,8,weights)
    expected=phase_objective(costs,frames,3,8,weights)
    torch.testing.assert_close((coefficients*costs).sum(),expected)
    full_gradient=torch.autograd.grad(expected,costs)[0]
    torch.testing.assert_close(coefficients,full_gradient)
    # A short interaction window keeps the original four-frame denominator.
    window=(frames>=5)&(frames<=7)
    assert float(coefficients[window].sum())==pytest.approx(.45/2)
    assert float(coefficients.sum())==pytest.approx(1.)


def test_missing_full_episode_phase_cannot_be_renormalized_as_a_window():
    with pytest.raises(ValueError,match='original scored phase'):
        phase_coefficients(torch.tensor([4,5,6]),3,8,dict(before=.1,interaction=.45,release=.45))


@pytest.mark.parametrize('checkpointed',[False,True])
def test_step_reference_preserves_baseline_and_differentiates_the_whole_history(checkpointed):
    class LinearDynamics(torch.nn.Module):
        def forward(self,state,p0,p1,theta,geometry,dt):
            kinematics=state[...,:6]*theta[:,None,:1]+theta[:,None,1:2]
            return torch.cat((kinematics,state[...,6:]),-1)
    model=LinearDynamics()
    reference=torch.zeros(1,5,1,24,dtype=torch.float64)
    reference[0,:,0,0]=torch.tensor([1.,2.,4.,8.,16.])
    pads=torch.zeros(1,5,4,13,dtype=torch.float64);times=torch.arange(5,dtype=torch.float64)[None]
    theta=torch.tensor([[.7,.1,0.,0.]],dtype=torch.float64,requires_grad=True)
    with torch.no_grad():
        predictions=torch.stack([model(reference[:,t],pads[:,t],pads[:,t+1],theta,{},times[:,t+1]-times[:,t])[...,:6] for t in range(4)],1)
    states=step_reference_rollout(model,reference,predictions,pads,theta,{},times,checkpointed)
    torch.testing.assert_close(states[...,:6],reference[...,:6],rtol=0,atol=0)
    gradient=torch.autograd.grad(states[0,-1,0,0],theta)[0]
    # All four transitions contribute: resetting at the last frame would give
    # derivatives 8 and 1 instead of these full-history values.
    torch.testing.assert_close(gradient,theta.new_tensor([[12.123,2.533,0.,0.]]))
    with pytest.raises(ValueError,match='Incomplete'):
        step_reference_rollout(model,reference,predictions[:,:-1],pads,theta,{},times)


@pytest.mark.parametrize('checkpointed',[False,True])
def test_deformation_reference_preserves_F_and_its_full_history_gradient(checkpointed):
    class DeformationDynamics(torch.nn.Module):
        deformation_memory=True
        kinematic_only=False
        def forward(self,state,p0,p1,theta,geometry,dt):
            result=state.clone()
            result[...,0]=state[...,0]+state[...,6]
            result[...,6]=state[...,6]+theta[:,None,0]+p1[:,None,0,0]
            return result
    model=DeformationDynamics()
    reference=torch.zeros(1,5,1,24,dtype=torch.float64)
    reference[0,:,0,0]=torch.tensor([1.,2.,4.,8.,16.])
    reference[0,:,0,6]=torch.tensor([.1,.2,.4,.8,1.6])
    pads=torch.zeros(1,5,4,13,dtype=torch.float64,requires_grad=True)
    theta=torch.tensor([[.3,0.,0.,0.]],dtype=torch.float64,requires_grad=True)
    times=torch.arange(5,dtype=torch.float64)[None]
    with torch.no_grad():
        predictions=torch.stack([model(reference[:,t],pads[:,t],pads[:,t+1],theta,{},times[:,t+1]-times[:,t])[...,:15] for t in range(4)],1)
    result=step_reference_rollout(model,reference,predictions,pads,theta,{},times,checkpointed)
    torch.testing.assert_close(result[...,:15],reference[...,:15],atol=0,rtol=0)
    material_gradient,action_gradient=torch.autograd.grad(result[0,-1,0,0],(theta,pads))
    torch.testing.assert_close(material_gradient,theta.new_tensor([[6.,0.,0.,0.]]))
    torch.testing.assert_close(action_gradient[0,:,0,0],theta.new_tensor([0.,3.,2.,1.,0.]))
    with pytest.raises(ValueError,match='representation'):
        step_reference_rollout(model,reference,predictions[...,:6],pads,theta,{},times)
