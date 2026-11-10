import pytest
import torch

from softdynamics.shared_world_physics import geometric_deformation,corotated_stress,physical_state_features
from softdynamics.shared_world_model import SharedWorldModel,rollout,apply_gripper_action,rotation_matrix
from softdynamics.shared_world_training import load_model


def example(dtype=torch.float64):
    torch.manual_seed(173)
    canonical=torch.randn(1,9,3,dtype=dtype)*.1
    neighbors=torch.stack([torch.tensor([k for k in range(9) if k!=i]) for i in range(9)])[None]
    geometry=dict(canonical=canonical,neighbors=neighbors,scale_m=torch.tensor([.15],dtype=dtype),
        ground_y=torch.tensor([-.2],dtype=dtype),surface_mask=torch.ones(1,9,dtype=torch.bool))
    state=torch.zeros(1,9,24,dtype=dtype)
    state[...,:3]=canonical*1.03
    state[...,3:6]=torch.randn(1,9,3,dtype=dtype)*.01
    pads=torch.zeros(1,4,13,dtype=dtype)
    pads[...,:3]=torch.randn(1,4,3,dtype=dtype)*.05
    pads[...,3:12]=torch.eye(3,dtype=dtype).flatten()
    pads[...,12]=1
    theta=torch.tensor([[3500.,.27,.8,.4]],dtype=dtype)
    return state,pads,theta,geometry


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_geometry_rest_deformation_is_identity_and_translation_invariant(dtype):
    state,pads,theta,g=example(dtype)
    rest=geometric_deformation(g['canonical'],g['canonical'],g['neighbors'])
    torch.testing.assert_close(rest,torch.eye(3,dtype=dtype).expand_as(rest),atol=0,rtol=0)
    shifted=geometric_deformation(g['canonical']+1,g['canonical'],g['neighbors'])
    torch.testing.assert_close(shifted,rest,atol=2e-6 if dtype==torch.float32 else 1e-12,rtol=0)


def test_corotated_stress_matches_rigid_rotation_and_uniform_stretch():
    _,_,theta,_=example()
    rotation=rotation_matrix(theta.new_tensor([[.3,-.2,.8]]))[:,None]
    torch.testing.assert_close(corotated_stress(rotation,theta),torch.zeros_like(rotation),atol=1e-8,rtol=0)
    stretch=1.05
    F=torch.eye(3,dtype=theta.dtype)[None,None]*stretch
    E,nu=theta[0,:2]
    mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));J=stretch**3
    expected=(2*mu*(stretch-1)*stretch+lam*J*(J-1))*torch.eye(3,dtype=theta.dtype)[None,None]
    torch.testing.assert_close(corotated_stress(F,theta),expected,atol=1e-8,rtol=1e-10)


def test_corotated_stress_has_finite_correct_gradient_at_identity():
    _,_,theta,_=example()
    F=torch.eye(3,dtype=theta.dtype)[None,None].requires_grad_()
    gradient=torch.autograd.grad(corotated_stress(F,theta).sum(),F)[0]
    direction=torch.arange(9,dtype=theta.dtype).reshape_as(F)/100
    eps=1e-5
    finite=(corotated_stress(F+eps*direction,theta).sum()-corotated_stress(F-eps*direction,theta).sum())/(2*eps)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(finite,(gradient*direction).sum(),atol=1e-6,rtol=1e-7)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_physical_feature_initialization_preserves_parent_and_input_gradients(dtype):
    state,pads,theta,g=example(dtype)
    source=SharedWorldModel(16,1,4,direct_displacement=True,contact_geometry=True).to(dtype=dtype)
    augmented=SharedWorldModel(16,1,4,direct_displacement=True,contact_geometry=True,physical_features=True).to(dtype=dtype)
    augmented.initialize_physical_features(source.state_dict())
    source.requires_grad_(False);augmented.requires_grad_(False)
    theta.requires_grad_()
    action=torch.full((1,2,7),.03,dtype=dtype,requires_grad=True)
    end=apply_gripper_action(pads,action,g['scale_m'])
    poses=torch.stack((pads,end,end),1)
    times=torch.arange(3,dtype=dtype)[None]/30
    before=rollout(source,state,poses,theta,g,times)
    after=rollout(augmented,state,poses,theta,g,times)
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    gradients=[torch.autograd.grad(value[...,:3].square().sum(),(theta,action),retain_graph=True) for value in [before,after]]
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_physical_features_use_observed_geometry_and_preserve_candidate_derivatives():
    state,pads,theta,g=example()
    state.requires_grad_();theta.requires_grad_()
    dt=theta.new_tensor([1/30])
    features=physical_state_features(state,theta,g,dt)
    assert features.shape==(1,9,15) and torch.isfinite(features).all()
    changed=state.detach().clone();changed[...,6:]+=100
    torch.testing.assert_close(physical_state_features(changed,theta,g,dt),features,atol=0,rtol=0)
    gradients=torch.autograd.grad(features.square().sum(),(state,theta))
    directions=(torch.full_like(state,.01),theta.new_tensor([[100.,.01,.02,.03]]))
    eps=1e-5
    positive=physical_state_features(state+eps*directions[0],theta+eps*directions[1],g,dt).square().sum()
    negative=physical_state_features(state-eps*directions[0],theta-eps*directions[1],g,dt).square().sum()
    expected=sum((gradient*direction).sum() for gradient,direction in zip(gradients,directions))
    torch.testing.assert_close((positive-negative)/(2*eps),expected,atol=1e-8,rtol=1e-5)
    assert all(torch.isfinite(gradient).all() and gradient.abs().max()>0 for gradient in gradients)


@pytest.mark.parametrize('reference',[None,[3547.0699,.278514,.853391,.359543]])
def test_physical_feature_checkpoint_preserves_reference_and_learned_encoder(tmp_path,reference):
    model=SharedWorldModel(16,1,4,direct_displacement=True,physical_features=True,physical_feature_reference=reference)
    with torch.no_grad():model.physical_encoder[-1].weight.normal_(std=.01)
    path=tmp_path/'world.pt'
    torch.save(dict(architecture=model.architecture,model=model.state_dict()),path)
    restored,_=load_model(path,'cpu')
    assert restored.architecture==model.architecture
    state,pads,theta,g=example(torch.float32)
    torch.testing.assert_close(model(state,pads,pads,theta,g,theta.new_tensor([1/30])),
        restored(state,pads,pads,theta,g,theta.new_tensor([1/30])),atol=0,rtol=0)


@pytest.mark.parametrize('extra',[{},dict(direct_displacement=True,affine_material=True),
    dict(direct_displacement=True,residual_material=True),dict(direct_displacement=True,deformation_memory=True)])
def test_physical_features_do_not_silently_change_other_material_representations(extra):
    with pytest.raises(ValueError,match='Physical features'):
        SharedWorldModel(physical_features=True,**extra)
