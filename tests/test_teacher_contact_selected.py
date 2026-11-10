import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation,Slerp

from softdynamics.teacher_contact import (CONTACT_ID,SUPPORT_ID,FACE_SIZE_M,GRID_DX_M,
    selected_architecture,require_selected_architecture,pack_pads,replace_visual_actions,
    load_completed_actions,verify_selected_kernels)
from softdynamics.teacher_contact_support import finite_pad_grid_swept_support
from softdynamics.shared_world_model import SharedWorldModel
from softdynamics.world_data import controller_features

def inputs(points):
    state=torch.zeros(1,len(points),24,dtype=torch.float64)
    state[0,:,:3]=torch.tensor(points,dtype=torch.float64)
    pads=torch.zeros(1,4,13,dtype=state.dtype);pads[...,3:12]=torch.eye(3).flatten();pads[0,0,12]=1
    return state,pads,torch.ones(1,dtype=state.dtype),torch.full((1,),1/30,dtype=state.dtype)

def test_finite_support_excludes_noncontact_and_respects_grid_transfer_neighborhood():
    s,p,scale,dt=inputs([[-.003,0,0],[.03,0,0],[-.003,.05,0],[-.003,0,.03],[.004,0,0]])
    assert finite_pad_grid_swept_support(s,p,p,scale,dt).squeeze(0)[:,0].tolist()==[1,0,0,0,1]
    assert not finite_pad_grid_swept_support(s,p,p,scale,dt)[...,1:].any()

def test_both_pad_side_normal_conventions():
    s,p,scale,dt=inputs([[-.010,0,0],[.010,0,0]])
    p[0,1,12]=1
    g=finite_pad_grid_swept_support(s,p,p,scale,dt)[0,:,:2]
    assert g.tolist()==[[1,0],[0,1]]

def test_fast_translation_and_particle_crossings_are_not_lost_between_endpoints():
    s,p,scale,dt=inputs([[0,0,0]]);a=p.clone();b=p.clone();a[0,0,0]=-.2;b[0,0,0]=.2
    assert finite_pad_grid_swept_support(s,a,a,scale,dt)[0,0,0]==0
    assert finite_pad_grid_swept_support(s,b,b,scale,dt)[0,0,0]==0
    assert finite_pad_grid_swept_support(s,a,b,scale,dt)[0,0,0]==1
    s[0,0,0]=-.2;s[0,0,3]=.2
    assert finite_pad_grid_swept_support(s,p,p,scale,dt)[0,0,0]==1

def test_rotation_sweep_reaches_point_outside_both_endpoint_faces():
    s,p,scale,dt=inputs([[-.003,.03/2**.5,.03/2**.5]])
    b=p.clone();b[0,0,3:12]=torch.tensor(Rotation.from_rotvec([np.pi/2,0,0]).as_matrix().reshape(9))
    assert finite_pad_grid_swept_support(s,p,p,scale,dt)[0,0,0]==0
    assert finite_pad_grid_swept_support(s,b,b,scale,dt)[0,0,0]==0
    assert finite_pad_grid_swept_support(s,p,b,scale,dt)[0,0,0]==1

@pytest.mark.parametrize('side',[0,1])
def test_sampled_teacher_contacts_and_grid_support_are_covered_by_sweep(side):
    rng=np.random.default_rng(725+side)
    for _ in range(12):
        R0=Rotation.random(random_state=rng);axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        R1=Rotation.from_rotvec(axis*rng.uniform(0,np.pi))*R0
        c0=rng.uniform(-.1,.1,3);c1=rng.uniform(-.1,.1,3);times=rng.uniform(0,1,24)
        rotations=Slerp([0,1],Rotation.from_matrix([R0.as_matrix(),R1.as_matrix()]))(times).as_matrix()
        local=rng.uniform([-2*GRID_DX_M,-FACE_SIZE_M[0]/2,-FACE_SIZE_M[1]/2],[0,FACE_SIZE_M[0]/2,FACE_SIZE_M[1]/2],(24,3))
        if side:local[:,0]*=-1
        node=np.einsum('nij,nj->ni',rotations,local)+c0+times[:,None]*(c1-c0)
        particles=node+rng.uniform(-1.5*GRID_DX_M,1.5*GRID_DX_M,(24,3))
        s,p,scale,dt=inputs(particles);p[...,12]=0;p[0,side,12]=1;p[0,side,:3]=torch.tensor(c0);p[0,side,3:12]=torch.tensor(R0.as_matrix().reshape(9))
        b=p.clone();b[0,side,:3]=torch.tensor(c1);b[0,side,3:12]=torch.tensor(R1.as_matrix().reshape(9))
        assert finite_pad_grid_swept_support(s,p,b,scale,dt)[0,:,side].all()

def test_metric_units_and_exact_half_turn_are_finite():
    s,p,scale,dt=inputs([[-.003,.02,.005],[.03,0,0]])
    b=p.clone();b[0,0,3:12]=torch.tensor(Rotation.from_rotvec([np.pi,0,0]).as_matrix().reshape(9))
    first=finite_pad_grid_swept_support(s,p,b,scale,dt)
    s[...,:6]*=10;p[...,:3]*=10;b[...,:3]*=10
    torch.testing.assert_close(first,finite_pad_grid_swept_support(s,p,b,scale/10,dt),atol=0,rtol=0)

def fixture(selected=True,far=False):
    torch.manual_seed(725)
    state,pads,scale,dt=inputs([[.03 if far else -.003,y,0] for y in [-.01,0,.01,.02]])
    geo=dict(canonical=state[...,:3].clone(),neighbors=torch.tensor([[[1,2],[0,2],[1,3],[1,2]]]),
        ground_y=torch.tensor([-.1],dtype=state.dtype),scale_m=scale,surface_mask=torch.ones(1,4,dtype=torch.bool))
    arch=dict(hidden=16,layers=1,heads=4,direct_displacement=True,kinematic_only=True,material_dim=2,contact_geometry=True,pad_locality=True)
    if selected:arch=selected_architecture(arch)
    model=SharedWorldModel(**arch).double().eval().requires_grad_(False)
    theta=state.new_tensor([[3500.,.27]],requires_grad=True)
    return model,state,pads,theta,geo,dt

def test_actual_network_has_no_remote_pad_action_injection_and_keeps_noncontact_motion():
    model,s,p,theta,g,dt=fixture(far=True);moved=p.clone();moved[0,0,0]+=.001
    static=model(s,p,p,theta,g,dt);moving=model(s,p,moved,theta,g,dt)
    torch.testing.assert_close(static,moving,atol=0,rtol=0)
    # Locality constrains the pad contribution, not the entire predicted motion.
    assert not torch.equal(static[...,:3],s[...,:3])

def test_near_action_and_material_derivatives_remain_available():
    model,s,p,theta,g,dt=fixture();moving=p.clone().requires_grad_()
    result=model(s,p,moving,theta,g,dt)[...,:3].square().sum()
    pg,tg=torch.autograd.grad(result,(moving,theta))
    assert torch.isfinite(pg).all() and pg[0,0,:3].abs().sum()>0
    assert pg[0,1:].abs().max()==0 and torch.isfinite(tg).all() and (tg.abs()>0).all()
    direction=theta.new_tensor([[100.,.01]]);eps=1e-5
    hi=model(s,p,p,theta+eps*direction,g,dt)[...,:3].square().sum()
    lo=model(s,p,p,theta-eps*direction,g,dt)[...,:3].square().sum()
    torch.testing.assert_close((hi-lo)/(2*eps),(tg*direction).sum(),rtol=3e-4,atol=1e-12)

def test_checkpoint_architecture_distinguishes_selected_and_legacy_behavior():
    new,*_=fixture();old,*_=fixture(selected=False)
    require_selected_architecture(new.architecture)
    with pytest.raises(ValueError,match='predates'):require_selected_architecture(old.architecture)
    assert 'contact_contract' not in old.architecture
    rebuilt=SharedWorldModel(**new.architecture).double();rebuilt.load_state_dict(new.state_dict())
    assert rebuilt.contact_contract==CONTACT_ID and rebuilt.pad_locality_geometry==SUPPORT_ID

def test_selected_kernel_source_is_frozen():
    verify_selected_kernels(Path(__file__).resolve().parents[1]/'runtime')

def test_one_action_packer_is_shared_by_dynamics_and_confidence():
    poses=np.broadcast_to(np.eye(4),(4,1,2,4,4)).copy();poses[...,0,3]=np.arange(4)[:,None,None]*.001
    a=dict(poses=poses,valid=np.ones((4,1,2),bool),source_timestamps_us=np.array([0,30000,65000,100000]))
    origin=np.array([.1,.2,.3]);scale=.2
    p=pack_pads(a,origin,scale)
    np.testing.assert_array_equal(p,controller_features(SimpleNamespace(poses=poses,valid=a['valid'],times=np.arange(4)),origin,scale))
    visual=dict(origin_m=origin,scale_m=scale,times=a['source_timestamps_us']*1e-6,actions=np.zeros((4,52)))
    first=replace_visual_actions(visual,a)['actions'].reshape(4,4,13)
    matrices=first[...,:12].reshape(4,4,3,4)
    recovered=np.concatenate([matrices[...,3],matrices[...,:3].reshape(4,4,9),first[...,12:]],-1)
    np.testing.assert_array_equal(p,recovered)
    assert not p[:,2:,12].any()

def test_completed_export_not_observed_mask_controls_gripper_presence():
    a,meta=load_completed_actions('047-rectangle-sponge',0)
    assert a['completed'].sum()==13 and a['valid'].all()
    packed=pack_pads(a,np.zeros(3),1.)
    assert packed[:,:2,12].all() and not packed[:,2:,12].any()
    rabbit,meta=load_completed_actions('125-rabbit',5)
    assert rabbit['poses'].shape[1]==1 and meta['grippers']==1
    assert not pack_pads(rabbit,np.zeros(3),1.)[:,2:,12].any()
