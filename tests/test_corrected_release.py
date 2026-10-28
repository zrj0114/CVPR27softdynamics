import sys
from pathlib import Path

import numpy as np
import torch

from softdynamics.motion import CausalMotion, relative_actions
from softdynamics.adapters.pad_empm import simulation_domain
from softdynamics.teacher import select_windows
from softdynamics.shared_pipeline import domain_failure


def test_independent_pads_and_missing_reference():
    a = torch.zeros(16,52)
    for side in range(4):
        a[:,side*13:side*13+12] = torch.eye(4)[:3].reshape(12)
        a[:,side*13+12] = 1
    a[:,3] = torch.arange(16)*.01
    a[:,16] = .05
    a[-1,12] = 0
    a[-1,:12] = 0
    z = relative_actions(a,-1)
    assert torch.equal(z,a)
    assert torch.all(z[-1,:12]==0)
    assert torch.all(z[:,16]==.05)
    model = CausalMotion(action_dim=52)
    assert model.action(torch.cat([z,torch.zeros(16,1)],1)).shape==(16,64)


def test_domain_ignores_distant_nonoperation_controller():
    initial = np.array([[0,0,0],[.1,.1,.1]])
    poses = np.broadcast_to(np.eye(4),(20,1,2,4,4)).copy()
    poses[:5,...,:3,3] = 10
    valid = np.ones((20,1,2),bool)
    grid,shift = simulation_domain(initial,poses,valid,.003125,operation=(5,15))
    assert grid<=128
    assert (initial+shift>2*.003125).all()
    assert (initial+shift<grid*.003125-2*.003125).all()


def test_explicit_windows_keep_phase_mass():
    windows=[dict(index=i,end=i,start=i-1,phase=p,weight=w) for i,p,w in
             [(2,'before',.1),(8,'interaction',.45),(16,'release',.45)]]
    result=select_windows({'frames':np.arange(20)},{'teacher_windows':windows},
                           {'before':.1,'interaction':.45,'release':.45})
    assert result==windows


def test_only_finite_positive_boundary_failures_are_retried():
    for jac,finite,margin,expected in [(0.1,True,.00625,True),
                                     (-.01,True,.00625,False),
                                     (.1,False,.00625,False),
                                     (.1,True,.05,False)]:
        failure={'error':str(dict(finite=finite,min_J=jac,domain_margin_m=margin))}
        assert domain_failure(failure)==expected
    initial=np.array([[0,0,0],[.1,.1,.1]])
    poses=np.broadcast_to(np.eye(4),(2,1,2,4,4)).copy()
    valid=np.ones((2,1,2),bool)
    grid,shift=simulation_domain(initial,poses,valid,.003125,margin=.075)
    expanded,moved=simulation_domain(initial,poses,valid,.003125,margin=.15)
    assert expanded>grid
    assert (initial+moved>initial+shift).all()


def test_pad_contact_impulse_is_finite_unilateral_and_masked():
    root=Path(__file__).resolve().parents[1]/'runtime'
    sys.path.insert(0,str(root/'gs-layer'))
    import warp as wp
    from material_contact_kernels import MaterialContact,MPMStateStruct,MPMModelStruct
    from pad_contact_kernels import accumulate_pad_contact
    wp.init()
    n=8
    state=MPMStateStruct();model=MPMModelStruct();c=MaterialContact()
    state.grid_m=wp.ones((n,n,n),dtype=float,device='cpu')
    velocity=np.zeros((n,n,n,3),np.float32);velocity[...,0]=-1
    state.grid_v_out=wp.array(velocity,dtype=wp.vec3,device='cpu')
    c.enabled=wp.array([1,0],dtype=int,device='cpu')
    transform=np.array([[.01,.01,.01,0,0,0,1],[0,0,0,0,0,0,1]],np.float32)
    c.transforms=wp.array(transform,dtype=wp.transform,device='cpu')
    c.linear=wp.zeros(2,dtype=wp.vec3,device='cpu');c.angular=wp.zeros(2,dtype=wp.vec3,device='cpu')
    c.momentum=wp.zeros((n,n,n),dtype=wp.vec3,device='cpu');c.weight=wp.zeros((n,n,n),dtype=float,device='cpu')
    c.inv_dx=100.;c.frame_time=0.;c.footprint_y=.032;c.footprint_z=.0125;c.radius=.02
    c.contact_friction=2.;c.sliding_friction=.8
    def launch():
        c.momentum.zero_();c.weight.zero_()
        wp.launch(accumulate_pad_contact,dim=(n,n,n),inputs=[0.,.001,state,model,c],device='cpu')
        return c.momentum.numpy()
    delta=launch()
    np.testing.assert_allclose(delta[0,1,1],[1,0,0])
    np.testing.assert_allclose(delta[0,5,1],[0,0,0])
    velocity[...,0]=1;state.grid_v_out.assign(velocity)
    assert not np.any(launch())
    c.enabled.zero_();velocity[...,0]=-1;state.grid_v_out.assign(velocity)
    assert not np.any(launch())
    c.enabled.assign(np.array([0,1],np.int32));velocity[...,0]=1;state.grid_v_out.assign(velocity)
    np.testing.assert_allclose(launch()[1,1,1],[-1,0,0])
