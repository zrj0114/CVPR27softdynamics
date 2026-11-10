from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from softdynamics.rectangular_grid import allocation, RectangularObservation


@pytest.mark.parametrize('shape,offset', [([64,64,64],[1,0,0]), ([64,64,64],[-1,0,0]),
                                       ([65,64,64],[0,0,0]), ([1024,1024,1024],[0,0,0])])
def test_refuses_allocations_that_lose_original_cells_or_exceed_budget(shape,offset):
    with pytest.raises(ValueError):
        allocation(shape,offset,64)


def test_negative_coordinate_g2p_preserves_uniform_translation_and_deformation():
    import warp as wp
    root=Path(__file__).resolve().parents[1]/'runtime/gs-layer'
    sys.path.insert(0,str(root))
    from rectangular_g2p import g2p_rectangular, MPMModelStruct, MPMStateStruct
    wp.init()
    x=np.array([[-2.2,0.,-3.2],[-.6,0.,-.6],[.6,0.,.6],[2.2,0.,3.2]],dtype='f')
    velocity=np.array([.125,-.25,.3],dtype='f')
    shape=(16,16,32)
    state=MPMStateStruct();next_state=MPMStateStruct();model=MPMModelStruct()
    model.grid_dim_x,model.grid_dim_y,model.grid_dim_z=shape
    model.dx=model.inv_dx=1.;model.n_particles=len(x)
    state.particle_selection=wp.zeros(len(x),dtype=int,device='cpu')
    state.particle_x=wp.array(x,dtype=wp.vec3,device='cpu')
    state.particle_F=wp.array(np.tile(np.eye(3,dtype='f'),(len(x),1,1)),dtype=wp.mat33,device='cpu')
    state.grid_v_out=wp.array(np.broadcast_to(velocity,shape+(3,)).copy(),dtype=wp.vec3,device='cpu')
    for name,dtype in [('particle_x',wp.vec3),('particle_v',wp.vec3),('particle_C',wp.mat33),('particle_F_trial',wp.mat33)]:
        setattr(next_state,name,wp.zeros(len(x),dtype=dtype,device='cpu'))
    wp.launch(g2p_rectangular,dim=len(x),inputs=[state,next_state,model,.1,wp.vec3i(8,8,16)],device='cpu')
    np.testing.assert_allclose(next_state.particle_x.numpy(),x+.1*velocity,rtol=0,atol=3e-7)
    np.testing.assert_allclose(next_state.particle_v.numpy(),np.broadcast_to(velocity,x.shape),rtol=0,atol=3e-7)
    np.testing.assert_allclose(next_state.particle_C.numpy(),0,rtol=0,atol=3e-7)
    np.testing.assert_allclose(next_state.particle_F_trial.numpy(),np.tile(np.eye(3),(len(x),1,1)),rtol=0,atol=3e-7)


def test_observer_checks_unsampled_particles_and_signed_domain_bounds():
    import warp as wp
    wp.init()
    x=np.array([[-2.,0.,-3.],[-.6,0.,-.6],[.6,0.,.6],[2.,0.,3.]],dtype='f')
    F=np.tile(np.eye(3,dtype='f'),(len(x),1,1))
    state=SimpleNamespace(particle_x=wp.array(x,dtype=wp.vec3,device='cpu'),
                          particle_v=wp.zeros(len(x),dtype=wp.vec3,device='cpu'),
                          particle_C=wp.zeros(len(x),dtype=wp.mat33,device='cpu'),
                          particle_F_trial=wp.array(F,dtype=wp.mat33,device='cpu'))
    sim=SimpleNamespace(states=[state],current=0,model=SimpleNamespace(dx=1.),
                        rectangular_lower=np.array([-8,-8,-16],dtype='f'),
                        rectangular_upper=np.array([8,8,16],dtype='f'))
    observer=RectangularObservation(sim,np.array([0]))
    assert observer.check()['valid']
    F[3,0,0]=-1;state.particle_F_trial.assign(F)
    assert not observer.check()['valid']
    F[3,0,0]=1;state.particle_F_trial.assign(F)
    x[3,0]=-6;state.particle_x.assign(x)
    result=observer.check()
    assert not result['valid'] and result['failure_kind']=='domain_boundary'
