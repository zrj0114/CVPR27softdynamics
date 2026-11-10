import numpy as np
import torch

from softdynamics.cma_search import BlockCMA
from softdynamics.joint_acceleration import expand_active_coordinates, normalized_weights
from softdynamics.joint_world import JointWorld, context_batch, interaction_context, save_joint, load_joint, confidence_loss
from test_joint_world import example


def test_two_heads_and_checkpoint_retain_explicit_parameter_scope(tmp_path):
    model=JointWorld(dict(hidden=16,layers=1,heads=2,kinematic_only=True,
                         direct_displacement=True,contact_geometry=True),['E_pa','nu'])
    context=context_batch([interaction_context(example(),dict(clip_start=10,clip_end=20))],'cpu')
    theta=torch.tensor([[3500.,.27,.8,.4]])
    expected=model.confidence(context,theta)
    assert expected.shape==(1,2)
    path=tmp_path/'model.pt'
    save_joint(path,model,dict(training_objects=[],validation_objects=[]),0,{})
    restored,bundle=load_joint(path,'cpu')
    assert restored.confidence_parameters==('E_pa','nu')
    torch.testing.assert_close(restored.confidence(context,theta),expected)

    class Labels:
        def batch(self,*args):
            # Invalid nuisance targets must never enter the two-parameter loss.
            return context,theta,torch.tensor([[.2,.7,float('nan'),float('nan')]]),torch.ones(1,4,dtype=torch.bool)
    loss=confidence_loss(restored,Labels(),[0],'cpu',np.ones(4))
    assert torch.isfinite(loss)
    loss.backward()
    assert restored.world.state_encoder[0].weight.grad.abs().sum()>0


def test_active_gradient_cannot_update_fixed_nuisance_parameters():
    active=torch.tensor([.001,-.002],requires_grad=True)
    full=expand_active_coordinates(active,[0,1])
    assert torch.equal(full[2:],torch.zeros(2))
    gradient=torch.autograd.grad((full*torch.tensor([2.,3.,999.,888.])).sum(),active)[0]
    torch.testing.assert_close(gradient,torch.tensor([2.,3.]))
    weights=normalized_weights(torch.tensor([[1.,4.],[3.,0.]]))
    torch.testing.assert_close(weights,torch.tensor([[.25,1.],[.75,0.]]))


def test_cma_only_searches_selected_material_axes_and_resumes(tmp_path):
    optimizer=BlockCMA(np.zeros(4),np.ones((2,2)),cycles=2,block_generations=1,population=4,
                       active_parameters=['E_pa','nu'])
    target=np.array([.007,-.008,0.,0.])
    seen=[]
    while not optimizer.done:
        proposed=optimizer.ask()
        assert np.array_equal(proposed['q'][:,2:],np.zeros((4,2)))
        seen.append(proposed['parameter'])
        losses=np.sum((proposed['q']-target)**2,axis=1)
        incumbent=np.sum((proposed['incumbent']-target)**2)
        optimizer.tell(np.stack([losses,losses],1),[incumbent,incumbent])
        path=tmp_path/'cma.pkl';optimizer.save(path);optimizer=BlockCMA.load(path)
    assert seen==['E_pa','nu','E_pa','nu']
    assert optimizer.logical_candidates==16
    assert np.array_equal(optimizer.current[2:],np.zeros(2))
