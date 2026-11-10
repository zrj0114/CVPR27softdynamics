import numpy as np
import pytest
import torch

from softdynamics.elastic_world_data import MaterialPairs
from softdynamics.elastic_world_objective import autoregressive, pair_losses, curriculum
from softdynamics.elastic_world_search import optimize


class AccumulatingDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(.3))
        self.register_buffer('loss_scale', torch.ones(24))

    def forward(self, state, pads0, pads1, theta, geometry, dt):
        change = state.new_zeros(state.shape)
        change[..., 0] = theta[:, 0, None]*self.rate
        return state+change


def example():
    batch = dict(states=torch.zeros(2,6,3,24), pads=torch.zeros(2,6,4,13),
        times=torch.arange(6).float().expand(2,-1), theta=torch.tensor([[1.,0.],[2.,0.]],requires_grad=True),
        phase=torch.tensor([0,1,1,2,2]), parameter_distance=torch.tensor(1.))
    return batch, dict(scale_m=torch.ones(2))


def test_full_prefix_derivative_keeps_all_history():
    model = AccumulatingDynamics()
    batch, geo = example()
    prediction = autoregressive(model,batch,geo,gradients=True)
    derivative = torch.autograd.grad(prediction[0,-1,0,0],batch['theta'],retain_graph=True)[0]
    assert float(derivative[0,0]) == pytest.approx(5*.3)
    prediction.sum().backward()
    assert model.rate.grad != 0
    # Future teacher targets cannot affect predictions.
    batch['states'][:,1:] = 1000
    other = autoregressive(model,batch,geo,gradients=False)
    torch.testing.assert_close(prediction,other)


def test_response_loss_is_signed_and_backpropagates_to_dynamics():
    model = AccumulatingDynamics()
    batch, geo = example()
    predicted = autoregressive(model,batch,geo,gradients=True)
    batch = dict(batch, states=torch.cat((batch['states'][:,:1].clone(),predicted.detach()),1))
    _, correct = pair_losses(predicted,batch,geo,model)
    _, reversed_response = pair_losses(predicted.flip(0),batch,geo,model)
    assert correct == 0 and float(reversed_response.detach()) == pytest.approx(4.)
    batch = dict(batch, states=batch['states']*2)
    _, mismatch = pair_losses(predicted,batch,geo,model)
    mismatch.backward()
    assert torch.isfinite(model.rate.grad) and model.rate.grad < 0


def test_pair_rejects_initial_state_or_action_changes():
    source = dict(minimum_start=0,pads=np.zeros((2,4,13)),times=np.arange(2),phase=np.zeros(2),
        origin_m=np.zeros(3),scale_m=1.,state=np.zeros((2,3,24)),theta=np.array([3500.,.27]),
        record=dict(source_hashes={'initial':'same'}),task=dict(parameters={'E_pa':3500.,'nu':.27,'friction':1.},dtmax=.001))
    import copy
    other = copy.deepcopy(source)
    other['theta'][0] = 5000
    assert MaterialPairs.mismatch(source,other) is None
    other['state'][0,0,0] = .01
    assert MaterialPairs.mismatch(source,other) == 'Different initial state'
    other['state'][0,0,0] = 0
    other['pads'][1,0,0] = .1
    assert MaterialPairs.mismatch(source,other) == 'Different pads'


def test_neural_search_updates_both_parameters_and_scores_last_point():
    config = dict(baseline={},reference_E_pa=3500.,reference_nu=.27,nu_scale=.2,
                  E_range_pa=[1000.,12000.],nu_range=[.08,.42])
    target = torch.tensor([.8,.6])
    def objective(q):
        assert torch.all(torch.isfinite(q))
        return (q-target).square().sum()
    result = optimize(objective,config,starts=3,steps=100)
    np.testing.assert_allclose(result['q'],target.numpy(),atol=.015)
    assert result['original_mpm_calls']==0 and result['model_evaluations']==303
    assert result['model_loss'] == pytest.approx(float(objective(torch.tensor(result['q']))))


def test_long_rollout_curriculum_and_unbounded_final_prefix():
    assert curriculum(0)==(1,8)
    assert curriculum(2000)==(8,32)
    assert curriculum(8000)==(16,96)
    assert curriculum(16000)==(32,192)
    assert curriculum(28000)==(64,None)
