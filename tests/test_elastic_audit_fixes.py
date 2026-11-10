import copy
import numpy as np
import pytest
import torch

from softdynamics.elastic_world_search import optimize, parameter_weighted_gradient
from softdynamics.elastic_step_observation import ObservedStepEpisode, reconstruct_observed_states
from softdynamics.elastic_step_validation import evaluate_confidence
from softdynamics.joint_world import JointWorld, context_batch, interaction_context
from test_joint_world import example
from test_elastic_step import observed_example


def test_each_parameter_uses_its_own_episode_weights_and_no_weight_derivative():
    q = torch.zeros(2, requires_grad=True)
    losses = torch.stack((q@torch.tensor([1., 2.]), q@torch.tensor([9., 8.])))
    weights = torch.tensor([[3., 0.], [0., 7.]], requires_grad=True)
    gradient = parameter_weighted_gradient(losses, q, weights)
    torch.testing.assert_close(gradient, torch.tensor([1., 8.]))
    assert weights.grad is None


def test_zero_confidence_column_falls_back_to_uniform_only_for_that_parameter():
    q = torch.zeros(2, requires_grad=True)
    losses = torch.stack((q@torch.tensor([1., 2.]), q@torch.tensor([9., 8.])))
    gradient = parameter_weighted_gradient(losses, q, torch.tensor([[0., 1.], [0., 0.]]))
    torch.testing.assert_close(gradient, torch.tensor([5., 2.]))


def test_search_weights_change_update_direction_and_are_exported():
    config = dict(baseline={}, reference_E_pa=3500., reference_nu=.27, nu_scale=.2,
        E_range_pa=[1000.,12000.], nu_range=[.08,.42])
    visited = []
    def objective(q):
        visited.append(q.detach().clone())
        return torch.stack(((q-torch.tensor([.8,-.8])).square().sum(),
                            (q-torch.tensor([-1.,.8])).square().sum()))
    result = optimize(objective, config, starts=1, steps=2, weights=torch.tensor([[1.,0.],[0.,1.]]))
    assert torch.all(visited[1]>visited[0])
    assert result['episode_parameter_weights']==[[1.,0.],[0.,1.]]
    assert result['original_mpm_calls']==0 and result['weights_frozen_during_search']
    assert 'equal-episode' in result['selection']
    visited.clear()
    optimize(objective, config, starts=1, steps=2)
    assert visited[1][0]<visited[0][0]


def model_and_context():
    torch.manual_seed(4)
    model = JointWorld(dict(hidden=16,layers=2,heads=2,kinematic_only=True,
        direct_displacement=True,contact_geometry=True,material_dim=2),('E_pa','nu'))
    context = context_batch([interaction_context(example(),dict(clip_start=10,clip_end=20))],'cpu')
    return model, context, torch.tensor([[3500.,.27]])


@pytest.mark.parametrize('missing_value', [1000., float('nan')])
def test_invalid_points_cannot_contaminate_confidence_attention(missing_value):
    model, context, theta = model_and_context()
    context['valid'][:,:,0] = False
    original = model.confidence(context,theta)
    changed = copy.deepcopy(context)
    changed['states'][:,:,0,:6] = missing_value
    changed['states'].requires_grad_()
    revised = model.confidence(changed,theta)
    torch.testing.assert_close(revised,original,rtol=0,atol=0)
    revised.sum().backward()
    assert torch.isfinite(changed['states'].grad).all()
    assert not changed['states'].grad[:,:,0].any()


def test_empty_context_frames_do_not_bias_temporal_pooling():
    model, context, theta = model_and_context()
    original = model.confidence(context,theta)
    changed = {k:torch.cat((v,v[:,:1]),dim=1) for k,v in context.items() if k!='geometry'}
    changed['geometry'] = {k:torch.cat((v,v[:,:1]),dim=1) for k,v in context['geometry'].items()}
    changed['valid'][:,-1] = False
    torch.testing.assert_close(model.confidence(changed,theta),original)
    changed['valid'][:] = False
    with pytest.raises(ValueError,match='valid observed frame'):
        model.confidence(changed,theta)


def test_real_search_does_not_assume_unobserved_initial_velocity():
    inputs = list(observed_example())
    _, valid, _ = reconstruct_observed_states(*inputs,scale_m=.1)
    assert not valid[0] and valid[1:].all()
    inputs[2][0] = False
    _, valid, _ = reconstruct_observed_states(*inputs,scale_m=.1)
    assert not valid[0] and not valid[1]


def test_search_fails_on_invalid_predictions_instead_of_dropping_targets():
    episode = object.__new__(ObservedStepEpisode)
    episode.frames = torch.tensor([1,2,3])
    episode.observed_states = torch.zeros(1,4,2,24)
    episode.observed_tracks = torch.zeros(4,2,3)
    episode.observed_visible = torch.ones(4,2,dtype=torch.bool)
    episode.projection = {'track_matrix':torch.eye(2)}
    episode.scale = 1.
    episode.meta = dict(clip_start=2,clip_end=2)
    episode.config = dict(huber_delta_mm=5.,phase_weights=dict(before=.1,interaction=.45,release=.45))
    prediction = torch.zeros(4,2,24)
    prediction[2,0,0] = float('inf')
    with pytest.raises(FloatingPointError,match='scored observed track'):
        episode.loss(prediction)


def test_confidence_validation_covers_labels_beyond_first_twelve():
    class Data:
        rows = [dict(object='a') for _ in range(16)]
        def batch(self, ids, device, tau):
            target=torch.zeros(len(ids),2)
            prediction=torch.tensor([[0.,0.] if i<12 else [1.,1.] for i in ids])
            return prediction,None,target,torch.ones_like(target,dtype=torch.bool)
    class Model:
        def confidence(self, context, theta): return context
    result = evaluate_confidence(Model(),Data(),None,'cpu')
    assert result['evaluated_rows']==result['available_rows']==16
    assert result['mean_mae']==pytest.approx(.25)


def test_masking_fix_preserves_fully_observed_network_predictions():
    model, context, theta = model_and_context()
    geometry={k:v[:,0] for k,v in context['geometry'].items()}
    args=(context['states'][:,0],context['pads0'][:,0],context['pads1'][:,0],theta)
    original=model.world(*args,geometry,context['dt'][:,0])
    geometry['node_valid']=torch.ones(original.shape[:2],dtype=torch.bool)
    torch.testing.assert_close(model.world(*args,geometry,context['dt'][:,0]),original)
