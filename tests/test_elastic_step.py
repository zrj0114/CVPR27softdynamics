import numpy as np
import pytest
import torch

from softdynamics.elastic_step_objective import independent_steps, StepWorld
from softdynamics.elastic_step_observation import reconstruct_observed_states
from softdynamics.elastic_step_validation import summarize
from softdynamics.elastic_world_search import optimize


class ConditionalStep(torch.nn.Module):
    kinematic_only = True

    def forward(self, state, p0, p1, theta, geometry, dt):
        delta = torch.zeros_like(state)
        delta[..., 0] = theta[:, 0, None]+p0[:, 0, 0, None]
        delta[..., 1] = theta[:, 1, None]
        return state+delta


def example():
    states = torch.arange(6).float()[None, :, None, None].expand(1, 6, 3, 24).clone()
    return dict(states=states, pads=torch.zeros(1, 6, 4, 13), times=torch.arange(6).float()[None],
                theta=torch.tensor([[.2, .3]], requires_grad=True))


def test_outputs_are_independent_and_material_derivative_has_no_history_chain():
    model, batch = ConditionalStep(), example()
    prediction = independent_steps(model, batch, {}, chunk_size=2)
    assert prediction[0, -1, 0, 0].item() == pytest.approx(4.2)
    derivative = torch.autograd.grad(prediction[0, -1, 0, 0], batch['theta'])[0]
    torch.testing.assert_close(derivative, torch.tensor([[1., 0.]]))
    changed = dict(batch, pads=batch['pads'].clone())
    changed['pads'][:, 0, 0, 0] = 1000
    other = independent_steps(model, changed, {}, gradients=False, chunk_size=5)
    assert (other[0, 0, 0, 0]-prediction[0, 0, 0, 0]).item() == pytest.approx(1000.)
    torch.testing.assert_close(other[:, 1:], prediction[:, 1:])
    future = dict(batch, states=batch['states'].clone())
    future['states'][:, 3:] = -1000
    torch.testing.assert_close(independent_steps(model, future, {}, False)[:, :3], prediction[:, :3])


def test_step_training_rejects_multiple_frames():
    model = StepWorld(dict(hidden=16, layers=1, heads=4, material_dim=2), ('E_pa', 'nu'))
    with pytest.raises(ValueError, match='exactly one transition'):
        model(example(), {}, None)


def observed_example():
    initial = np.zeros((4, 24), np.float32)
    initial[:, :3] = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, 0]])
    times = np.arange(6)/30
    displacement = np.arange(6)[:, None, None]*np.array([.001, .002, 0])[None, None]
    tracks = initial[None, :, :3]*.1+displacement
    neighbors = np.array([[1, 2], [0, 3], [0, 3], [1, 2]])
    return initial, tracks, np.ones((6, 4), bool), times, np.eye(4), neighbors


def test_state_reconstruction_and_velocity_use_only_current_and_past_observations():
    inputs = observed_example()
    states, valid, stats = reconstruct_observed_states(*inputs, scale_m=.1)
    np.testing.assert_allclose(states[1:, :, 3:6], np.broadcast_to([.01, .02, 0], (5, 4, 3)), atol=1e-6)
    changed = list(inputs); changed[1] = changed[1].copy(); changed[1][3:] += 100
    revised, _, _ = reconstruct_observed_states(*changed, scale_m=.1)
    np.testing.assert_array_equal(states[:3], revised[:3])
    assert not valid[0] and valid[1:].all() and stats['prediction_feedback'] is False
    _, known_valid, _ = reconstruct_observed_states(*inputs, scale_m=.1, initial_velocity_known=True)
    assert known_valid.all()
    assert stats['mean_track_reconstruction_rmse_mm'] < .001


def test_missing_observations_cannot_silently_supply_scored_states():
    inputs = list(observed_example()); inputs[2][2] = False; inputs[1][2] = np.nan
    states, valid, stats = reconstruct_observed_states(*inputs, scale_m=.1)
    assert np.isfinite(states).all()
    assert not valid[1] and not valid[2] and not valid[3]
    assert stats['usable_transitions'] < stats['total_transitions']


def test_independent_parameter_search_recovers_two_shared_parameters():
    model, batch = ConditionalStep(), example()
    target_q = torch.tensor([.45, -.35])
    # Actual toy trajectory; each measured current state is fixed for fitting.
    batch['states'].zero_()
    batch['states'][0, :, :, 0] = torch.arange(6)[:, None]*target_q[0]
    batch['states'][0, :, :, 1] = torch.arange(6)[:, None]*target_q[1]
    def objective(q):
        prediction = independent_steps(model, dict(batch, theta=q[None]), {})
        return (prediction[..., :2]-batch['states'][:, 1:, :, :2]).square().mean()
    config = dict(baseline={}, reference_E_pa=3500., reference_nu=.27, nu_scale=.2,
                  E_range_pa=[1000., 12000.], nu_range=[.08, .42])
    result = optimize(objective, config, starts=2, steps=100)
    np.testing.assert_allclose(result['q'], target_q.numpy(), atol=.015)
    assert result['original_mpm_calls'] == 0


def test_step_validation_is_balanced_and_keeps_both_baselines():
    rows = [dict(object='a', phase=0, position_mm=10., static_mm=20., constant_velocity_mm=5.)]*20
    rows += [dict(object='a', phase=p, position_mm=0., static_mm=0., constant_velocity_mm=0.) for p in (1, 2)]
    rows += [dict(object='b', phase=p, position_mm=3., static_mm=6., constant_velocity_mm=1.5) for p in (0, 1, 2)]
    result = summarize(rows)
    assert result['position_mm'] == pytest.approx(2.)
    assert result['static_mm'] == pytest.approx(4.)
    assert result['constant_velocity_mm'] == pytest.approx(1.)
