from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from softdynamics.elastic_world_data import MaterialPairs, balanced_validation_windows
from softdynamics.elastic_world_objective import autoregressive, local_horizon_v2
from softdynamics.elastic_world_train import gradients
from softdynamics.elastic_world_validation import summarize_windows, macro_mean
from softdynamics.shared_world_model import SharedWorldModel, swept_pad_influence


def scene():
    torch.manual_seed(91)
    state = torch.zeros(1, 4, 24, dtype=torch.float64)
    state[0, :, 1] = torch.arange(4)*.01
    pads = torch.zeros(1, 4, 13, dtype=state.dtype)
    pads[..., 3:12] = torch.eye(3).flatten()
    pads[..., 12] = 1
    pads[..., 0] = 3.  # Far in metric space even at scale .1.
    pads[:, 0, 0] = .03
    geometry = dict(canonical=state[..., :3].clone(), neighbors=torch.tensor([[[1, 2], [0, 2], [1, 3], [1, 2]]]),
        ground_y=torch.tensor([-.1], dtype=state.dtype), scale_m=torch.tensor([.1], dtype=state.dtype),
        surface_mask=torch.ones(1, 4, dtype=torch.bool))
    theta = torch.tensor([[3500., .27]], dtype=state.dtype, requires_grad=True)
    model = SharedWorldModel(hidden=16, layers=1, heads=4, direct_displacement=True,
        contact_geometry=True, material_dim=2, pad_locality=True).double()
    model.requires_grad_(False)
    return model, state, pads, theta, geometry


def test_far_pose_and_validity_cannot_rescale_near_contact_but_near_action_has_gradient():
    model, state, pads, theta, geo = scene()
    dt = theta.new_tensor([1/30])
    moving = pads.clone().requires_grad_()
    baseline = model(state, pads, moving, theta, geo, dt)
    changed = pads.clone()
    changed[:, 1:, 0] = 20
    changed[:, 1:, 3:12] = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]]).flatten()
    torch.testing.assert_close(baseline, model(state, pads, changed, theta, geo, dt), rtol=0, atol=0)
    changed[:, 1:, 12] = 0
    torch.testing.assert_close(baseline, model(state, pads, changed, theta, geo, dt), rtol=0, atol=0)
    pg, tg = torch.autograd.grad(baseline[..., :3].square().sum(), (moving, theta))
    assert pg[:, 1:].abs().max() == 0
    assert pg[:, 0, :3].abs().max() > 0
    assert torch.isfinite(tg).all() and (tg.abs() > 0).all()
    eps = 1e-5
    direction = theta.new_tensor([[100., .01]])
    high = model(state, pads, pads, theta+eps*direction, geo, dt)[..., :3].square().sum()
    low = model(state, pads, pads, theta-eps*direction, geo, dt)[..., :3].square().sum()
    torch.testing.assert_close((high-low)/(2*eps), (tg*direction).sum(), atol=1e-12, rtol=2e-4)


def test_swept_gate_detects_crossing_and_respects_metric_scale():
    state = torch.zeros(1, 1, 24, dtype=torch.float64)
    a = torch.zeros(1, 1, 13, dtype=state.dtype); a[..., 0] = -2
    b = a.clone(); b[..., 0] = 2
    scale = torch.tensor([.1], dtype=state.dtype)
    assert swept_pad_influence(state, a, a, scale).item() == 0
    assert swept_pad_influence(state, b, b, scale).item() == 0
    assert swept_pad_influence(state, a, b, scale).item() == 1
    a[..., 0] = .6; b[..., 0] = .65
    first = swept_pad_influence(state, a, b, scale)
    assert 0 < first.item() < 1
    a[..., :3] *= 10; b[..., :3] *= 10
    torch.testing.assert_close(first, swept_pad_influence(state, a, b, scale/10))


class ExpandingDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(1.3))

    def forward(self, state, pads0, pads1, theta, geometry, dt):
        return state*self.rate + theta[:, :1, None]


def history(frames=5):
    return dict(states=torch.zeros(2, frames+1, 2, 24), pads=torch.zeros(2, frames+1, 4, 13),
        times=torch.arange(frames+1).float().expand(2, -1),
        theta=torch.ones(2, 2, requires_grad=True))


def test_training_detach_keeps_forward_and_search_keeps_full_derivative():
    model, batch = ExpandingDynamics(), history()
    full = autoregressive(model, batch, {}, gradients=True)
    short = autoregressive(model, batch, {}, gradients=True, detach_interval=2)
    torch.testing.assert_close(full, short, atol=0, rtol=0)
    fg = torch.autograd.grad(full[0, -1, 0, 0], batch['theta'])[0][0, 0]
    sg = torch.autograd.grad(short[0, -1, 0, 0], batch['theta'])[0][0, 0]
    assert fg.item() == pytest.approx(sum(1.3**i for i in range(5)), rel=1e-6)
    assert sg.item() == 1
    corrupted = dict(batch, states=batch['states'].clone())
    corrupted['states'][:, 1:] = 10000
    torch.testing.assert_close(full, autoregressive(model, corrupted, {}, gradients=False), atol=0, rtol=0)


class WeightObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.world = ExpandingDynamics()

    def forward(self, batch, detach_interval=None, first_step_fraction=0):
        prediction = autoregressive(self.world, batch, {}, True, detach_interval)
        return prediction[:, -1].square().mean().reshape(1)


def test_one_step_gradient_guard_clips_without_updating_weights():
    model, batch = WeightObjective(), history(1)
    batch['states'][:, 0] = 1
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.world.rate.detach().clone()
    losses, norm, info = gradients(model, model, optimizer, (batch,), torch.ones(1), 0, max_raw_norm=100.)
    assert norm > 1 and abs(model.world.rate.grad.item()) <= 1
    assert len(info['gradient_attempts']) == 1
    torch.testing.assert_close(model.world.rate, before)
    torch.testing.assert_close(losses, model(batch))
    with pytest.raises(FloatingPointError, match='Excessive one-step gradient'):
        gradients(model, model, optimizer, (batch,), torch.ones(1), 0, max_raw_norm=.001)


def test_phase_endpoints_have_no_early_prefix_cap_and_validation_contains_each_phase():
    source = dict(minimum_start=0, pads=np.zeros((180, 4, 13)), times=np.arange(180),
        phase=np.repeat([0, 1, 2], [50, 70, 60]), origin_m=np.zeros(3), scale_m=1.,
        state=np.zeros((180, 3, 24)), theta=np.array([3500., .27]),
        record=dict(source_hashes={'initial': 'same'}),
        task=dict(object='toy', episode=0, parameters={'E_pa':3500., 'nu':.27}, dtmax=.001))
    other = deepcopy(source); other['theta'][0] = 5000.; other['task']['parameters']['E_pa'] = 5000.
    replay = SimpleNamespace(trajectories={'a':source, 'b':other})
    pairs = MaterialPairs(replay); rng = np.random.default_rng(42)
    for phase in (0, 1, 2):
        choices = [pairs.sample_phase(rng, phase) for _ in range(100)]
        assert all(source['phase'][end] == phase for _, _, end in choices)
        if phase == 2:
            assert min(end for _, _, end in choices) >= 120
            assert any(end == 179 for _, _, end in choices)
    selection = balanced_validation_windows(replay, 32)
    assert len(selection) == 12 and {p for _, _, p in selection} == {0, 1, 2}
    for key, start, phase in selection:
        assert np.all(replay.trajectories[key]['phase'][start+1:start+33] == phase)
    assert pairs.report()['pairs'] == 1 and pairs.validation_choices() == [('a', 'b', 179)]


def test_validation_macro_balance_and_fixed_local_horizons():
    rows = [dict(object='a', phase=0, position_mm=100., static_mm=10.)]*10
    rows += [dict(object='a', phase=p, position_mm=0., static_mm=0.) for p in (1, 2)]
    rows += [dict(object='b', phase=p, position_mm=2., static_mm=1.) for p in (0, 1, 2)]
    summary = summarize_windows(rows)
    assert summary['mean_mm'] == pytest.approx(6.)
    assert summary['static_mean_mm'] == pytest.approx(1.)
    assert macro_mean([dict(object='a', value=1.)]*10+[dict(object='b', value=3.)], 'value') == 2
    assert [local_horizon_v2(s) for s in (0, 1000, 8000, 20000, 39999)] == [1, 8, 16, 32, 32]
