import pytest
import torch

from softdynamics.shared_world_control import (
    pose_rollout, reference_step_anchors, reference_step_corrections,
)


class AffineWorld(torch.nn.Module):
    kinematic_only = True

    def __init__(self, gain, offset):
        super().__init__()
        self.gain = gain
        self.offset = offset

    def forward(self, state, pads0, pads1, theta, geometry, dt):
        signal = theta[:, :1, None] + pads1[:, :1, :1]
        dynamic = self.gain * state[..., :6] + self.offset + signal
        return torch.cat((dynamic, state[..., 6:]), -1)


def inputs(dtype, horizon=32):
    reference = torch.linspace(.1, .7, horizon+1, dtype=dtype)[None, :, None, None].expand(1, -1, 2, 24).clone()
    pads = torch.zeros((1, horizon+1, 4, 13), dtype=dtype)
    theta = torch.zeros((1, 4), dtype=dtype)
    times = torch.arange(horizon+1, dtype=dtype)[None] / 30
    return reference, pads, theta, times


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_stable_anchor_reproduces_reference_despite_large_cancellation(dtype):
    reference, pads, theta, times = inputs(dtype)
    world = AffineWorld(gain=3., offset=1e8 if dtype == torch.float32 else 1e16)
    residual = reference_step_corrections(world, reference, pads, theta, {}, times)
    old = pose_rollout(world, reference[:, 0], pads, theta, {}, times, step_corrections=residual)
    assert (old[..., :6]-reference[..., :6]).abs().max() > .01
    anchors = reference_step_anchors(world, reference, pads, theta, {}, times)
    stable = pose_rollout(world, reference[:, 0], pads, theta, {}, times, step_anchors=anchors)
    torch.testing.assert_close(stable[..., :6], reference[..., :6], rtol=0, atol=0)


@pytest.mark.parametrize('use_checkpoint', [False, True])
def test_stable_anchor_preserves_material_action_and_state_gradients(use_checkpoint):
    reference, pads, theta, times = inputs(torch.float64, horizon=5)
    reference.requires_grad_()
    world = AffineWorld(gain=1.2, offset=8.)
    anchors = reference_step_anchors(world, reference, pads, theta, {}, times)
    assert all(not value.requires_grad for value in anchors)
    initial = reference[:, 0].detach().clone().requires_grad_()
    actions = pads.clone().requires_grad_()
    material = theta.clone().requires_grad_()
    def objective(start, controls, parameters):
        result = pose_rollout(world, start, controls, parameters, {}, times,
            use_checkpoint=use_checkpoint, step_anchors=anchors)
        return result[:, -1, :, :6].square().sum()
    loss = objective(initial, actions, material)
    gradients = torch.autograd.grad(loss, (initial, actions, material))
    directions = (torch.full_like(initial, .03), torch.full_like(actions, -.02), torch.full_like(material, .04))
    expected = sum((gradient*direction).sum() for gradient, direction in zip(gradients, directions))
    eps = 1e-5
    positive = objective(*[value+eps*direction for value,direction in zip((initial,actions,material),directions)])
    negative = objective(*[value-eps*direction for value,direction in zip((initial,actions,material),directions)])
    torch.testing.assert_close((positive-negative)/(2*eps), expected, rtol=1e-7, atol=1e-8)
    assert all(torch.isfinite(gradient).all() and gradient.abs().max()>0 for gradient in gradients)
    assert reference.grad is None


def test_anchor_and_residual_are_alternative_representations():
    reference, pads, theta, times = inputs(torch.float64, horizon=3)
    world = AffineWorld(gain=1.1, offset=.2)
    anchors = reference_step_anchors(world, reference, pads, theta, {}, times)
    corrections = reference_step_corrections(world, reference, pads, theta, {}, times)
    with pytest.raises(ValueError, match='one reference'):
        pose_rollout(world, reference[:, 0], pads, theta, {}, times,
            step_corrections=corrections, step_anchors=anchors)
    with pytest.raises(ValueError, match='reference step anchors'):
        pose_rollout(world, reference[:, 0], pads, theta, {}, times, step_anchors=(anchors[0][:, :1], anchors[1]))
