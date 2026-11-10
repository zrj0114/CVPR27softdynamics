"""Differentiable autoregressive dynamics and finite material-response loss."""
import torch
from torch.utils.checkpoint import checkpoint

from .joint_world import JointWorld
from .shared_world_model import dynamics_loss


def autoregressive(model, batch, geometry, gradients=True, detach_interval=None):
    if detach_interval is not None and detach_interval < 1:
        raise ValueError('Positive training BPTT interval required')
    state = batch['states'][:, 0]
    predictions = []
    for t in range(batch['pads'].shape[1]-1):
        # Truncate the network-weight training graph, never the forward
        # trajectory. Inference/search callers use the default full graph.
        if gradients and detach_interval and t and t % detach_interval == 0:
            state=state.detach()
        args = (state, batch['pads'][:, t], batch['pads'][:, t+1], batch['theta'],
                geometry, batch['times'][:, t+1]-batch['times'][:, t])
        state = checkpoint(model, *args, use_reentrant=False, preserve_rng_state=False) if gradients else model(*args)
        predictions.append(state)
    return torch.stack(predictions, 1)


def phase_weights(phase, dtype):
    weights = torch.zeros_like(phase, dtype=dtype)
    for index, mass in enumerate((.1, .45, .45)):
        mask = phase == index
        weights = weights + mask.to(dtype) * mass / mask.sum().clamp_min(1)
    return weights / weights.sum()


def pair_losses(predicted, batch, geometry, model):
    """Supervise signed trajectory secants with a 3 mm / unit-q noise floor.

    Both branches are supervised and start from an identical known initial
    state. All later states are predicted. Training may truncate the weight
    graph; the default search rollout retains every parameter-history path.
    """
    target = batch['states'][:, 1:]
    weights = phase_weights(batch['phase'], predicted.dtype)
    scale = geometry['scale_m'][0]
    position_error = ((predicted[..., :3]-target[..., :3])*scale/.003).square().mean((0, 2, 3))
    velocity_error = ((predicted[..., 3:6]-target[..., 3:6])/model.loss_scale[3:6]).square().mean((0, 2, 3))
    absolute = ((position_error+.2*velocity_error)*weights).sum()
    actual = (target[1, ..., :3]-target[0, ..., :3])*scale
    forecast = (predicted[1, ..., :3]-predicted[0, ..., :3])*scale
    energy = (actual.square().mean((1, 2))*weights).sum()
    floor = (.003*batch['parameter_distance'].clamp_min(.1)).square()
    response = (((forecast-actual).square().mean((1, 2))*weights).sum()/torch.maximum(energy, floor))
    return absolute, response


class SearchWorld(JointWorld):
    def forward(self, batch, geometry, confidence_batch, pair=None, detach_interval=None,
                first_step_fraction=0.):
        if not 0 <= first_step_fraction <= 1:
            raise ValueError('Invalid single-step loss fraction')
        predicted = autoregressive(self.world, batch, geometry, self.training, detach_interval)
        local = dynamics_loss(predicted, batch['states'][:, 1:], self.world, geometry['scale_m'])
        if first_step_fraction:
            first=dynamics_loss(predicted[:,:1],batch['states'][:,1:2],self.world,geometry['scale_m'])
            local=(1-first_step_fraction)*local+first_step_fraction*first
        context, theta, target, mask = confidence_batch
        confidence = (self.confidence(context, theta)-target).square()[mask].mean()
        absolute = local.new_zeros(())
        response = local.new_zeros(())
        if pair is not None:
            pair_batch, pair_geometry = pair
            pair_prediction = autoregressive(self.world, pair_batch, pair_geometry, self.training, detach_interval)
            absolute, response = pair_losses(pair_prediction, pair_batch, pair_geometry, self.world)
        return torch.stack((local, confidence, absolute, response))


def curriculum(step, total=40000):
    fraction = step/total
    for end, local, prefix in ((.05, 1, 8), (.2, 8, 32), (.4, 16, 96), (.7, 32, 192), (1.01, 64, None)):
        if fraction < end:
            return local, prefix
    raise ValueError('Step exceeds budget')


def local_horizon_v2(step,total=40000):
    fraction=step/total
    return 1 if fraction<.025 else 8 if fraction<.2 else 16 if fraction<.5 else 32
