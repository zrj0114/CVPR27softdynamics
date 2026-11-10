"""Independent, observation/teacher-conditioned one-step dynamics."""
import torch
from torch.utils.checkpoint import checkpoint

from .joint_world import JointWorld
from .shared_world_model import dynamics_loss


def independent_steps(model, batch, geometry, gradients=True, chunk_size=16):
    """Predict each transition from its supplied state, with no recurrent edge.

    A time dimension is only a batch of independent examples. Predictions are
    never assigned to the input of any other transition. Targets at t+1 are
    not inputs when predicting that transition.
    """
    if chunk_size < 1:
        raise ValueError('Positive independent-step batch size required')
    states, pads, times = batch['states'], batch['pads'], batch['times']
    B, T, N, C = states.shape
    if T < 2 or pads.shape[:2] != (B, T) or times.shape != (B, T):
        raise ValueError('Aligned states, actions, and times required')
    intervals = T-1
    flat_states = states[:, :-1].reshape(B*intervals, N, C)
    ids = torch.arange(B*intervals, device=states.device)
    values = []
    for index in ids.split(chunk_size):
        b, t = index//intervals, index%intervals
        geo = {k: v[b] for k, v in geometry.items()}
        args = (flat_states[index], pads[b, t], pads[b, t+1], batch['theta'][b],
                geo, times[b, t+1]-times[b, t])
        output = checkpoint(model, *args, use_reentrant=False, preserve_rng_state=False) if gradients else model(*args)
        values.append(output)
    return torch.cat(values, 0).reshape(B, intervals, N, C)


class StepWorld(JointWorld):
    def forward(self, batch, geometry, confidence_batch):
        if batch['states'].shape[1] != 2:
            raise ValueError('Training accepts exactly one transition per example')
        predicted = independent_steps(self.world, batch, geometry, self.training)
        local = dynamics_loss(predicted, batch['states'][:, 1:], self.world, geometry['scale_m'])
        context, theta, target, mask = confidence_batch
        confidence = (self.confidence(context, theta)-target).square()[mask].mean()
        return torch.stack((local, confidence))
