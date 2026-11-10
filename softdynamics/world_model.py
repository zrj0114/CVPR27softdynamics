"""Action-conditioned, autoregressive material-node dynamics for a fixed asset.

The teacher inherits the interaction-confidence-weighted fit through fit_handoff.
This network distills its fixed physics rather than sharing confidence weights. Differentiability alone does
not establish that the learned action derivatives match the physics teacher.
"""
import numpy as np
import torch
from torch import nn

DT_REF = 1 / 30


def normalize_state(state, origin, scale):
    """Physical x,v,F,C -> dimensionless state; accepts NumPy arrays."""
    result = np.array(state, dtype=np.float32, copy=True)
    result[..., :3] = (result[..., :3] - origin) / scale
    result[..., 3:6] *= DT_REF / scale
    result[..., 6:15] -= np.eye(3, dtype=np.float32).reshape(9)
    result[..., 15:] *= DT_REF
    return result


def physical_state(state, origin, scale):
    result = np.array(state, dtype=np.float32, copy=True)
    result[..., :3] = result[..., :3] * scale + origin
    result[..., 3:6] *= scale / DT_REF
    result[..., 6:15] += np.eye(3, dtype=np.float32).reshape(9)
    result[..., 15:] /= DT_REF
    return result


def fit_statistics(trajectories):
    """Only training episodes may determine feature/output scales."""
    if not trajectories or any(t['split'] != 'train' for t in trajectories):
        raise ValueError('Normalization requires training trajectories exclusively')
    states = np.concatenate([t['state'].reshape(-1, 24) for t in trajectories])
    mean = states.mean(0, dtype=np.float64).astype('f')
    std = np.maximum(states.std(0, dtype=np.float64), 1e-3).astype('f')
    deltas = []
    for t in trajectories:
        delta = t['state'][1:] - t['state'][:-1]
        delta[..., :3] -= t['state'][:-1, :, 3:6] * (np.diff(t['times']) / DT_REF)[:, None, None]
        deltas.append(delta.reshape(-1, 24))
    rms = np.sqrt(np.mean(np.concatenate(deltas).astype('d') ** 2, axis=0)).astype('f')
    # A single scale per physical group avoids arbitrary coordinate preferences.
    for start, end in ((0, 3), (3, 6), (6, 15), (15, 24)):
        std[start:end] = max(float(np.sqrt(np.mean(std[start:end]**2))), 1e-3)
        rms[start:end] = max(float(np.sqrt(np.mean(rms[start:end]**2))), 1e-4)
    loss_scale = std.copy()
    # Position supervision explicitly targets millimetres, not object diameter.
    loss_scale[:3] = .003 / trajectories[0]['scale_m']
    return dict(mean=mean, std=std, output_scale=rms, loss_scale=loss_scale,
                training_ids=[t['id'] for t in trajectories], dt_ref=DT_REF)


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class MaterialDynamics(nn.Module):
    def __init__(self, canonical, neighbors, statistics, hidden=96, layers=2, position_update='independent'):
        super().__init__()
        if position_update not in ('independent','velocity_consistent'):
            raise ValueError('Unknown position integration rule')
        self.position_update = position_update
        self.register_buffer('canonical', torch.as_tensor(canonical, dtype=torch.float32))
        self.register_buffer('neighbors', torch.as_tensor(neighbors, dtype=torch.long))
        for key in ('mean', 'std', 'output_scale', 'loss_scale'):
            self.register_buffer(key, torch.as_tensor(statistics[key], dtype=torch.float32))
        # Each pad: local position 3, translation 3, R 9, dR 9, eligible 1, valid 1.
        self.encoder = mlp(24 + 3 + 1 + 1 + 4*26, hidden, hidden)
        self.messages = nn.ModuleList([mlp(hidden + 9, hidden, hidden) for _ in range(layers)])
        self.updates = nn.ModuleList([mlp(hidden*2, hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.decoder = mlp(hidden, hidden, 24 if position_update == 'independent' else 21)
        # Small nonzero output gives testable action gradients even at step zero.
        nn.init.normal_(self.decoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, state, pads0, pads1, eligibility, dt, ground_y):
        """One control interval. Pads are corrected pad centers, not EEF poses.

        Only an initial state and action endpoints enter this function. Validity
        requires both endpoints, matching the frozen EMPM interval_motion rule.
        ground_y is the normalized world ground height for this episode origin.
        """
        B, N, _ = state.shape
        ratio = (dt / DT_REF).reshape(B, 1, 1)
        active = (pads0[..., 12] > .5) & (pads1[..., 12] > .5)
        p0 = torch.where(active[..., None], pads0[..., :12], 0.)
        p1 = torch.where(active[..., None], pads1[..., :12], 0.)
        relative = state[:, :, None, :3] - p0[:, None, :, :3]
        rotation = p0[..., 3:12].reshape(B, 4, 3, 3)
        local = torch.einsum('bnpi,bpij->bnpj', relative, rotation)
        action = torch.cat([(p1[..., :3]-p0[..., :3]) / ratio,
                            p0[..., 3:12], p1[..., 3:12]-p0[..., 3:12]], -1)
        pad_features = torch.cat([local, action[:, None].expand(-1, N, -1, -1),
                                  eligibility[..., None],
                                  active[:, None, :, None].expand(-1, N, -1, -1)], -1)
        pad_features = pad_features * active[:, None, :, None]
        ground = state[..., 1:2] - ground_y.reshape(B, 1, 1)
        inputs = torch.cat([(state-self.mean)/self.std,
                            self.canonical[None].expand(B, -1, -1), ground,
                            ratio.expand(-1, N, -1), pad_features.flatten(2)], -1)
        h = self.encoder(inputs)
        j = self.neighbors
        edges = torch.cat([state[:, j, :3]-state[:, :, None, :3],
                           state[:, j, 3:6]-state[:, :, None, 3:6],
                           (self.canonical[j]-self.canonical[:, None])[None].expand(B, -1, -1, -1)], -1)
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            aggregate = message(torch.cat([h[:, j]-h[:, :, None], edges], -1)).mean(2)
            h = norm(h + update(torch.cat([h, aggregate], -1)))
        if self.position_update == 'independent':
            residual = self.decoder(h) * self.output_scale * ratio
            advected = torch.cat([state[..., :3]+state[..., 3:6]*ratio, state[..., 3:]], -1)
            return advected + residual
        # Couple displacement to the velocity update. No independent position
        # bias can accumulate while the predicted material velocity is zero.
        next_dynamic = state[..., 3:] + self.decoder(h)*self.output_scale[3:]*ratio
        next_x = state[..., :3] + .5*(state[..., 3:6]+next_dynamic[..., :3])*ratio
        return torch.cat([next_x, next_dynamic], -1)


def rollout(model, initial, pads, eligibility, times, ground_y):
    """No teacher forcing, including during multi-step training."""
    state = initial
    predictions = []
    for t in range(pads.shape[1]-1):
        state = model(state, pads[:, t], pads[:, t+1], eligibility,
                      times[:, t+1]-times[:, t], ground_y)
        predictions.append(state)
    return torch.stack(predictions, 1)


def rollout_loss(prediction, target, loss_scale):
    error = ((prediction-target)/loss_scale)**2
    return (error[..., :3].mean() + .2*error[..., 3:6].mean()
            + .05*error[..., 6:15].mean() + .05*error[..., 15:].mean())


def position_error_mm(prediction, target, scale_m):
    return np.linalg.norm(np.asarray(prediction)[..., :3]-np.asarray(target)[..., :3], axis=-1)*scale_m*1000
