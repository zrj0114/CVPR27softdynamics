"""Standalone inference: normalized visual tracks, initial geometry and EEF actions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .motion import CausalMotion, relative_actions

PARAMETERS = ['E_pa', 'nu', 'ground_friction', 'velocity_decay_rate_s_inv']


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(580, 32), nn.GELU(), nn.Dropout(.1),
                          nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
            for _ in range(4)
        ])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=1)


def validate_input(data):
    required = ['positions', 'valid', 'geometry', 'actions', 'times']
    if any(k not in data for k in required):
        raise ValueError('Required input arrays: ' + ', '.join(required))
    positions = np.asarray(data['positions'])
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError('positions must have shape [T,N,3]')
    T, N = positions.shape[:2]
    if T < 16 or N < 128:
        raise ValueError('At least 16 frames and 128 point tracks are required')
    if np.shape(data['actions']) not in [(T,28), (T,52)]:
        raise ValueError('actions must be [T,28] EEF or [T,52] independent pads')
    for key, shape in [('valid', (T, N)), ('geometry', (N, 15)), ('times', (T,))]:
        if np.shape(data[key]) != shape:
            raise ValueError(f'{key} must have shape {shape}')
    if np.asarray(data['valid']).dtype != np.bool_:
        raise ValueError('valid must be a Boolean observation mask')
    for key in ['positions', 'geometry', 'actions', 'times']:
        if not np.isfinite(data[key]).all():
            raise ValueError(f'{key} must be finite; retain tracked/imputed coordinates and set valid=False for occluded samples')
    if not np.all(np.diff(data['times']) > 0):
        raise ValueError('timestamps must be strictly increasing, in seconds')


@torch.inference_mode()
def extract(encoder, d):
    rows = []
    T = len(d['times'])
    device = d['positions'].device
    starts = np.unique(np.r_[np.arange(0, T - 16 + 1, 8), T - 16])
    ids = torch.as_tensor(np.random.default_rng(8200).choice(
        d['positions'].shape[1], 128, replace=False), device=device)
    for start in starts:
        x = d['positions'][start:start + 16, ids]
        valid = d['valid'][start:start + 16, ids]
        times = d['times'][start:start + 16]
        actions = relative_actions(d['actions'][start:start + 16], -1)
        batch = dict(past=x[None], valid=valid[None],
                     geometry=d['geometry'][ids][None], anchor=x[-1][None],
                     past_times=(times - times[-1])[None])
        h = encoder.encode(batch)
        ah, _ = encoder.action_history(encoder.action(torch.cat(
            [actions[None], (times - times[-1])[None, :, None]], -1)))
        rows.append(torch.cat([h.mean(1), h.amax(1), ah[:, -1]], -1)[0].cpu().numpy())
    z = np.stack(rows)
    return np.concatenate([z.mean(0), z.std(0), z.max(0)])


class ConfidenceModel:
    def __init__(self, checkpoint=None, device='cpu'):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if checkpoint is None:
            raise ValueError('An explicit trained checkpoint is required')
        self.device = torch.device(device)
        self.bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.encoder = CausalMotion(actions=True, action_dim=self.bundle.get('action_dim',28)).to(self.device)
        self.encoder.load_state_dict(self.bundle['encoder'])
        self.encoder.eval()
        self.models = []
        for state in self.bundle['heads']:
            model = Heads().to(self.device)
            model.load_state_dict(state['model'])
            model.eval()
            self.models.append((model, state))

    @torch.inference_mode()
    def feature(self, data):
        validate_input(data)
        tensors = {k: torch.as_tensor(data[k], device=self.device,
                   dtype=torch.bool if k == 'valid' else torch.float32)
                   for k in ['positions', 'valid', 'geometry', 'actions', 'times']}
        return extract(self.encoder, tensors)

    @torch.inference_mode()
    def predict_feature(self, feature, physical_parameters):
        p = np.asarray(physical_parameters, dtype=np.float64)
        if p.shape != (4,) or not np.isfinite(p).all():
            raise ValueError('Supply E_pa, nu, ground_friction, velocity_decay_rate_s_inv')
        if p[0] <= 0 or not -1 < p[1] < .5 or np.any(p[2:] < 0):
            raise ValueError('Physical parameters are outside the allowed domain')
        q = (p - self.bundle['base'].numpy()) / self.bundle['scales'].numpy()
        x = np.r_[feature, q / self.bundle['query_step']][None]
        predictions = []
        for model, state in self.models:
            z = model(torch.as_tensor((x - state['mu'].numpy()) / state['std'].numpy(),
                      dtype=torch.float32, device=self.device)).cpu().numpy()[0]
            u = (z * state['target_std'].numpy() + state['target_mean'].numpy()).clip(0, 10)
            s = state['tau'].numpy() * np.expm1(u)
            s[~state['ready'].numpy()] = np.nan
            predictions.append(s)
        sensitivity = np.mean(predictions, axis=0)
        score = sensitivity / (sensitivity + self.bundle['tau'].numpy())
        return dict(sensitivity=sensitivity, confidence=score, q=q,
                    within_training_query_box=bool(np.all(np.abs(q) <= self.bundle['query_step'] + 1e-7)))

    def predict(self, data, physical_parameters):
        return self.predict_feature(self.feature(data), physical_parameters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--parameters', type=float, nargs=4, required=True,
                        metavar=('E_PA', 'NU', 'GROUND_FRICTION', 'DECAY_S_INV'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with np.load(args.input, allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    result = ConfidenceModel(args.checkpoint, args.device).predict(data, args.parameters)
    encode = lambda values: {k: float(v) if np.isfinite(v) else None
                             for k, v in zip(PARAMETERS, values)}
    output = dict(confidence=encode(result['confidence']), sensitivity=encode(result['sensitivity']),
                  within_training_query_box=result['within_training_query_box'],
                  interpretation='Four independent relative motion-sensitivity scores; not probabilities of parameter accuracy.')
    text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
