"""Frozen encoder features joined with offline labels by explicit manifest rows."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .inference import extract, validate_input
from .io import load_npz, read_json, relative_path
from .motion import CausalMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.encoder, map_location='cpu', weights_only=True)
    if not checkpoint.get('use_actions', checkpoint.get('variant', 'actions') == 'actions'):
        raise ValueError('Four-parameter confidence uses the actions-enabled encoder')
    encoder = CausalMotion(actions=True, action_dim=checkpoint.get('action_dim',28)).to(args.device)
    encoder.load_state_dict(checkpoint['model'])
    encoder.eval()
    query_step = read_json(args.config)['query_step']
    features, sensitivities, masks, groups = [], [], [], []
    cache = {}
    for row in read_json(args.manifest):
        path = relative_path(args.manifest, row['input'])
        if str(path) not in cache:
            data = load_npz(path)
            validate_input(data)
            tensors = {k: torch.as_tensor(data[k], device=args.device,
                       dtype=torch.bool if k == 'valid' else torch.float32)
                       for k in ['positions', 'valid', 'geometry', 'actions', 'times']}
            cache[str(path)] = extract(encoder, tensors)
        teacher = load_npz(relative_path(args.manifest, row['teacher']))
        features.append(np.r_[cache[str(path)], teacher['q']/query_step])
        sensitivities.append(teacher['S'])
        masks.append(teacher['valid'])
        groups.append(str(row['episode_id']))
    if not features:
        raise ValueError('Empty feature manifest')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=features, S=sensitivities, valid=masks,
                        episode_ids=np.asarray(groups))


if __name__ == '__main__':
    main()
