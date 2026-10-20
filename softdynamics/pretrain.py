"""Object-disjoint motion pretraining with observed history and future actions."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .io import load_npz, read_json, relative_path, write_json
from .inference import validate_input
from .motion import CausalMotion, PAST, FUTURE, make_batch, baselines


def load_episode(path, device):
    arrays = load_npz(path)
    validate_input(arrays)
    if len(arrays['times']) < PAST + FUTURE or 'scale_m' not in arrays:
        raise ValueError('Pretraining requires at least 32 frames and scale_m')
    return {key: torch.as_tensor(arrays[key], device=device)
            for key in ['positions', 'valid', 'geometry', 'actions', 'times', 'scale_m']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--heldout-object', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-actions', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('steps and batch-size must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read_json(args.manifest)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Episode IDs must be unique')
    train = [r for r in rows if r['object'] != args.heldout_object]
    test = [r for r in rows if r['object'] == args.heldout_object]
    if not train or not test:
        raise ValueError('Provide both training objects and the held-out object')
    data = {r['id']: load_episode(relative_path(args.manifest, r['path']), args.device)
            for r in train}
    keys = list(data)
    model = CausalMotion(actions=not args.no_actions).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(1300 + args.seed)
    history = []
    for step in range(args.steps):
        batch, target, mask, _ = make_batch(data, rng.choice(keys, args.batch_size).tolist(), rng)
        if not mask.any():
            raise ValueError('Batch has no visible future points')
        prediction = model(batch)
        loss = F.smooth_l1_loss(prediction[mask], target[mask], beta=.01)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 200 == 0 or step == args.steps - 1:
            row = dict(step=step, loss=float(loss.detach()))
            history.append(row)
            print(row, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    # Freeze before loading any held-out trajectories; no checkpoint selection on test.
    torch.save(dict(model=model.state_dict(), training_episodes=keys,
                    heldout=args.heldout_object, use_actions=not args.no_actions),
               args.output / 'encoder.pt')
    model.eval()
    evaluation = []
    with torch.inference_mode():
        for row in test:
            episode = load_episode(relative_path(args.manifest, row['path']), args.device)
            starts = np.unique(np.linspace(0, len(episode['times'])-PAST-FUTURE, 32).astype(int))
            totals, count = np.zeros(3), 0
            for start in starts:
                batch, target, mask, scale = make_batch({row['id']: episode}, [row['id']],
                    np.random.default_rng(17100 + int(start)), starts=[int(start)])
                hold, cv = baselines(batch)
                for j, pred in enumerate([model(batch), hold, cv]):
                    errors = torch.linalg.vector_norm(pred-target, dim=-1)*scale[:, None, None]*1000
                    totals[j] += float(errors[mask].sum())
                count += int(mask.sum())
            if count == 0:
                raise ValueError('Held-out episode has no eligible predictions')
            evaluation.append(dict(episode_id=row['id'], model_mm=totals[0]/count,
                                   hold_mm=totals[1]/count, constant_velocity_mm=totals[2]/count))
    write_json(args.output / 'evaluation.json', dict(history=history, episodes=evaluation,
               heldout_object=args.heldout_object, training_episodes=keys))


if __name__ == '__main__':
    main()
