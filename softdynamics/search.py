"""Per-parameter weighted cyclic search; all directions come from full losses."""
import argparse
from pathlib import Path

import numpy as np

from .io import load_npz, read_json, write_json
from .parameters import ParameterSpace, PARAMETERS


def choice(options, costs, current):
    finite = np.isfinite(costs)
    if not finite.any():
        return current
    best = np.min(np.asarray(costs)[finite])
    scale = max(float(np.max(np.abs(np.asarray(costs)[finite]))), 1e-12)
    tied = [option for option, cost in zip(options, costs)
            if np.isfinite(cost) and cost <= best+scale*1e-10]
    return current if current in tied else min(tied, key=lambda value: abs(value-1))


def objective(loss, confidence):
    loss, confidence = np.asarray(loss), np.asarray(confidence)
    mask = np.isfinite(confidence)
    if loss.shape != confidence.shape or np.any(confidence[mask] < 0):
        raise ValueError('One nonnegative score or NaN per episode is required')
    if not np.isfinite(loss).all() or not mask.any():
        return np.inf
    return float(np.mean(confidence[mask]*loss[mask]))


def cycle(losses, confidence, start, levels=(-.025, 0., .025), passes=3):
    """losses(q) returns all episode losses; confidence is frozen throughout.

    A teacher-derived validity mask is only used when explicitly present in
    confidence. This does not whiten observations or resolve parameter coupling.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.shape != (3,) or not np.all(np.diff(levels) > 0) or levels[1] != 0:
        raise ValueError('Three increasing levels centered on zero are required')
    confidence = np.asarray(confidence, dtype=float)
    if confidence.ndim != 2 or confidence.shape[1] != 4 or passes < 1:
        raise ValueError('Expected confidence[episode,4] and positive passes')
    start = np.asarray(start, dtype=float)
    if start.shape != (4,) or not np.isfinite(start).all():
        raise ValueError('Expected four finite initial normalized coordinates')
    indices = [int(np.argmin(abs(levels-value))) for value in start]
    if not np.allclose(levels[indices], start, atol=1e-10, rtol=0):
        raise ValueError('Initial guess must belong to the candidate grid')
    cache, trace = {}, []
    for pass_id in range(passes):
        for j in range(4):
            trials, costs = [], []
            for value in range(3):
                candidate = list(indices)
                candidate[j] = value
                key = tuple(candidate)
                if key not in cache:
                    cache[key] = np.asarray(losses(levels[candidate]), dtype=float)
                trials.append(key)
                costs.append(objective(cache[key], confidence[:, j]))
            old = indices[j]
            indices[j] = choice(list(range(3)), costs, old)
            trace.append(dict(cycle=pass_id, parameter=PARAMETERS[j],
                              tested_indices=trials, costs=costs, old_level=old,
                              new_level=indices[j]))
    return levels[indices], trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True,
                   help='NPZ: confidence[method,episode,4], methods, episodes, start[4]')
    p.add_argument('--loss-bank', type=Path,
                   help='Optional NPZ: q[candidate,4], losses[candidate,episode], episodes')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--workspace', type=Path)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--cache', type=Path, default=Path('runs/rollouts'))
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config, weights = read_json(args.config), load_npz(args.weights)
    if weights['confidence'].shape != (len(weights['methods']), len(weights['episodes']), 4):
        raise ValueError('Weight methods, episode IDs and confidence dimensions differ')
    space = ParameterSpace.from_config(config)
    if args.loss_bank:
        bank = load_npz(args.loss_bank)
        np.testing.assert_array_equal(bank['episodes'], weights['episodes'])
        def loss_function(q):
            matches = np.flatnonzero(np.all(np.isclose(bank['q'], q, atol=1e-9, rtol=0), axis=1))
            if len(matches) != 1:
                raise ValueError(f'Missing or duplicate full-trajectory candidate {q}')
            return bank['losses'][matches[0]]
    else:
        if not args.manifest or not args.workspace:
            p.error('Use --loss-bank or provide --manifest and --workspace')
        from .rollouts import CachedRollouts
        loss_function = CachedRollouts(args.manifest, args.workspace, args.cache,
                                       config, space, args.gpus, args.runtime)
        np.testing.assert_array_equal([r['episode'] for r in loss_function.rows], weights['episodes'])
    rows = []
    for name, confidence in zip(weights['methods'], weights['confidence']):
        q, trace = cycle(loss_function, confidence, weights['start'],
                         config['grid_levels'], config['coordinate_passes'])
        final_losses = np.asarray(loss_function(q))
        valid = bool(np.isfinite(final_losses).all())
        rows.append(dict(method=str(name), q=q, parameters=space.parameters(q), trace=trace,
                         valid=valid, unweighted_objective_mm=float(final_losses.mean()) if valid else None))
        write_json(args.output, dict(completed_methods=rows, weights_frozen=True,
                   shared_teacher_mask=bool(weights.get('shared_teacher_mask', False))))


if __name__ == '__main__':
    main()
