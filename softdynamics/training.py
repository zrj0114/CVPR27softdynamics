"""Four independent heads with whole-episode splits and train-only calibration."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .inference import Heads
from .io import load_npz, read_json, write_json
from .parameters import PARAMETERS, ParameterSpace


def fit(X, S, valid, episode_ids, train, validation, seed=0, epochs=600,
        patience=70, device='cpu'):
    """The prediction target is standardized log1p(S/tau); masks never become zero labels."""
    X, S = np.asarray(X), np.asarray(S)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(S) & (S >= 0)
    groups = np.asarray(episode_ids).astype(str)
    train, validation = np.asarray(train, dtype=int), np.asarray(validation, dtype=int)
    if X.shape != (len(S), 580) or S.shape != (len(groups), 4) or valid.shape != S.shape:
        raise ValueError('Expected X[M,580], S[M,4], valid[M,4], episode_ids[M]')
    if not np.isfinite(X).all() or epochs < 1 or patience < 1:
        raise ValueError('Finite features and positive epochs/patience are required')
    if not len(train) or not len(validation) or set(groups[train]) & set(groups[validation]):
        raise ValueError('Training and validation must contain disjoint whole episodes')
    torch.manual_seed(seed)
    counts = np.array([len(set(groups[train][valid[train, j]])) for j in range(4)])
    ready = counts >= 3
    tau = np.array([max(np.median(S[train, j][valid[train, j]]), 1e-8)
                    if valid[train, j].any() else 1. for j in range(4)])
    y = np.log1p(np.where(valid, S, 0)/tau)
    mu, std = X[train].mean(0), np.maximum(X[train].std(0), 1e-4)
    ym = np.array([y[train, j][valid[train, j]].mean() if valid[train, j].any() else 0.
                   for j in range(4)])
    ys = np.array([max(y[train, j][valid[train, j]].std(), .01) if valid[train, j].any() else 1.
                   for j in range(4)])
    x = torch.tensor((X-mu)/std, dtype=torch.float32, device=device)
    target = torch.tensor((y-ym)/ys, dtype=torch.float32, device=device)
    mask = torch.tensor(valid & ready[None], device=device)
    if not mask[train].any() or not mask[validation].any():
        raise ValueError('Need reliable training and validation labels on supported heads')
    model = Heads().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    best, best_epoch, best_state, history = float('inf'), 0, None, []
    for epoch in range(epochs):
        model.train()
        loss = ((model(x[train])-target[train])**2)[mask[train]].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(((model(x[validation])-target[validation])**2)[mask[validation]].mean())
        history.append([float(loss.detach()), val_loss])
        if val_loss < best-1e-7:
            best, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch-best_epoch >= patience:
            break
    state = {k: torch.as_tensor(v).cpu() for k, v in dict(mu=mu, std=std,
             target_mean=ym, target_std=ys, tau=tau, ready=ready).items()}
    state['model'] = {k: v.detach().cpu() for k, v in best_state.items()}
    return state, dict(best_epoch=best_epoch, validation_loss=best, history=history,
                      reliable_training_episode_counts=counts)


def predict(state, X, device='cpu'):
    model = Heads().to(device)
    model.load_state_dict(state['model'])
    model.eval()
    with torch.inference_mode():
        z = model(torch.tensor((X-state['mu'].numpy())/state['std'].numpy(),
                               dtype=torch.float32, device=device)).cpu().numpy()
    u = (z*state['target_std'].numpy()+state['target_mean'].numpy()).clip(0, 10)
    S = state['tau'].numpy()*np.expm1(u)
    S[:, ~state['ready'].numpy()] = np.nan
    return S


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--validation', nargs='+', required=True)
    p.add_argument('--test', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--patience', type=int, default=70)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d = load_npz(args.dataset)
    groups = d['episode_ids'].astype(str)
    if set(args.validation) & set(args.test) or not (set(args.validation) | set(args.test)) <= set(groups):
        raise ValueError('Validation/test IDs must exist and be disjoint')
    train = np.flatnonzero(~np.isin(groups, args.validation+args.test))
    val, test = np.flatnonzero(np.isin(groups, args.validation)), np.flatnonzero(np.isin(groups, args.test))
    heads, records = [], []
    for seed in args.seeds:
        state, record = fit(d['X'], d['S'], d['valid'], groups, train, val, seed,
                            args.epochs, args.patience, args.device)
        heads.append(state)
        records.append(record)
    tau = heads[0]['tau'].numpy()
    predictions = np.mean([predict(state, d['X'], args.device) for state in heads], axis=0)
    scores, targets = predictions/(predictions+tau), d['S']/(d['S']+tau)
    valid = d['valid'].astype(bool) & np.isfinite(d['S']) & np.isfinite(scores)
    constant = np.array([-np.expm1(-np.log1p(d['S'][train, j][valid[train, j]]/tau[j]).mean())
                         if valid[train, j].any() else np.nan for j in range(4)])
    evaluation = []
    for j, key in enumerate(PARAMETERS):
        ids = test[valid[test, j]]
        evaluation.append(dict(parameter=key, count=len(ids),
            mae=float(np.abs(scores[ids, j]-targets[ids, j]).mean()) if len(ids) else None,
            constant_mae=float(np.abs(constant[j]-targets[ids, j]).mean()) if len(ids) else None))
    config = read_json(args.config)
    space = ParameterSpace.from_config(config)
    encoder = torch.load(args.encoder, map_location='cpu', weights_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(encoder=encoder['model'], heads=heads, parameters=list(PARAMETERS),
                    base=torch.tensor([space.base[k] for k in PARAMETERS], dtype=torch.float64),
                    scales=torch.tensor(space.scales), tau=heads[0]['tau'], query_step=config['query_step'],
                    action_dim=encoder.get('action_dim',28)),
               args.output/'model.pt')
    write_json(args.output/'training.json', dict(train_episodes=sorted(set(groups[train])),
               validation_episodes=args.validation, test_episodes=args.test, seeds=records,
               test_evaluation=evaluation, calibration_from_training_only=True))


if __name__ == '__main__':
    main()
