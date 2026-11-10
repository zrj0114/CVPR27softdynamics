"""Immutable registered labels, causal histories and aligned joint responses."""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import os
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from .io import read_json, write_json, sha256, load_npz
from .elastic_world_data import freeze_data, MaterialPairs
from .shared_world_model import DT_REF


def snapshot(source, destination):
    """Pin completed files by hard link; copy/rebase mutable JSON metadata.

No candidate repair is admitted until it appears in the registration index.
The source collection is neither paused nor edited.
"""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Snapshot destination already exists')
    destination.mkdir(parents=True)
    plan = read_json(source/'plan.json')
    retries = read_json(source/'retry_index.json')
    frozen = copy.deepcopy(plan)
    frozen['tasks'] = []
    provenance = []
    counts = Counter()
    for task in plan['tasks']:
        original = source/'trajectories'/task['id']
        folder = source/retries[task['id']]['folder'] if task['id'] in retries else original
        if not (folder/'complete.json').exists():
            continue
        receipt = read_json(folder/'complete.json')
        if not receipt.get('valid') or not receipt.get('full_rollout'):
            continue
        counts[task['split']] += 1
        if task['split'] == 'test':
            continue
        if task['object'] in plan['excluded_objects']:
            raise ValueError('Held-out object in fitting split')
        digest = sha256(folder/'complete.json')
        if task['id'] in retries:
            rr = retries[task['id']]
            if digest != rr['complete_sha256'] or sha256(original/'complete.json') != rr['original_failure_sha256']:
                raise ValueError('Registration changed while taking snapshot')
        out = destination/'trajectories'/task['id']
        out.mkdir(parents=True)
        manifest = read_json(folder/'teacher_manifest.json')
        for src, dst in [(folder/'trajectory.npz', out/'trajectory.npz'),
                         (Path(manifest['input']), out/'confidence_input.npz')]:
            os.link(src, dst)
        nodes = destination/'node_maps'/(receipt['nodes_sha256']+'.npz')
        nodes.parent.mkdir(exist_ok=True)
        if not nodes.exists():
            os.link(receipt['nodes'], nodes)
        teacher = out/'teacher'
        teacher.mkdir()
        for row in manifest['records']:
            os.link(Path(manifest['teacher_directory'])/row['path'], teacher/row['path'])
        shutil.copyfile(manifest['metadata'], out/'metadata.json')
        manifest.update(input=str(out/'confidence_input.npz'), metadata=str(out/'metadata.json'),
                        teacher_directory=str(teacher))
        write_json(out/'teacher_manifest.json', manifest)
        receipt.update(nodes=str(nodes), teacher_manifest=str(out/'teacher_manifest.json'))
        write_json(out/'complete.json', receipt)
        effective = dict(task, **receipt['task'])
        for k in ('id', 'object', 'episode', 'split'):
            if effective[k] != task[k]:
                raise ValueError('Repair identity/split changed')
        frozen['tasks'].append(effective)
        provenance.append(dict(id=task['id'], source_folder=str(folder), source_receipt_sha256=digest,
            trajectory_sha256=receipt['trajectory_sha256'], confidence_input_sha256=manifest['input_sha256']))
    frozen.update(snapshot_source=str(source), snapshot_at=time.time())
    write_json(destination/'plan.json', frozen)
    result = dict(source=str(source), counts_at_snapshot=dict(counts), included=len(provenance),
                  held_out_objects=plan['excluded_objects'], files='Hard-linked completed arrays; copied JSON metadata',
                  records=provenance, created_at=time.time())
    write_json(destination/'snapshot.json', result)
    return result


def history_batch(replay, choices, device, count=3):
    values = defaultdict(list)
    for identity, start in choices:
        t = replay.trajectories[identity]
        indices = np.arange(start-count, start)
        present = indices >= t['minimum_start']
        indices = np.maximum(indices, t['minimum_start'])
        values['states'].append(t['state'][indices])
        values['pads0'].append(t['pads'][indices])
        values['pads1'].append(t['pads'][indices+1])
        values['dt'].append(t['times'][indices+1]-t['times'][indices])
        values['present'].append(present)
    return {k: torch.as_tensor(np.stack(v), device=device,
                dtype=torch.bool if k=='present' else torch.float32) for k, v in values.items()}


def temporal_batch(replay, choices, horizon, device):
    batch, geo = replay.batch(choices, horizon, device)
    batch['history'] = history_batch(replay, choices, device)
    return batch, geo


class TemporalPairs(MaterialPairs):
    def batch(self, choice, device):
        batch, geo = super().batch(choice, device)
        batch['history'] = history_batch(self.replay, [(choice[0], 0), (choice[1], 0)], device)
        return batch, geo


def fit_statistics_streaming(replay):
    if replay.splits != {'train'}:
        raise ValueError('Only training objects calibrate the model')
    total = 0
    first = np.zeros(24, np.float64)
    second = first.copy()
    delta2 = np.zeros(21, np.float64)
    for t in replay.trajectories.values():
        ids = np.linspace(t['minimum_start'], len(t['state'])-2,
                          min(64, len(t['state'])-1-t['minimum_start'])).astype(int)
        states = t['state'][ids].reshape(-1, 24).astype(np.float64)
        delta = ((t['state'][ids+1, :, 3:]-t['state'][ids, :, 3:]) /
                 (np.diff(t['times'])[ids]/DT_REF)[:, None, None]).reshape(-1, 21).astype(np.float64)
        first += states.sum(0)
        second += (states*states).sum(0)
        delta2 += (delta*delta).sum(0)
        total += len(states)
    mean = first/total
    std = np.sqrt(np.maximum(second/total-mean*mean, 1e-6))
    output = np.sqrt(delta2/total)
    for lo, hi in ((0, 3), (3, 6), (6, 15), (15, 24)):
        std[lo:hi] = max(float(np.sqrt(np.mean(std[lo:hi]**2))), 1e-3)
    for lo, hi in ((0, 3), (3, 12), (12, 21)):
        output[lo:hi] = max(float(np.sqrt(np.mean(output[lo:hi]**2))), 1e-4)
    return dict(state_mean=mean, state_std=std, output_scale=output, loss_scale=std,
                trajectories=sorted(replay.trajectories), objects=replay.object_names())


def to_device(value, device):
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    a = np.asarray(value)
    dtype = torch.bool if a.dtype == bool else torch.long if np.issubdtype(a.dtype, np.integer) else torch.float32
    return torch.as_tensor(a, device=device, dtype=dtype)


def teacher_context(visual, label, windows, base_theta):
    """Exact teacher point order and endpoint masks; prefix inputs are causal.

Synthetic point states used by local physics distillation are fully available.
The scoring encoder separately masks points absent from the observed history.
No future position enters either current-state encoder.
"""
    lookup = {int(v): i for i, v in enumerate(visual['point_ids'])}
    ids = np.array([lookup[int(v)] for v in label['point_ids']])
    x = visual['positions'][:, ids]
    visible = visual['valid'][:, ids]
    times = visual['times'].astype(np.float64)
    if not np.isfinite(x).all() or not (np.diff(times) > 0).all():
        raise ValueError('Invalid synthetic point history')
    starts, ends = label['starts'], label['ends']
    if not np.array_equal(ends-starts, np.ones(len(starts))) or any(
            w['start'] != s or w['end'] != e for w, s, e in zip(windows, starts, ends)):
        raise ValueError('Teacher windows are not aligned local transitions')
    states = np.zeros((*x.shape[:2], 24), np.float32)
    states[..., :3] = x
    states[1:, :, 3:6] = np.diff(x, axis=0)/(np.diff(times)/DT_REF)[:, None, None]
    source = visual['actions'].reshape(len(times), 4, 13)
    matrix = source[..., :12].reshape(len(times), 4, 3, 4)
    pads = np.concatenate((matrix[..., 3], matrix[..., :3].reshape(len(times), 4, 9), source[..., 12:]), -1)
    canonical = x[0]
    neighbors = cKDTree(canonical).query(canonical, k=min(25, len(ids)))[1][:, 1:]
    scale = float(visual['scale_m'])
    geo = dict(canonical=canonical, neighbors=neighbors, surface_mask=np.ones(len(ids), bool),
               scale_m=np.float32(scale), ground_y=np.float32((-.01-visual['origin_m'][1])/scale))
    geometry = {k: np.broadcast_to(v, (len(starts), *np.shape(v))).copy() for k, v in geo.items()}
    past = starts[:, None] + np.arange(-3, 0)[None]
    present = past >= 0
    past = np.maximum(past, 0)
    history = dict(states=states[past], pads0=pads[past], pads1=pads[past+1],
                   dt=(times[past+1]-times[past]).astype('f'), present=present)
    weights = np.array([w['weight'] for w in windows])
    counts = visible[ends].sum(1)
    if not (counts > 0).all() or label['J_fine'].shape != (int(counts.sum())*3, 2):
        raise ValueError('Teacher residual layout and observed point masks differ')
    phase = np.array([('before', 'interaction', 'release').index(w['phase']) for w in windows])
    time_features = np.c_[times[starts]/max(times[-1], 1e-6),
                          (times[ends]-times[starts])/DT_REF, np.eye(3)[phase]].astype('f')
    return dict(states=states[starts], pads0=pads[starts], pads1=pads[ends],
        dt=(times[ends]-times[starts]).astype('f'), geometry=geometry, history=history,
        observed=visible[starts] & visible[np.maximum(starts-1, 0)], target_visible=visible[ends],
        response_scale=(scale*1000*np.sqrt(weights/counts)).astype('f'),
        time_features=time_features, base_theta=np.broadcast_to(base_theta, (len(starts), 2)).astype('f'))


class JointLabels:
    def __init__(self, root, replay):
        self.root, self.replay = Path(root), replay
        self.rows, self.contexts = [], {}
        self.by_object = defaultdict(list)
        for identity, trajectory in replay.trajectories.items():
            folder = self.root/'trajectories'/identity
            manifest = read_json(folder/'teacher_manifest.json')
            if manifest['parameters'] != ['E_pa', 'nu'] or sha256(manifest['input']) != manifest['input_sha256']:
                raise ValueError('Changed sensitivity input or coordinates')
            visual = None
            for record in manifest['records']:
                path = Path(manifest['teacher_directory'])/record['path']
                if sha256(path) != record['sha256']:
                    raise ValueError('Changed sensitivity label')
                z = load_npz(path)
                valid = z['valid'] & np.isfinite(z['S']) & (z['S'] >= 0) & np.isfinite(z['J_fine']).all(0)
                if not valid.any():
                    continue
                if identity not in self.contexts:
                    visual = load_npz(manifest['input'])
                    self.contexts[identity] = teacher_context(visual, z, record['windows'], trajectory['theta'])
                if not np.allclose(np.linalg.norm(z['J_fine'][:, valid], axis=0), z['S'][valid], rtol=1e-6):
                    raise ValueError('Sensitivity norm and signed response disagree')
                fine = replay.plan['config']['teacher']['radii'][-1]
                refined = replay.plan['config']['teacher']['refinement_radii'][-1]
                radii = np.array([refined if j in record['refined_axes'] else fine for j in range(2)], 'f')
                correlation_valid = bool(valid.all() and np.isfinite(z['correlation'][0, 1]) and (z['S'] > 1e-10).all())
                row = dict(id=identity, object=trajectory['task']['object'], q=z['q'], valid=valid,
                    S=np.where(valid, z['S'], 0.), rho=float(z['correlation'][0, 1]) if correlation_valid else 0.,
                    correlation_valid=correlation_valid, J=np.where(valid[None], z['J_fine'], 0.).astype('f'),
                    radii=radii, source=str(path), sha256=record['sha256'])
                self.by_object[row['object']].append(len(self.rows))
                self.rows.append(row)
        if not self.rows:
            raise ValueError('No valid joint response supervision')

    def calibrate(self):
        if self.replay.splits != {'train'}:
            raise ValueError('Response scales require training data')
        return np.array([max(np.median([r['S'][j] for r in self.rows if r['valid'][j]]), 1e-8) for j in range(2)])

    def sample(self, count, rng):
        objects = sorted(self.by_object)
        return [int(rng.choice(self.by_object[str(rng.choice(objects))])) for _ in range(count)]

    def batch(self, ids, tau, device):
        rows = [self.rows[i] for i in ids]
        result = {k: np.stack([r[k] for r in rows]) for k in ('q', 'valid', 'rho', 'correlation_valid', 'radii')}
        result['score'] = np.stack([r['S']/(r['S']+tau) for r in rows])
        result['J'] = [r['J'] for r in rows]
        result['contexts'] = [self.contexts[r['id']] for r in rows]
        return to_device(result, device)
