"""Frozen MPM replay and signed, full-history material response pairs.

Pairs use the SAME episode, initial state, geometry, times and actions. Their
only changed physical inputs are E and nu. Responses are finite parameter
secants, not labels of an infinitesimal simulator Jacobian.
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .io import read_json, sha256, write_json
from .shared_world_training import Replay
from .source_equivalence import physical_sources


def freeze_data(data_root, output):
    data_root, output = Path(data_root).resolve(), Path(output)
    path = output / 'data_manifest.json'
    if path.exists():
        result = read_json(path)
        if result['data_root'] != str(data_root) or result['plan_sha256'] != sha256(data_root/'plan.json'):
            raise ValueError('Changed frozen data source')
        return result
    plan = read_json(data_root/'plan.json')
    retries=read_json(data_root/'retry_index.json') if (data_root/'retry_index.json').exists() else {}
    records = []
    for task in plan['tasks']:
        if plan.get('collection_includes_test') and task['split']=='test':
            if task['object'] not in plan['excluded_objects']:raise ValueError('Test collection split mismatch')
            continue
        folder = data_root/'trajectories'/task['id']
        if task['id'] in retries:
            retry=retries[task['id']]
            if sha256(folder/'complete.json')!=retry['original_failure_sha256']:raise ValueError('Changed original retry failure')
            folder=data_root/retry['folder']
            if sha256(folder/'complete.json')!=retry['complete_sha256']:raise ValueError('Changed successful numerical retry')
        receipt = folder/'complete.json'
        if not receipt.exists():
            continue
        complete = read_json(receipt)
        if not complete['valid'] or not complete['full_rollout']:
            continue
        if task['object'] in plan['excluded_objects']:
            raise ValueError('Held-out object entered training replay')
        records.append(dict(id=task['id'], split=task['split'], object=task['object'],
            receipt_sha256=sha256(receipt), trajectory_sha256=complete['trajectory_sha256'],
            record_folder=str(folder.relative_to(data_root))))
    result = dict(schema='softdynamics.world-search-data.v1', data_root=str(data_root),
        plan_sha256=sha256(data_root/'plan.json'), records=records,
        reused_mpm_trajectories=True, reused_network_weights=False,
        excluded_objects=plan['excluded_objects'])
    write_json(path, result)
    return result


def frozen_replay(manifest, split):
    root = Path(manifest['data_root'])
    for r in manifest['records']:
        folder=root/r.get('record_folder','trajectories/'+r['id'])
        if r['split'] == split and sha256(folder/'complete.json') != r['receipt_sha256']:
            raise ValueError('Changed trajectory receipt')
    replay = Replay(root, [split], observed_velocity=True)
    replay.refresh()
    allowed = {r['id'] for r in manifest['records'] if r['split'] == split}
    replay.trajectories = {k: v for k, v in sorted(replay.trajectories.items()) if k in allowed}
    if set(replay.trajectories) != allowed:
        raise ValueError('Incomplete frozen replay')
    return replay


class MaterialPairs:
    def __init__(self, replay):
        self.replay = replay
        groups = defaultdict(list)
        for identity, t in replay.trajectories.items():
            groups[(t['task']['object'], t['task']['episode'])].append(identity)
        self.by_object = defaultdict(list)
        self.rejected = []
        for (obj, _), ids in sorted(groups.items()):
            for a, b in combinations(sorted(ids), 2):
                x, y = replay.trajectories[a], replay.trajectories[b]
                reason = self.mismatch(x, y)
                if reason:
                    self.rejected.append(dict(a=a, b=b, reason=reason))
                else:
                    self.by_object[obj].append((a, b))
        if not self.by_object:
            raise ValueError('No physically matched material pairs')

    @staticmethod
    def mismatch(x, y):
        if x['minimum_start'] != 0 or y['minimum_start'] != 0:
            return 'Missing causal initial state'
        for key in ('pads', 'times', 'phase', 'origin_m', 'scale_m'):
            if not np.array_equal(x[key], y[key]):
                return 'Different ' + key
        if not np.array_equal(x['state'][0], y['state'][0]):
            return 'Different initial state'
        if physical_sources(x['record']) != physical_sources(y['record']):
            return 'Different physical sources'
        if x['record'].get('domain') != y['record'].get('domain'):
            return 'Different simulation domain'
        for key, value in x['task']['parameters'].items():
            if key not in ('E_pa', 'nu') and y['task']['parameters'][key] != value:
                return 'Different fixed physical parameter'
        if x['task']['dtmax'] != y['task']['dtmax']:
            return 'Different timestep'
        if np.array_equal(x['theta'], y['theta']):
            return 'Identical material'
        return None

    def sample(self, rng, max_horizon):
        obj = str(rng.choice(sorted(self.by_object)))
        a, b = self.by_object[obj][int(rng.integers(len(self.by_object[obj])))]
        t = self.replay.trajectories[a]
        last = len(t['state']) - 1
        if max_horizon is not None:
            last = min(last, max_horizon)
        # Every fourth pair spans the entire currently allowed prefix. Other
        # endpoints are phase-balanced, always rolling from frame ZERO.
        if rng.random() < .25:
            end = last
        else:
            phases = np.unique(t['phase'][1:last+1])
            phase = int(rng.choice(phases))
            eligible = np.flatnonzero(t['phase'][1:last+1] == phase) + 1
            end = int(rng.choice(eligible))
        return a, b, end

    def batch(self, choice, device):
        a, b, horizon = choice
        batch, geometry = self.replay.batch([(a, 0), (b, 0)], horizon, device)
        t = self.replay.trajectories[a]
        batch['phase'] = torch.as_tensor(t['phase'][1:horizon+1], device=device)
        cfg = self.replay.plan['config']
        q = np.array([[np.log(self.replay.trajectories[k]['theta'][0]/cfg['reference_E_pa']),
            (self.replay.trajectories[k]['theta'][1]-cfg['reference_nu'])/cfg['nu_scale']] for k in (a, b)])
        batch['parameter_distance'] = torch.tensor(float(np.linalg.norm(q[1]-q[0])), device=device)
        return batch, geometry

    def sample_phase(self, rng, phase):
        """Choose an observed phase endpoint, with no cap on its frame index."""
        if phase not in (0,1,2):raise ValueError('Unknown phase')
        eligible={}
        for obj,choices in self.by_object.items():
            valid=[(a,b) for a,b in choices if np.any(self.replay.trajectories[a]['phase'][1:]==phase)]
            if valid:eligible[obj]=valid
        if not eligible:raise ValueError('No pairs cover requested phase')
        obj=str(rng.choice(sorted(eligible)))
        a,b=eligible[obj][int(rng.integers(len(eligible[obj])))]
        phases=self.replay.trajectories[a]['phase']
        frames=np.flatnonzero(phases[1:]==phase)+1
        end=len(phases)-1 if phase==2 and rng.random()<.25 else int(rng.choice(frames))
        return a,b,end

    def validation_choices(self):
        # Fixed, metadata-only pair selection. No choice based on model error.
        choices = []
        for obj, pairs in sorted(self.by_object.items()):
            a, b = pairs[0]
            choices.append((a, b, len(self.replay.trajectories[a]['state'])-1))
        return choices

    def report(self):
        return dict(per_object={o: len(v) for o, v in sorted(self.by_object.items())},
                    pairs=sum(map(len, self.by_object.values())), rejected=self.rejected)


def balanced_validation_windows(replay, horizon=32, per_phase=2):
    """Fixed windows from every trajectory, wholly contained in each phase."""
    result=[]
    for key,t in sorted(replay.trajectories.items()):
        for phase in (0,1,2):
            starts=[i for i in range(t['minimum_start'],len(t['state'])-horizon)
                    if np.all(t['phase'][i+1:i+horizon+1]==phase)]
            if starts:
                for j in np.unique(np.linspace(0,len(starts)-1,min(per_phase,len(starts))).astype(int)):
                    result.append((key,int(starts[j]),phase))
    if not result:raise ValueError('No balanced validation windows')
    return result
