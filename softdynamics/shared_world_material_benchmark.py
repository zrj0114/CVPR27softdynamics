"""Verify frozen world-model material proposals with the original full solver.

Support alone determines acceptance. Query observations are scored only after
that decision is frozen. No old candidate cache is imported across timesteps.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .io import read_json, sha256, write_json
from .recovery_data import qkey, trajectory_folder, simulate, score_candidate


def bounded_export(q):
    """Canonicalize only the exact float32 representation of the clamp bounds."""
    values=np.asarray(q,dtype=np.float64)
    if values.shape!=(4,) or not np.isfinite(values).all():raise ValueError('Invalid material coordinates')
    outside=np.abs(values)>.025
    float32_bound=float(np.float32(.025))
    if np.any(outside & (np.abs(values)!=float32_bound)):
        raise ValueError('Material proposal lies outside the frozen bounds')
    return np.clip(values,-.025,.025).tolist()


def prepare(data_root, root, selections):
    data_root, root = Path(data_root), Path(root)
    if (root/'plan.json').exists():
        raise ValueError('Physical benchmark plan is frozen')
    data = read_json(data_root/'plan.json')
    proposals = [read_json(p) for p in selections]
    first = proposals[0]; obj = first['object']
    for p in proposals:
        if any(p[k] != first[k] for k in ['object', 'initial_q', 'support', 'query', 'model_sha256', 'confidence_sha256']):
            raise ValueError('Material proposals do not share an initial condition and protocol')
        if p['query_observations_used'] or not p['world_weights_frozen'] or not p['recorded_actions_fixed']:
            raise ValueError('Invalid world-model material proposal protocol')
    if set(first['support']) & set(first['query']):
        raise ValueError('Overlapping support and query episodes')
    plan = copy.deepcopy(read_json(Path(data['fit_run'])/'plan.json'))
    plan['objects'] = [o for o in plan['objects'] if o['object'] == obj]
    if len(plan['objects']) != 1:
        raise ValueError('Missing downstream object')
    item = plan['objects'][0]
    if item['support'] != first['support'] or item['query'] != first['query']:
        raise ValueError('Original first-module split changed')
    for entry in item['episodes']:
        task = next(t for t in data['tasks'] if t['object'] == obj and t['episode'] == entry['episode'] and t['variant'] == 0)
        entry['dtmax'] = task['dtmax']; entry['domain_margin'] = task['domain_margin']
    plan['conditions'] = [c for c in plan['conditions'] if c['truth'] == 'real']
    if len(plan['conditions']) != 1:
        raise ValueError('Expected one unchanged real-observation objective')
    plan.update(schema='softdynamics.shared-world-physical-material.v1',
        data_root=str(data_root.resolve()), data_plan_sha256=sha256(data_root/'plan.json'),
        source_fit_plan_sha256=sha256(Path(data['fit_run'])/'plan.json'),
        proposals={p['method']:dict(q=bounded_export(p['proposed_q']),serialized_source_q=p['proposed_q'], source=str(path), source_sha256=sha256(path),
            surrogate_seconds=p['history'][-1]['elapsed_s']) for p,path in zip(proposals,selections)},
        incumbent_q=bounded_export(first['initial_q']), world_checkpoint_sha256=first['model_sha256'],
        confidence_sha256=first['confidence_sha256'], created_at=time.time(),
        cache_policy='New physical rollouts at the world-data timestep; no original CMA cache imported',
        boundary_precision='Only exact float32 clamp endpoints are serialized as exact float64 +/-0.025',
        acceptance='Strict unweighted physical support improvement over the completed first-module incumbent',
        query_policy='Score raw and accepted candidates only after the support decision is frozen')
    write_json(root/'plan.json', plan)
    return plan


def worker(root, obj, episode, q):
    folder = trajectory_folder(root, obj, episode, q); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        simulate(root, obj, episode, q)
        score_candidate(root, obj, episode, q)


def select_support(incumbent, proposed):
    valid = all(v is not None and np.isfinite(v) for v in incumbent+proposed)
    return bool(valid and np.mean(proposed) < np.mean(incumbent))


def run(root, gpus):
    root = Path(root); plan = read_json(root/'plan.json'); item = plan['objects'][0]; obj = item['object']
    if (root/'result.json').exists():
        raise ValueError('Physical material result is frozen')
    with (root/'benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        candidates = {'incumbent':plan['incumbent_q'], **{k:v['q'] for k,v in plan['proposals'].items()}}
        unique = {qkey(q):q for q in candidates.values()}
        active = {}; completed = []; claim = threading.Lock(); started = time.monotonic()
        def stage(episodes, name):
            jobs = [(episode,q) for episode in episodes for q in unique.values()]
            def execute(gpu):
                while True:
                    with claim:
                        if not jobs:return
                        episode,q = jobs.pop(0)
                    command = [sys.executable,'-u','-m','softdynamics.shared_world_material_benchmark','worker',
                        '--root',str(root),'--object',obj,'--episode',str(episode),'--q',*map(str,q)]
                    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    logs = root/'logs';logs.mkdir(exist_ok=True)
                    with (logs/(str(episode)+'_'+qkey(q)+'.log')).open('a') as log:
                        p = subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,stdout=log,stderr=subprocess.STDOUT)
                        with claim:
                            active[str(gpu)] = dict(pid=p.pid,episode=episode,q=q)
                            write_json(root/'status.json',dict(pid=os.getpid(),stage=name,active=active,completed=completed,updated_at=time.time()))
                        code = p.wait()
                    with claim:
                        active.pop(str(gpu),None); completed.append(dict(episode=episode,q=q,exit_code=code))
                        write_json(root/'status.json',dict(pid=os.getpid(),stage=name,active=active,completed=completed,updated_at=time.time()))
                    if code:
                        raise RuntimeError('Physical material worker failed; see '+str(logs))
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                list(pool.map(execute,gpus))
        condition = plan['conditions'][0]['key']
        def losses(q,episodes):
            values = []
            for episode in episodes:
                score = read_json(trajectory_folder(root,obj,episode,q)/'scores.json')['conditions'][condition]
                values.append(score['objective_mm'] if score['valid'] else None)
            return values
        stage(item['support'],'physical_support')
        support = {name:losses(q,item['support']) for name,q in candidates.items()}
        accepted = {name:select_support(support['incumbent'],support[name]) for name in plan['proposals']}
        selection = dict(support_losses_mm=support,accepted=accepted,
            selected_q={name:candidates[name] if good else candidates['incumbent'] for name,good in accepted.items()},
            query_observations_used=False,frozen_at=time.time())
        write_json(root/'support_selection.json',selection)
        stage(item['query'],'held_out_query')
        query = {name:losses(q,item['query']) for name,q in candidates.items()}
        result = dict(object=obj,candidates=candidates,support=item['support'],query=item['query'],
            support_selection=selection,raw_query_losses_mm=query,
            accepted_query_losses_mm={name:query[name] if good else query['incumbent'] for name,good in accepted.items()},
            physical_rollouts=len(unique)*(len(item['support'])+len(item['query'])),
            shared_cache_wall_time_is_not_a_method_speedup=True,elapsed_s=time.monotonic()-started,
            plan_sha256=sha256(root/'plan.json'),query_used_for_acceptance=False)
        write_json(root/'result.json',result)
        write_json(root/'status.json',dict(pid=os.getpid(),stage='complete',updated_at=time.time()))
        return result


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='mode',required=True)
    a = sub.add_parser('prepare');a.add_argument('--data-root',type=Path,required=True);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--selections',type=Path,nargs='+',required=True)
    a = sub.add_parser('run');a.add_argument('--root',type=Path,required=True);a.add_argument('--gpus',type=int,nargs='+',required=True)
    a = sub.add_parser('worker');a.add_argument('--root',type=Path,required=True);a.add_argument('--object',required=True)
    a.add_argument('--episode',type=int,required=True);a.add_argument('--q',type=float,nargs=4,required=True)
    a = p.parse_args()
    if a.mode=='prepare':prepare(a.data_root,a.root,a.selections)
    elif a.mode=='worker':worker(a.root,a.object,a.episode,a.q)
    else:run(a.root,a.gpus)


if __name__=='__main__':main()
