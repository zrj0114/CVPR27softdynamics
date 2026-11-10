"""Use spare farm GPUs for remaining episodes of the current support candidate.

Existing optimizer processes keep their candidate order and consume the same
locked physical cache. No future CMA candidate or held-out episode is launched.
"""
import hashlib
import os
from pathlib import Path
import pickle

import numpy as np

from .elastic_downstream import bounds,simulation_coordinates,wide_key
from .experiment_farm import available_gpus
from .io import read_json


def process_live(pid):
    path=Path('/proc')/str(pid)/'stat'
    try:return path.read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def current_candidate(folder,method,config):
    history_path=folder/'history.json'
    history=read_json(history_path) if history_path.exists() else []
    if method=='physical_cma':
        path=folder/'optimizer.pkl'
        if not path.exists():return None
        state=pickle.loads(path.read_bytes());pending=state['pending']
        if pending is None or len(pending['costs'])>=len(pending['candidates']):return None
        lo,hi=bounds(config)
        q=lo+np.asarray(pending['candidates'][len(pending['costs'])])*(hi-lo)
        return len(state['history'])+1,q
    path=folder/'proposal.json'
    if not path.exists():return None
    proposal=read_json(path)
    if proposal['round']!=len(history)+1:return None
    return proposal['round'],np.asarray(proposal['candidate_q'],float)


def update_jobs(run,plan,python,gpu_choices=None):
    run=Path(run);spec=read_json(run/'protocol.json');physics=read_json(run/'physics/plan.json')
    farm=read_json(run/'farm_status.json') if (run/'farm_status.json').exists() else dict(active={})
    active={k:v for k,v in farm['active'].items() if process_live(v['pid'])}
    jobs=list(plan['jobs']);removed=[]
    # A serial optimizer can finish an episode before its queued helper starts.
    # Drop such unused helpers, while preserving any job already active/done.
    for job in list(jobs):
        info=job.get('prefetch')
        if info and job['id'] not in active and not (run/'job_results'/f'{job["id"]}.json').exists() and (Path(info['folder'])/'complete.json').exists():
            jobs.remove(job);removed.append(job['id'])
    pending=sum(j['id'].startswith('prefetch_') and j['id'] not in active and
        not (run/'job_results'/f'{j["id"]}.json').exists() for j in jobs)
    if gpu_choices is None:gpu_choices=available_gpus(plan['gpus'],plan.get('shared_memory_limit_mb',0))
    busy={v['gpu'] for v in active.values()}
    spare=min(sum(index not in busy for index,_ in gpu_choices),plan.get('max_active',8)-len(active))-pending
    if spare<=0:return jobs,[],removed
    existing={j['id'] for j in jobs};candidates=[]
    for item in physics['objects']:
        obj=item['object'];baseline_path=run/'baselines'/f'{obj}.json'
        if not baseline_path.exists():continue
        baseline=read_json(baseline_path)
        if not baseline['valid']:continue
        duration={r['episode']:r['isolated_execution_s'] for r in baseline['records']}
        for method in spec['methods']:
            if f'inverse_{obj}_{method}' not in active:continue
            folder=run/'inverse'/obj/method;candidate=current_candidate(folder,method,spec['elastic_config'])
            if candidate is None:continue
            round_index,q=candidate
            storage=wide_key(simulation_coordinates(q,spec['elastic_config'],physics['config']))
            unfinished=[]
            for episode in item['support']:
                path=run/'physics/trajectories'/obj/f'episode_{episode}'/storage
                if not (path/'complete.json').exists():unfinished.append((episode,path))
            # The optimizer's own GPU handles the first unfinished episode.
            for episode,path in unfinished[1:]:
                key=f'prefetch_{obj}_{method}_{round_index}_{episode}_{storage[-12:]}'
                if key in existing:continue
                command=[python,'-m','softdynamics.elastic_downstream','--run',str(run),
                    '--mode','physics','--object',obj,'--episode',str(episode),'--q',
                    *[repr(float(v)) for v in q],'--device','{device}','--requester',f'{obj}/{method}']
                row=dict(id=key,command=command,after=[f'baseline_{obj}'],
                    prefetch=dict(object=obj,method=method,round=round_index,episode=episode,folder=str(path),
                                  current_candidate_only=True,query_observations_used=False))
                candidates.append((duration.get(episode,0.),row))
    candidates.sort(key=lambda row:(-row[0],row[1]['id']))
    additions=[row for _,row in candidates[:spare]]
    return jobs+additions,[j['id'] for j in additions],removed
