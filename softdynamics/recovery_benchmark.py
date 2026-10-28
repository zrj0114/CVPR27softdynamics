"""Resumable uniform-versus-learned parameter recovery, using exact full rollouts."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .io import load_npz,read_json,sha256,write_json
from .parameters import PARAMETERS,ParameterSpace
from .recovery_data import (condition_key,find_entry,observation_for,qkey,
                            trajectory_folder)
from .search import choice,objective
from .shared_pipeline import alive,gpu_state


def shuffled_scores(values):
    result=np.array(values,copy=True)
    for j in range(4):
        order=np.random.default_rng(7300+j).permutation(len(values))
        if len(values)>1 and np.array_equal(order,np.arange(len(values))):order=np.roll(order,1)
        result[:,j]=values[order,j]
    return result


def frontier(bank,confidence,start,passes=3,levels=(-.025,0.,.025)):
    """Same updates/tie rules as search.cycle; pause only for missing exact losses."""
    levels=np.asarray(levels);indices=[int(np.argmin(abs(levels-v))) for v in start]
    if not np.allclose(levels[indices],start,atol=1e-9,rtol=0):raise ValueError('Off-grid start')
    trace=[];visited=set()
    for pass_id in range(passes):
        for j in range(4):
            candidates=[]
            for value in range(3):
                q=indices.copy();q[j]=value;candidates.append(levels[q])
            missing=[q for q in candidates if qkey(q) not in bank]
            if missing:return None,trace,missing
            costs=[objective(bank[qkey(q)],confidence[:,j]) for q in candidates]
            visited.update(qkey(q) for q in candidates)
            old=indices[j];indices[j]=choice(list(range(3)),costs,old)
            trace.append(dict(cycle=pass_id,parameter=PARAMETERS[j],costs=costs,
                         tested_q=[q.tolist() for q in candidates],old_level=old,new_level=indices[j]))
    return dict(q=levels[indices],trace=trace,logical_candidate_vectors=passes*4*3,
                unique_candidate_vectors=len(visited)),trace,[]


def compute_weights(root,obj):
    import torch
    from .inference import ConfidenceModel
    root=Path(root);plan=read_json(root/'plan.json');output=root/'weights'/f'{obj}.json'
    if output.exists():return read_json(output)
    if sha256(plan['checkpoint'])!=plan['model_sha256']:raise ValueError('Frozen checkpoint changed')
    torch.set_num_threads(2);model=ConfidenceModel(plan['checkpoint'],'cpu')
    item=next(o for o in plan['objects'] if o['object']==obj)
    if obj in model.bundle['training_objects']:raise ValueError('Downstream test object leaked into training')
    contexts=[];started=time.monotonic()
    for condition in plan['conditions']:
        features=[]
        for episode in item['support']:
            entry=find_entry(plan,obj,episode);row=read_json(entry['row']);data=load_npz(row['real'])
            cache=root/'features'/obj/condition['key']/f'episode_{episode}.npy'
            if cache.exists():feature=np.load(cache)
            else:
                _,tracks=observation_for(root,plan,obj,episode,condition)
                data['positions']=((tracks-data['origin_m'])/float(data['scale_m'])).astype(np.float32)
                feature=model.feature(data);cache.parent.mkdir(parents=True,exist_ok=True);np.save(cache,feature)
            features.append(feature)
        guesses=[plan['real_initial_guess']] if condition['truth']=='real' else plan['initial_guesses']
        for start_no,start in enumerate(guesses):
            p=model.bundle['base'].numpy()+np.array(start)*model.bundle['scales'].numpy()
            learned=np.stack([model.predict_feature(f,p)['confidence'] for f in features])
            if not np.isfinite(learned).all() or np.any(learned<0):raise ValueError('Invalid model scores')
            contexts.append(dict(id=condition['key']+f'_start{start_no}',condition=condition,start=start,
                           weights=dict(uniform=np.ones_like(learned),learned=learned,shuffled=shuffled_scores(learned))))
        print('WEIGHTS',obj,condition['key'],flush=True)
    result=dict(object=obj,support=item['support'],query=item['query'],contexts=contexts,
                model_sha256=plan['model_sha256'],teacher_scores_used=False,elapsed_s=time.monotonic()-started)
    write_json(output,result);return result


def score_path(root,obj,episode,q):
    return trajectory_folder(root,obj,episode,q)/'scores.json'


def bank_for(root,item,condition):
    banks=[]
    for episode in item['support']:
        values={}
        for path in (Path(root)/'trajectories'/item['object']/f'episode_{episode}').glob('*/scores.json'):
            r=read_json(path)['conditions'].get(condition['key'])
            if r is not None:values[path.parent.name]=r['objective_mm'] if r['valid'] else np.inf
        banks.append(values)
    return {key:np.array([b[key] for b in banks]) for key in set.intersection(*(set(b) for b in banks))}


def write_results(root,plan):
    rows=[read_json(p) for p in (root/'results').glob('*/*/*.json')]
    synthetic=[r for r in rows if r['condition']['truth']!='real']
    summary=[]
    for noise in [0.,.5,2.]:
        for method in plan['methods']:
            selected=[r for r in synthetic if r['method']==method and r['condition']['noise_mm']==noise]
            if not selected:continue
            per_object=[]
            for obj in sorted({r['object'] for r in selected}):
                group=[r for r in selected if r['object']==obj]
                good=[r for r in group if r['query_valid']]
                per_object.append(dict(object=obj,trials=len(group),query_valid=len(good),
                    parameter_error_steps=float(np.mean([r['parameter_error_steps'] for r in group])),
                    per_parameter_error_steps=np.mean([r['per_parameter_error_steps'] for r in group],0),
                    exact_fraction=float(np.mean([r['exact'] for r in group])),
                    query_objective_mm=float(np.mean([r['query_objective_mm'] for r in good])) if good else None))
            summary.append(dict(noise_mm=noise,method=method,objects=len(per_object),trials=len(selected),
                           parameter_error_steps=float(np.mean([r['parameter_error_steps'] for r in per_object])),
                           per_parameter_error_steps=np.mean([r['per_parameter_error_steps'] for r in per_object],0),
                           exact_fraction=float(np.mean([r['exact_fraction'] for r in per_object])),
                           query_objective_mm=float(np.mean([r['query_objective_mm'] for r in per_object if r['query_objective_mm'] is not None])) if any(r['query_objective_mm'] is not None for r in per_object) else None,
                           per_object=per_object))
    # Paired deltas use identical object/truth/noise/seed/start and never the best seed.
    lookup={(r['object'],r['context'],r['method']):r for r in synthetic};paired=[]
    for row in synthetic:
        if row['method']=='uniform':continue
        base=lookup.get((row['object'],row['context'],'uniform'))
        if base is None:continue
        paired.append(dict(object=row['object'],context=row['context'],method=row['method'],
                       parameter_error_steps_delta=row['parameter_error_steps']-base['parameter_error_steps'],
                       query_mm_delta=row['query_objective_mm']-base['query_objective_mm'] if row['query_valid'] and base['query_valid'] else None,
                       same_parameters=row['selected_q']==base['selected_q']))
    result=dict(updated_at=time.time(),completed_trials=len(rows),synthetic_summary=summary,
                real=[r for r in rows if r['condition']['truth']=='real'],paired=paired,
                aggregate='Each object has equal weight; repeated starts/noise seeds are not independent objects.',
                model_sha256=plan['model_sha256'])
    write_json(root/'report.json',result)
    return result


def supervise(root,gpus,score_workers=6):
    root=Path(root).resolve();plan=read_json(root/'plan.json');repo=Path(__file__).resolve().parents[1]
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=read_json(root/'jobs.json') if (root/'jobs.json').exists() else {};processes={}
    for name in ['logs','results','selections','weights']: (root/name).mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(repo),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def save():write_json(root/'jobs.json',jobs)
    def job_done(key):return jobs.get(key,{}).get('status') in ['exited','failed']
    def launch(key,kind,obj,episode=None,q=None,gpu=None):
        job=jobs.get(key,{})
        if job.get('status') in ['running','failed']:return False
        if job.get('attempt',0)>=2:jobs[key]['status']='failed';save();return False
        command=[sys.executable,'-u','-m','softdynamics.recovery_benchmark' if kind=='weights' else 'softdynamics.recovery_data',
                 '--root',str(root),'--mode',kind,'--object',obj]
        if episode is not None:command+=['--episode',str(episode),'--q',*map(str,q)]
        worker_env=env.copy()
        if gpu is not None:worker_env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=root/'logs'/(key.replace('/','_')+'.log')
        with log.open('a') as stream:
            p=subprocess.Popen(command,cwd=repo,env=worker_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[key]=dict(pid=p.pid,status='running',kind=kind,gpu=gpu,command=command,log=str(log),
                       started_at=time.time(),attempt=job.get('attempt',0)+1)
        processes[key]=p;save();print('START',key,'gpu',gpu,'pid',p.pid,flush=True);return True
    def refresh():
        for key,j in jobs.items():
            if j['status']!='running':continue
            if key in processes:
                rc=processes[key].poll()
                if rc is None:continue
            elif alive(j['pid'],str(root)):continue
            else:rc=None
            j.update(status='exited',exit_code=rc,finished_at=time.time());print('EXIT',key,rc,flush=True)
        save()
    def request_sim(requests,obj,e,q,priority):
        folder=trajectory_folder(root,obj,e,q)
        if not (folder/'complete.json').exists():
            key=f'simulate/{obj}/{e}/{qkey(q)}'
            if key not in requests or priority<requests[key][0]:requests[key]=(priority,obj,e,list(q))
    def request_score(requests,obj,e,q,priority):
        folder=trajectory_folder(root,obj,e,q)
        if (folder/'complete.json').exists() and not (folder/'scores.json').exists():
            key=f'score/{obj}/{e}/{qkey(q)}'
            if key not in requests or priority<requests[key][0]:requests[key]=(priority,obj,e,list(q))
    def resolved_scores(obj,e,q,condition):
        p=score_path(root,obj,e,q)
        return read_json(p)['conditions'][condition] if p.exists() else None
    while True:
        refresh();sim_requests={};score_requests={};weight_requests=[];object_status={};finished=0
        # Smallest ready cases first; every test object remains in the frozen plan.
        ordered=sorted(plan['objects'],key=lambda o:sum(e['particles']*e['frames'] for e in o['episodes']))
        for rank,item in enumerate(ordered):
            obj=item['object'];truth_ready=True;truth_invalid=[]
            for entry in item['episodes']:
                e=entry['episode']
                for truth in plan['config']['truths']:
                    q=truth['q'];request_sim(sim_requests,obj,e,q,(rank,0,0,e))
                    p=trajectory_folder(root,obj,e,q)/'complete.json'
                    if not p.exists():truth_ready=False
                    elif not read_json(p)['valid']:truth_invalid.append(dict(episode=e,truth=truth['name']))
            if not truth_ready:object_status[obj]=dict(stage='truth_rollouts');continue
            if truth_invalid:
                object_status[obj]=dict(stage='numerically_unavailable',invalid_truths=truth_invalid)
                write_json(root/'results'/f'{obj}_unavailable.json',object_status[obj]);finished+=1;continue
            wpath=root/'weights'/f'{obj}.json'
            if not wpath.exists():weight_requests.append(obj)
            # Score truths too: zero-noise truth must have zero loss.
            for entry in item['episodes']:
                for truth in plan['config']['truths']:
                    request_score(score_requests,obj,entry['episode'],truth['q'],(rank,0,0,entry['episode']))
            if not wpath.exists():object_status[obj]=dict(stage='weights');continue
            weights=read_json(wpath);done=0;total=len(weights['contexts'])*len(plan['methods'])
            for context_index,context in enumerate(weights['contexts']):
                cond=context['condition'];bank=bank_for(root,item,cond)
                priority=(0 if context_index==0 else 1 if cond['truth']=='real'
                          else 2 if cond['noise_mm']==0 else 3+context_index)
                for method in plan['methods']:
                    output=root/'results'/obj/context['id']/f'{method}.json'
                    if output.exists():done+=1;continue
                    selection=root/'selections'/obj/context['id']/f'{method}.json'
                    if selection.exists():chosen=read_json(selection)
                    else:
                        fit,trace,missing=frontier(bank,np.array(context['weights'][method]),context['start'],plan['passes'],plan['grid_levels'])
                        for q in missing:
                            for e in item['support']:
                                request_sim(sim_requests,obj,e,q,(rank,2,priority,e));request_score(score_requests,obj,e,q,(rank,2,priority,e))
                        if fit is None:continue
                        q=fit['q'];loss=bank[qkey(q)];initial=bank[qkey(context['start'])]
                        support_valid=bool(np.isfinite(loss).all());initial_valid=bool(np.isfinite(initial).all())
                        accepted=True
                        if cond['truth']=='real':
                            accepted=bool(support_valid and initial_valid and loss.mean()<initial.mean()-1e-12)
                        selected=q if accepted else np.array(context['start'])
                        chosen=dict(object=obj,context=context['id'],condition=cond,method=method,start=context['start'],
                                    candidate_q=q,selected_q=selected,accepted=accepted,support_valid=support_valid,
                                    support_candidate_mm=float(loss.mean()) if support_valid else None,
                                    support_initial_mm=float(initial.mean()) if initial_valid else None,
                                    logical_episode_evaluations=fit['logical_candidate_vectors']*len(item['support']),
                                    unique_candidate_vectors=fit['unique_candidate_vectors'],trace=trace,
                                    selection_frozen_at=time.time(),query_observations_used=False)
                        if cond['truth']!='real':
                            truth=next(t for t in plan['config']['truths'] if t['name']==cond['truth'])
                            errors=abs(np.array(selected)-truth['q'])/.025
                            chosen.update(truth_q=truth['q'],per_parameter_error_steps=errors,parameter_error_steps=float(errors.mean()),
                                          parameter_absolute_errors=errors*.025*np.array(plan['config']['coordinate_scales']),exact=bool(np.max(errors)<1e-7))
                        write_json(selection,chosen)
                    q=chosen['selected_q'];query_rows=[]
                    target=condition_key(cond['truth'],0.,19)
                    for e in item['query']:
                        request_sim(sim_requests,obj,e,q,(rank,1,priority,e));request_score(score_requests,obj,e,q,(rank,1,priority,e))
                        value=resolved_scores(obj,e,q,target)
                        if value is not None:query_rows.append(dict(episode=e,**value))
                    if len(query_rows)!=len(item['query']):continue
                    valid=all(r['valid'] for r in query_rows)
                    chosen.update(query=query_rows,query_valid=valid,
                                  query_objective_mm=float(np.mean([r['objective_mm'] for r in query_rows])) if valid else None,
                                  query_surface_mm=float(np.mean([r['surface_mm'] for r in query_rows])) if valid else None,
                                  query_track_mm=float(np.mean([r['track_mm'] for r in query_rows])) if valid else None)
                    write_json(output,chosen);done+=1
                    print('RESULT',obj,context['id'],method,'q',q,'query',chosen['query_objective_mm'],flush=True)
            object_status[obj]=dict(stage='complete' if done==total else 'recovery',completed=done,total=total)
            if done==total:finished+=1
        used={j['gpu'] for j in jobs.values() if j['status']=='running' and j['gpu'] is not None}
        hardware=gpu_state();available=[g for g in gpus if g not in used and hardware[g][0]<1800 and hardware[g][1]<40]
        disk=shutil.disk_usage(root).free;can_launch=disk>10*1024**3
        if can_launch:
            for key,(_,obj,e,q) in sorted(sim_requests.items(),key=lambda kv:kv[1][0]):
                if not available:break
                if jobs.get(key,{}).get('status')=='running':continue
                if launch(key,'simulate',obj,e,q,available[0]):available.pop(0)
            cpu_active=sum(j['status']=='running' and j['kind']!='simulate' for j in jobs.values())
            for obj in weight_requests:
                if cpu_active>=score_workers:break
                if launch('weights/'+obj,'weights',obj):cpu_active+=1
            for key,(_,obj,e,q) in sorted(score_requests.items(),key=lambda kv:kv[1][0]):
                if cpu_active>=score_workers:break
                if launch(key,'score',obj,e,q):cpu_active+=1
        report=write_results(root,plan)
        active=[dict(key=k,**j) for k,j in jobs.items() if j['status']=='running']
        failed=[k for k,j in jobs.items() if j['status']=='failed']
        status=dict(pid=os.getpid(),updated_at=time.time(),objects=object_status,completed_trials=report['completed_trials'],
                    active_workers=active,requested_simulations=len(sim_requests),requested_scoring=len(score_requests),
                    infrastructure_failures=failed,disk_free_gb=disk/1024**3,
                    status='complete' if finished==len(plan['objects']) else ('failed' if failed else 'running'))
        write_json(root/'status.json',status)
        print('STATUS',object_status,'active',len(active),flush=True)
        if status['status'] in ['complete','failed']:return
        time.sleep(15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['supervise','weights'],required=True);p.add_argument('--object')
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));p.add_argument('--score-workers',type=int,default=6)
    a=p.parse_args()
    if a.mode=='weights':compute_weights(a.root,a.object)
    else:supervise(a.root,a.gpus,a.score_workers)


if __name__=='__main__':main()
