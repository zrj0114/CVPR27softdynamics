"""Multi-GPU, resumable ours versus equal-weight parameter-block CMA-ES."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .cma_search import BlockCMA
from .io import read_json,sha256,write_json
from .parameters import ParameterSpace
from .recovery_benchmark import compute_weights
from .recovery_data import condition_key, qkey, trajectory_folder, simulate, score_candidate
from .shared_pipeline import alive,gpu_state


def foreign_compute_gpus():
    """Do not join other users' training during a transient idle instant."""
    device_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={r.split(',')[1].strip():int(r.split(',')[0]) for r in device_rows.strip().splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy=set()
    for row in apps.strip().splitlines():
        uuid,pid=[v.strip() for v in row.split(',')]
        try:foreign=Path('/proc',pid).stat().st_uid!=os.getuid()
        except FileNotFoundError:continue
        if foreign:busy.add(mapping[uuid])
    return busy


def snapshot_code(root, previous_plan):
    """Make the runtime named by the new plan real before importing any cache."""
    repo = Path(__file__).resolve().parents[1]
    previous_runtime = Path(previous_plan['runtime'])
    physics = ['run_episode.py', 'gs-layer/material_contact_kernels.py',
               'gs-layer/pad_contact_kernels.py', 'gs-layer/deterministic_transfer.py']
    physics += [str(p.relative_to(repo/'runtime')) for p in sorted((repo/'runtime/empm').rglob('*.py'))]
    comparisons = []
    for name in physics:
        previous, current = previous_runtime/name, repo/'runtime'/name
        if not previous.is_file() or sha256(previous) != sha256(current):
            raise ValueError(f'Physics differs from previous recovery; refuse cache reuse: {name}')
        comparisons.append(dict(file=name, sha256=sha256(current)))
    # The adapters are part of the simulation contract as well as the solver.
    for name in ('adapters/empm.py', 'adapters/pad_empm.py'):
        previous = previous_runtime.parent/'softdynamics'/name
        current = repo/'softdynamics'/name
        if not previous.is_file() or sha256(previous) != sha256(current):
            raise ValueError(f'Adapter differs from previous recovery: {name}')
        comparisons.append(dict(file='softdynamics/'+name, sha256=sha256(current)))
    code = root/'code'
    if code.exists():
        raise FileExistsError('Incomplete destination already has a code snapshot; use a new run directory')
    code.mkdir(parents=True)
    for name in ('softdynamics', 'runtime'):
        shutil.copytree(repo/name, code/name, ignore=shutil.ignore_patterns('__pycache__', '.warp-cache'))
    write_json(root/'code_manifest.json',dict(source_files={str(p.relative_to(code)):sha256(p)
               for p in sorted(code.rglob('*.py'))}))
    write_json(root/'physics_parity.json',dict(method='Byte-identical active solver and adapter sources',
               previous_runtime=str(previous_runtime), files=comparisons,
               trajectory_replay_performed=False))


def prepare(root, previous):
    """Freeze protocol before search; reuse only identical completed rollouts."""
    root, previous = Path(root).resolve(),Path(previous).resolve()
    if (root/'plan.json').exists():return read_json(root/'plan.json')
    root.mkdir(parents=True,exist_ok=True)
    plan = copy.deepcopy(read_json(previous/'plan.json'))
    snapshot_code(root, plan)
    plan.update(schema='softdynamics.block-cma-recovery.v1',created_at=time.time(),
        previous_run=str(previous),runtime=str(root/'code/runtime'), methods=['uniform','learned'],
        optimizer=dict(name='parameter-block CMA-ES',library='cma==4.5.0',dimensions_per_block=1,
            population=4,block_generations=2,cycles=3,sigma=.2,seed=9101,
            bounds_q=[-.025,.025],CMA_active=False,maxstd='disabled (pycma N=1 bug)',
            budgets=[32,64,96],early_stop=False),
        interpretation='Continuous local parameter-block search; no joint four-dimensional covariance or physics gradients. Four independent objectives are preserved.',
        fairness='Identical seeds, block order, population and forward budget. Model weights fixed at the same initial guess. Candidate rollouts are shared; cache hits and shared wall time are not method speedups.',
        weight_query='Frozen four per-parameter confidence columns; normalized within a column, never averaged across parameters. No teacher labels or query motion used.',
        selection_policy='At each block end retain its best valid weighted-loss candidate including the entering incumbent. Freeze held-out evaluation at budgets 32,64,96.',
        success='Synthetic max(abs(q-truth)/.025)<=0.1 (one tenth of the previous grid spacing), plus continuous absolute errors and held-out motion error.',
        context_selection='Predeclared real, center/mixed noise-free with both starts, and mixed 2mm seed19 with both starts, all four held-out objects. No contexts selected by prior improvement.',
        real_acceptance='Same strict unweighted support improvement over q=0; report raw and accepted choices; query never used for acceptance.')
    selected={'real','center_noise0_seed19','mixed_noise0_seed19','mixed_noise2_seed19'}
    plan['conditions']=[c for c in plan['conditions'] if c['key'] in selected]
    plan.pop('grid_levels',None);plan.pop('passes',None);plan.pop('logical_candidates_per_fit',None)
    write_json(root/'config.json',plan['config'])
    imported=[]
    for item in plan['objects']:
        for entry in item['episodes']:
            source=Path(entry['input']);dest=root/'inputs'/item['object']/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,dest,dirs_exist_ok=True);entry['input']=str(dest)
            base=previous/'trajectories'/item['object']/f'episode_{entry["episode"]}'
            for complete in base.glob('*/complete.json'):
                result=read_json(complete)
                if not result.get('valid') or result.get('smoke'):continue
                # Verify payload and the per-episode physics provenance before
                # linking immutable outputs. Never reuse incomplete candidates.
                for name,digest in result['payload_sha256'].items():
                    if sha256(complete.parent/(name+'.npy'))!=digest:raise ValueError('Old payload changed')
                for key,digest in [('initial',entry['initial_sha256']),('actions',entry['actions_sha256']),('asset',entry['asset_sha256'])]:
                    if result['source_hashes'][key]!=digest:raise ValueError('Old simulation source changed')
                if result['dtmax']!=entry['dtmax']:raise ValueError('Old time step changed')
                if result['domain'].get('margin_m')!=entry['domain_margin']:raise ValueError('Old domain changed')
                dest=trajectory_folder(root,item['object'],entry['episode'],result['q'])
                dest.mkdir(parents=True,exist_ok=True)
                # Scores depend on the new list of conditions: re-score locally.
                for name in ['complete.json','validation.json','tracks.npy','surface.npy']:
                    os.link(complete.parent/name,dest/name)
                imported.append(str(dest))
        old_weights=previous/'weights'/f'{item["object"]}.json'
        if old_weights.exists():
            weights=read_json(old_weights)
            if weights['model_sha256']!=plan['model_sha256']:raise ValueError('Old weights model changed')
            weights['contexts']=[c for c in weights['contexts'] if c['condition']['key'] in selected]
            for c in weights['contexts']:c['weights'].pop('shuffled',None)
            write_json(root/'weights'/old_weights.name,weights)
    if sha256(plan['checkpoint'])!=plan['model_sha256']:raise ValueError('Frozen model changed')
    write_json(root/'cache_import.json',dict(previous=str(previous),verified_complete_trajectories=imported,
        physical_payloads_hardlinked=True,scores_recomputed=True))
    write_json(root/'plan.json',plan)
    return plan


def summary(root, unavailable=None):
    rows=[read_json(p) for p in (root/'results').glob('*/*/*/*.json')]
    paired=[];lookup={(r['object'],r['context'],r['budget'],r['method']):r for r in rows}
    for row in rows:
        if row['method']!='learned':continue
        base=lookup.get((row['object'],row['context'],row['budget'],'uniform'))
        if base is None:continue
        paired.append(dict(object=row['object'],context=row['context'],budget=row['budget'],
            parameter_error_delta=row.get('parameter_error_steps',0)-base.get('parameter_error_steps',0)
                if row['condition']['truth']!='real' else None,
            query_mm_delta=row['query_objective_mm']-base['query_objective_mm']
                if row['query_valid'] and base['query_valid'] else None,
            uniform_query_valid=base['query_valid'],learned_query_valid=row['query_valid']))
    write_json(root/'report.json',dict(updated_at=time.time(),completed_budget_evaluations=len(rows),
        paired=paired,rows=rows,numerically_unavailable=unavailable or {},
        aggregation='Paired per context; object equal weighting required for aggregate claims.'))
    return len(rows)


def reference_status(root, item, truths):
    """Inspect all references before requesting work; retain numerical failures."""
    missing=[];invalid=[]
    for episode in item['support']+item['query']:
        for truth in truths:
            path=trajectory_folder(root,item['object'],episode,truth['q'])/'complete.json'
            if not path.exists():
                missing.append((episode,truth['q']))
                continue
            result=read_json(path)
            if not result['valid']:
                invalid.append(dict(episode=episode,truth=truth['name'],q=truth['q'],
                    complete_json=str(path),invalid_frame=result.get('invalid_frame'),
                    check=result.get('check')))
    return missing,invalid


def supervise(root,gpus,score_workers=6):
    root=Path(root).resolve()
    with (root/'pipeline.lock').open('a') as lock:
        # A rejected second supervisor must not overwrite the owner's status.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            _supervise(root,gpus,score_workers)
        except Exception as error:
            status=read_json(root/'status.json') if (root/'status.json').exists() else {}
            status.update(pid=os.getpid(),updated_at=time.time(),status='failed',
                supervisor_error=dict(type=type(error).__name__,message=str(error)))
            write_json(root/'status.json',status)
            raise


def _supervise(root,gpus,score_workers=6):
    root=Path(root).resolve();plan=read_json(root/'plan.json')
    repo=root/'code' if (root/'code/softdynamics/cma_benchmark.py').is_file() else Path(__file__).resolve().parents[1]
    jobs=read_json(root/'jobs.json') if (root/'jobs.json').exists() else {};processes={};states={}
    for name in ['logs','results','selections','weights','search']: (root/name).mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(repo),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def launch(key,kind,obj,episode=None,q=None,gpu=None):
        old=jobs.get(key,{})
        if old.get('status') in ['running','failed']:return False
        if old.get('attempt',0)>=2:
            old['status']='failed';jobs[key]=old;return False
        command=[sys.executable,'-u','-m','softdynamics.cma_benchmark','--root',str(root),'--mode',kind,'--object',obj]
        if episode is not None:command+=['--episode',str(episode),'--q',*map(str,q)]
        worker_env=env.copy()
        if gpu is not None:worker_env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=root/'logs'/(key.replace('/','_')+'.log')
        with log.open('a') as stream:
            p=subprocess.Popen(command,cwd=repo,env=worker_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[key]=dict(pid=p.pid,status='running',kind=kind,gpu=gpu,command=command,log=str(log),
            started_at=time.time(),attempt=old.get('attempt',0)+1)
        processes[key]=p;write_json(root/'jobs.json',jobs)
        print('START',key,'GPU',gpu,'PID',p.pid,flush=True);return True
    def refresh():
        for key,j in jobs.items():
            if j['status']!='running':continue
            if key in processes:
                rc=processes[key].poll()
                if rc is None:continue
            elif alive(j['pid'],str(root)):continue
            else:rc=None
            j.update(status='exited',exit_code=rc,finished_at=time.time())
            print('EXIT',key,rc,flush=True)
        write_json(root/'jobs.json',jobs)
    ordered=sorted(plan['objects'],key=lambda o:sum(e['particles']*e['frames'] for e in o['episodes']))
    while True:
        refresh();requests={};scoring={};weights_needed=[];progress={};total_done=0;unavailable={}
        def losses(item,q,condition,episodes,priority):
            result=[];ready=True
            for e in episodes:
                folder=trajectory_folder(root,item['object'],e,q);key=f'{item["object"]}/{e}/{qkey(q)}'
                if not (folder/'complete.json').exists():
                    requests[key]=(priority,item['object'],e,list(q));ready=False;continue
                if not (folder/'scores.json').exists():
                    scoring[key]=(priority,item['object'],e,list(q));ready=False;continue
                value=read_json(folder/'scores.json')['conditions'][condition]
                result.append(value['objective_mm'] if value['valid'] else np.inf)
            return np.array(result) if ready else None
        for rank,item in enumerate(ordered):
            obj=item['object'];wpath=root/'weights'/f'{obj}.json'
            if not wpath.exists():
                missing,invalid=reference_status(root,item,plan['config']['truths'])
                if invalid:
                    unavailable[obj]=dict(reason='invalid_reference_under_frozen_protocol',references=invalid)
                    progress[obj]=dict(stage='numerically_unavailable',**unavailable[obj])
                    continue
                for e,q in missing:
                    requests[f'{obj}/{e}/{qkey(q)}']=((rank,0,0),obj,e,q)
                if not missing:weights_needed.append(obj)
                progress[obj]=dict(stage='weights_or_truth');continue
            contexts=read_json(wpath)['contexts']
            contexts.sort(key=lambda c:(c['condition']['truth']!='real',c['condition']['noise_mm'],c['id']))
            done=0
            for context_no,ctx in enumerate(contexts):
                cond=ctx['condition'];condition=cond['key']
                for method in plan['methods']:
                    checkpoint=root/'search'/obj/ctx['id']/(method+'.pkl');sid=str(checkpoint)
                    if sid not in states:
                        opt=plan['optimizer']
                        states[sid]=BlockCMA.load(checkpoint) if checkpoint.exists() else BlockCMA(
                            ctx['start'],ctx['weights'][method],opt['seed'],opt['cycles'],opt['block_generations'],opt['population'],opt['sigma'])
                    state=states[sid]
                    # Evaluate every complete sweep before advancing to the next.
                    milestone=state.logical_candidates>0 and state.logical_candidates%32==0
                    budget=state.logical_candidates
                    output=root/'results'/obj/ctx['id']/str(budget)/(method+'.json')
                    if milestone and not output.exists():
                        selection=root/'selections'/obj/ctx['id']/str(budget)/(method+'.json')
                        if not selection.exists():
                            current=losses(item,state.current,condition,item['support'],(rank,0,context_no))
                            initial=losses(item,ctx['start'],condition,item['support'],(rank,0,context_no))
                            if current is None or initial is None:continue
                            valid=bool(np.isfinite(current).all());initial_valid=bool(np.isfinite(initial).all())
                            accepted=valid if cond['truth']!='real' else bool(valid and initial_valid and current.mean()<initial.mean()-1e-12)
                            selected=state.current if accepted else np.array(ctx['start'])
                            chosen=dict(object=obj,context=ctx['id'],method=method,condition=cond,budget=budget,
                                selected_q=selected,candidate_q=state.current,start=ctx['start'],accepted=accepted,
                                support_initial_mm=float(initial.mean()) if initial_valid else None,
                                support_candidate_mm=float(current.mean()) if valid else None,
                                trace=state.trace,unique_candidate_vectors=len(state.seen),
                                logical_episode_evaluations=budget*len(item['support']),
                                episode_evaluations_including_start=(budget+1)*len(item['support']),
                                selection_frozen_at=time.time(),query_observations_used=False)
                            if cond['truth']!='real':
                                truth=next(t['q'] for t in plan['config']['truths'] if t['name']==cond['truth'])
                                error=abs(selected-np.array(truth))/.025
                                chosen.update(truth_q=truth,per_parameter_error_steps=error,
                                    parameter_error_steps=float(error.mean()),success=bool(error.max()<=.1),
                                    parameter_absolute_errors=error*.025*np.array(plan['config']['coordinate_scales']))
                            write_json(selection,chosen)
                        chosen=read_json(selection)
                        # Ensure noise-free truths exist on the held-out episode.
                        if cond['truth']!='real':
                            truth=next(t['q'] for t in plan['config']['truths'] if t['name']==cond['truth'])
                            for e in item['query']:
                                p=trajectory_folder(root,obj,e,truth)/'complete.json'
                                if not p.exists():requests[f'{obj}/{e}/{qkey(truth)}']=((rank,-1,context_no),obj,e,truth)
                            if any(not (trajectory_folder(root,obj,e,truth)/'complete.json').exists() for e in item['query']):continue
                        query=losses(item,chosen['selected_q'],condition_key(cond['truth']),item['query'],(rank,-1,context_no))
                        if query is None:continue
                        valid=bool(np.isfinite(query).all())
                        chosen.update(query_valid=valid,query_objective_mm=float(query.mean()) if valid else None)
                        write_json(output,chosen)
                        print('RESULT',obj,ctx['id'],method,budget,chosen['query_objective_mm'],flush=True)
                    if state.done:done+=1;continue
                    p=state.ask();state.save(checkpoint)
                    priority=(rank,1+budget//32,context_no)
                    rows=[losses(item,q,condition,item['support'],priority) for q in p['q']]
                    incumbent=losses(item,p['incumbent'],condition,item['support'],priority)
                    if incumbent is None or any(row is None for row in rows):continue
                    state.tell(np.array(rows),incumbent);state.save(checkpoint)
                    write_json(checkpoint.with_suffix('.json'),dict(block=state.block,generation=state.generation,
                        candidate_vectors=state.logical_candidates,current_q=state.current,trace=state.trace))
                    print('TELL',obj,ctx['id'],method,state.logical_candidates,state.current,flush=True)
            progress[obj]=dict(stage='complete' if done==len(contexts)*2 else 'search',completed=done,total=len(contexts)*2,
                searches=[dict(context=c['id'],method=m,candidates=states[str(root/'search'/obj/c['id']/(m+'.pkl'))].logical_candidates)
                    for c in contexts for m in plan['methods']])
            total_done+=done
        used={j['gpu'] for j in jobs.values() if j['status']=='running' and j['gpu'] is not None}
        hardware=gpu_state()
        # One worker per GPU. Idle service contexts with ample free memory can
        # coexist; busy foreign training jobs are left alone.
        foreign=foreign_compute_gpus()
        available=[g for g in gpus if g not in used and g not in foreign and hardware[g][0]<20000 and hardware[g][1]<35]
        disk=shutil.disk_usage(root).free
        if disk>10*1024**3:
            for key,(_,obj,e,q) in sorted(requests.items(),key=lambda kv:kv[1][0]):
                if not available:break
                if launch('simulate/'+key,'simulate',obj,e,q,available[0]):available.pop(0)
            cpu=sum(j['status']=='running' and j['kind']!='simulate' for j in jobs.values())
            for obj in weights_needed:
                if cpu>=score_workers:break
                if launch('weights/'+obj,'weights',obj):cpu+=1
            for key,(_,obj,e,q) in sorted(scoring.items(),key=lambda kv:kv[1][0]):
                if cpu>=score_workers:break
                if launch('score/'+key,'score',obj,e,q):cpu+=1
        completed=summary(root,unavailable);failed=[k for k,j in jobs.items() if j['status']=='failed']
        planned_contexts=sum(1 if c['truth']=='real' else len(plan['config']['initial_guesses']) for c in plan['conditions'])
        expected_by_object={o['object']:len(read_json(root/'weights'/f'{o["object"]}.json')['contexts'])*len(plan['methods'])
                     if (root/'weights'/f'{o["object"]}.json').exists() else planned_contexts*len(plan['methods']) for o in ordered}
        expected=sum(expected_by_object.values());unavailable_searches=sum(expected_by_object[o] for o in unavailable)
        active_workers=[dict(key=k,**j) for k,j in jobs.items() if j['status']=='running']
        waiting=[]
        if disk<=10*1024**3:waiting.append('disk_free_below_10_GiB')
        if requests and not available:waiting.append('no_available_gpu')
        phase='running'
        if failed:phase='failed'
        elif total_done==expected-unavailable_searches:phase='complete_with_unavailable' if unavailable else 'complete'
        elif not active_workers and waiting:phase='waiting_resources'
        status=dict(pid=os.getpid(),updated_at=time.time(),status=phase,
            objects=progress,completed_searches=total_done,total_searches=expected,completed_budget_evaluations=completed,
            unavailable_searches=unavailable_searches,eligible_searches=expected-unavailable_searches,
            numerically_unavailable=unavailable,waiting_reasons=waiting,active_workers=active_workers,
            requested_simulations=len(requests),requested_scoring=len(scoring),
            infrastructure_failures=failed,disk_free_gb=disk/1024**3,gpus=gpus,foreign_compute_gpus=sorted(foreign))
        write_json(root/'status.json',status);print('STATUS',total_done,'searches',completed,'evaluations',flush=True)
        if status['status'] in ['complete','complete_with_unavailable','failed']:return
        time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path)
    p.add_argument('--mode',choices=['prepare','supervise','simulate','score','weights'],required=True)
    p.add_argument('--object');p.add_argument('--episode',type=int);p.add_argument('--q',type=float,nargs=4)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));p.add_argument('--score-workers',type=int,default=6)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.previous)
    elif a.mode=='supervise':supervise(a.root,a.gpus,a.score_workers)
    elif a.mode=='simulate':simulate(a.root,a.object,a.episode,a.q)
    elif a.mode=='score':score_candidate(a.root,a.object,a.episode,a.q)
    else:compute_weights(a.root,a.object)


if __name__=='__main__':main()
