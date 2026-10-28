"""Resumable local scheduling: preparation -> offline labels -> model training.

Run this module detached. Its durable job records allow a later invocation to
adopt surviving workers. It never signals another process or edits source data.
"""
import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .io import read_json, sha256, write_json


def alive(pid, expected):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return expected in cmd
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return False


def gpu_state():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(r.split(',')[0]):tuple(map(int,r.split(',')[1:])) for r in text.strip().splitlines()}


def shareable_own_gpus():
    """Allow overlap with this user's older EMPM jobs while they finish."""
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={r.split(',')[1].strip():int(r.split(',')[0]) for r in devices.strip().splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    owned={};seen=set()
    for line in apps.strip().splitlines():
        uuid,pid,memory=[v.strip() for v in line.split(',')]
        if not memory.isdigit() or int(memory)<1024:continue
        gpu=mapping[uuid];seen.add(gpu)
        try:
            path=Path('/proc')/pid
            cmd=(path/'cmdline').read_bytes().decode(errors='replace')
            ok=path.stat().st_uid==os.getuid() and ('empm_deterministic_transfer_20261013' in cmd or 'softdynamics.shared_teacher' in cmd)
        except (FileNotFoundError,PermissionError):ok=False
        owned[gpu]=owned.get(gpu,True) and ok
    return {g for g in seen if owned[g]}


def domain_failure(failure):
    if failure.get('classification')=='domain_boundary':return True
    message=failure.get('error','')
    margin=re.search(r"domain_margin_m': ([0-9.eE+-]+)",message)
    jac=re.search(r"min_J': ([0-9.eE+-]+)",message)
    return bool(margin and jac and "'finite': True" in message and
                float(jac.group(1))>0 and float(margin.group(1))<=2.01*.003125)


def run(root, gpus, prep_workers=4):
    root=Path(root).resolve();repo=Path(__file__).resolve().parents[1]
    lock=(root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_path=root/'config.json';config=read_json(config_path)
    selection=read_json(root/'selection.json')
    selected=[r for r in selection['rows'] if r['candidate']]
    objects=sorted({r['object'] for r in selected},key=lambda o:(o not in config['dataset']['pilot_objects'],o))
    pilot_ids=set()
    for obj in config['dataset']['pilot_objects']:
        candidates=sorted([r for r in selected if r['object']==obj],key=lambda r:r['episode'])
        pilot_ids.update(r['id'] for r in candidates[:config['dataset']['pilot_episodes_per_object']])
    jobs_path=root/'jobs.json'
    jobs=read_json(jobs_path) if jobs_path.exists() else {}
    processes={}
    base_env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    base_env['PYTHONPATH']=str(repo)
    teacher_root=root/'labels_v1'
    training_root=root/'training'
    (root/'logs').mkdir(exist_ok=True);training_root.mkdir(exist_ok=True)
    write_json(root/'plan.json',dict(candidate_episodes=len(selected),teacher_tasks=len(selected)*len(config['truths']),
                objects=objects,pilot_episode_ids=sorted(pilot_ids),gpus=gpus,prep_workers=prep_workers,
                config_sha256=sha256(config_path),source_code=str(repo),
                stages=['prepare','pilot_offline_labels','pilot_train','remaining_offline_labels','full_train']))

    def save():
        write_json(jobs_path,jobs)

    def launch(key,module,args,kind,gpu=None):
        old=jobs.get(key,{})
        log=root/'logs'/(key.replace('/','_')+'.log')
        env=base_env.copy()
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        command=[sys.executable,'-u','-m',module,*map(str,args)]
        with log.open('a') as stream:
            process=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[key]=dict(pid=process.pid,status='running',kind=kind,gpu=gpu,module=module,
                       command=command,log=str(log),started_at=time.time(),attempt=old.get('attempt',0)+1)
        processes[key]=process
        save()
        print('START',key,'gpu',gpu,'pid',process.pid,flush=True)

    def refresh():
        for key,job in jobs.items():
            if job['status']!='running':continue
            if key in processes:
                rc=processes[key].poll()
                if rc is None:continue
            elif alive(job['pid'],job['module']):continue
            else:rc=None
            job.update(status='exited',exit_code=rc,finished_at=time.time())
            print('EXIT',key,rc,flush=True)
        save()

    def terminal_teacher(row,truth):
        if not row['candidate']:return 'preparation_rejected'
        prep=root/'objects'/row['object']/'prepare.json'
        complete=root/'objects'/row['object']/'prepared'/f'episode_{row["episode"]}'/'complete.json'
        if not complete.exists():
            if jobs.get('prepare/'+row['object'],{}).get('status')=='running':
                return 'waiting_preparation'
            if prep.exists() and read_json(prep)['status'] in ['complete','asset_failed']:
                return 'preparation_rejected'
            return 'waiting_preparation'
        out=teacher_root/row['object']/f'episode_{row["episode"]}'/truth
        if (out/'manifest.json').exists():return 'complete'
        key='teacher/'+row['id']+'/'+truth
        job=jobs.get(key,{})
        if job.get('status')=='running':return 'running'
        failure=out/'failure.json'
        if failure.exists():
            if domain_failure(read_json(failure)) and job.get('attempt',1)<3:
                return 'pending'
            return 'failed'
        if job.get('status')=='failed':return 'failed'
        return 'pending'

    def completed_teachers(pilot=False):
        paths=[]
        for row in selected:
            if pilot and row['id'] not in pilot_ids:continue
            for truth in config['truths']:
                out=teacher_root/row['object']/f'episode_{row["episode"]}'/truth['name']/'manifest.json'
                if out.exists():paths.append(str(out))
        return paths

    def start_training(name,available):
        key='train/'+name
        output=training_root/name
        if (output/'report.json').exists():return
        job=jobs.get(key,{})
        if job.get('status') in ['running','failed']:return
        if job.get('status')=='exited':
            jobs[key]['status']='failed';save();return
        if not available:return
        paths=completed_teachers(name=='pilot')
        manifest=root/(name+'_training_manifest.json')
        write_json(manifest,dict(teachers=paths,selection_sha256=sha256(root/'selection.json'),
                   offline_labels_complete_before_training=True,config_sha256=sha256(config_path)))
        args=['--manifest',manifest,'--config',config_path,'--output',output]
        if name=='pilot':args.append('--pilot')
        launch(key,'softdynamics.shared_training',args,'train',available.pop(0))

    while True:
        refresh()
        disk=shutil.disk_usage(root).free
        # Never remove other experiments to make room; completed small outputs remain intact.
        can_launch=disk>5*1024**3
        prep_running=sum(j['kind']=='prepare' and j['status']=='running' for j in jobs.values())
        for obj in objects:
            if prep_running>=prep_workers or not can_launch:break
            path=root/'objects'/obj/'prepare.json';key='prepare/'+obj
            if path.exists() and read_json(path)['status']=='complete':continue
            job=jobs.get(key,{})
            if job.get('status') in ['running','failed']:continue
            if job.get('status')=='exited':
                jobs[key]['status']='failed';save();continue
            launch(key,'softdynamics.shared_data',['--selection',root/'selection.json','--object',obj,
                   '--output',root,'--runtime',repo/'runtime'],'prepare')
            prep_running+=1
        states=[]
        for row in selected:
            for truth in config['truths']:
                states.append((row,truth['name'],terminal_teacher(row,truth['name'])))
        terminal={'complete','failed','preparation_rejected'}
        pilot_finished=all(s in terminal for r,t,s in states if r['id'] in pilot_ids)
        full_finished=all(s in terminal for r,t,s in states)
        used={j['gpu'] for j in jobs.values() if j['status']=='running' and j['gpu'] is not None}
        hardware=gpu_state()
        own=shareable_own_gpus()
        available=[g for g in gpus if g not in used and
                   ((hardware[g][0]<1800 and hardware[g][1]<35) or (g in own and hardware[g][0]<6000))] if can_launch else []
        if pilot_finished:start_training('pilot',available)
        if full_finished:start_training('full',available)
        # Freeze each source episode's pilot membership before labels are produced.
        pending=sorted([(r,t) for r,t,s in states if s=='pending'],key=lambda v:(v[0]['id'] not in pilot_ids,
                       v[0]['episode'],v[1]!='center',v[0]['object']))
        for row,truth in pending:
            if not available:break
            key='teacher/'+row['id']+'/'+truth
            output=teacher_root/row['object']/f'episode_{row["episode"]}'/truth
            job=jobs.get(key,{})
            if job.get('status') in ['exited','failed']:
                boundary=(output/'failure.json').exists() and domain_failure(read_json(output/'failure.json'))
                if job.get('attempt',0)>=(3 if boundary else 2):
                    jobs[key]['status']='failed';save();continue
            if output.exists():
                output.rename(output.with_name(output.name+f'.interrupted_{time.time_ns()}'))
            margin=[.075,.15,.25][min(job.get('attempt',0),2)]
            launch(key,'softdynamics.shared_teacher',['--row',root/'objects'/row['object']/'prepared'/f'episode_{row["episode"]}'/'complete.json',
                   '--config',config_path,'--runtime',repo/'runtime','--truth',truth,'--output',output,
                   '--domain-margin',margin],'teacher',available.pop(0))
        counts={s:sum(v==s for r,t,v in states) for s in ['waiting_preparation','pending','running','complete','failed','preparation_rejected']}
        model_status={name:('complete' if (training_root/name/'report.json').exists() else jobs.get('train/'+name,{}).get('status','waiting')) for name in ['pilot','full']}
        status=dict(pid=os.getpid(),updated_at=time.time(),stage='offline_labels' if not full_finished else 'training',
                    candidates=len(selected),label_tasks=counts,models=model_status,
                    active_workers=[dict(key=k,pid=j['pid'],kind=j['kind'],gpu=j['gpu']) for k,j in jobs.items() if j['status']=='running'],
                    disk_free_gb=disk/1024**3,launch_allowed=can_launch)
        if model_status['full']=='complete':status.update(stage='complete',status='complete')
        elif full_finished and model_status['full']=='failed':status.update(stage='training_failed',status='failed')
        else:status['status']='running'
        write_json(root/'status.json',status)
        write_json(root/'label_status.json',[dict(id=r['id'],truth=t,status=s) for r,t,s in states])
        print('STATUS',counts,model_status,flush=True)
        if status['status'] in ['complete','failed']:break
        time.sleep(20)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[1,3,4,2,5,6,7])
    p.add_argument('--prep-workers',type=int,default=4)
    a=p.parse_args();run(a.root,a.gpus,a.prep_workers)


if __name__=='__main__':
    main()
