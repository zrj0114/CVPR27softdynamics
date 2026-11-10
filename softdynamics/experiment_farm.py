"""Run dependency-ordered experiment jobs on currently unoccupied GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .io import read_json, write_json


def available_gpus(allowed,shared_memory_limit_mb=0):
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    occupied=set(applications.splitlines())
    result=[]
    for line in devices.splitlines():
        index,uuid,memory,utilization=[s.strip() for s in line.split(',')]
        if int(index) in allowed and ((uuid not in occupied and int(memory)<1000) or int(memory)<shared_memory_limit_mb):
            result.append((int(index),uuid,int(memory),int(utilization)))
    result.sort(key=lambda x:(x[2]>=1000,x[3],x[2],x[0]))
    return [(i,u) for i,u,_,_ in result]


def run(root):
    root=Path(root);active={};done={};started=time.time()
    with (root/'farm.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            plan=read_json(root/'jobs.json')
            for key,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close()
                done[key]=dict(returncode=code,ended_at=time.time(),gpu=item['gpu'])
                write_json(root/'job_results'/f'{key}.json',done[key]);del active[key]
            jobs={j['id']:j for j in plan['jobs']}
            for key,j in jobs.items():
                if key in active or key in done:continue
                receipt=root/'job_results'/f'{key}.json'
                if receipt.exists():done[key]=read_json(receipt)
            ready=[]
            for key,j in jobs.items():
                if key in done or key in active:continue
                if all(dep in done and done[dep]['returncode']==0 for dep in j.get('after',[])):
                    ready.append(j)
            used={v['gpu'] for v in active.values()}
            free=[(i,u) for i,u in available_gpus(plan['gpus'],plan.get('shared_memory_limit_mb',0)) if i not in used]
            free=free[:max(0,plan.get('max_active',len(plan['gpus']))-len(active))]
            for job,(index,uuid) in zip(ready,free):
                logpath=root/'logs'/f"{job['id']}.log";logpath.parent.mkdir(exist_ok=True)
                stream=logpath.open('a')
                command=[v.replace('{device}','cuda:0') for v in job['command']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                         PYTHONUNBUFFERED='1',PYTHONPATH=plan['code_root'])
                process=subprocess.Popen(command,cwd=plan['code_root'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[job['id']]=dict(process=process,log=stream,gpu=index,started_at=time.time())
                print('LAUNCH',job['id'],'gpu',index,'pid',process.pid,flush=True)
            blocked={key:[d for d in j.get('after',[]) if d in done and done[d]['returncode']!=0]
                for key,j in jobs.items() if key not in done and key not in active}
            blocked={k:v for k,v in blocked.items() if v}
            write_json(root/'farm_status.json',dict(pid=os.getpid(),started_at=started,updated_at=time.time(),
                active={k:dict(pid=v['process'].pid,gpu=v['gpu'],started_at=v['started_at']) for k,v in active.items()},
                completed=done,waiting=[j['id'] for j in ready if j['id'] not in active],blocked=blocked,
                jobs=len(jobs),allow_more_jobs=not plan.get('sealed',False)))
            if plan.get('sealed',False) and not active and all(k in done or k in blocked for k in jobs):
                write_json(root/'farm_complete.json',dict(completed=done,blocked=blocked,ended_at=time.time()));return
            time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
