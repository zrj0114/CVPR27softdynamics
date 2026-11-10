"""Parallel fresh-data collection followed by distributed joint training."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read_json, write_json


class AdoptedProcess:
    """Observe a surviving worker after a controller-only restart."""
    def __init__(self,pid,root):
        self.pid=pid;self.root=str(root);self.returncode=None
    def poll(self):
        try:
            cmd=Path(f'/proc/{self.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            stat=Path(f'/proc/{self.pid}/stat').read_text().split()
            if self.root in cmd and stat[2]!='Z':return None
        except FileNotFoundError:pass
        self.returncode=0
        return self.returncode


def gpu_capacity():
    result=[]
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
    for line in output.splitlines():
        index,uuid,free,util=[v.strip() for v in line.split(',')]
        result.append(dict(index=int(index),uuid=uuid,free=int(free),util=int(util)))
    return sorted(result,key=lambda x:(x['util'], -x['free']))


def run(root,code_root,steps=40000,max_workers=7,training_gpus=4):
    root=Path(root).resolve();code_root=Path(code_root).resolve();plan=read_json(root/'plan.json')
    logs=root/'logs';logs.mkdir(exist_ok=True)
    active={};training=None;reserved=set();started=time.time()
    with (root/'farm.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'status.json').exists():
            previous=read_json(root/'status.json')
            for identity,item in previous.get('active',{}).items():
                process=AdoptedProcess(item['pid'],root)
                if process.poll() is None:
                    stream=(logs/(identity.replace('/','__')+'.log')).open('a')
                    active[identity]=dict(process=process,log=stream,gpu=item['gpu'])
            old=previous.get('training')
            if old:
                process=AdoptedProcess(old['pid'],root)
                if process.poll() is None:
                    training=dict(process=process,log=(logs/'training.log').open('a'),gpus=old['gpus'])
                    reserved=set(old['gpus'])
        while True:
            for identity,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                if isinstance(item['process'],AdoptedProcess):
                    receipt=root/'trajectories'/identity/'complete.json'
                    code=0 if receipt.exists() and read_json(receipt)['valid'] else 2
                item['log'].close();del active[identity]
                write_json(root/'worker_results'/(identity.replace('/','__')+'.json'),
                    dict(returncode=code,gpu=item['gpu'],finished_at=time.time()))
            done={};counts=Counter();confidence_counts=Counter();label_counts=Counter();failed=[]
            for task in plan['tasks']:
                receipt=root/'trajectories'/task['id']/'complete.json'
                if not receipt.exists():continue
                record=read_json(receipt);done[task['id']]=record['valid']
                if record['valid']:
                    counts[task['object']]+=1
                    manifest=read_json(record['teacher_manifest'])
                    if any(any(row['valid']) for row in manifest['records']):confidence_counts[task['object']]+=1
                    for row in manifest['records']:
                        for j,valid in enumerate(row['valid']):
                            if valid:label_counts[f"{task['split']}:{j}"]+=1
                else:failed.append(dict(id=task['id'],error=record.get('error')))
            # Start only with every training/validation object represented, and
            # at least two independent parameter rollouts per object.
            minimum=(all(counts[o['object']]>=2 and confidence_counts[o['object']]>=2 for o in plan['objects'])
                     and all(label_counts[f'{split}:{j}']>=8 for split in ('train','validation') for j in range(2)))
            if minimum and training is None and not (root/'training/complete.json').exists() and not (root/'disable_training').exists():
                busy={v['gpu'] for v in active.values()}
                candidates=[g for g in gpu_capacity() if g['free']>=22000 and
                            (g['util']<95 or g['index'] in busy)]
                if len(reserved)<training_gpus:
                    additions=[g['index'] for g in candidates if g['index'] not in reserved]
                    reserved.update(additions[:training_gpus-len(reserved)])
                # Training peaks below 2 GiB in preflight. With >=22 GiB free,
                # temporarily share our collectors' cards, then let them drain.
                # Existing collector processes are never killed for training.
                free=[g for g in candidates if g['index'] in reserved and
                      (g['index'] in busy or g['util']<95)]
                if len(free)==len(reserved) and len(free)>=min(2,training_gpus):
                    # Select a fixed number of ranks for the run, up to four.
                    free=free[:training_gpus];reserved={g['index'] for g in free}
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in free),
                        PYTHONPATH=str(code_root),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                        PYTHONUNBUFFERED='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
                    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
                        f'--nproc_per_node={len(free)}','-m','softdynamics.elastic_train',
                        '--root',str(root),'--steps',str(steps)]
                    if (root/'training/latest.pt').exists():command+=['--resume']
                    if not (root/'training_data_start.json').exists():
                        write_json(root/'training_data_start.json',dict(
                            ids=sorted(k for k,v in done.items() if v),created_at=time.time()))
                    stream=(logs/'training.log').open('a')
                    process=subprocess.Popen(command,cwd=code_root,env=env,stdout=stream,
                        stderr=subprocess.STDOUT,start_new_session=True)
                    training=dict(process=process,log=stream,gpus=sorted(reserved),command=command)
                    write_json(root/'training_launch.json',dict(pid=process.pid,command=command,
                        gpus=sorted(reserved),started_at=time.time(),steps=steps))
                    print('TRAINING_LAUNCH',process.pid,sorted(reserved),flush=True)
            if training is not None and training['process'].poll() is not None:
                code=training['process'].returncode;training['log'].close()
                if isinstance(training['process'],AdoptedProcess):
                    code=0 if (root/'training/complete.json').exists() else 1
                write_json(root/'training_exit.json',dict(returncode=code,ended_at=time.time()))
                if code!=0:
                    print('TRAINING_FAILED',code,flush=True)
                    # Preserve completed work; require diagnosis before restarting.
                    write_json(root/'attention_required.json',dict(reason='training_failed',returncode=code))
                    training=None;reserved=set();minimum=False
                    # Keep collectors running but prevent automatic crash loops.
                    (root/'disable_training').touch()
                else:
                    print('TRAINING_COMPLETE',flush=True)
                    while active:
                        for identity,item in list(active.items()):
                            if item['process'].poll() is not None:
                                item['log'].close();del active[identity]
                        if active:time.sleep(10)
                    write_json(root/'complete.json',dict(steps=steps,finished_at=time.time()));return
            if (root/'disable_training').exists():minimum=False
            busy={v['gpu'] for v in active.values()}|reserved
            free=[g for g in gpu_capacity() if g['index'] not in busy and g['free']>=22000 and g['util']<95]
            pending=[t for t in plan['tasks'] if t['id'] not in done and t['id'] not in active
                     and not (root/'worker_results'/(t['id'].replace('/','__')+'.json')).exists()]
            for task,gpu in zip(pending,free[:max(0,max_workers-len(active))]):
                identity=task['id'];stream=(logs/(identity.replace('/','__')+'.log')).open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONPATH=str(code_root),
                         OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                command=[sys.executable,'-u','-m','softdynamics.elastic_data','collect',
                         '--root',str(root),'--task',identity]
                process=subprocess.Popen(command,cwd=code_root,env=env,stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
                active[identity]=dict(process=process,log=stream,gpu=gpu['index'])
                print('COLLECT_LAUNCH',identity,gpu['index'],process.pid,flush=True)
            status=dict(pid=os.getpid(),started_at=started,updated_at=time.time(),total_steps=steps,
                tasks=len(plan['tasks']),valid=sum(done.values()),failed=failed,
                coverage=dict(counts),confidence_coverage=dict(confidence_counts),reliable_labels=dict(label_counts),
                active={k:dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in active.items()},
                training=(dict(pid=training['process'].pid,gpus=training['gpus']) if training else None),
                training_ready=minimum,reserved_gpus=sorted(reserved),
                checkpoint_policy=['best.pt','latest.pt'])
            write_json(root/'status.json',status)
            if not active and not pending and training is None:
                write_json(root/'attention_required.json',dict(reason='Insufficient valid data for training',coverage=dict(counts)))
                return
            time.sleep(15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--max-workers',type=int,default=7)
    p.add_argument('--training-gpus',type=int,default=4)
    a=p.parse_args();run(a.root,Path(__file__).resolve().parents[1],a.steps,a.max_workers,a.training_gpus)


if __name__=='__main__':main()
