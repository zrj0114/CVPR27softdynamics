"""Size-aware scheduling and restart bookkeeping; no changes to teacher physics."""
from collections import Counter
from pathlib import Path
import math,time
from .io import read_json,write_json,sha256

def workload(plan):
    episodes={};objects={}
    for episode in plan['episodes']:
        path=Path(episode['preparation'])
        if not path.exists():continue
        receipt=read_json(path);row=read_json(receipt['row'])
        particles=int(receipt['particles'])
        episodes[episode['id']]=dict(particles=particles,frames_estimate=int(row['frames'])+5)
        objects[episode['object']]=max(objects.get(episode['object'],0),particles)
    return dict(episodes=episodes,object_particles=objects)

def size_tier(particles):
    return 0 if particles<100_000 else 1 if particles<1_000_000 else 2

def task_cost(task,work):
    e=work['episodes'][f"{task['object']}/episode_{task['episode']}"]
    # Complete nominal rollout plus the fixed-state sensitivity branches.
    return e['particles']*(e['frames_estimate']+240)

def required_free_mib(task,work):
    particles=work['episodes'][f"{task['object']}/episode_{task['episode']}"]['particles']
    # 8 GiB covers context, the largest grid, and reserve; particle storage is
    # conservatively budgeted separately. One collector is allowed per GPU.
    return 8192+math.ceil(particles*2048/1024**2)

def choose_task(pending,running_objects,work,free_mib):
    if not pending:return None
    tier=min(size_tier(work['object_particles'][t['object']]) for t in pending)
    available=[t for t in pending if size_tier(work['object_particles'][t['object']])==tier
               and required_free_mib(t,work)<=free_mib]
    counts=Counter(running_objects)
    return min(available,key=lambda t:(counts[t['object']],task_cost(t,work),t['variant'],t['episode'],t['id'])) if available else None

def result_path(root,task,retries):
    root=Path(root);entry=retries.get(task['id'])
    path=(root/entry['folder'] if entry else root/'trajectories'/task['id'])/'complete.json'
    if entry and (not path.exists() or sha256(path)!=entry['complete_sha256']):
        raise ValueError('Registered retry receipt changed: '+task['id'])
    return path

def archive_incomplete(out):
    """Caller holds task.lock. Preserve interrupted work before a fresh rollout."""
    out=Path(out)
    if (out/'complete.json').exists():return None
    entries=[p for p in out.iterdir() if p.name not in ('task.lock','interrupted')]
    if not entries:return None
    destination=out/'interrupted'/str(time.time_ns());destination.mkdir(parents=True)
    names=[]
    for p in entries:names.append(p.name);p.replace(destination/p.name)
    write_json(destination/'interruption.json',dict(archived_at=time.time(),files=names,
        reason='Previous collection stopped before its final receipt; restart from the current prepared initial state.'))
    return destination
