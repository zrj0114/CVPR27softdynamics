"""Versioned episode-domain changes with explicit cache compatibility checks."""
import fcntl
from pathlib import Path

from .io import read_json,sha256
from .recovery_data import find_entry


def read_protocol(run):
    run=Path(run)
    with (run/'physics/plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        spec=read_json(run/'protocol.json');plan=read_json(run/'physics/plan.json')
        if 'physics_plan_sha256' in spec and sha256(run/'physics/plan.json')!=spec['physics_plan_sha256']:
            raise ValueError('Physics plan and protocol are inconsistent')
        return spec,plan


def cache_compatible(run,digest,obj,episode,spec=None,plan=None):
    run=Path(run)
    if spec is None or plan is None:spec,plan=read_protocol(run)
    if digest==spec['physics_plan_sha256']:return True
    archive=spec.get('compatible_physics_plans',{}).get(digest)
    if archive is None:return False
    path=Path(archive)
    if sha256(path)!=digest:raise ValueError('Archived physics protocol changed')
    previous=read_json(path)
    for key in ('config','runtime','conditions','data_plan_sha256','source_plan_sha256'):
        if previous.get(key)!=plan.get(key):return False
    old_item=next((o for o in previous['objects'] if o['object']==obj),None)
    item=next((o for o in plan['objects'] if o['object']==obj),None)
    if old_item is None or item is None:return False
    if any(old_item.get(key)!=item.get(key) for key in ('support','query')):return False
    return find_entry(previous,obj,episode)==find_entry(plan,obj,episode)
