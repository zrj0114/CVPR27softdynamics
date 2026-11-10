"""Count executed trajectories once, including retired numerical failures."""
import fcntl
from pathlib import Path

from .io import read_json


def executions(run):
    run=Path(run)
    with (run/'physics/plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        return _executions(run)


def _executions(run):
    run=Path(run);seen=set();rows=[]
    # Main cache takes precedence over the diagnostic copy adopted as baseline.
    roots=[run/'physics/trajectories',run/'physics/retired',run/'diagnostics/domain_187_episode0']
    for root in roots:
        for path in sorted(root.rglob('elastic_receipt.json')):
            receipt=read_json(path);key=(receipt['execution_pid'],receipt['complete_sha256'])
            if key in seen:continue
            execution=read_json(path.parent/'execution.json');record=read_json(path.parent/'complete.json')
            seen.add(key)
            rows.append(dict(folder=str(path.parent),requester=receipt.get('requester') or '',
                pid=receipt['execution_pid'],seconds=execution['elapsed_s'],valid=record['valid'],
                retired=root==run/'physics/retired'))
    return rows
