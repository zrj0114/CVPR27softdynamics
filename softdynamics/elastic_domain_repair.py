"""Publish a fully validated support-domain repair without changing material.

The configured probe order, never the fitting score, determines the choice.
Historical failures and the original protocol remain available for auditing.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import time

import numpy as np

from .io import read_json,write_json,sha256,load_npz
from .parameters import ParameterSpace
from .physics_protocol import read_protocol,cache_compatible
from .recovery_data import find_entry


def validate_probe(run,root,obj,episode,failed):
    run,root=Path(run),Path(root);spec,plan=read_protocol(run);probe_spec,probe=read_protocol(root)
    entry=find_entry(plan,obj,episode);new_entry=find_entry(probe,obj,episode)
    if probe['config']!=plan['config'] or probe['runtime']!=plan['runtime']:
        raise ValueError('A domain repair cannot change material, fitting, or solver settings')
    numerical={'grid_padding_cells','grid_lower_cells'}
    if {k:v for k,v in entry.items() if k not in numerical}!={k:v for k,v in new_entry.items() if k not in numerical}:
        raise ValueError('A domain repair changed non-domain episode inputs')
    paths=list((root/'physics/trajectories'/obj/f'episode_{episode}').glob('*/complete.json'))
    if len(paths)!=1:raise ValueError('Expected exactly one completed nominal probe')
    folder=paths[0].parent;record=read_json(paths[0])
    if not record['valid']:return None
    if record.get('smoke',True):raise ValueError('Partial smoke rollout cannot replace the baseline')
    if not np.array_equal(record['q'],np.zeros(4)):raise ValueError('Probe changed the nominal material coordinates')
    expected=ParameterSpace.from_config(plan['config']).parameters(np.zeros(4))
    if record['parameters']!=expected:raise ValueError('Probe changed a physical parameter')
    if record['source_hashes']!=failed['source_hashes']:raise ValueError('Probe changed the source physics or episode')
    if record['dtmax']!=entry['dtmax'] or record['domain']['dx_m']!=.003125:
        raise ValueError('Probe changed time or spatial discretization')
    if record['frames']!=entry['frames']:raise ValueError('Incomplete observed trajectory')
    for name,digest in record['payload_sha256'].items():
        if sha256(folder/(name+'.npy'))!=digest:raise ValueError('Probe trajectory payload changed')
    state_record=read_json(folder/'state.json')
    if sha256(folder/'state.npz')!=state_record['state_sha256']:raise ValueError('Probe node trajectory changed')
    if not state_record['full_rollout'] or state_record['query_observations_used']:
        raise ValueError('Probe is not a complete support-only state trajectory')
    if state_record['physics_complete_sha256']!=sha256(folder/'complete.json'):
        raise ValueError('Probe states refer to a different physical trajectory')
    data=read_json(Path(spec['data_root'])/'plan.json')
    nodes=next(o for o in data['objects'] if o['object']==obj)
    if state_record['nodes_sha256']!=nodes['nodes_sha256'] or sha256(nodes['nodes'])!=nodes['nodes_sha256']:
        raise ValueError('Probe used a different material-node map')
    receipt=read_json(folder/'elastic_receipt.json')
    if receipt['complete_sha256']!=sha256(folder/'complete.json') or not cache_compatible(
            root,receipt['physics_plan_sha256'],obj,episode,probe_spec,probe):
        raise ValueError('Probe receipt does not match its protocol and trajectory')
    state=load_npz(folder/'state.npz');checks=read_json(folder/'validation.json')
    actions=load_npz(Path(entry['workspace'])/'queries'/f'episode_{episode}'/'actions.npz')
    total=len(actions['source_timestamps_us'])
    if len(state['state'])!=total or len(state['times'])!=total or state_record['frames']!=total:
        raise ValueError('Probe did not finish all recorded action frames')
    if len(checks)!=total-1 or [c['frame'] for c in checks]!=list(range(1,total)) or not all(c['valid'] for c in checks):
        raise ValueError('Original per-frame validity checks did not all pass')
    if not np.isfinite(state['state']).all():raise ValueError('Nonfinite probe node states')
    scores=read_json(folder/'scores.json')
    if not scores['valid'] or not np.isfinite(scores['conditions']['real']['objective_mm']):
        raise ValueError('Full support objective is unavailable')
    return dict(folder=folder,entry=new_entry,record=record,
                seconds=read_json(folder/'execution.json')['elapsed_s'],frames=total)


def _commit(run,journal):
    """Resume any interrupted directory/manifest publication from its journal."""
    target,retired,staging=[Path(journal[k]) for k in ('target','retired','staging')]
    receipt=journal['receipt'];old_digest=receipt['old_physics_plan_sha256'];new_digest=receipt['new_physics_plan_sha256']
    with (run/'physics/plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=read_json(run/'protocol.json')
        if current not in (journal['old_spec'],journal['new_spec']):
            raise ValueError('Protocol changed outside the pending domain repair')
        if sha256(run/'physics/plan.json') not in (old_digest,new_digest):
            raise ValueError('Physics plan changed outside the pending domain repair')
        if sha256(journal['new_plan'])!=new_digest:
            raise ValueError('Journaled replacement physics plan changed')
        if not retired.exists():
            if sha256(target/'complete.json')!=journal['failed_complete_sha256']:
                raise ValueError('Original failed attempt changed before retirement')
            with (target/'elastic.lock').open('a') as episode_lock:
                fcntl.flock(episode_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                target.rename(retired)
        if sha256(retired/'complete.json')!=journal['failed_complete_sha256']:
            raise ValueError('Archived failed attempt changed')
        if not target.exists():
            if sha256(staging/'elastic_receipt.json')!=journal['adopted_receipt_sha256']:
                raise ValueError('Staged adopted trajectory changed')
            staging.rename(target)
        if sha256(target/'elastic_receipt.json')!=journal['adopted_receipt_sha256']:
            raise ValueError('Replacement trajectory has unexpected provenance')
        if sha256(target/'complete.json')!=journal['adopted_complete_sha256']:
            raise ValueError('Replacement physical trajectory changed')
        # Keep immutable copies for recovery if the process exits between the
        # two atomic file replacements. Readers serialize on plan.lock.
        write_json(run/'physics/plan.json',read_json(journal['new_plan']))
        if sha256(run/'physics/plan.json')!=new_digest:raise ValueError('Plan serialization changed')
        write_json(run/'protocol.json',journal['new_spec'])
        write_json(run/'domain_repair_complete.json',receipt)
    return receipt


def _publish(run):
    run=Path(run);request=read_json(run/'numerical_repair_plan.json');obj=request['object'];episode=request['episode']
    receipt_path=run/'domain_repair_complete.json'
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if not cache_compatible(run,receipt['new_physics_plan_sha256'],obj,episode):
            raise ValueError('Published repair no longer matches the episode protocol')
        return receipt
    journal_path=run/'domain_repair_pending.json'
    if journal_path.exists():return _commit(run,read_json(journal_path))
    failed_path=Path(request['failed_initial']);failed=read_json(failed_path)
    if failed['valid'] or failed['check'].get('failure_kind')!='domain_boundary':
        raise ValueError('Expected the recorded domain-boundary failure')
    if failed['object']!=obj or failed['episode']!=episode:raise ValueError('Repair object or episode mismatch')
    spec,plan=read_protocol(run);old_digest=spec['physics_plan_sha256'];selected=None;probe_results=[]
    for probe in request['probes']:
        candidate=validate_probe(run,probe['root'],obj,episode,failed)
        paths=list((Path(probe['root'])/'physics/trajectories'/obj/f'episode_{episode}').glob('*/execution.json'))
        if len(paths)!=1:raise ValueError('A configured diagnostic has not finished')
        probe_results.append(dict(id=probe['id'],root=probe['root'],valid=candidate is not None,
            seconds=read_json(paths[0])['elapsed_s']))
        if selected is None and candidate is not None:selected=dict(candidate,probe=probe)
    if selected is None:raise ValueError('Neither predefined domain probe completed a valid full trajectory')
    old_seconds=read_json(failed_path.parent/'execution.json')['elapsed_s']
    extra_seconds=old_seconds+sum(p['seconds'] for p in probe_results)-selected['seconds']
    new_plan=copy.deepcopy(plan);entry=find_entry(new_plan,obj,episode)
    for key in ('grid_padding_cells','grid_lower_cells'):
        if key in selected['entry']:entry[key]=selected['entry'][key]
        else:entry.pop(key,None)
    revisions=run/'physics/plan_revisions';revisions.mkdir(parents=True,exist_ok=True)
    candidate_plan=revisions/'candidate.json';write_json(candidate_plan,new_plan);new_digest=sha256(candidate_plan)
    frozen_plan=revisions/f'{new_digest}.json';candidate_plan.replace(frozen_plan)
    archive=revisions/f'{old_digest}.json'
    if not archive.exists():shutil.copy2(run/'physics/plan.json',archive)
    if sha256(archive)!=old_digest:raise ValueError('Cannot archive the exact original physics plan')
    staging=run/'physics'/f'domain_repair_staging_{obj}_{episode}'
    if staging.exists():shutil.rmtree(staging)
    shutil.copytree(selected['folder'],staging,ignore=shutil.ignore_patterns('*.lock','*.tmp'))
    source_receipt=read_json(staging/'elastic_receipt.json')
    adopted=dict(source_receipt,physics_plan_sha256=new_digest,requester=f'initial/{obj}',
        adopted_from=str(selected['folder']),adopted_source_receipt_sha256=sha256(selected['folder']/'elastic_receipt.json'))
    write_json(staging/'elastic_receipt.json',adopted)
    target=failed_path.parent
    retired=run/'physics/retired'/obj/f'episode_{episode}'/old_digest/target.name
    retired.parent.mkdir(parents=True,exist_ok=True)
    new_spec=copy.deepcopy(spec)
    new_spec.setdefault('compatible_physics_plans',{})[old_digest]=str(archive)
    new_spec['physics_plan_sha256']=new_digest
    new_spec.setdefault('numerical_domain_repairs',{})[f'{obj}/{episode}']=dict(
        grid_padding_cells=entry['grid_padding_cells'],grid_lower_cells=entry.get('grid_lower_cells'),
        receipt=str(receipt_path),same_material=True,same_timestep=True)
    receipt=dict(status='complete',object=obj,episode=episode,published_at=time.time(),
        selected_probe=selected['probe']['id'],selection='First valid full trajectory in the predefined probe order; no fitting-score selection',
        old_physics_plan_sha256=old_digest,new_physics_plan_sha256=new_digest,
        original_plan=str(archive),retired_initial=str(retired),adopted_trajectory=str(target),
        all_action_frames=selected['frames'],parameters=selected['record']['parameters'],
        domain=selected['record']['domain'],dtmax=selected['record']['dtmax'],probes=probe_results,
        numerical_preflight_seconds=extra_seconds,numerical_preflight_episode_rollouts=len(probe_results),
        source_physics_unchanged=True,query_observations_used=False)
    journal=dict(target=str(target),retired=str(retired),staging=str(staging),new_plan=str(frozen_plan),
        old_spec=spec,new_spec=new_spec,receipt=receipt,failed_complete_sha256=sha256(failed_path),
        adopted_complete_sha256=sha256(staging/'complete.json'),adopted_receipt_sha256=sha256(staging/'elastic_receipt.json'))
    write_json(journal_path,journal)
    return _commit(run,journal)


def publish(run):
    run=Path(run)
    with (run/'domain_repair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _publish(run)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    publish(parser.parse_args().run)
