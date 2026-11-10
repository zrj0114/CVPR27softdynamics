"""Reject label/action mismatches before training under the selected contact rule."""
from pathlib import Path
import numpy as np
from .io import read_json,write_json,load_npz
from .teacher_contact import (CONTACT_ID,SUPPORT_ID,KERNEL_SHA256,STATIC_FRICTION,
    SLIDING_FRICTION,load_completed_actions,pack_pads,contract_sha256)

def audit_replay_contact(manifest,output=None,action_root=None):
    root=Path(manifest['data_root']);plan=read_json(root/'plan.json')
    tasks={t['id']:t for t in plan['tasks']};records=[];cache={}
    retries=read_json(root/'retry_index.json') if (root/'retry_index.json').exists() else {}
    for row in manifest['records']:
        identity=row['id'];task=tasks[identity];obj,episode=task['object'],task['episode'];key=(obj,episode)
        if key not in cache:cache[key]=load_completed_actions(obj,episode,root=action_root)
        actions,provenance=cache[key]
        folder=root/'trajectories'/identity
        if identity in retries:folder=root/retries[identity]['folder']
        receipt=read_json(folder/'complete.json');reasons=[]
        if not receipt.get('valid') or not receipt.get('full_rollout'):reasons.append('incomplete_teacher')
        hashes=receipt.get('source_hashes',{})
        if hashes.get('pad_kernel')!=KERNEL_SHA256['pad_contact_kernels.py']:reasons.append('different_teacher_pad_rule')
        parameters=task['parameters']
        if not np.isclose(parameters['gripper_static_friction'],STATIC_FRICTION,rtol=0,atol=1e-6) or not np.isclose(parameters['gripper_sliding_friction'],SLIDING_FRICTION,rtol=0,atol=1e-6):
            reasons.append('different_gripper_friction')
        if task.get('randomized_actions'):
            if hashes.get('actions_logical')!=provenance['logical_sha256'] or hashes.get('contact_contract')!=contract_sha256():
                reasons.append('randomized_actions_require_selected_base_provenance')
        else:
            with np.load(folder/'trajectory.npz',allow_pickle=False) as z:
                expected=pack_pads(actions,z['origin_m'],float(z['scale_m']))
                if not np.array_equal(z['pads'],expected):reasons.append('completed_actions_change_teacher_controls')
                times=(actions['source_timestamps_us']-actions['source_timestamps_us'][0])*1e-6
                if z['times'].shape!=times.shape or not np.allclose(z['times'],times,atol=1e-10,rtol=0):reasons.append('action_timeline_mismatch')
        records.append(dict(id=identity,ready=not reasons,reasons=reasons,actions=provenance))
    result=dict(contact_id=CONTACT_ID,support_id=SUPPORT_ID,contact_sha256=contract_sha256(),
        records=records,teacher_records=len(records),ready_records=sum(r['ready'] for r in records),
        need_regeneration=sum(not r['ready'] for r in records),ready=all(r['ready'] for r in records),
        source_pose_quality_filter=False,label_arrays_modified=False)
    if output is not None:write_json(output,result)
    return result

def require_aligned_replay(manifest,output=None):
    result=audit_replay_contact(manifest,output)
    if not result['ready']:
        raise ValueError(f"{result['need_regeneration']}/{result['teacher_records']} teacher trajectories do not match the selected contact/actions; regenerate or explicitly revalidate those labels before training. Alignment report: {output}")
    return result
