"""Collect ordinary on-policy MPM transitions for the next shared-world update."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from .io import load_npz,read_json,sha256,write_json
from .shared_world_policy import load_policy
from .shared_world_control import PhysicalControl


def collect(root,policy_checkpoint,obj,episode,round_no=0,horizon=32,variant=0,device='cuda:0'):
    return collect_many(root,[policy_checkpoint],obj,episode,round_no,horizon,variant,device)[0]


def collect_many(root,policy_checkpoints,obj,episode,round_no=0,horizon=32,variant=0,device='cuda:0'):
    """Different frozen policies share one exact physical prefix and goal."""
    if not policy_checkpoints:raise ValueError('At least one frozen policy is required')
    root=Path(root);plan=read_json(root/'plan.json');item=next(o for o in plan['objects'] if o['object']==obj)
    if item['split']!='train':raise ValueError('On-policy training experience must exclude validation/test objects')
    entries=[]
    for policy_checkpoint in policy_checkpoints:
        digest=sha256(policy_checkpoint)
        identity=f'round_{round_no}/{obj}/episode_{episode}/policy_{digest[:12]}_material_{variant}'
        output=root/'experience'/identity;output.mkdir(parents=True,exist_ok=True)
        entries.append((policy_checkpoint,digest,identity,output))
        if (output/'complete.json').exists():
            previous=read_json(output/'complete.json')
            if previous['planned_frames']!=horizon+1 or previous['source_data_plan_sha256']!=sha256(root/'plan.json'):
                raise ValueError('Existing policy experience uses another horizon or data plan')
    if len({v[1] for v in entries})!=len(entries):raise ValueError('Repeated policy checkpoint in collection batch')
    parent=entries[0][3].parent
    with (parent/f'material_{variant}_batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[v for v in entries if not (v[3]/'complete.json').exists()]
        if not pending:return [read_json(v[3]/'complete.json') for v in entries]
        torch.set_num_threads(2)
        task=next(t for t in plan['tasks'] if t['object']==obj and t['episode']==episode and t['variant']==variant)
        physical=PhysicalControl(root,obj,episode,horizon,device,'baseline',task['parameters'])
        goal_folder=parent/f'goal_material_{variant}'
        if (goal_folder/'complete.json').exists():
            ref=read_json(goal_folder/'complete.json')
            if ref['planned_frames']!=horizon+1 or ref['snapshot_sha256']!=physical.snapshot_sha256 or sha256(goal_folder/'trajectory.npz')!=ref['trajectory_sha256']:
                raise ValueError('Changed physical policy goal cache')
            reference=load_npz(goal_folder/'trajectory.npz')['state']
        else:reference,ref=physical.rollout(physical.pads[physical.start:physical.end+1],goal_folder)
        if not ref['valid']:raise ValueError('Invalid reachable goal for policy experience')
        goal=reference[-1,:,:3]
        for policy_checkpoint,digest,identity,output in pending:
            with (output/'collect.lock').open('a') as output_lock:
                fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                actor,bundle=load_policy(policy_checkpoint,device);actor.requires_grad_(False)
                states,pads,result=physical.rollout_policy(actor,goal,device,observed_velocity=bundle['manifest'].get('observed_velocity',False))
                physical.save(output,states,pads,result)
                mask=physical.nodes['surface_mask']
                error=float(np.linalg.norm(states[-1,mask,:3]-goal[mask],axis=-1).mean()*1000) if result['valid'] else None
                result.update(schema='softdynamics.policy-experience.v1',
                    task=dict(id='experience/'+identity,object=obj,episode=episode,split='train',variant=1000+round_no),
                    source_data_plan_sha256=sha256(root/'plan.json'),policy_checkpoint=str(policy_checkpoint),policy_sha256=digest,
                    policy_step=bundle['step'],full_rollout=bool(result['valid'] and len(states)==horizon+1),
                    full_source_episode=False,trajectory_kind='complete_bounded_policy_episode',
                    source_start_frame=physical.start,source_end_frame=physical.end,nodes_sha256=item['nodes_sha256'],
                    material_parameters=physical.parameters,terminal_goal_mm=error,round=round_no,
                    shared_physical_goal=str(goal_folder),shared_goal_sha256=ref['trajectory_sha256'],
                    response_labels_used=False,world_model_targets='Actual MPM x/v/F/C at every policy step',created_at=time.time())
                write_json(output/'complete.json',result)
                print('POLICY_EXPERIENCE',identity,result['valid'],error,flush=True)
                del actor,bundle
        return [read_json(v[3]/'complete.json') for v in entries]


def evaluate(root,policy_checkpoint,obj,episode,output,horizon=32,device='cuda:0'):
    actor,bundle=load_policy(policy_checkpoint,device);actor.requires_grad_(False)
    if obj in bundle['training_objects']:raise ValueError('Physical policy evaluation requires an unseen object')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise ValueError('Evaluation is frozen')
    physical=PhysicalControl(root,obj,episode,horizon,device,'first_fit')
    reference,rr=physical.rollout(physical.pads[physical.start:physical.end+1],output/'reference')
    if not rr['valid']:raise ValueError('Invalid policy target')
    goal=reference[-1,:,:3];mask=physical.nodes['surface_mask']
    hold_pads=np.repeat(physical.pads[physical.start][None],horizon+1,axis=0)
    hold,hr=physical.rollout(hold_pads,output/'hold')
    predicted,pads,pr=physical.rollout_policy(actor,goal,device,output/'policy',bundle['manifest'].get('observed_velocity',False))
    def error(states):return float(np.linalg.norm(states[-1,mask,:3]-goal[mask],axis=-1).mean()*1000)
    result=dict(object=obj,episode=episode,policy_sha256=sha256(policy_checkpoint),policy_step=bundle['step'],
        policy_terminal_mm=error(predicted) if pr['valid'] else None,hold_terminal_mm=error(hold) if hr['valid'] else None,
        physical_policy_valid=pr['valid'],physical_hold_valid=hr['valid'],policy_seconds=pr['elapsed_s'],
        material_source=physical.material_source,parameters=physical.parameters,horizon=horizon,
        snapshot_sha256=physical.snapshot_sha256,actor_observes_actual_surface_state=True,actor_internal_F_C_input=False,
        actor_has_future_reference_actions=False,training_objects=bundle['training_objects'],real_robot_deployment=False)
    write_json(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['collect','evaluate'],required=True)
    p.add_argument('--root',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--policy-checkpoint',type=Path);group.add_argument('--policy-checkpoints',type=Path,nargs='+')
    p.add_argument('--object',required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--round',type=int,default=0);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--variant',type=int,default=0);p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='collect':collect_many(a.root,a.policy_checkpoints or [a.policy_checkpoint],a.object,a.episode,a.round,a.horizon,a.variant,a.device)
    else:
        if a.output is None:p.error('--output is required for evaluation')
        if a.policy_checkpoint is None:p.error('Single-policy evaluation requires --policy-checkpoint')
        evaluate(a.root,a.policy_checkpoint,a.object,a.episode,a.output,a.horizon,a.device)


if __name__=='__main__':main()
