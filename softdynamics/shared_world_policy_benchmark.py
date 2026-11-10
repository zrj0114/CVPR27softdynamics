"""Compare frozen policies from one identical full physical snapshot.

Imitation, validation-selected, and final gradient-trained policies receive the
same surface goal and actual current state. A verified physical reference cache
can avoid repeating identical goal/hold rollouts; logical costs remain explicit.
"""
import argparse
import fcntl
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .io import load_npz,read_json,sha256,write_json
from .shared_world_control import PhysicalControl,goal_cost,bounded_reference_commands
from .shared_world_policy import load_policy,policy_rollout
from .shared_world_training import load_model


def actor_digest(actor):
    digest=hashlib.sha256(str(bool(getattr(actor,'bounded_actions',False))).encode())
    for key,value in sorted(actor.state_dict().items()):
        array=value.detach().cpu().numpy()
        digest.update(key.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def validate_reference_case(case,physical,goal_spec=None):
    for key,value in dict(object=physical.obj,episode=physical.episode,horizon=physical.horizon,
                          start=physical.start,end=physical.end,parameters=physical.parameters,
                          snapshot_sha256=physical.snapshot_sha256).items():
        if case[key]!=value:raise ValueError('Physical reference cache differs: '+key)
    if goal_spec is not None and case.get('goal_spec',{'source':'recorded'})!=goal_spec:
        raise ValueError('Physical reference cache uses a different goal specification')


def cached_rollout(source,name,physical,goal_spec=None):
    source=Path(source);validate_reference_case(read_json(source/'case.json'),physical,goal_spec)
    folder=source/name;record=read_json(folder/'complete.json')
    if not record['valid'] or record['frames']!=physical.horizon+1 or record['snapshot_sha256']!=physical.snapshot_sha256:
        raise ValueError('Incomplete or mismatched physical reference')
    if sha256(folder/'trajectory.npz')!=record['trajectory_sha256']:raise ValueError('Changed cached trajectory')
    trajectory=load_npz(folder/'trajectory.npz')
    np.testing.assert_array_equal(trajectory['origin_m'],physical.origin)
    np.testing.assert_array_equal(trajectory['theta'],physical.theta)
    np.testing.assert_allclose(trajectory['times'],physical.times,atol=1e-10,rtol=0)
    if float(trajectory['scale_m'])!=physical.scale:raise ValueError('Changed physical node scale')
    return trajectory['state'],record


def benchmark(root,checkpoints,obj,episode,output,horizon=32,device='cuda:0',reference_case=None,start_frame=None,
              domain_padding_cells=None,goal_source='recorded',goal_seed=0,goal_knot_std=.15,prefix_cache_root=None):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'result.json').exists():raise ValueError('Policy comparison is immutable')
    if reference_case is not None and domain_padding_cells is not None:
        raise ValueError('Explicit domain retry must recompute the common physical reference and hold')
    if goal_source not in ('recorded','bounded_knots'):raise ValueError('Unknown physical goal source')
    if goal_source=='bounded_knots' and (goal_seed<0 or not np.isfinite(goal_knot_std) or goal_knot_std<=0):
        raise ValueError('Bounded goals require a nonnegative seed and positive finite knot scale')
    goal_spec={'source':'recorded'} if goal_source=='recorded' else dict(source=goal_source,seed=goal_seed,
        knot_std=goal_knot_std,knots=4,raw_command_step_bound=.249,translation_m=.012,rotation_rad=.08,aperture_m=.006)
    torch.set_num_threads(2);actors={};bundles={}
    for name,path in checkpoints.items():
        actor,bundle=load_policy(path,device);actor.requires_grad_(False)
        if obj in bundle['training_objects']:raise ValueError('Evaluation object was used for policy training')
        actors[name]=actor;bundles[name]=bundle
    first=next(iter(bundles.values()))
    if goal_source=='bounded_knots' and not first['manifest'].get('bounded_actions'):
        raise ValueError('Bounded task goals require policies with the matching action class')
    for bundle in bundles.values():
        for key in ['world_sha256','data_plan_sha256','bounded_actions','observed_velocity']:
            if bundle['manifest'].get(key)!=first['manifest'].get(key):raise ValueError('Compared policies use different '+key)
    if first['manifest']['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Changed physical data plan')
    world_path=Path(first['manifest']['world_checkpoint'])
    if sha256(world_path)!=first['manifest']['world_sha256']:raise ValueError('Changed shared world checkpoint')
    world,world_bundle=load_model(world_path,device);world.requires_grad_(False)
    if obj in world_bundle['training_objects']:raise ValueError('Evaluation object was used for world-model training')
    physical=PhysicalControl(root,obj,episode,horizon,device,'first_fit',start_frame=start_frame,
        domain_padding_cells=domain_padding_cells,prefix_cache_root=prefix_cache_root)
    write_json(output/'physical_setup.json',dict(object=obj,episode=episode,horizon=horizon,
        start=physical.start,end=physical.end,parameters=physical.parameters,
        material_source=physical.material_source,snapshot_sha256=physical.snapshot_sha256,
        domain_padding_cells=physical.grid_padding,lower_padding_cells=physical.grid_lower_padding,
        requested_start_frame=start_frame,prefix_seconds=physical.prefix_s,prefix_cache=physical.prefix_cache_info,goal_spec=goal_spec))
    if reference_case is None:
        commands=physical.pads[physical.start:physical.end+1]
        if goal_source=='bounded_knots':
            poses,actions,raw=bounded_reference_commands(torch.as_tensor(commands[:1],device=device),
                torch.tensor([physical.scale],device=device),horizon,goal_seed,goal_knot_std)
            commands=poses[0].cpu().numpy()
            np.savez_compressed(output/'reference_actions.npz',policy_actions=actions.cpu().numpy(),raw_knots=raw.cpu().numpy())
        reference,rr=physical.rollout(commands,output/'reference')
        hold_pads=np.repeat(physical.pads[physical.start][None],horizon+1,axis=0)
        hold,hr=physical.rollout(hold_pads,output/'hold')
    else:
        reference,rr=cached_rollout(reference_case,'reference',physical,goal_spec)
        hold,hr=cached_rollout(reference_case,'hold',physical,goal_spec)
    if not rr['valid'] or not hr['valid']:raise ValueError('Valid physical goal and baseline are required')
    goal=reference[-1,:,:3];mask=physical.nodes['surface_mask']
    def error(states):return float(np.linalg.norm(states[-1,mask,:3]-goal[mask],axis=-1).mean()*1000)
    initial,pads0,theta,geometry,times=physical.tensors(device,world.direct_displacement)
    target=torch.tensor((goal-physical.origin)/physical.scale,device=device)[None]
    case=dict(object=obj,episode=episode,horizon=horizon,start=physical.start,end=physical.end,
        domain_padding_cells=physical.grid_padding,lower_padding_cells=physical.grid_lower_padding,
        snapshot_sha256=physical.snapshot_sha256,parameters=physical.parameters,material_source=physical.material_source,
        world_sha256=first['manifest']['world_sha256'],data_plan_sha256=sha256(root/'plan.json'),
        policies={name:dict(checkpoint=str(path),checkpoint_sha256=sha256(path),step=bundles[name]['step'],
            actor_sha256=actor_digest(actors[name])) for name,path in checkpoints.items()},
        reference_case=str(reference_case) if reference_case is not None else None,
        cached_reference_sha256=rr['trajectory_sha256'],cached_hold_sha256=hr['trajectory_sha256'],
        prefix_seconds=physical.prefix_s,prefix_cache=physical.prefix_cache_info,hold_terminal_mm=error(hold),
        actual_current_surface_state_input=True,velocity_from_past_positions=world.direct_displacement,
        future_reference_actions_are_policy_input=False,internal_F_C_is_policy_input=False,
        goal_spec=goal_spec,reference_commands_within_policy_action_class=True if goal_source=='bounded_knots' else None,
        control_task_informative=error(hold)>1e-8,
        bounded_actions=bool(getattr(next(iter(actors.values())),'bounded_actions',False)),
        independent_test_object=physical.item['split']=='test',real_robot_deployment=False)
    write_json(output/'case.json',case);rows={};executed={};started=time.monotonic()
    for name,actor in actors.items():
        digest=case['policies'][name]['actor_sha256'];destination=output/name
        with torch.no_grad():
            imagined,_,_=policy_rollout(world,actor,initial,pads0,theta,geometry,times,target,False)
            _,predicted_mm=goal_cost(imagined[:,-1],target,geometry)
        if digest in executed:
            states,record,source=executed[digest]
        else:
            states,_,record=physical.rollout_policy(actor,goal,device,destination,world.direct_displacement)
            source=name;executed[digest]=(states,record,source)
        metric=error(states) if record['valid'] else None
        rows[name]=dict(step=bundles[name]['step'],valid=record['valid'],error=record['error'],
            terminal_mm=metric,imagined_terminal_mm=float(predicted_mm),
            improvement_over_hold_mm=case['hold_terminal_mm']-metric if metric is not None else None,
            physical_seconds=record['elapsed_s'],logical_policy_rollouts=1,
            reused_identical_actor_measurement=source if source!=name else None)
        write_json(output/'progress.json',dict(rows=rows,updated_at=time.time(),elapsed_s=time.monotonic()-started))
    result=dict(**case,rows=rows,unique_policy_rollouts=len(executed),
        physical_goal_rollouts_executed=int(reference_case is None),physical_hold_rollouts_executed=int(reference_case is None),
        reference_generation_excluded_equally_from_all_methods=True,shared_cache_is_not_claimed_as_policy_speedup=True,
        elapsed_s=time.monotonic()-started,completed_at=time.time())
    write_json(output/'result.json',result);lock.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--policy',action='append',nargs=2,metavar=('NAME','CHECKPOINT'),required=True)
    p.add_argument('--object',required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--reference-case',type=Path);p.add_argument('--start-frame',type=int)
    p.add_argument('--prefix-cache-root',type=Path)
    p.add_argument('--domain-padding-cells',type=int)
    p.add_argument('--goal-source',choices=['recorded','bounded_knots'],default='recorded')
    p.add_argument('--goal-seed',type=int,default=0);p.add_argument('--goal-knot-std',type=float,default=.15)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    checkpoints=dict(a.policy)
    if len(checkpoints)!=len(a.policy):p.error('Policy names must be unique')
    benchmark(a.root,checkpoints,a.object,a.episode,a.output,a.horizon,a.device,a.reference_case,a.start_frame,
              a.domain_padding_cells,a.goal_source,a.goal_seed,a.goal_knot_std,a.prefix_cache_root)


if __name__=='__main__':main()
