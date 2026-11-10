"""One goal-conditioned actor/value trained through the frozen shared dynamics.

Warm up the actor using recorded gripper motion, then optimize task return by
backpropagating through imagined dynamics. No material value or world-model
weight is changed by task reward. Actor and value inputs exclude internal F/C.
"""
import argparse
from collections import defaultdict
import copy
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .io import read_json,sha256,write_json
from .shared_world_model import GoalPolicy,GoalValue,apply_gripper_action,policy_action_step,initialize_goal_module
from .shared_world_training import Replay,load_model
from .shared_world_control import goal_cost,reference_step_corrections,corrected_step


def bounded_experience_start(trajectory,horizon):
    """Use goals reached by commands representable in the current actor domain.

    The original episode start fixes the bound origin. A cropped window would
    silently recenter those bounds. Random-knot commands additionally obey the
    actor's per-step increment limit; policy-generated commands already do.
    """
    record=trajectory.get('record',{})
    if trajectory.get('minimum_start',0)!=0 or len(trajectory['pads'])<=horizon:return False
    if record.get('response_labels_used') is not False:return False
    if record.get('schema')=='softdynamics.policy-experience.v1':
        return record.get('bounded_actions') is True
    if record.get('schema')!='softdynamics.control-experience.v1':return False
    expected={'total_translation_bound_m_per_axis':.012,'total_rotation_bound_rad_per_axis':.08,'total_aperture_bound_m':.006}
    if any(record.get(key)!=value for key,value in expected.items()):return False
    spread=record['knot_std'];rng=np.random.default_rng(record['random_seed'])
    raw=np.zeros((14,4)) if spread==0 else rng.normal(0,spread,(1,14,4))[0].astype(np.float32)
    knots=np.concatenate((np.zeros((14,1)),raw),axis=1)
    coordinates=np.linspace(0,4,len(trajectory['pads']))[:horizon+1]
    commands=np.stack([np.interp(coordinates,np.arange(5),channel) for channel in knots])
    return bool(np.abs(np.diff(commands,axis=1)).max(initial=0)<=.25+1e-6)


def eligible_windows(replay,horizon,goal_source='all'):
    if goal_source not in ('all','bounded_experience'):raise ValueError('Unknown policy goal source')
    choices=defaultdict(list)
    for identity,t in replay.trajectories.items():
        if t.get('record',{}).get('policy_training_eligible') is False:continue
        valid=t['pads'][...,12]>.5;starts=[]
        if goal_source=='bounded_experience':
            if not bounded_experience_start(t,horizon):continue
            candidates=[0]
        else:candidates=range(t.get('minimum_start',0),len(valid)-horizon)
        for start in candidates:
            if not (t['phase'][start:start+horizon+1]==1).all():continue
            window=valid[start:start+horizon+1]
            if np.array_equal(window,np.broadcast_to(window[0],window.shape)) and window[0].reshape(2,2).all(1).any():
                starts.append(start)
        if starts:choices[t['task']['object']].append((identity,np.array(starts)))
    return choices


def sample_windows(eligible,count,rng):
    objects=sorted(eligible);result=[]
    if not objects:raise ValueError('No complete, valid policy windows')
    for _ in range(count):
        obj=objects[int(rng.integers(len(objects)))];trajectories=eligible[obj]
        identity,starts=trajectories[int(rng.integers(len(trajectories)))]
        result.append((identity,int(rng.choice(starts))))
    return result


def policy_rollout(world,actor,initial,pads,theta,geometry,times,goal,gradients=True,step_corrections=None):
    if step_corrections is not None and step_corrections.shape[1]!=times.shape[1]-1:
        raise ValueError('Incorrect number of policy reference corrections')
    state=initial;states=[state];poses=[pads];actions=[]
    reference=pads;command=state.new_zeros((len(state),2,7))
    for t in range(times.shape[1]-1):
        next_pads,action,command=policy_action_step(actor,state,geometry,goal,pads,theta,reference,command)
        args=(state,pads,next_pads,theta,geometry,times[:,t+1]-times[:,t])
        state=checkpoint(world,*args,use_reentrant=False,preserve_rng_state=False) if gradients else world(*args)
        if step_corrections is not None:state=corrected_step(state,step_corrections[:,t])
        pads=next_pads;states.append(state);poses.append(pads);actions.append(action)
    return torch.stack(states,1),torch.stack(poses,1),torch.stack(actions,1)


def imitation_loss(actor,batch,geometry):
    goal=batch['states'][:,-1,:,:3];state=batch['states'][:,0];pads=batch['pads'][:,0]
    prediction,_,_=policy_action_step(actor,state,geometry,goal,pads,batch['theta'],pads,state.new_zeros((len(state),2,7)))
    target=batch['pads'][:,1]
    active=(pads[...,12]>.5)&(target[...,12]>.5)
    translation=(prediction[...,:3]-target[...,:3])*geometry['scale_m'][:,None,None]/.004
    rotation=(prediction[...,3:12]-target[...,3:12])/.03
    per_pad=F.smooth_l1_loss(translation,torch.zeros_like(translation),reduction='none').mean(-1)
    per_pad=per_pad+.1*F.smooth_l1_loss(rotation,torch.zeros_like(rotation),reduction='none').mean(-1)
    return (per_pad*active).sum()/active.sum().clamp_min(1)


def imagined_objective(states,pads,actions,goal,theta,geometry,value_target,discount=.95):
    delta=(states[:,1:,:,:3]-goal[:,None])*geometry['scale_m'][:,None,None,None]
    mask=geometry['surface_mask'].to(delta.dtype)[:,None]
    distance=(delta.square().sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    rewards=-distance/(.01**2)-.001*actions.square().mean((-1,-2))
    terminal=value_target(states[:,-1],geometry,goal,theta,pads[:,-1])
    discounts=discount**torch.arange(rewards.shape[1],device=states.device)
    imagined_return=(rewards*discounts).sum(-1)+discount**rewards.shape[1]*terminal
    task,mm=goal_cost(states[:,-1],goal,geometry)
    # Discourage imagined policies from leaving the current replay action range.
    # The range can be expanded after collecting matching physical experience.
    displacement=(pads[:,:, :, :3]-pads[:,0:1,:, :3])*geometry['scale_m'][:,None,None,None]
    trust=(displacement.abs()/.012-1).relu().square().mean()
    loss=task.mean()-.1*imagined_return.mean()+100*trust
    target=terminal.detach();returns=[]
    for t in reversed(range(rewards.shape[1])):
        target=rewards[:,t].detach()+discount*target;returns.append(target)
    return loss,torch.stack(returns[::-1],1),mm


@torch.no_grad()
def validate_policy(world,actor,replay,selection,horizon,device):
    rows=[];actor.eval()
    for identity,start in selection:
        b,g=replay.batch([(identity,start)],horizon,device);goal=b['states'][:,-1,:,:3]
        states,_,_=policy_rollout(world,actor,b['states'][:,0],b['pads'][:,0],b['theta'],g,b['times'],goal,False)
        _,mm=goal_cost(states[:,-1],goal,g)
        rows.append(dict(id=identity,start=start,object=replay.trajectories[identity]['task']['object'],predicted_terminal_mm=float(mm[0])))
    per_object={obj:float(np.mean([r['predicted_terminal_mm'] for r in rows if r['object']==obj])) for obj in {r['object'] for r in rows}}
    return dict(object_macro_predicted_terminal_mm=float(np.mean(list(per_object.values()))),per_object=per_object,rows=rows,
        reference='World-model prediction versus held-out physical trajectory goal; physical execution still required')


def validate_policy_initialization(saved,plan_sha256,objects,observed_velocity,bounded_actions,horizon):
    manifest=saved['manifest']
    if manifest['data_plan_sha256']!=plan_sha256:raise ValueError('Policy initialization uses another physical data plan')
    if manifest.get('observed_velocity',False)!=observed_velocity:raise ValueError('Policy initialization changes the velocity convention')
    if manifest.get('bounded_actions',False)!=bounded_actions:raise ValueError('Policy initialization changes the action domain')
    if manifest['horizon']!=horizon:raise ValueError('Policy refinement must preserve the validation horizon')
    if any(obj not in objects or objects[obj]['split']!='train' for obj in saved['training_objects']):
        raise ValueError('Held-out object in policy initialization')


def train_policy(root,world_checkpoint,output,steps=2000,warmup=300,horizon=32,batch_size=4,seed=0,device='cuda:0',bounded_actions=False,
                 initialize_from=None,replay_manifest=None,goal_source='all',reference_step_correction=False,pointwise_goal=False):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'policy.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'last.pt').exists() or (output/'initial.pt').exists():raise ValueError('Refusing to overwrite existing policy training')
    if initialize_from is not None and warmup:raise ValueError('Policy refinement uses zero imitation warmup')
    if goal_source=='bounded_experience' and not bounded_actions:raise ValueError('Bounded experience requires the matching actor action domain')
    if reference_step_correction and goal_source!='bounded_experience':
        raise ValueError('Reference-corrected policy training requires goals from its bounded action class')
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    world,w=load_model(world_checkpoint,device);world.requires_grad_(False)
    if reference_step_correction and not world.direct_displacement:
        raise ValueError('Reference-corrected policy training requires the observed-velocity world model')
    world_hash=sha256(world_checkpoint)
    actor=GoalPolicy(128,pointwise_goal).to(device);value=GoalValue(128,pointwise_goal).to(device)
    actor.bounded_actions=bounded_actions
    target=copy.deepcopy(value).requires_grad_(False);target.eval()
    actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=1e-4,weight_decay=1e-5)
    value_optimizer=torch.optim.AdamW(value.parameters(),lr=3e-4,weight_decay=1e-5)
    replay=Replay(root,['train'],world.direct_displacement);replay.refresh()
    val=Replay(root,['validation'],world.direct_displacement);val.refresh()
    if replay_manifest is not None:
        frozen=read_json(replay_manifest)
        if frozen['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Frozen policy replay uses another data plan')
        for buffer,split in [(replay,'train'),(val,'validation')]:
            for identity,digest in frozen[split].items():
                if identity not in buffer.trajectories:raise ValueError('Missing frozen policy trajectory: '+identity)
                if buffer.trajectories[identity]['record']['trajectory_sha256']!=digest:raise ValueError('Changed frozen policy trajectory')
            buffer.trajectories={identity:buffer.trajectories[identity] for identity in frozen[split]}
    eligible=eligible_windows(replay,horizon,goal_source);val_eligible=eligible_windows(val,horizon)
    selection=[];initializer=None
    if initialize_from is not None:
        initializer=torch.load(initialize_from,map_location=device,weights_only=False)
        validate_policy_initialization(initializer,sha256(root/'plan.json'),replay.objects,world.direct_displacement,bounded_actions,horizon)
        for module,key in [(actor,'actor'),(value,'value'),(target,'value_target')]:
            initialize_goal_module(module,initializer[key],initializer['manifest']['architecture'])
        selection=initializer['manifest']['validation_windows']
        allowed={identity:set(starts.tolist()) for rows in val_eligible.values() for identity,starts in rows}
        if any(identity not in allowed or start not in allowed[identity] for identity,start in selection):
            raise ValueError('Missing complete fixed policy validation window')
    else:
        val_rng=np.random.default_rng(9000+seed)
        for obj in sorted(val_eligible):selection.extend(sample_windows({obj:val_eligible[obj]},4,val_rng))
    if not selection or len(eligible)<2:raise ValueError('Need cross-object policy train/validation data')
    manifest=dict(world_checkpoint=str(world_checkpoint),world_sha256=world_hash,world_step=w['step'],
        data_plan_sha256=sha256(root/'plan.json'),architecture=actor.architecture,steps=steps,warmup=warmup,
        horizon=horizon,batch_size=batch_size,seed=seed,validation_windows=selection,
        training_goal_source=goal_source,validation_goal_source='Original fixed physical trajectory windows',
        reference_step_correction_during_training=reference_step_correction,
        reference_step_correction_during_validation_or_execution=False,
        correction_source='Ordinary simulated replay states and recorded replay actions' if reference_step_correction else None,
        world_model_frozen=True,materials_frozen_during_reward_update=True,
        policy_inputs='Current surface x/v, canonical surface geometry, goal, metric scale, current pads, fixed material estimate',
        actor_internal_F_C_input=False,tactile_input=False,bounded_actions=bounded_actions,
        pointwise_goal_offset_before_pooling=pointwise_goal,
        action_domain='Whole-window offsets: translation 12 mm per axis, rotation .08 rad per axis, aperture 6 mm' if bounded_actions else 'Legacy incremental commands',
        command_update='command += .25 * actor_action; bounded pose = tanh(command) relative to the initial grippers' if bounded_actions else None)
    manifest['observed_velocity']=world.direct_displacement
    if replay_manifest is not None:
        manifest.update(replay_manifest=str(Path(replay_manifest).resolve()),replay_manifest_sha256=sha256(replay_manifest))
    if initializer is not None:
        manifest.update(initialize_from=str(Path(initialize_from).resolve()),initialization_sha256=sha256(initialize_from),
            initialization_step=initializer['step'],initialization_world_sha256=initializer['manifest']['world_sha256'],
            optimizer_reset_for_new_world=True,fixed_parent_validation_windows=True)
        if pointwise_goal and not initializer['manifest']['architecture'].get('pointwise_goal',False):
            manifest['initialization_conversion']='Preserve all actor/value weights and add a zero goal-offset projection before pointwise activation'
    write_json(output/'manifest.json',manifest);started=time.monotonic();best=float('inf');seen=set((initializer or {}).get('training_ids',[]))
    def policy_checkpoint(step,result):
        return dict(schema='softdynamics.shared-policy.v1',manifest=manifest,actor=actor.state_dict(),value=value.state_dict(),
            value_target=target.state_dict(),step=step,training_ids=sorted(seen),
            training_objects=sorted(set(eligible)|set((initializer or {}).get('training_objects',[]))),
            validation=result,actor_optimizer=actor_optimizer.state_dict(),value_optimizer=value_optimizer.state_dict())
    if initializer is not None:
        result=validate_policy(world,actor,val,selection,horizon,device)
        best=result['object_macro_predicted_terminal_mm'];saved=policy_checkpoint(0,result)
        torch.save(saved,output/'initial.pt');torch.save(saved,output/'best.pt')
        write_json(output/'validation_0.json',dict(step=0,improved=True,**result))
    for step in range(1,steps+1):
        if step%100==1 and replay_manifest is None:replay.refresh();eligible=eligible_windows(replay,horizon,goal_source)
        choices=sample_windows(eligible,batch_size,rng);seen.update(i for i,_ in choices)
        b,g=replay.batch(choices,horizon,device);actor.train();actor_optimizer.zero_grad(set_to_none=True)
        bc=imitation_loss(actor,b,g);value_loss=None;terminal_mm=None
        if step<=warmup:
            loss=bc;loss.backward()
        else:
            goal=b['states'][:,-1,:,:3]
            corrections=reference_step_corrections(world,b['states'],b['pads'],b['theta'],g,b['times']) if reference_step_correction else None
            states,pads,actions=policy_rollout(world,actor,b['states'][:,0],b['pads'][:,0],b['theta'],g,b['times'],goal,
                step_corrections=corrections)
            task_loss,returns,mm=imagined_objective(states,pads,actions,goal,b['theta'],g,target)
            # Keep a small trajectory-imitation term while the policy explores.
            loss=task_loss+.1*bc;terminal_mm=float(mm.detach().mean())
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite differentiable policy objective')
            loss.backward()
            value_optimizer.zero_grad(set_to_none=True)
            indices=np.unique(np.linspace(0,horizon-1,4).astype(int));predicted=[]
            for t in indices:predicted.append(value(states[:,t].detach(),g,goal,b['theta'],pads[:,t].detach()))
            predicted=torch.stack(predicted,1)
            value_loss=F.smooth_l1_loss(predicted,returns[:,indices],beta=1.)
            value_loss.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),1.,error_if_nonfinite=True);value_optimizer.step()
            with torch.no_grad():
                for dst,src in zip(target.parameters(),value.parameters()):dst.lerp_(src,.01)
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);actor_optimizer.step()
        if step%20==0 or step==1:
            row=dict(step=step,stage='imitation_warmup' if step<=warmup else 'world_model_policy_gradient',
                loss=float(loss.detach()),imitation_loss=float(bc.detach()),value_loss=float(value_loss.detach()) if value_loss is not None else None,
                predicted_terminal_mm=terminal_mm,elapsed_s=time.monotonic()-started,train_objects=sorted(eligible),
                train_trajectories=len(replay.trajectories),updated_at=time.time())
            import json
            with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            write_json(output/'status.json',row);print('POLICY',step,row['loss'],terminal_mm,flush=True)
        if step%200==0 or step==steps or step==warmup:
            result=validate_policy(world,actor,val,selection,horizon,device)
            improved=np.isfinite(result['object_macro_predicted_terminal_mm']) and result['object_macro_predicted_terminal_mm']<best
            if improved:best=result['object_macro_predicted_terminal_mm']
            saved=policy_checkpoint(step,result)
            torch.save(saved,output/'last.tmp.pt');(output/'last.tmp.pt').replace(output/'last.pt')
            if step==warmup:torch.save(saved,output/'imitation.pt')
            if improved:torch.save(saved,output/'best.tmp.pt');(output/'best.tmp.pt').replace(output/'best.pt')
            write_json(output/f'validation_{step}.json',dict(step=step,improved=improved,**result))
    write_json(output/'complete.json',dict(steps=steps,world_sha256=world_hash,elapsed_s=time.monotonic()-started,
        best_predicted_validation_mm=best,physical_policy_evaluation_pending=True))
    lock.close()


def load_policy(path,device):
    saved=torch.load(path,map_location=device,weights_only=False)
    actor=GoalPolicy(**saved['manifest']['architecture']).to(device);actor.load_state_dict(saved['actor']);actor.eval()
    actor.bounded_actions=saved['manifest'].get('bounded_actions',False)
    return actor,saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--world-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--warmup',type=int,default=300);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--bounded-actions',action='store_true')
    p.add_argument('--initialize-from',type=Path);p.add_argument('--replay-manifest',type=Path)
    p.add_argument('--goal-source',choices=['all','bounded_experience'],default='all')
    p.add_argument('--reference-step-correction',action='store_true')
    p.add_argument('--pointwise-goal',action='store_true')
    a=p.parse_args();train_policy(a.root,a.world_checkpoint,a.output,a.steps,a.warmup,a.horizon,a.batch_size,a.seed,a.device,a.bounded_actions,
        a.initialize_from,a.replay_manifest,a.goal_source,a.reference_step_correction,a.pointwise_goal)


if __name__=='__main__':main()
