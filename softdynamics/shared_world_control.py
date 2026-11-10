"""Optimize bounded gripper actions and verify them in the original full MPM.

Each case freezes a physical snapshot, material estimate and reachable goal.
World-model optimization receives the current state and goal, never the future
reference actions. All methods restore the identical physical snapshot. Internal
F/C is available to the world model in this controlled simulator experiment;
deployment with an estimated hidden state is a separate evaluation condition.
"""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .io import load_npz,read_json,sha256,write_json
from .inference import PARAMETERS
from .shared_world_model import apply_gripper_action,policy_action_step,recurrent_channels
from .shared_world_training import load_model
from .world_data import controller_features
from .world_model import normalize_state


def choose_control_start(pads,clip_start,clip_end,horizon,preferred_start=None):
    """Nearest to the interaction midpoint with continuous, stable tool visibility."""
    valid=np.asarray(pads)[...,12]>.5
    starts=[]
    for start in range(max(0,clip_start),min(clip_end-horizon+1,len(valid)-horizon)):
        window=valid[start:start+horizon+1]
        if np.array_equal(window,np.broadcast_to(window[0],window.shape)) and window[0].reshape(2,2).all(1).any():
            starts.append(start)
    if not starts:raise ValueError('No complete control window with stable valid gripper poses')
    if preferred_start is not None:
        if preferred_start not in starts:raise ValueError('Requested start is not a complete valid interaction window')
        return int(preferred_start)
    center=(clip_start+clip_end-horizon)/2
    return min(starts,key=lambda x:(abs(x-center),x))


def expanded_actions(raw,horizon):
    # A small number of temporal knots defines a smooth, bounded command stream.
    return F.interpolate(raw,size=horizon,mode='linear',align_corners=True).transpose(1,2).reshape(-1,horizon,2,7).tanh()


def action_pads(initial_pads,actions,scale_m):
    values=[initial_pads]
    for t in range(actions.shape[1]):values.append(apply_gripper_action(values[-1],actions[:,t],scale_m))
    return torch.stack(values,1)


def bounded_pose_sequence(initial_pads,raw,horizon,scale_m,translation_m=.012,rotation_rad=.08,aperture_m=.006):
    """Total gripper offsets match the ordinary rollout randomization domain.

    Initial pose is fixed. Bounds apply to the whole horizon rather than being
    accumulated every frame. Both fingers still undergo one rigid transform.
    """
    knots=torch.cat((torch.zeros_like(raw[:,:,:1]),raw),-1)
    commands=expanded_actions(knots,horizon+1)
    poses=[initial_pads]
    for t in range(1,horizon+1):
        poses.append(apply_gripper_action(initial_pads,commands[:,t],scale_m,
            translation_m=translation_m,rotation_rad=rotation_rad,aperture_m=aperture_m))
    return torch.stack(poses,1),commands[:,1:]


def bounded_reference_commands(initial_pads,scale_m,horizon,seed,knot_std=.15,knots=4):
    """A goal-generating command sequence within the bounded policy action class.

    The reference actions are used only to generate a physical target. Limiting
    the raw command increments also respects policy_action_step's per-step
    accumulation bound, not merely the final pose limits.
    """
    if horizon<1 or knots<1 or not np.isfinite(knot_std) or knot_std<=0:
        raise ValueError('Positive horizon, knot count and finite knot scale required')
    raw=torch.as_tensor(np.random.default_rng(seed).normal(0,knot_std,(len(initial_pads),14,knots)),
                        device=initial_pads.device,dtype=initial_pads.dtype)
    commands=F.interpolate(torch.cat((torch.zeros_like(raw[:,:,:1]),raw),-1),
                           size=horizon+1,mode='linear',align_corners=True)
    largest=commands.diff(dim=-1).abs().amax((1,2),keepdim=True)
    # The small interior margin avoids requiring an exactly saturated tanh actor.
    factor=torch.clamp(.249/largest.clamp_min(1e-12),max=1.)
    raw=raw*factor
    commands=F.interpolate(torch.cat((torch.zeros_like(raw[:,:,:1]),raw),-1),
                           size=horizon+1,mode='linear',align_corners=True)
    policy_actions=(commands.diff(dim=-1)/.25).transpose(1,2).reshape(-1,horizon,2,7)
    if bool((policy_actions.abs()>1).any()):raise ValueError('Reference exceeds the policy command rate')
    # Accumulate the actual policy commands so even floating-point roundoff
    # matches the feedback controller's command path.
    command=initial_pads.new_zeros((len(initial_pads),2,7));poses=[initial_pads]
    for t in range(horizon):
        command=command+.25*policy_actions[:,t]
        poses.append(apply_gripper_action(initial_pads,command.tanh(),scale_m,
            translation_m=.012,rotation_rad=.08,aperture_m=.006))
    return torch.stack(poses,1),policy_actions,raw


@torch.no_grad()
def reference_step_corrections(model,states,pads,theta,geometry,times):
    """Fixed zero-order model errors along an ordinary physical trajectory.

    The reference material/actions and corrections receive no candidate
    gradients. This is a rollout correction, never a response/Jacobian label.
    Candidate derivatives still come entirely from the frozen learned dynamics.
    """
    if states.shape[1]!=pads.shape[1] or times.shape[1]!=states.shape[1]:
        raise ValueError('Reference states, controls and times must share a horizon')
    channels=recurrent_channels(model)
    corrections=[]
    for t in range(states.shape[1]-1):
        predicted=model(states[:,t],pads[:,t],pads[:,t+1],theta,geometry,times[:,t+1]-times[:,t])
        corrections.append(states[:,t+1,:,:channels]-predicted[...,:channels])
    return torch.stack(corrections,1).detach()


def corrected_step(state,correction):
    if state.shape[:-1]!=correction.shape[:-1] or correction.shape[-1]>state.shape[-1]:
        raise ValueError('Invalid state correction shape')
    channels=correction.shape[-1]
    return torch.cat((state[...,:channels]+correction.detach(),state[...,channels:]),-1)


@torch.no_grad()
def reference_step_anchors(model,states,pads,theta,geometry,times):
    """Keep the two terms of an ordinary trajectory correction separately.

    Evaluating reference_next + (prediction - reference_prediction) preserves
    exact cancellation at the reference. Forming a residual first can inject
    rounding errors which unstable learned dynamics amplify over a rollout.
    Both anchor terms are fixed; candidate state/action/material gradients
    still come entirely from the learned transition.
    """
    if states.shape[1]!=pads.shape[1] or times.shape[1]!=states.shape[1]:
        raise ValueError('Reference states, controls and times must share a horizon')
    channels=recurrent_channels(model)
    predictions=[]
    for t in range(states.shape[1]-1):
        predicted=model(states[:,t],pads[:,t],pads[:,t+1],theta,geometry,times[:,t+1]-times[:,t])
        predictions.append(predicted[...,:channels])
    return states[:,1:,:,:channels].detach().clone(),torch.stack(predictions,1).detach()


def anchored_step(state,reference_next,reference_prediction):
    if (reference_next.shape!=reference_prediction.shape or
        state.shape[:-1]!=reference_next.shape[:-1] or reference_next.shape[-1]>state.shape[-1]):
        raise ValueError('Invalid state anchor shapes')
    channels=reference_next.shape[-1]
    corrected=reference_next.detach()+(state[...,:channels]-reference_prediction.detach())
    return torch.cat((corrected,state[...,channels:]),-1)


def pose_rollout(model,initial,pads,theta,geometry,times,use_checkpoint=False,step_corrections=None,step_anchors=None):
    if step_corrections is not None and step_anchors is not None:
        raise ValueError('Choose one reference correction representation')
    if step_corrections is not None and step_corrections.shape[1]!=pads.shape[1]-1:
        raise ValueError('Incorrect number of reference step corrections')
    if step_anchors is not None:
        if (len(step_anchors)!=2 or step_anchors[0].shape!=step_anchors[1].shape or
            step_anchors[0].ndim!=4 or step_anchors[0].shape[1]!=pads.shape[1]-1):
            raise ValueError('Incorrect reference step anchors')
    state=initial;states=[state]
    for t in range(pads.shape[1]-1):
        args=(state,pads[:,t],pads[:,t+1],theta,geometry,times[:,t+1]-times[:,t])
        state=checkpoint(model,*args,use_reentrant=False,preserve_rng_state=False) if use_checkpoint else model(*args)
        if step_corrections is not None:state=corrected_step(state,step_corrections[:,t])
        if step_anchors is not None:state=anchored_step(state,step_anchors[0][:,t],step_anchors[1][:,t])
        states.append(state)
    return torch.stack(states,1)


def control_rollout(model,initial,initial_pads,actions,theta,geometry,times,use_checkpoint=False):
    state=initial;pads=initial_pads;states=[state];poses=[pads]
    for t in range(actions.shape[1]):
        next_pads=apply_gripper_action(pads,actions[:,t],geometry['scale_m'])
        args=(state,pads,next_pads,theta,geometry,times[:,t+1]-times[:,t])
        state=checkpoint(model,*args,use_reentrant=False,preserve_rng_state=False) if use_checkpoint else model(*args)
        pads=next_pads;states.append(state);poses.append(pads)
    return torch.stack(states,1),torch.stack(poses,1)


def goal_cost(state,goal,geometry):
    delta=(state[...,:3]-goal)*geometry['scale_m'][:,None,None]
    mask=geometry['surface_mask'].to(delta.dtype)
    squared=(delta.square().sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    mm=(torch.linalg.vector_norm(delta,dim=-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)*1000
    return squared/(.003**2),mm


def anchored_positions(predicted,baseline_prediction,physical_baseline):
    """Zero-order simulator correction; only the proposal receives gradients.

    This changes inference, not training: no paired-response/Jacobian targets.
    At the baseline action, the corrected positions equal the actual simulator.
    """
    if predicted.shape!=baseline_prediction.shape or predicted.shape!=physical_baseline.shape:
        raise ValueError('Anchor and prediction positions must have identical shapes')
    return physical_baseline.detach()+(predicted-baseline_prediction.detach())


def optimize_actions(model,initial,pads0,theta,geometry,times,goal,steps=60,knots=4,seed=0,output=None,anchor=None):
    model.eval().requires_grad_(False);torch.manual_seed(seed)
    horizon=times.shape[1]-1;raw=torch.zeros((len(initial),14,knots),device=initial.device,requires_grad=True)
    optimizer=torch.optim.Adam([raw],lr=.08);best=None;history=[];started=time.monotonic()
    baseline=None
    if anchor is not None:
        with torch.no_grad():
            baseline_pads=bounded_pose_sequence(pads0,raw,horizon,geometry['scale_m'])[0]
            baseline=pose_rollout(model,initial,baseline_pads,theta,geometry,times)[...,:3]
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        pads,actions=bounded_pose_sequence(pads0,raw,horizon,geometry['scale_m'])
        states=pose_rollout(model,initial,pads,theta,geometry,times,use_checkpoint=True)
        positions=states[...,:3] if anchor is None else anchored_positions(states[...,:3],baseline,anchor)
        task,mm=goal_cost(positions[:,-1],goal,geometry)
        loss=task.mean()+.001*actions.square().mean()+.001*(actions[:,1:]-actions[:,:-1]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Invalid differentiable control prediction')
        row=dict(step=step,objective=float(loss.detach()),predicted_terminal_mm=float(mm.detach().mean()),
                 elapsed_s=time.monotonic()-started)
        if best is None or row['objective']<best['objective']:
            best={**row,'actions':actions.detach().cpu().numpy(),'pads':pads.detach().cpu().numpy(),
                  'states':states.detach().cpu().numpy(),'corrected_positions':positions.detach().cpu().numpy(),
                  'raw':raw.detach().cpu().numpy()}
        if step<steps:
            loss.backward();row['gradient_norm']=float(torch.nn.utils.clip_grad_norm_([raw],10.,error_if_nonfinite=True))
            optimizer.step()
        history.append(row)
        if output is not None and (step%5==0 or step==steps):write_json(Path(output)/'optimization_progress.json',row)
    return best,history,time.monotonic()-started


def sample_state(sim,ids):
    s=sim.states[sim.current]
    return np.concatenate((s.particle_x.numpy()[ids]-sim.shift,s.particle_v.numpy()[ids],
        s.particle_F_trial.numpy()[ids].reshape(-1,9),s.particle_C.numpy()[ids].reshape(-1,9)),-1).astype('f')


def geometry_tensors(nodes,origin,device):
    values={k:torch.as_tensor(nodes[k][None],device=device) for k in ('canonical','neighbors','surface_mask')}
    values['scale_m']=torch.tensor([float(nodes['scale_m'])],device=device)
    values['ground_y']=torch.tensor([(-.01-origin[1])/float(nodes['scale_m'])],device=device,dtype=torch.float32)
    return values


class PhysicalControl:
    def __init__(self,root,obj,episode,horizon,device='cuda:0',material='first_fit',parameters_override=None,start_frame=None,
                 domain_padding_cells=None,prefix_cache_root=None):
        from .adapters.pad_empm import PadRuntime
        self.root=Path(root);self.plan=read_json(self.root/'plan.json');self.obj=obj;self.episode=episode
        self.item=next(o for o in self.plan['objects'] if o['object']==obj)
        task=next(t for t in self.plan['tasks'] if t['object']==obj and t['episode']==episode and t['variant']==0)
        self.parameters=dict(task['parameters']);self.material_source='pretrain_baseline'
        if material=='first_fit' and obj in self.plan['first_module_handoffs']:
            handoff=self.plan['first_module_handoffs'][obj]
            if handoff['selected_q'] is not None:
                fit=read_json(Path(self.plan['fit_run'])/'plan.json');cfg=fit['config']
                theta=np.array([cfg['baseline'][k] for k in PARAMETERS])+np.array(handoff['selected_q'])*cfg['coordinate_scales']
                self.parameters.update(zip(PARAMETERS,theta.tolist()));self.material_source=handoff['selection']
            else:self.material_source='baseline_no_completed_first_module_fit'
        if parameters_override is not None:
            self.parameters.update(parameters_override);self.material_source='explicit_simulation_material_sample'
        self.theta=np.array([self.parameters[k] for k in PARAMETERS],np.float32)
        for path,digest in task['input_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Changed original physical input: '+path)
        if sha256(self.item['nodes'])!=self.item['nodes_sha256']:raise ValueError('Changed node mapping')
        self.nodes=load_npz(self.item['nodes']);self.ids=self.nodes['material_indices'];self.horizon=horizon
        retries=read_json(self.root/'retry_index.json') if (self.root/'retry_index.json').exists() else {}
        self.grid_padding=retries.get(task['id'],{}).get('domain_padding_cells',0)
        self.grid_lower_padding=retries.get(task['id'],{}).get('lower_padding_cells')
        if domain_padding_cells is not None:
            self.grid_padding=domain_padding_cells;self.grid_lower_padding=None
        from .shared_world_observation import padded_simulation_grid
        with padded_simulation_grid(self.grid_padding,self.grid_lower_padding):
            self.sim=PadRuntime(episode,self.parameters,self.item['workspace'],self.plan['runtime'],task['dtmax'],device,task['domain_margin'])
        self.observer=None
        if self.sim.n>=200000:
            from .shared_world_observation import GPUObservation
            self.observer=GPUObservation(self.sim,self.ids);self.observer.install_validation()
        self.origin=self.sim.positions(np.arange(self.sim.n)).mean(0);self.scale=float(self.nodes['scale_m'])
        self.pads=controller_features(self.sim,self.origin,self.scale)
        self.start=choose_control_start(self.pads,self.sim.meta['clip_start'],self.sim.meta['clip_end'],horizon,start_frame)
        self.end=self.start+horizon;self.original_poses=self.sim.poses.copy();self.original_valid=self.sim.valid.copy()
        started=time.monotonic()
        def build_prefix():
            previous=None
            if self.start:
                if self.start>1:self.sim.advance(self.start-1)
                previous=self.capture()[:,:3]
                check=self.sim.advance(self.start)
                if not check['valid']:raise ValueError('Invalid physical prefix: '+str(check))
            return self.sim.snapshot(),previous
        self.prefix_cache_info=None
        if prefix_cache_root is None:self.snapshot,previous=build_prefix()
        else:
            if self.sim.model.material!=0:raise ValueError('Prefix cache supports elastic FCR only')
            from .shared_world_snapshot import physical_prefix
            runtime=Path(self.plan['runtime']);source=Path(__file__).resolve().parent
            identity=dict(schema='softdynamics.physical-prefix-identity.v1',data_plan_sha256=sha256(self.root/'plan.json'),
                object=obj,episode=episode,start_frame=self.start,particles=self.sim.n,sampled_nodes=len(self.ids),
                parameters={key:float(value) for key,value in self.parameters.items()},dtmax=float(self.sim.dtmax),
                domain=self.sim.domain,nodes_sha256=self.item['nodes_sha256'],physical_inputs=self.sim.source_hashes,
                runtime_python_sha256={str(p.relative_to(runtime)):sha256(p) for p in sorted(runtime.rglob('*.py'))},
                adapter_sha256=sha256(source/'adapters/empm.py'),control_sha256=sha256(Path(__file__)),
                observation_sha256=sha256(source/'shared_world_observation.py'),
                numpy_version=np.__version__,torch_version=torch.__version__,warp_version=self.sim.up.wp.__version__)
            self.snapshot,previous,self.prefix_cache_info=physical_prefix(prefix_cache_root,identity,build_prefix)
            if self.prefix_cache_info['reused']:self.sim.restore(self.snapshot,self.parameters)
        self.initial=self.capture()
        self.initial_observed_velocity=np.zeros_like(self.initial[:,:3]) if previous is None else (
            self.initial[:,:3]-previous)/(self.sim.times[self.start]-self.sim.times[self.start-1])
        self.prefix_s=time.monotonic()-started;self.snapshot_sha256=self.sim.digest(self.snapshot)
        self.times=self.sim.times[self.start:self.end+1]-self.sim.times[self.start]

    def capture(self):
        return self.observer.capture() if self.observer is not None else sample_state(self.sim,self.ids)

    def reset(self):
        self.sim.restore(self.snapshot,self.parameters)
        self.sim.poses=self.original_poses.copy();self.sim.valid=self.original_valid.copy()

    def set_controls(self,start,pads):
        for g in range(self.sim.poses.shape[1]):
            for side in range(2):
                k=2*g+side;good=pads[:,k,12]>.5
                values=np.tile(np.eye(4),(len(pads),1,1))
                values[good,:3,3]=pads[good,k,:3]*self.scale+self.origin
                values[good,:3,:3]=pads[good,k,3:12].reshape(-1,3,3);values[~good]=np.nan
                self.sim.poses[start:start+len(pads),g,side]=values
                self.sim.valid[start:start+len(pads),g,side]=good

    def save(self,output,states,pads,result):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'trajectory.npz',state=states,pads=pads[:len(states)],times=self.times[:len(states)],
            origin_m=self.origin,scale_m=self.scale,theta=self.theta,phase=np.ones(len(states),np.int32),
            initial_observed_velocity_m_s=self.initial_observed_velocity)
        result['trajectory_sha256']=sha256(output/'trajectory.npz');write_json(output/'complete.json',result)

    def rollout(self,pads,output=None):
        pads=np.asarray(pads)
        if pads.shape!=(self.horizon+1,4,13):raise ValueError('Incorrect physical control horizon')
        np.testing.assert_allclose(pads[0],self.pads[self.start],atol=2e-6,rtol=2e-6)
        started=time.monotonic();self.reset();self.set_controls(self.start,pads)
        states=[self.capture()];checks=[];failure=None
        try:
            for frame in range(self.start+1,self.end+1):
                check=self.sim.advance(frame);checks.append(check)
                if not check['valid']:raise ValueError('Invalid physical action rollout: '+str(check))
                states.append(self.capture())
        except (ValueError,RuntimeError) as exc:failure=str(exc)
        result=dict(valid=failure is None,error=failure,frames=len(states),planned_frames=self.horizon+1,
                    elapsed_s=time.monotonic()-started,snapshot_sha256=self.snapshot_sha256,checks=checks)
        states=np.stack(states)
        if output is not None:self.save(output,states,pads,result)
        return states,result

    @torch.no_grad()
    def rollout_policy(self,actor,goal,device,output=None,observed_velocity=False):
        """Closed-loop policy sees each actual next physical surface state."""
        self.reset();started=time.monotonic();states=[self.capture()]
        poses=[self.pads[self.start]];checks=[];failure=None
        geometry=geometry_tensors(self.nodes,self.origin,device);theta=torch.tensor(self.theta[None],device=device)
        target=torch.tensor((goal-self.origin)/self.scale,device=device)[None]
        reference=torch.tensor(poses[0][None],device=device);command=reference.new_zeros((1,2,7))
        try:
            for frame in range(self.start,self.end):
                current_state=states[-1].copy()
                if observed_velocity:
                    current_state[:,3:6]=self.initial_observed_velocity if len(states)==1 else (
                        states[-1][:,:3]-states[-2][:,:3])/(self.sim.times[frame]-self.sim.times[frame-1])
                state=torch.tensor(normalize_state(current_state,self.origin,self.scale)[None],device=device)
                current=torch.tensor(poses[-1][None],device=device)
                nxt,_,command=policy_action_step(actor,state,geometry,target,current,theta,reference,command)
                nxt=nxt[0].cpu().numpy()
                self.set_controls(frame,np.stack((poses[-1],nxt)));check=self.sim.advance(frame+1)
                checks.append(check)
                if not check['valid']:raise ValueError('Invalid physical policy rollout: '+str(check))
                poses.append(nxt);states.append(self.capture())
        except (ValueError,RuntimeError) as exc:failure=str(exc)
        result=dict(valid=failure is None,error=failure,frames=len(states),planned_frames=self.horizon+1,
            elapsed_s=time.monotonic()-started,snapshot_sha256=self.snapshot_sha256,checks=checks,
            actor_observes_actual_surface_state=True,actor_internal_F_C_input=False,
            bounded_actions=getattr(actor,'bounded_actions',False))
        states=np.stack(states);poses=np.stack(poses)
        if output is not None:self.save(output,states,poses,result)
        return states,poses,result

    def tensors(self,device,observed_velocity=False):
        state=self.initial.copy()
        if observed_velocity:state[:,3:6]=self.initial_observed_velocity
        return (torch.tensor(normalize_state(state,self.origin,self.scale)[None],device=device),
                torch.tensor(self.pads[self.start][None],device=device),
                torch.tensor(self.theta[None],device=device),geometry_tensors(self.nodes,self.origin,device),
                torch.tensor(self.times[None],device=device,dtype=torch.float32))


def benchmark(root,checkpoint_path,obj,episode,output,horizon=32,steps=60,candidates=8,seed=0,device='cuda:0',material='first_fit',anchor=False,equal_budget_search=False):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    search_setting=None
    if equal_budget_search:
        from .shared_world_control_search import search_protocol
        if not anchor:raise ValueError('Prospective amplitude search uses the declared physical baseline anchor')
        search_setting=search_protocol(candidates,seed)
    lock=(output/'case.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'result.json').exists():raise ValueError('Control result is frozen')
    world,bundle=load_model(checkpoint_path,device)
    if obj in bundle['training_objects']:raise ValueError('Downstream control object was used to train world model')
    world.requires_grad_(False);physical=PhysicalControl(root,obj,episode,horizon,device,material)
    initial,pads0,theta,g,times=physical.tensors(device,world.direct_displacement)
    reference,record=physical.rollout(physical.pads[physical.start:physical.end+1],output/'reference')
    if not record['valid']:raise ValueError('Invalid reference goal')
    goal=torch.tensor((reference[-1,:,:3]-physical.origin)/physical.scale,device=device)[None]
    mask=physical.nodes['surface_mask'];target=reference[-1,mask,:3]
    def error(states):return float(np.linalg.norm(states[-1,mask,:3]-target,axis=-1).mean()*1000)
    hold_pads=np.repeat(physical.pads[physical.start][None],horizon+1,axis=0)
    hold,hrec=physical.rollout(hold_pads,output/'hold')
    case=dict(object=obj,episode=episode,start=physical.start,end=physical.end,horizon=horizon,seed=seed,
        material_source=physical.material_source,parameters=physical.parameters,snapshot_sha256=physical.snapshot_sha256,
        checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),checkpoint_step=bundle['step'],
        goal='Exact terminal surface nodes from recorded actions at the fixed current material estimate',
        gradient_planner_receives_reference_actions=False,initialization='stationary grippers',
        initial_state_information=('Actual current simulated positions and F; velocity from past positions; privileged mid-episode F initialization'
            if getattr(world,'deformation_memory',False) else
            'Actual current simulated positions; velocity from past positions' if world.direct_displacement else 'Actual current simulated x/v/F/C'),
        world_reads_internal_F_C=not world.kinematic_only,
        world_reads_F=getattr(world,'deformation_memory',False) or not world.kinematic_only,
        world_reads_C=not world.kinematic_only and not getattr(world,'deformation_memory',False),
        simulator_anchored_prediction=anchor,
        action_parameterization='Absolute offsets from initial gripper pose, with a fixed initial knot',
        total_translation_bound_m_per_axis=.012,total_rotation_bound_rad_per_axis=.08,total_aperture_bound_m=.006,
        task_selection='Interaction midpoint; requires a full window of constant valid gripper identities',
        independent_test_object=physical.item['split']=='test',prefix_seconds=physical.prefix_s,
        reference_seconds=record['elapsed_s'],model_training_seconds_amortized_separately=True,
        equal_budget_search_protocol=search_setting)
    write_json(output/'case.json',case)
    anchor_positions=None
    if anchor:
        if not hrec['valid']:raise ValueError('Simulator anchoring requires a valid baseline rollout')
        anchor_positions=torch.tensor((hold[...,:3]-physical.origin)/physical.scale,device=device)[None]
    best,history,planning_s=optimize_actions(world,initial,pads0,theta,g,times,goal,steps=steps,seed=seed,output=output,anchor=anchor_positions)
    proposed,prec=physical.rollout(best['pads'][0],output/'world_gradient')
    np.savez_compressed(output/'world_prediction.npz',states=best['states'],pads=best['pads'],actions=best['actions'],
        corrected_positions=best['corrected_positions'],raw=best['raw'])
    write_json(output/'optimization.json',dict(history=history,selected_step=best['step'],planning_seconds=planning_s))
    hold_error=error(hold) if hrec['valid'] else None;proposed_error=error(proposed) if prec['valid'] else None
    random_rows=[];rng=np.random.default_rng(seed);best_random=hold_error;random_time=0.
    for index in range(candidates):
        raw=torch.tensor(rng.normal(0,.25,(1,14,4)),device=device,dtype=torch.float32)
        with torch.no_grad():pads=bounded_pose_sequence(pads0,raw,horizon,g['scale_m'])[0][0].cpu().numpy()
        states,r=physical.rollout(pads,output/'physical_shooting'/f'candidate_{index:03d}')
        value=error(states) if r['valid'] else None;random_time+=r['elapsed_s']
        if value is not None and (best_random is None or value<best_random):best_random=value
        random_rows.append(dict(candidate=index,valid=r['valid'],terminal_mm=value,best_terminal_mm=best_random,
                                elapsed_seconds=random_time,physical_rollouts=index+1))
        write_json(output/'shooting_progress.json',dict(rows=random_rows))
    search=None
    if equal_budget_search:
        from .shared_world_control_search import compare_search
        search=compare_search(physical,pads0,g,best['raw'],proposed,prec,target,hold_error,
            random_rows,candidates,seed,output,planning_s,hrec['elapsed_s'])
    result=dict(**case,hold_terminal_mm=hold_error,world_gradient_terminal_mm=proposed_error,
        world_predicted_terminal_mm=best['predicted_terminal_mm'],physical_shooting_terminal_mm=best_random,
        world_improved_over_hold=bool(proposed_error is not None and hold_error is not None and proposed_error<hold_error),
        world_improved_over_shooting=bool(proposed_error is not None and best_random is not None and proposed_error<best_random),
        world_planning_seconds=planning_s,world_validation_seconds=prec['elapsed_s'],
        world_total_online_seconds=planning_s+prec['elapsed_s']+(hrec['elapsed_s'] if anchor else 0),shooting_seconds=random_time,
        baseline_seconds=hrec['elapsed_s'],shooting_total_online_seconds=random_time+hrec['elapsed_s'],
        world_physical_rollouts_including_anchor=1+int(anchor),shooting_physical_rollouts_including_baseline=candidates+1,
        world_physical_candidate_rollouts=1,shooting_physical_candidate_rollouts=candidates,
        shooting_trace=random_rows,world_rollout_valid=prec['valid'],hold_rollout_valid=hrec['valid'],
        physical_state_oracle=True,real_robot_deployment_test=False,equal_budget_search=search,completed_at=time.time())
    write_json(output/'result.json',result);lock.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--object',required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--steps',type=int,default=60)
    p.add_argument('--candidates',type=int,default=8);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--material',choices=['first_fit','baseline'],default='first_fit')
    p.add_argument('--anchor',action='store_true')
    p.add_argument('--equal-budget-search',action='store_true')
    a=p.parse_args();benchmark(a.root,a.checkpoint,a.object,a.episode,a.output,a.horizon,a.steps,a.candidates,a.seed,a.device,a.material,a.anchor,a.equal_budget_search)


if __name__=='__main__':main()
