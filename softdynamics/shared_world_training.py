"""Shared-world training and held-out prediction evaluation.

Only ordinary multi-step physical-state losses are optimized. Data may arrive
from the simulator while training, as in a replay buffer. Every checkpoint
records exactly which object/trajectory data were used. Test objects are never
used for fitting statistics, gradient updates or checkpoint selection.
"""
import argparse
from collections import defaultdict
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from .io import load_npz,read_json,sha256,write_json
from .parameters import PARAMETERS
from .shared_world_model import SharedWorldModel,rollout,dynamics_loss,DT_REF
from .world_model import normalize_state


class Replay:
    def __init__(self,root,splits,observed_velocity=False):
        self.root=Path(root);self.plan=read_json(self.root/'plan.json')
        self.objects={o['object']:o for o in self.plan['objects']}
        self.splits=set(splits);self.trajectories={};self.geometry={}
        self.failures={}
        self.observed_velocity=observed_velocity

    def refresh(self):
        count=0
        entries=[(task,self.root/'trajectories'/task['id']) for task in self.plan['tasks']]
        retry_path=self.root/'retry_index.json';retries=read_json(retry_path) if retry_path.exists() else {}
        for index,(task,folder) in enumerate(entries):
            if task['id'] not in retries:continue
            retry=retries[task['id']]
            if sha256(folder/'complete.json')!=retry['original_failure_sha256']:raise ValueError('Changed original retry failure')
            repaired=self.root/retry['folder']
            if sha256(repaired/'complete.json')!=retry['complete_sha256']:raise ValueError('Changed successful numerical retry')
            entries[index]=(task,repaired);self.failures.pop(task['id'],None)
        for path in sorted((self.root/'experience').glob('*/*/*/*/complete.json')):
            record=read_json(path)
            if record.get('schema') not in ('softdynamics.policy-experience.v1','softdynamics.control-experience.v1',
                                           'softdynamics.material-experience.v1'):continue
            task=record['task'];obj=task['object']
            if self.objects[obj]['split']!='train' or task['split']!='train':raise ValueError('Held-out object in policy replay')
            if record['source_data_plan_sha256']!=sha256(self.root/'plan.json'):raise ValueError('Policy experience uses another data plan')
            if record['nodes_sha256']!=self.objects[obj]['nodes_sha256']:raise ValueError('Policy experience uses another node mapping')
            entries.append((task,path.parent))
        for task,folder in entries:
            if task['split'] not in self.splits or task['id'] in self.trajectories or task['id'] in self.failures:
                continue
            record=folder/'complete.json'
            if not record.exists():continue
            r=read_json(record)
            if not r['valid']:
                self.failures[task['id']]=r.get('error','invalid');continue
            if not r['full_rollout']:raise ValueError('Truncated simulation cannot enter replay')
            if sha256(folder/'trajectory.npz')!=r['trajectory_sha256']:raise ValueError('Changed trajectory')
            obj=task['object'];info=self.objects[obj]
            geometry_key=obj
            if r.get('episode_preparation'):
                info=dict(nodes=r['nodes'],nodes_sha256=r['nodes_sha256'])
                geometry_key=info['nodes']
            if geometry_key not in self.geometry:
                if sha256(info['nodes'])!=info['nodes_sha256']:raise ValueError('Changed material nodes')
                self.geometry[geometry_key]=load_npz(info['nodes'])
            t=load_npz(folder/'trajectory.npz');scale=float(t['scale_m'])
            t['state']=normalize_state(t['state'],t['origin_m'],scale)
            t['minimum_start']=0
            if self.observed_velocity:
                t['state'][1:,:,3:6]=np.diff(t['state'][...,:3],axis=0)/(np.diff(t['times'])[:,None,None]/DT_REF)
                t['state'][0,:,3:6]=0
                if 'initial_observed_velocity_m_s' in t:
                    t['state'][0,:,3:6]=t['initial_observed_velocity_m_s']*DT_REF/scale
                elif r.get('trajectory_kind')=='complete_bounded_policy_episode':
                    # Never derive the initial velocity from a future frame.
                    t['minimum_start']=1
            t.update(task=task,id=task['id'],split=task['split'],record=r,geometry_key=geometry_key,
                     ground_y=float((-.01-t['origin_m'][1])/scale))
            self.trajectories[task['id']]=t;count+=1
        return count

    def object_names(self,split=None):
        return sorted({t['task']['object'] for t in self.trajectories.values() if split is None or t['split']==split})

    def sample(self,count,horizon,rng,include_prefix=None,exclude_prefix=None,include_ids=None,exclude_ids=None):
        by_object=defaultdict(list)
        for identity,t in self.trajectories.items():
            if include_prefix is not None and not identity.startswith(include_prefix):continue
            if exclude_prefix is not None and identity.startswith(exclude_prefix):continue
            if include_ids is not None and identity not in include_ids:continue
            if exclude_ids is not None and identity in exclude_ids:continue
            if len(t['state'])>horizon+t['minimum_start']:by_object[t['task']['object']].append(identity)
        objects=sorted(by_object)
        if count and not objects:raise ValueError('No eligible trajectories in the requested replay group')
        result=[]
        for _ in range(count):
            obj=objects[int(rng.integers(len(objects)))];identity=rng.choice(by_object[obj]);t=self.trajectories[identity]
            phase=int(rng.integers(3));starts=np.flatnonzero(t['phase'][:len(t['state'])-horizon]==phase)
            starts=starts[starts>=t['minimum_start']]
            if not len(starts):starts=np.arange(t['minimum_start'],len(t['state'])-horizon)
            result.append((str(identity),int(rng.choice(starts))))
        return result

    def batch(self,selection,horizon,device):
        values=defaultdict(list);geo=defaultdict(list)
        for identity,start in selection:
            t=self.trajectories[identity];nodes=self.geometry[t.get('geometry_key',t['task']['object'])];end=start+horizon+1
            if start<t['minimum_start']:raise ValueError('Window lacks a past-only initial velocity estimate')
            values['states'].append(t['state'][start:end]);values['pads'].append(t['pads'][start:end])
            values['times'].append(t['times'][start:end]-t['times'][start]);values['theta'].append(t['theta'])
            for k in ('canonical','neighbors','surface_mask'):geo[k].append(nodes[k])
            geo['ground_y'].append(t['ground_y']);geo['scale_m'].append(float(t['scale_m']))
        tensors={k:torch.as_tensor(np.stack(v),dtype=torch.float32,device=device) for k,v in values.items()}
        geometry={k:torch.as_tensor(np.stack(v),dtype=torch.long if k=='neighbors' else torch.bool if k=='surface_mask' else torch.float32,device=device) for k,v in geo.items()}
        return tensors,geometry


def sample_recent_mixture(replay,count,horizon,rng,prefix=None,fraction=0.,identities=None):
    if prefix is None and identities is None:
        if fraction:raise ValueError('A recent-data fraction requires an explicit training experience prefix')
        return replay.sample(count,horizon,rng)
    if (prefix is not None and (identities is not None or not prefix.startswith('experience/'))) or not 0<fraction<1:
        raise ValueError('Recent replay must mix training experience with historical trajectories')
    exact=count*fraction;recent=int(exact)
    if exact!=recent:recent+=int(rng.random()<exact-recent)
    choices=replay.sample(recent,horizon,rng,include_prefix=prefix,include_ids=identities)
    choices+=replay.sample(count-recent,horizon,rng,exclude_prefix=prefix,exclude_ids=identities)
    return choices


def fit_statistics(replay):
    if replay.splits!={'train'}:raise ValueError('Statistics require train-only replay')
    states=[];deltas=[]
    for t in replay.trajectories.values():
        # Deterministic bounded samples give every trajectory equal influence.
        ids=np.linspace(t['minimum_start'],len(t['state'])-2,min(64,len(t['state'])-1-t['minimum_start'])).astype(int)
        states.append(t['state'][ids].reshape(-1,24))
        ratio=np.diff(t['times'])[ids]/DT_REF
        deltas.append(((t['state'][ids+1,:,3:]-t['state'][ids,:,3:])/ratio[:,None,None]).reshape(-1,21))
    states=np.concatenate(states);deltas=np.concatenate(deltas)
    mean=states.mean(0,dtype=np.float64).astype('f')
    std=np.maximum(states.std(0,dtype=np.float64),1e-3).astype('f')
    output=np.sqrt(np.mean(deltas.astype('d')**2,axis=0)).astype('f')
    for lo,hi in ((0,3),(3,6),(6,15),(15,24)):
        std[lo:hi]=max(float(np.sqrt(np.mean(std[lo:hi]**2))),1e-3)
    for lo,hi in ((0,3),(3,12),(12,21)):
        output[lo:hi]=max(float(np.sqrt(np.mean(output[lo:hi]**2))),1e-4)
    return dict(state_mean=mean,state_std=std,output_scale=output,loss_scale=std,
                trajectories=sorted(replay.trajectories),objects=replay.object_names())


@torch.no_grad()
def validate(model,replay,selection,horizon,batch_size,device):
    per_object=defaultdict(list);model.eval()
    deformation_errors=[];minimum_J=float('inf')
    for start in range(0,len(selection),batch_size):
        choices=selection[start:start+batch_size];batch,g=replay.batch(choices,horizon,device)
        predicted=rollout(model,batch['states'][:,0],batch['pads'],batch['theta'],g,batch['times'])
        mm=torch.linalg.vector_norm(predicted[...,:3]-batch['states'][:,1:,:,:3],dim=-1)*g['scale_m'][:,None,None]*1000
        values=mm.mean((1,2)).cpu().numpy()
        if model.deformation_memory:
            error=predicted[...,6:15]-batch['states'][:,1:,:,6:15]
            deformation_errors.extend(torch.linalg.vector_norm(error,dim=-1).mean((1,2)).cpu().tolist())
            F=predicted[...,6:15].reshape(*predicted.shape[:-1],3,3)+torch.eye(3,device=predicted.device,dtype=predicted.dtype)
            minimum_J=min(minimum_J,float(torch.linalg.det(F).min()))
        for (identity,_),value in zip(choices,values):
            per_object[replay.trajectories[identity]['task']['object']].append(float(value))
    means={o:float(np.mean(v)) for o,v in per_object.items()}
    result=dict(mean_mm=float(np.mean(list(means.values()))),per_object_mm=means)
    if model.deformation_memory:
        result.update(deformation_error_frobenius=float(np.mean(deformation_errors)),minimum_predicted_J=minimum_J,
            deformation_start_source='True simulated window start; not observation-only state estimation',
            future_deformation_inputs=False)
    return result


def checkpoint(model,step,root,train,validation,statistics,setting,result):
    return dict(schema='softdynamics.shared-world-checkpoint.v1',model=model.state_dict(),
        architecture=model.architecture,step=step,data_root=str(Path(root).resolve()),
        data_plan_sha256=sha256(Path(root)/'plan.json'),training=setting,
        statistics=statistics,training_ids=sorted(train.trajectories),training_objects=train.object_names(),
        validation_ids=sorted(validation.trajectories),validation_objects=validation.object_names(),
        validation=result,object_specific_weights=False,response_supervision=False)


def train(root,output,device='cuda:0',steps=None,seed=0,resume=None,wait_for_data=False,horizon=None,batch_size=None,
          kinematic_only=False,memory_fraction=None,direct_displacement=False,contact_geometry=False,replay_manifest=None,
          initialize_from=None,learning_rate=None,allow_tf32=False,recent_prefix=None,recent_fraction=0.,validation_windows_path=None,recent_manifest=None,
          affine_material=False,first_step_loss_fraction=0.,residual_material=False,deformation_memory=False,
          physical_features=False,physical_feature_reference=None):
    root,output=Path(root).resolve(),Path(output).resolve();plan=read_json(root/'plan.json')
    if not 0<=first_step_loss_fraction<1:raise ValueError('Initial-transition loss fraction must be in [0,1)')
    if affine_material and residual_material:raise ValueError('Choose one material output representation')
    if deformation_memory and (not direct_displacement or kinematic_only or affine_material or residual_material):
        raise ValueError('Deformation memory requires the original observed-velocity model')
    if physical_features and (not direct_displacement or affine_material or residual_material or deformation_memory):
        raise ValueError('Physical features require the original observed-velocity model')
    if physical_feature_reference is not None and not physical_features:
        raise ValueError('A physical feature reference requires physical features')
    if residual_material and not (initialize_from or resume):raise ValueError('A frozen material residual requires a trained parent')
    setting=dict(plan['config']['training']);setting['seed']=seed
    setting['first_step_loss_fraction']=first_step_loss_fraction
    setting.setdefault('minimum_randomized_training_objects',2)
    if steps is not None:setting['steps']=steps
    if horizon is not None:setting['horizon']=horizon
    if batch_size is not None:setting['batch_size']=batch_size
    if learning_rate is not None:setting['learning_rate']=learning_rate
    if resume and initialize_from:raise ValueError('Choose either exact resume or a new fine-tuning stage')
    setting['allow_tf32']=allow_tf32
    if (recent_prefix is None and recent_manifest is None and recent_fraction) or (recent_prefix is not None and
        (not recent_prefix.startswith('experience/') or not 0<recent_fraction<1)):
        raise ValueError('Declare a training experience prefix and a mixture fraction strictly between zero and one')
    setting.update(recent_prefix=recent_prefix,recent_fraction=recent_fraction)
    recent_ids=None
    if recent_manifest is not None:
        if recent_prefix is not None or not 0<recent_fraction<1:raise ValueError('Use one recent replay selector and a nonzero mixture fraction')
        recent_data=read_json(recent_manifest)
        if recent_data['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Recent replay belongs to another data plan')
        recent_ids=set(recent_data['trajectories'])
        if not recent_ids or any(not i.startswith('experience/') for i in recent_ids):raise ValueError('Recent manifest must list ordinary training experience')
        setting.update(recent_manifest=str(Path(recent_manifest).resolve()),recent_manifest_sha256=sha256(recent_manifest))
    if validation_windows_path is not None:
        setting.update(validation_windows_path=str(Path(validation_windows_path).resolve()),
            validation_windows_sha256=sha256(validation_windows_path))
    kinematic_only=(kinematic_only or direct_displacement) and not deformation_memory
    setting['kinematic_only']=kinematic_only;setting['observed_velocity']=direct_displacement
    if deformation_memory:
        setting.update(deformation_memory=True,deformation_supervision='Ordinary simulated state F',
            short_window_deformation_start='Simulated window start; full-episode inference recurs from the initial asset',
            future_deformation_inputs=False,apic_C_inputs=False)
    setting['contact_geometry']=contact_geometry
    setting.update(physical_features=physical_features,physical_feature_reference=physical_feature_reference)
    if output.exists() and (output/'best.pt').exists() and resume is None:
        raise ValueError('Refusing to overwrite an existing trained experiment')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    if memory_fraction is not None:torch.cuda.set_per_process_memory_fraction(memory_fraction,device)
    torch.backends.cuda.matmul.allow_tf32=allow_tf32
    training=Replay(root,['train'],direct_displacement);validation=Replay(root,['validation'],direct_displacement)
    frozen=None
    if replay_manifest is not None:
        frozen=read_json(replay_manifest)
        if frozen['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Frozen replay uses a different data plan')
        setting['replay_manifest']=str(Path(replay_manifest).resolve())
        setting['replay_manifest_sha256']=sha256(replay_manifest)
    while True:
        training.refresh();validation.refresh()
        if recent_ids is not None:
            for identity in recent_ids:
                if identity not in training.trajectories:raise ValueError('Missing valid recent training trajectory: '+identity)
                if training.trajectories[identity]['record']['trajectory_sha256']!=recent_data['trajectories'][identity]:
                    raise ValueError('Changed recent training trajectory')
        if frozen is not None:
            for replay,split in [(training,'train'),(validation,'validation')]:
                for identity,digest in frozen[split].items():
                    if identity not in replay.trajectories:raise ValueError('Missing frozen replay trajectory: '+identity)
                    if replay.trajectories[identity]['record']['trajectory_sha256']!=digest:raise ValueError('Changed frozen replay trajectory')
                replay.trajectories={identity:replay.trajectories[identity] for identity in frozen[split]}
        randomized={t['task']['object'] for t in training.trajectories.values() if t['task']['variant']>0}
        ready=(len(training.object_names())>=setting['minimum_training_objects'] and
               len(validation.object_names())>=setting['minimum_validation_objects'] and
               len(randomized)>=setting['minimum_randomized_training_objects'])
        status=dict(pid=os.getpid(),stage='waiting_for_data',train_objects=len(training.object_names()),
            validation_objects=len(validation.object_names()),train_trajectories=len(training.trajectories),
            randomized_training_objects=len(randomized),updated_at=time.time())
        if ready:break
        if not wait_for_data:raise ValueError('Insufficient cross-object data: '+str(status))
        write_json(output/'status.json',status);time.sleep(10)
    architecture={**plan['config']['model'],'kinematic_only':kinematic_only,
        'direct_displacement':direct_displacement,'contact_geometry':contact_geometry}
    if deformation_memory:architecture['deformation_memory']=True
    if physical_features:architecture['physical_features']=True
    if physical_feature_reference is not None:architecture['physical_feature_reference']=list(physical_feature_reference)
    if affine_material or residual_material:
        nominal={tuple(task['parameters'][key] for key in PARAMETERS) for task in plan['tasks']
            if task['split']=='train' and task['variant']==0}
        if len(nominal)!=1:raise ValueError('Affine material output requires one declared training reference material')
        architecture.update(material_reference=list(next(iter(nominal))))
        architecture['affine_material' if affine_material else 'residual_material']=True
        setting['material_output']='Affine in normalized log-E, nu, ground friction and decay; ordinary state supervision' if affine_material else 'Zero-reference material residual on the frozen parent; ordinary state supervision'
        if residual_material:setting['freeze_original_dynamics']=True
    initializer=None
    migrate_material_head=False
    migrate_material_residual=False
    migrate_deformation_memory=False
    migrate_physical_features=False
    if initialize_from:
        initializer=torch.load(initialize_from,map_location=device,weights_only=False)
        if initializer['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Fine-tuning requires the same physical data plan')
        if initializer['architecture']!=architecture:
            source_architecture={k:v for k,v in architecture.items() if k not in ('affine_material','material_reference','residual_material','deformation_memory','physical_features','physical_feature_reference')}
            if deformation_memory:source_architecture['kinematic_only']=True
            migrate_material_head=affine_material and initializer['architecture']==source_architecture
            migrate_material_residual=residual_material and initializer['architecture']==source_architecture
            migrate_deformation_memory=deformation_memory and initializer['architecture']==source_architecture
            migrate_physical_features=physical_features and initializer['architecture']==source_architecture
            if not (migrate_material_head or migrate_material_residual or migrate_deformation_memory or migrate_physical_features):raise ValueError('Fine-tuning cannot silently change the model architecture')
            setting['initialization_conversion']=('Preserve all parent weights and input gradients; add a zero-output encoder of geometry-derived physical features' if migrate_physical_features else
                'Preserve all parent weights and initial behavior; add zero-output F encoder and multiplicative F transition' if migrate_deformation_memory else
                'Preserve all parent weights; add zero-output material residual' if migrate_material_residual else
                'Copy original velocity head into intercept; zero four material fields; preserve predictions only at reference material')
        if any(training.objects[obj]['split']!='train' for obj in initializer['training_objects']):
            raise ValueError('Held-out object in initialization checkpoint')
        statistics=initializer['statistics']
        setting.update(initialize_from=str(Path(initialize_from).resolve()),initialization_sha256=sha256(initialize_from),
            initialization_step=initializer['step'],preserve_initialization_statistics=True,fixed_finetuning_horizon=True)
    else:statistics=fit_statistics(training)
    model=SharedWorldModel(**architecture).to(device);model.set_statistics(statistics)
    if initializer is not None:
        if migrate_material_head:model.initialize_affine_material(initializer['model'])
        elif migrate_material_residual:model.initialize_residual_material(initializer['model'])
        elif migrate_deformation_memory:model.initialize_deformation_memory(initializer['model'])
        elif migrate_physical_features:model.initialize_physical_features(initializer['model'])
        else:model.load_state_dict(initializer['model'])
    start_step=0;old=None
    if resume:
        old=torch.load(resume,map_location=device,weights_only=False)
        if old['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Resume requires the same data plan')
        if old['architecture'].get('kinematic_only',False)!=kinematic_only:raise ValueError('Resume cannot silently change recurrent state representation')
        if old['architecture'].get('direct_displacement',False)!=direct_displacement:raise ValueError('Resume cannot change velocity convention')
        if old['architecture'].get('deformation_memory',False)!=deformation_memory:raise ValueError('Resume cannot change deformation memory')
        if old['architecture'].get('contact_geometry',False)!=contact_geometry:raise ValueError('Resume cannot change pad contact representation')
        if (old['architecture'].get('physical_features',False)!=physical_features or
            old['architecture'].get('physical_feature_reference')!=architecture.get('physical_feature_reference')):
            raise ValueError('Resume cannot change physical material/state features')
        if (old['architecture'].get('affine_material',False)!=affine_material or
            old['architecture'].get('material_reference')!=architecture.get('material_reference')):
            raise ValueError('Resume cannot change material output representation')
        if (old['architecture'].get('residual_material',False)!=residual_material or
            old['training'].get('freeze_original_dynamics',False)!=residual_material):
            raise ValueError('Resume cannot change frozen material adaptation')
        if old['training'].get('replay_manifest_sha256')!=setting.get('replay_manifest_sha256'):raise ValueError('Resume cannot change frozen replay membership')
        if old['training'].get('allow_tf32',True)!=allow_tf32:raise ValueError('Exact resume must preserve the matrix multiplication precision')
        if old['training'].get('recent_prefix')!=recent_prefix or old['training'].get('recent_fraction',0.)!=recent_fraction:
            raise ValueError('Exact resume cannot change replay mixing')
        if old['training'].get('recent_manifest_sha256')!=setting.get('recent_manifest_sha256'):
            raise ValueError('Exact resume cannot change the recent experience manifest')
        if old['training'].get('validation_windows_sha256')!=setting.get('validation_windows_sha256'):
            raise ValueError('Exact resume cannot change frozen validation windows')
        if old['training'].get('first_step_loss_fraction',0.)!=first_step_loss_fraction:
            raise ValueError('Exact resume cannot change temporal loss weights')
        model.load_state_dict(old['model']);statistics=old['statistics'];start_step=old['step']
    if residual_material:model.train_material_residual_only()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=setting['learning_rate'],weight_decay=1e-5)
    rng=np.random.default_rng(seed);val_rng=np.random.default_rng(1000+seed)
    if old is not None:
        optimizer.load_state_dict(old['optimizer'])
        rng.bit_generator.state=old['sampling_rng']
        torch.set_rng_state(old['torch_rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in old['cuda_rng']])
    # Freeze checkpoint-selection windows, including their object membership.
    val_h=setting['validation_horizon'];selection=[]
    if old is not None:
        selection=old['validation_windows']
    elif validation_windows_path is not None:
        frozen_windows=read_json(validation_windows_path)
        if frozen_windows['horizon']!=val_h:raise ValueError('Validation horizon differs from the fixed comparison')
        selection=frozen_windows['selection']
        for identity,start in selection:
            if identity not in validation.trajectories:raise ValueError('Missing held-out validation trajectory')
            t=validation.trajectories[identity]
            if start<t['minimum_start'] or start+val_h>=len(t['state']):raise ValueError('Invalid fixed validation window')
    else:
        for obj in validation.object_names():
            ids=[i for i,t in validation.trajectories.items() if t['task']['object']==obj and len(t['state'])>val_h]
            if not ids:raise ValueError('Validation object has no sufficiently long full trajectory')
            for _ in range(setting['validation_windows_per_object']):
                identity=str(val_rng.choice(ids));T=len(validation.trajectories[identity]['state'])
                selection.append((identity,int(val_rng.integers(T-val_h))))
    write_json(output/'validation_windows.json',dict(horizon=val_h,selection=selection))
    source=Path(__file__).resolve().parent
    if not (output/'code/softdynamics').exists():
        shutil.copytree(source,output/'code/softdynamics',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(output/'manifest.json',dict(setting=setting,statistics=statistics,architecture=model.architecture,
        data_plan_sha256=sha256(root/'plan.json'),world_model_shared_across_objects=True,
        response_supervision=False,test_objects_used=False,
        code_sha256={p.name:sha256(p) for p in source.glob('*.py')}))
    best=old['best_validation_mm'] if old is not None else float('inf');started=time.monotonic()
    seen=set((old or initializer or {}).get('sampled_training_ids',[]))
    if initializer is not None:
        result=validate(model,validation,selection,val_h,setting['batch_size'],device)
        best=result['mean_mm'];ckpt=checkpoint(model,0,root,training,validation,statistics,setting,result)
        ckpt.update(optimizer=optimizer.state_dict(),sampling_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            best_validation_mm=best,validation_windows=selection,sampled_training_ids=sorted(seen))
        torch.save(ckpt,output/'initial.pt');torch.save(ckpt,output/'best.pt')
        write_json(output/'validation_0.json',dict(step=0,improved=True,**result))
        print('VALIDATION',0,result,flush=True)
    for step in range(start_step+1,setting['steps']+1):
        if step%setting['evaluate_every']==1 and frozen is None:training.refresh()
        horizon=1 if step<=100 else min(4,setting['horizon']) if step<=300 else min(8,setting['horizon']) if step<=1500 else min(16,setting['horizon']) if step<=3000 else setting['horizon']
        if initializer is not None:horizon=setting['horizon']
        actual_batch=min(setting['batch_size'],max(1,setting['batch_size']*8//horizon))
        choices=sample_recent_mixture(training,actual_batch,horizon,rng,recent_prefix,recent_fraction,recent_ids)
        seen.update(identity for identity,_ in choices)
        batch,g=training.batch(choices,horizon,device)
        model.train();optimizer.zero_grad(set_to_none=True)
        prediction=rollout(model,batch['states'][:,0],batch['pads'],batch['theta'],g,batch['times'])
        rollout_loss=dynamics_loss(prediction,batch['states'][:,1:],model,g['scale_m'])
        first_step_loss=None;loss=rollout_loss
        if first_step_loss_fraction:
            # Ordinary absolute state targets, emphasizing the observed start
            # where randomized-material examples share state and controls.
            first_step_loss=dynamics_loss(prediction[:,:1],batch['states'][:,1:2],model,g['scale_m'])
            loss=(1-first_step_loss_fraction)*rollout_loss+first_step_loss_fraction*first_step_loss
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite world-model prediction loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step%20==0 or step==1:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),horizon=horizon,
                     rollout_loss=float(rollout_loss.detach()),
                     first_step_loss=float(first_step_loss.detach()) if first_step_loss is not None else None,
                     actual_batch_size=actual_batch,
                     recent_examples=sum(identity in recent_ids for identity,_ in choices) if recent_ids is not None else
                         sum(identity.startswith(recent_prefix) for identity,_ in choices) if recent_prefix else 0,
                     elapsed_s=time.monotonic()-started,train_objects=len(training.object_names()),
                     trajectories=len(training.trajectories),updated_at=time.time())
            with (output/'history.jsonl').open('a') as f:
                import json;f.write(json.dumps(row)+'\n')
            write_json(output/'status.json',dict(pid=os.getpid(),stage='training',**row))
            print('TRAIN',step,round(row['loss'],5),round(row['elapsed_s'],1),flush=True)
        if step%setting['evaluate_every']==0 or step==setting['steps']:
            result=validate(model,validation,selection,val_h,setting['batch_size'],device)
            ckpt=checkpoint(model,step,root,training,validation,statistics,setting,result)
            improved=np.isfinite(result['mean_mm']) and result['mean_mm']<best
            if improved:best=result['mean_mm']
            ckpt.update(optimizer=optimizer.state_dict(),sampling_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                best_validation_mm=best,validation_windows=selection,sampled_training_ids=sorted(seen))
            torch.save(ckpt,output/'last.tmp.pt');(output/'last.tmp.pt').replace(output/'last.pt')
            if improved:
                torch.save(ckpt,output/'best.tmp.pt');(output/'best.tmp.pt').replace(output/'best.pt')
            write_json(output/f'validation_{step}.json',dict(step=step,improved=improved,**result))
            print('VALIDATION',step,result,flush=True)
    write_json(output/'training_complete.json',dict(step=setting['steps'],best_validation_mm=best,
        elapsed_s=time.monotonic()-started,train_objects=training.object_names(),
        trajectory_count=len(training.trajectories),all_planned_data_complete=(root/'collection_complete.json').exists()))
    write_json(output/'status.json',dict(pid=os.getpid(),stage='training_complete',updated_at=time.time()))
    lock.close()


def load_model(path,device):
    c=torch.load(path,map_location=device,weights_only=False)
    if 'allow_tf32' in c.get('training',{}):torch.backends.cuda.matmul.allow_tf32=c['training']['allow_tf32']
    model=SharedWorldModel(**c['architecture']).to(device);model.load_state_dict(c['model']);model.eval()
    return model,c


def prediction_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[('nominal' if row['variant']==0 else 'randomized',str(row['horizon']),row['method'])].append(row)
    summary=[]
    for (regime,horizon,method),group in groups.items():
        by_object=defaultdict(list)
        for row in group:by_object[row['object']].append(row)
        per_object={}
        for obj,values in by_object.items():
            finite=[r for r in values if np.isfinite(r['mean_mm']) and np.isfinite(r['endpoint_mm'])]
            per_object[obj]=dict(cases=len(values),finite_cases=len(finite),
                mean_mm=float(np.mean([r['mean_mm'] for r in finite])) if finite else None,
                endpoint_mm=float(np.mean([r['endpoint_mm'] for r in finite])) if finite else None)
        good=[v for v in per_object.values() if v['finite_cases']==v['cases']]
        summary.append(dict(regime=regime,horizon=horizon,method=method,objects=len(per_object),
            fully_finite_objects=len(good),per_object=per_object,
            object_macro_mean_mm=float(np.mean([r['mean_mm'] for r in good])) if len(good)==len(per_object) else None,
            object_macro_endpoint_mm=float(np.mean([r['endpoint_mm'] for r in good])) if len(good)==len(per_object) else None))
    return summary


@torch.no_grad()
def evaluate(root,checkpoint_path,output,split='validation',device='cuda:0',full=True):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'evaluation.json').exists():raise ValueError('Evaluation is immutable')
    model,c=load_model(checkpoint_path,device);replay=Replay(root,[split],model.direct_displacement);replay.refresh()
    if set(replay.object_names())&set(c['training_objects']):raise ValueError('Object leakage')
    rows=[]
    for identity,t in replay.trajectories.items():
        # Nominal episodes assess extrapolation across geometry; randomized
        # parameter/action samples are reported separately, never mixed silently.
        for horizon in replay.plan['config']['evaluation_horizons']:
            if len(t['state'])<=horizon:continue
            starts=np.unique(np.linspace(0,len(t['state'])-horizon-1,8).astype(int))
            for start in starts:
                b,g=replay.batch([(identity,int(start))],horizon,device)
                pred=rollout(model,b['states'][:,0],b['pads'],b['theta'],g,b['times'])
                truth=b['states'][:,1:,:,:3];initial=b['states'][:,0,:,:3]
                persistence=initial[:,None].expand_as(truth)
                velocity=initial[:,None]+b['states'][:,0,None,:,3:6]*(b['times'][:,1:,None,None]/DT_REF)
                for name,p in [('world',pred[...,:3]),('persistence',persistence),('constant_velocity',velocity)]:
                    error=torch.linalg.vector_norm(p-truth,dim=-1)*g['scale_m'][:,None,None]*1000
                    rows.append(dict(id=identity,object=t['task']['object'],variant=t['task']['variant'],
                        horizon=horizon,start=int(start),method=name,mean_mm=float(error.mean()),endpoint_mm=float(error[:,-1].mean())))
        if full and t['task']['variant']==0:
            T=len(t['state'])-1;b,g=replay.batch([(identity,0)],T,device)
            pred=rollout(model,b['states'][:,0],b['pads'],b['theta'],g,b['times'])
            error=torch.linalg.vector_norm(pred[...,:3]-b['states'][:,1:,:,:3],dim=-1)*g['scale_m'][:,None,None]*1000
            rows.append(dict(id=identity,object=t['task']['object'],variant=0,horizon='full',method='world',
                mean_mm=float(error.mean()),endpoint_mm=float(error[:,-1].mean()),finite=bool(torch.isfinite(pred).all())))
            np.savez_compressed(output/(identity.replace('/','__')+'.npz'),prediction=pred.cpu().numpy(),
                                origin_m=t['origin_m'],scale_m=t['scale_m'])
            initial=b['states'][:,0,:,:3];truth=b['states'][:,1:,:,:3]
            for name,baseline in [('persistence',initial[:,None].expand_as(truth)),
                ('constant_velocity',initial[:,None]+b['states'][:,0,None,:,3:6]*(b['times'][:,1:,None,None]/DT_REF))]:
                error=torch.linalg.vector_norm(baseline-truth,dim=-1)*g['scale_m'][:,None,None]*1000
                rows.append(dict(id=identity,object=t['task']['object'],variant=0,horizon='full',method=name,
                    mean_mm=float(error.mean()),endpoint_mm=float(error[:,-1].mean()),finite=bool(torch.isfinite(baseline).all())))
        write_json(output/'progress.json',dict(last_id=identity,rows=len(rows),updated_at=time.time()))
    write_json(output/'evaluation.json',dict(checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),
        split=split,objects=replay.object_names(),rows=rows,summary=prediction_summary(rows),failures=replay.failures,
        error_reference='Same material-node states from the unchanged full-resolution physical simulator',
        velocity_convention='backward_difference_of_sampled_positions' if model.direct_displacement else 'MPM_instantaneous_velocity',
        training_objects=c['training_objects'],test_objects_used_for_training=False))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('train');a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--device',default='cuda:0');a.add_argument('--steps',type=int);a.add_argument('--seed',type=int,default=0)
    a.add_argument('--resume');a.add_argument('--wait-for-data',action='store_true')
    a.add_argument('--horizon',type=int);a.add_argument('--batch-size',type=int)
    a.add_argument('--kinematic-only',action='store_true');a.add_argument('--memory-fraction',type=float)
    a.add_argument('--direct-displacement',action='store_true')
    a.add_argument('--contact-geometry',action='store_true');a.add_argument('--replay-manifest',type=Path)
    a.add_argument('--affine-material',action='store_true')
    a.add_argument('--residual-material',action='store_true')
    a.add_argument('--deformation-memory',action='store_true',help='Predict and recur F from ordinary physical-state supervision; C is not consumed')
    a.add_argument('--physical-features',action='store_true',help='Use geometry-derived stress and material/state descriptors; ordinary trajectory supervision only')
    a.add_argument('--physical-feature-reference',type=float,nargs=4,help='Use a fixed material only in the new descriptors for a matched geometry-feature control')
    a.add_argument('--first-step-loss-fraction',type=float,default=0.)
    a.add_argument('--initialize-from',type=Path);a.add_argument('--learning-rate',type=float)
    a.add_argument('--tf32',action='store_true',help='Explicitly use the legacy reduced-mantissa matrix multiplication mode')
    a.add_argument('--recent-prefix');a.add_argument('--recent-fraction',type=float,default=0.)
    a.add_argument('--recent-manifest',type=Path)
    a.add_argument('--validation-windows',type=Path)
    a=s.add_parser('evaluate');a.add_argument('--root',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--split',choices=['validation','test'],default='validation')
    a.add_argument('--device',default='cuda:0');a.add_argument('--no-full',action='store_true')
    a=p.parse_args()
    try:
        if a.mode=='train':train(a.root,a.output,a.device,a.steps,a.seed,a.resume,a.wait_for_data,a.horizon,a.batch_size,a.kinematic_only,a.memory_fraction,a.direct_displacement,a.contact_geometry,a.replay_manifest,a.initialize_from,a.learning_rate,a.tf32,a.recent_prefix,a.recent_fraction,a.validation_windows,a.recent_manifest,a.affine_material,a.first_step_loss_fraction,a.residual_material,a.deformation_memory,a.physical_features,a.physical_feature_reference)
        else:evaluate(a.root,a.checkpoint,a.output,a.split,a.device,not a.no_full)
    except Exception as exc:
        import traceback
        failure=dict(pid=os.getpid(),stage='failed',error=repr(exc),traceback=traceback.format_exc(),updated_at=time.time())
        write_json(a.output/'failure.json',failure);write_json(a.output/'status.json',failure)
        raise


if __name__=='__main__':main()
