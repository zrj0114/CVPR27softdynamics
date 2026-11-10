"""From-scratch two-head training; only best.pt and latest.pt are persisted."""
import argparse
from collections import defaultdict
from datetime import timedelta
import fcntl
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from .elastic_data import ElasticSpace, PARAMETERS
from .io import load_npz, read_json, sha256, write_json
from .joint_world import JointWorld, interaction_context, context_batch
from .shared_world_model import dynamics_loss
from .shared_world_training import Replay, fit_statistics, validate


def horizon_for_step(step, total=40000):
    # 2k / 8k / 15k / 15k at the requested 40k budget.
    fraction=step/total
    return 1 if fraction<.05 else 8 if fraction<.25 else 16 if fraction<.625 else 32


class ConfidenceReplay:
    def __init__(self, root, split):
        self.root=Path(root);self.plan=read_json(self.root/'plan.json')
        self.split=split;self.rows=[];self.contexts={};self.loaded=set()
        self.space=ElasticSpace(self.plan['config'])

    def refresh(self):
        for task in self.plan['tasks']:
            if task['split']!=self.split or task['id'] in self.loaded:continue
            folder=self.root/'trajectories'/task['id'];receipt=folder/'complete.json'
            if not receipt.exists() or not read_json(receipt)['valid']:continue
            manifest=read_json(folder/'teacher_manifest.json')
            if manifest['parameters']!=list(PARAMETERS):raise ValueError('Teacher is not two-dimensional')
            if sha256(manifest['input'])!=manifest['input_sha256']:raise ValueError('Teacher input changed')
            context=interaction_context(load_npz(manifest['input']),read_json(manifest['metadata']))
            self.contexts[task['id']]=context
            for r in manifest['records']:
                path=Path(manifest['teacher_directory'])/r['path']
                if sha256(path)!=r['sha256']:raise ValueError('Teacher label changed')
                data=load_npz(path)
                if data['S'].shape!=(2,):raise ValueError('Unexpected confidence dimension')
                mask=data['valid'] & np.isfinite(data['S']) & (data['S']>=0)
                if not mask.any():continue
                self.rows.append(dict(object=task['object'],episode=task['episode'],
                    context=task['id'],q=data['q'],S=data['S'],valid=mask,
                    source=str(path),sha256=r['sha256']))
            self.loaded.add(task['id'])

    def calibrate(self):
        if self.split!='train':raise ValueError('Calibration must use training objects only')
        values=[[float(r['S'][j]) for r in self.rows if r['valid'][j]] for j in range(2)]
        if any(len(v)<8 for v in values):raise ValueError('Insufficient reliable two-parameter labels')
        return np.array([max(float(np.median(v)),1e-8) for v in values])

    def sample(self, count, rng):
        groups=defaultdict(list)
        for i,r in enumerate(self.rows):groups[r['object']].append(i)
        objects=sorted(groups)
        if not objects:raise ValueError('No confidence supervision')
        return [int(rng.choice(groups[str(rng.choice(objects))])) for _ in range(count)]

    def batch(self, ids, device, tau):
        rows=[self.rows[i] for i in ids]
        c=context_batch([self.contexts[r['context']] for r in rows],device)
        theta=torch.tensor([[self.space.parameters(r['q'])[k] for k in PARAMETERS] for r in rows],
                           dtype=torch.float32,device=device)
        mask=np.stack([r['valid'] for r in rows]);S=np.stack([r['S'] for r in rows])
        target=np.where(mask,S/(S+tau),0.)
        return c,theta,torch.tensor(target,dtype=torch.float32,device=device),torch.tensor(mask,device=device)


class TrainingJoint(JointWorld):
    def forward(self, batch, geo, confidence_batch):
        context,theta,target,mask=confidence_batch
        prediction=self.confidence(context,theta)
        conf=(prediction-target).square()[mask].mean()
        states=[];state=batch['states'][:,0]
        for t in range(batch['pads'].shape[1]-1):
            args=(state,batch['pads'][:,t],batch['pads'][:,t+1],batch['theta'],geo,
                  batch['times'][:,t+1]-batch['times'][:,t])
            state=checkpoint(self.world,*args,use_reentrant=False) if self.training else self.world(*args)
            states.append(state)
        predicted=torch.stack(states,1)
        dynamic=dynamics_loss(predicted,batch['states'][:,1:],self.world,geo['scale_m'])
        return dynamic,conf


@torch.no_grad()
def evaluate(model, replay, confidence, selection, confidence_ids, tau, device):
    model.eval()
    dynamics=validate(model.world,replay,selection,32,1,device)
    errors=defaultdict(list);pairs=defaultdict(list)
    for offset in range(0,len(confidence_ids),2):
        ids=confidence_ids[offset:offset+2]
        c,theta,target,mask=confidence.batch(ids,device,tau)
        pred=model.confidence(c,theta)
        for n,index in enumerate(ids):
            row=confidence.rows[index]
            for j in range(2):
                if mask[n,j]:
                    errors[(row['object'],j)].append(float(abs(pred[n,j]-target[n,j])))
                    pairs[(row['object'],j)].append((row['episode'],float(target[n,j]),float(pred[n,j])))
    mae=[float(np.mean([np.mean(v) for (o,k),v in errors.items() if k==j])) for j in range(2)]
    # Compare episodes after averaging over material workpoints; object-balanced.
    accuracy=[]
    for j in range(2):
        per_object=[]
        for (obj,k),values in pairs.items():
            if k!=j:continue
            episodes=defaultdict(list)
            for e,t,p in values:episodes[e].append((t,p))
            means=[np.mean(v,axis=0) for v in episodes.values()];correct=[]
            for a in range(len(means)):
                for b in range(a):
                    delta=means[a]-means[b]
                    if abs(delta[0])>=.02:correct.append(float(delta[0]*delta[1]>0))
            if correct:per_object.append(float(np.mean(correct)))
        accuracy.append(float(np.mean(per_object)) if per_object else None)
    return dict(dynamics=dynamics,confidence_mae=mae,
                mean_confidence_mae=float(np.mean(mae)),episode_pair_accuracy=accuracy)


def atomic_checkpoint(path,payload):
    path=Path(path)
    if path.name not in ('best.pt','latest.pt'):raise ValueError('Only best/latest model files are allowed')
    temp=path.with_name('.'+path.name+'.tmp')
    torch.save(payload,temp);os.replace(temp,path)


@torch.no_grad()
def clip_gradients(parameters, max_norm=1., loss_scale=1.):
    """Keep the usual clipping rule when finite FP32 gradients overflow its norm."""
    parameters=list(parameters)
    if loss_scale==1.:
        try:
            return torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)
        except RuntimeError as error:
            if 'total norm' not in str(error) or 'non-finite' not in str(error):raise
    grads=[p.grad for p in parameters if p.grad is not None]
    norms=torch.stack([torch.linalg.vector_norm(g.double()) for g in grads])
    total=torch.linalg.vector_norm(norms)/loss_scale
    if not torch.isfinite(total):raise RuntimeError('Gradient total norm is non-finite')
    coefficient=(max_norm/(total+1e-6)).clamp(max=1.)/loss_scale
    for gradient in grads:
        gradient.copy_((gradient.double()*coefficient).to(gradient.dtype))
    return total


def joint_gradients(wrapper, model, optimizer, batch, geo, confidence_batch, world_size=1):
    """Retry the same joint loss at a smaller backward scale if FP32 overflows.

    Full rollout, examples, objective and clipping threshold stay fixed. All
    ranks agree before any optimizer update; failed attempts never count as steps.
    """
    for loss_scale in (1.,2.**-16,2.**-32,2.**-64):
        optimizer.zero_grad(set_to_none=True)
        dynamic,conf=wrapper(batch,geo,confidence_batch);loss=dynamic+conf
        finite=torch.isfinite(loss).to(torch.int32)
        if world_size>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite:raise FloatingPointError('Nonfinite joint loss')
        (loss*loss_scale).backward()
        valid=torch.ones((),device=loss.device,dtype=torch.int32)
        try:gradient=clip_gradients(model.parameters(),1.,loss_scale)
        except RuntimeError as error:
            if 'non-finite' not in str(error):raise
            valid.zero_()
        if world_size>1:dist.all_reduce(valid,op=dist.ReduceOp.MIN)
        if valid:return dynamic,conf,gradient,loss_scale
    raise FloatingPointError('Nonfinite gradients persisted through same-batch backward scaling')


def train(root, steps=40000, resume=False, validation_interval=500, save_interval=100):
    root=Path(root);plan=read_json(root/'plan.json');out=root/'training';out.mkdir(exist_ok=True)
    rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'));device=f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank);torch.set_num_threads(2)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world_size>1:dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device(device))
    lock=None
    if rank==0:
        lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'latest.pt').exists() and not resume:raise ValueError('Existing training requires explicit resume')
    old=torch.load(out/'latest.pt',map_location='cpu',weights_only=False) if resume else None
    if old and (old['data_plan_sha256']!=sha256(root/'plan.json') or old['total_steps']!=steps):
        raise ValueError('Resume changes data or total step budget')
    replay=Replay(root,['train'],True);val=Replay(root,['validation'],True)
    replay.refresh();val.refresh()
    confidence=ConfidenceReplay(root,'train');vc=ConfidenceReplay(root,'validation')
    confidence.refresh();vc.refresh()
    initial_ids=set(read_json(root/'training_data_start.json')['ids'])
    training_ids=set(old['training_ids']) if old else initial_ids
    replay.trajectories={k:v for k,v in replay.trajectories.items() if k in training_ids}
    val.trajectories={k:v for k,v in val.trajectories.items() if k in initial_ids}
    confidence.rows=[r for r in confidence.rows if r['context'] in training_ids]
    confidence.loaded &= training_ids
    vc.rows=[r for r in vc.rows if r['context'] in initial_ids]
    if old:
        if set(replay.trajectories)!=training_ids:raise ValueError('Resume replay is incomplete')
        current={r['source']:r['sha256'] for r in confidence.rows}
        expected={r['source']:r['sha256'] for r in old['confidence_sources']}
        if current!=expected:raise ValueError('Resume confidence supervision changed')
    replay.trajectories=dict(sorted(replay.trajectories.items()))
    confidence.rows.sort(key=lambda r:r['source'])
    if set(replay.object_names()) & set(plan['excluded_objects']):raise ValueError('Test data leakage')
    statistics=old['statistics'] if old else fit_statistics(replay)
    tau=np.asarray(old['tau'] if old else confidence.calibrate())
    model=TrainingJoint(plan['config']['model'],PARAMETERS).to(device)
    model.world.set_statistics(statistics)
    if old:model.load_state_dict(old['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    if old:optimizer.load_state_dict(old['optimizer'])
    wrapper=DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False) if world_size>1 else model
    rng=np.random.default_rng(20261030+rank)
    start=int(old['step']) if old else 0;best=float(old['best_score']) if old else float('inf')
    if old and len(old['rng_states'])==world_size:
        state=old['rng_states'][rank];rng.bit_generator.state=state['numpy']
        torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'],device)
    if old:
        selection=[tuple(v) for v in old['validation_windows']]
        wanted=set(old['validation_confidence_sources'])
        confidence_ids=[i for i,r in enumerate(vc.rows) if r['source'] in wanted]
    else:
        selection=[]
        vrng=np.random.default_rng(712)
        for obj in val.object_names():
            ids={i for i,t in val.trajectories.items() if t['task']['object']==obj}
            selection+=val.sample(4,32,vrng,include_ids=ids)
        confidence_ids=list(range(min(192,len(vc.rows))))
    protocol=dict(parameters=list(PARAMETERS),total_steps=steps,initialization='random',
        old_weights_used=False,old_targets_used=False,both_heads_from_step_one=True,
        checkpoint_files=['best.pt','latest.pt'],world_size=world_size,
        dynamics_batch_per_rank=1,confidence_batch_per_rank=2,learning_rate=3e-4,
        validation_interval=validation_interval,latest_interval=save_interval,
        gradient_clipping=dict(max_norm=1.,overflow_fallback='float64 norm and scaling',
            backward_retry_scales=[1.,2.**-16,2.**-32,2.**-64],
            nonfinite_gradient_values='retry same batch at smaller backward scale; fail if persistent'),
        tau=tau.tolist(),tau_calibration_sources=(old['protocol']['tau_calibration_sources'] if old
            else [r['source'] for r in confidence.rows]),
        training_objects=replay.object_names(),validation_objects=val.object_names(),
        excluded_objects=plan['excluded_objects'],data_plan_sha256=sha256(root/'plan.json'),
        data_policy='Fresh trajectories stream in; train-only normalization and teacher calibration remain frozen',
        curriculum=[dict(end_step=int(steps*x),horizon=h) for x,h in ((.05,1),(.25,8),(.625,16),(1.,32))])
    if rank==0:write_json(out/'protocol.json',protocol)
    started=time.monotonic();last_metrics=old.get('metrics') if old else None
    norm_recoveries=int(old.get('gradient_norm_recoveries',0)) if old else 0
    for step in range(start,steps):
        if step>0 and step%500==0:
            ready=[None]
            if rank==0:
                ready[0]=[t['id'] for t in plan['tasks'] if t['split']=='train'
                    and (root/'trajectories'/t['id']/'complete.json').exists()
                    and read_json(root/'trajectories'/t['id']/'complete.json')['valid']]
            if world_size>1:dist.broadcast_object_list(ready,src=0)
            allowed=set(ready[0])
            replay.refresh();confidence.refresh()
            replay.trajectories={k:v for k,v in replay.trajectories.items() if k in allowed}
            confidence.rows=[r for r in confidence.rows if r['context'] in allowed]
            confidence.loaded &= allowed
            replay.trajectories=dict(sorted(replay.trajectories.items()))
            confidence.rows.sort(key=lambda r:r['source'])
        horizon=horizon_for_step(step,steps)
        choices=replay.sample(1,horizon,rng);batch,geo=replay.batch(choices,horizon,device)
        cb=confidence.batch(confidence.sample(2,rng),device,tau)
        model.train()
        lr=(3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*step/steps)))*min(1.,(step+1)/500.)
        for group in optimizer.param_groups:group['lr']=lr
        try:
            dynamic,conf,gradient,backward_scale=joint_gradients(wrapper,model,optimizer,batch,geo,cb,world_size)
        except FloatingPointError as error:
            write_json(out/f'numerical_failure_rank_{rank}.json',dict(step=step+1,
                choices=choices,error=str(error),created_at=time.time()))
            raise
        if gradient.dtype==torch.float64:
            norm_recoveries+=1
            if rank==0:
                recovery=dict(step=step+1,norm_float64=float(gradient),
                    backward_scale=backward_scale,
                    reason=('Float32 norm overflow' if backward_scale==1. else 'Same-batch backward overflow recovery'),
                    created_at=time.time())
                with (out/'numerical_recoveries.jsonl').open('a') as stream:
                    stream.write(__import__('json').dumps(recovery)+'\n')
                print('ELASTIC_GRADIENT_NORM_RECOVERED',recovery,flush=True)
        optimizer.step()
        if (step+1)%20==0 or step==start:
            values=torch.stack((dynamic.detach(),conf.detach()))
            if world_size>1:dist.all_reduce(values);values/=world_size
            if rank==0:
                status=dict(pid=os.getpid(),step=step+1,total_steps=steps,horizon=horizon,
                    dynamics_loss=float(values[0]),confidence_loss=float(values[1]),gradient_norm=float(gradient),
                    learning_rate=lr,world_size=world_size,elapsed_s=time.monotonic()-started,
                    gradient_norm_recoveries=norm_recoveries,
                    training_trajectories=len(replay.trajectories),confidence_labels=len(confidence.rows),updated_at=time.time())
                write_json(out/'status.json',status);print('ELASTIC_TRAIN',status,flush=True)
        validation_due=(step+1)%validation_interval==0 or step==steps-1
        save_due=(step+1)%save_interval==0 or validation_due
        improved=False
        if validation_due:
            if world_size>1:dist.barrier()
            if rank==0:
                last_metrics=evaluate(model,val,vc,selection,confidence_ids,tau,device)
                # Fixed scale across all curriculum stages; both tasks affect selection.
                score=last_metrics['dynamics']['mean_mm']/3.+last_metrics['mean_confidence_mae']/.1
                improved=score<best
                if improved:best=score
                row=dict(step=step+1,score=score,best_score=best,**last_metrics)
                with (out/'metrics.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
                print('ELASTIC_VALIDATE',row,flush=True)
            if world_size>1:dist.barrier()
        if save_due:
            local=dict(numpy=rng.bit_generator.state,torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
            states=[None]*world_size
            if world_size>1:dist.all_gather_object(states,local)
            else:states=[local]
            if rank==0:
                payload=dict(schema='softdynamics.elastic-joint-checkpoint.v1',
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    architecture=model.world.architecture,confidence_parameters=list(PARAMETERS),
                    optimizer=optimizer.state_dict(),step=step+1,total_steps=steps,best_score=best,
                    gradient_norm_recoveries=norm_recoveries,
                    statistics=statistics,tau=tau.tolist(),rng_states=states,protocol=protocol,
                    metrics=last_metrics,training_ids=sorted(replay.trajectories),
                    confidence_sources=[dict(source=r['source'],sha256=r['sha256']) for r in confidence.rows],
                    validation_windows=selection,validation_confidence_sources=[vc.rows[i]['source'] for i in confidence_ids],
                    data_plan_sha256=sha256(root/'plan.json'))
                atomic_checkpoint(out/'latest.pt',payload)
                if improved:atomic_checkpoint(out/'best.pt',payload)
        if world_size>1:dist.barrier()
    if rank==0:write_json(out/'complete.json',dict(status='complete',steps=steps,finished_at=time.time(),
        best=str(out/'best.pt'),latest=str(out/'latest.pt'),best_score=best))
    if world_size>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--resume',action='store_true')
    p.add_argument('--validation-interval',type=int,default=500);p.add_argument('--save-interval',type=int,default=100)
    a=p.parse_args();train(a.root,a.steps,a.resume,a.validation_interval,a.save_interval)


if __name__=='__main__':main()
