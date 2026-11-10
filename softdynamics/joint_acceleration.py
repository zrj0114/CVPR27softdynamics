"""Low-physics-budget material optimization with the shared dual-head model.

Each round performs many differentiable imagined updates, validates one material
vector in the original simulator, and adds its support transitions to replay.
Test-query observations are read only after a support selection is frozen.
"""
import argparse
import copy
import fcntl
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .io import load_npz, read_json, sha256, write_json
from .parameters import PARAMETERS, ParameterSpace
from .recovery_data import trajectory_folder, find_entry
from .shared_world_model import physical_parameters, dynamics_loss, rollout, observation_loss, DT_REF
from .shared_world_inverse import RealEpisode, project_observations
from .joint_world import (load_joint, interaction_context, context_batch, ConfidenceData,
                          confidence_loss, save_joint)
from .world_model import normalize_state


def prepare_physics(run):
    run=Path(run);spec=read_json(run/'protocol.json');out=run/'physics'
    if (out/'plan.json').exists():return
    plan=read_json(Path(spec['fit_run'])/'plan.json')
    plan['objects']=[o for o in plan['objects'] if o['object'] in spec['test_objects']]
    for item in plan['objects']:
        for entry in item['episodes']:
            padding=spec.get('simulation_grid_padding',{}).get(item['object'],{}).get(str(entry['episode']),0)
            if padding:entry['grid_padding_cells']=padding
    if {o['object'] for o in plan['objects']}!=set(spec['test_objects']):raise ValueError('Missing test object')
    plan.update(schema='softdynamics.joint-acceleration-physics.v1',
        runtime=str(run/'code/runtime'),conditions=[dict(key='real',truth='real',noise_mm=0.,seed=19)],
        source_plan_sha256=sha256(Path(spec['fit_run'])/'plan.json'),data_plan_sha256=sha256(Path(spec['data_root'])/'plan.json'))
    write_json(out/'plan.json',plan)


def collect_physics(run,obj,episode,q,device='cuda:0',capture=True):
    """Original simulate(), with a read-only node observer attached in process."""
    from . import recovery_data as rd
    from .shared_world_observation import GPUObservation,padded_simulation_grid
    from .world_data import controller_features
    run=Path(run);root=run/'physics';spec=read_json(run/'protocol.json');plan=read_json(root/'plan.json')
    item=next(o for o in plan['objects'] if o['object']==obj)
    if capture and episode not in item['support']:raise ValueError('Query states cannot enter experience')
    folder=trajectory_folder(root,obj,episode,q);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'complete.json').exists():
            record=read_json(folder/'complete.json')
            if record['valid'] and capture and not (folder/'state.npz').exists():
                raise ValueError('Dense cache lacks required support state observation')
            return record
        # Publish completion only after dense scores and optional state replay
        # are ready. Two methods may request the same vector concurrently.
        pending=folder/'complete.pending.json'
        original_read,original_write=rd.read_json,rd.write_json
        def staged_write(path,value):
            return original_write(pending if Path(path)==folder/'complete.json' else path,value)
        def staged_read(path):
            return original_read(pending if Path(path)==folder/'complete.json' and pending.exists() else path)
        rd.read_json,rd.write_json=staged_read,staged_write
        data=read_json(Path(spec['data_root'])/'plan.json')
        info=next(o for o in data['objects'] if o['object']==obj)
        nodes=load_npz(info['nodes']);captured=[];simulators=[];original=rd.PadRuntime
        class ObservedRuntime(original):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                observer=GPUObservation(self,nodes['material_indices'])
                # Checking all particles on GPU retains the original numerical
                # rules; CPU shadow checks assert agreement at fixed intervals.
                observer.install_validation();self._observer=observer
                if capture:captured.append(observer.capture())
                simulators.append(self)
            def advance(self,*args,**kwargs):
                check=super().advance(*args,**kwargs)
                if capture and check['valid']:captured.append(self._observer.capture())
                return check
        rd.PadRuntime=ObservedRuntime
        started=time.monotonic()
        try:
            entry=find_entry(plan,obj,episode)
            padding=entry.get('grid_padding_cells',0)
            with padded_simulation_grid(padding,entry.get('grid_lower_cells')):
                record=rd.simulate(root,obj,episode,q,device)
            if record['valid'] and capture:
                sim=simulators[0]
                if len(captured)!=len(sim.times):raise ValueError('Truncated support experience')
                origin=load_npz(Path(find_entry(plan,obj,episode)['workspace'])/f'initial_states/episode_{episode}.npz')['positions_world_m'].mean(0)
                scale=float(nodes['scale_m'])
                np.savez_compressed(folder/'state.npz',state=np.stack(captured),times=sim.times,
                    pads=controller_features(sim,origin,scale),origin_m=origin,scale_m=scale,
                    theta=np.array([record['parameters'][k] for k in PARAMETERS],np.float32))
                write_json(folder/'state.json',dict(object=obj,episode=episode,q=q,frames=len(captured),
                    state_sha256=sha256(folder/'state.npz'),nodes_sha256=info['nodes_sha256'],
                    full_rollout=True,query_observations_used=False,physics_complete_sha256=sha256(pending)))
            rd.score_candidate(root,obj,episode,q)
            write_json(folder/'execution.json',dict(elapsed_s=time.monotonic()-started,captured=capture,
                valid=record['valid'],pid=os.getpid()))
            pending.replace(folder/'complete.json')
        finally:
            rd.PadRuntime=original
            rd.read_json,rd.write_json=original_read,original_write
        return record


def physical_result(run,obj,episodes,q,device,capture=True):
    run=Path(run);losses=[];records=[]
    targets=read_json(run/'protocol.json').get('target_parameters',PARAMETERS)
    if any(q[j]!=0 for j,name in enumerate(PARAMETERS) if name not in targets):
        raise ValueError('A fixed physical parameter was changed')
    for episode in episodes:
        folder=trajectory_folder(run/'physics',obj,episode,q)
        cached=(folder/'complete.json').exists() and (folder/'execution.json').exists()
        worker_pid=None
        if not cached:
            command=[sys.executable,'-m','softdynamics.joint_acceleration','physics','--run',str(run),
                     '--object',obj,'--episode',str(episode),'--q',*[str(float(v)) for v in q],
                     '--device',device]
            if not capture:command.append('--no-capture')
            log=run/'logs'/f'physics_{obj}_{episode}_{folder.name}.log'
            with log.open('a') as stream:
                worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);worker_pid=worker.pid
                code=worker.wait()
                if code:raise subprocess.CalledProcessError(code,command)
        record=read_json(folder/'complete.json')
        if capture and record['valid'] and not (folder/'state.json').exists():raise ValueError('Missing full support states')
        scores=read_json(folder/'scores.json')
        value=scores['conditions']['real']['objective_mm'] if record['valid'] else float('inf')
        losses.append(value)
        execution=read_json(folder/'execution.json')
        cached=cached or execution['pid']!=worker_pid
        records.append(dict(episode=episode,valid=record['valid'],cached=cached,
            isolated_execution_s=execution['elapsed_s'],physics_s=record['elapsed_s'],folder=str(folder)))
    return np.asarray(losses),records


def normalized_weights(values):
    if values.ndim!=2 or not len(values) or not 1<=values.shape[1]<=4 or not torch.isfinite(values).all() or (values<0).any():
        raise ValueError('Expected nonnegative finite confidence[interactions,parameters]')
    values=values.detach();sums=values.sum(0,keepdim=True)
    return torch.where(sums>0,values/sums.clamp_min(1e-12),torch.full_like(values,1/len(values)))


def weighted_parameter_gradient(losses, q, weights):
    columns=[torch.autograd.grad(loss,q,retain_graph=True)[0] for loss in losses]
    return (torch.stack(columns)*normalized_weights(weights)).sum(0)


def expand_active_coordinates(active,indices):
    """Differentiable embedding; inactive coordinates remain exactly zero."""
    if active.ndim!=1 or len(active)!=len(indices) or len(set(indices))!=len(indices):
        raise ValueError('Expected one coordinate per unique target parameter')
    index=torch.as_tensor(indices,device=active.device,dtype=torch.long)
    return torch.zeros(4,device=active.device,dtype=active.dtype).scatter(0,index,active)


class EndpointWindows:
    """Stratified short imagined windows, anchored to the latest support physics.

    Six endpoint observations estimate the three-phase objective for the inner
    loop. Acceptance and reporting always use every original scored frame.
    """
    def __init__(self,run,obj,episode,device):
        self.run=Path(run);self.obj=obj;self.episode=episode;self.device=device
        self.spec=read_json(self.run/'protocol.json')
        self.real=RealEpisode(self.spec['data_root'],obj,episode,device)
        self.horizon=self.spec['inverse']['window_horizon']
        frames=self.real.frames.cpu().numpy()
        frames=frames[frames>=self.horizon]
        phase=np.where(frames<self.real.meta['clip_start'],0,np.where(frames<=self.real.meta['clip_end'],1,2))
        endpoints=[];weights=[]
        for j,name in enumerate(['before','interaction','release']):
            eligible=frames[phase==j]
            if not len(eligible):raise ValueError('No observed endpoint in scored phase')
            ids=(np.linspace(.25,.75,self.spec['inverse']['windows_per_phase'])*(len(eligible)-1)).astype(int)
            endpoints.extend(eligible[ids]);weights.extend([self.real.config['phase_weights'][name]/len(ids)]*len(ids))
        self.endpoints=torch.tensor(endpoints,device=device)
        self.frame_weights=torch.tensor(weights,dtype=torch.float32,device=device)
        self.geo={k:v.expand(len(endpoints),*v.shape[1:]) for k,v in self.real.geometry.items()}
        self.indices=self.endpoints[:,None]-self.horizon+torch.arange(self.horizon+1,device=device)[None]
        prepared=read_json(Path(self.spec['pretrain'])/'objects'/obj/'prepared'/f'episode_{episode}'/'complete.json')
        visual=load_npz(prepared['real']);observations=load_npz(prepared['observation'])
        # Use the measured world tracks in the documented normalized frame.
        visual['positions']=((observations['tracks']-visual['origin_m'])/float(visual['scale_m'])).astype('f')
        self.confidence_context=interaction_context(visual,self.real.meta)
        self.surface_ids=torch.tensor(np.linspace(0,len(self.real.projection['surface_initial'])-1,256).astype(int),device=device)
        self.track_ids=torch.tensor(np.linspace(0,len(self.real.projection['track_initial'])-1,256).astype(int),device=device)
        frame_map={int(f):i for i,f in enumerate(self.real.frames.cpu().tolist())}
        self.observed_indices=torch.tensor([frame_map[int(f)] for f in endpoints],device=device)

    @torch.no_grad()
    def set_anchor(self,model,q,theta):
        folder=trajectory_folder(self.run/'physics',self.obj,self.episode,q)
        record=read_json(folder/'state.json')
        if record['state_sha256']!=sha256(folder/'state.npz'):raise ValueError('Changed online support experience')
        a=load_npz(folder/'state.npz');raw=a['state'].copy()
        raw[1:,:,3:6]=np.diff(raw[:,:,:3],axis=0)/np.diff(a['times'])[:,None,None]
        normalized=normalize_state(raw,self.real.origin,self.real.scale)
        self.reference=torch.tensor(normalized,dtype=torch.float32,device=self.device)[self.indices]
        material=theta[None].expand(len(self.indices),theta.numel());self.nominal=[]
        for t in range(self.horizon):
            f=self.indices[:,t]
            self.nominal.append(model(self.reference[:,t],self.real.pads[0,f],self.real.pads[0,f+1],
                material,self.geo,self.real.times[0,f+1]-self.real.times[0,f])[...,:6].detach())
        xyz=self.reference[:,-1,:,:3]*self.real.scale+torch.as_tensor(self.real.origin,device=self.device)
        self.projected_anchor=project_observations(xyz,self.real.projection)
        self.physical_anchor=tuple(torch.tensor(np.load(folder/(k+'.npy'))[self.endpoints.cpu()],device=self.device)
                                   for k in ['surface','tracks'])
        self.online_state=normalized;self.online_pads=a['pads'];self.online_times=a['times'];self.online_theta=a['theta']

    def loss(self,model,theta):
        state=self.reference[:,0].detach();material=theta[None].expand(len(self.indices),theta.numel())
        for t in range(self.horizon):
            f=self.indices[:,t]
            raw=checkpoint(model,state,self.real.pads[0,f],self.real.pads[0,f+1],material,self.geo,
                self.real.times[0,f+1]-self.real.times[0,f],use_reentrant=False,preserve_rng_state=False)
            state=torch.cat((self.reference[:,t+1,:,:6]+(raw[...,:6]-self.nominal[t]),raw[...,6:]),-1)
        xyz=state[...,:3]*self.real.scale+torch.as_tensor(self.real.origin,device=self.device)
        predicted=project_observations(xyz,self.real.projection)
        surface,tracks=[phys+(pred-base) for phys,pred,base in zip(self.physical_anchor,predicted,self.projected_anchor)]
        terms=[]
        for k,index in enumerate(self.observed_indices):
            count=int(self.real.counts[int(index)])
            real_ids=torch.linspace(0,count-1,min(count,256),device=self.device).long()
            visible=self.real.visible[index,self.track_ids]
            if not visible.any():raise ValueError('Sampled endpoint lacks visible tracks')
            terms.append(observation_loss(surface[k:k+1,self.surface_ids],self.real.surface[index:index+1,real_ids],
                tracks[k:k+1,self.track_ids],self.real.tracks[index:index+1,self.track_ids],visible[None],
                self.real.config['surface_weight'],self.real.config['track_weight'],self.real.config['huber_delta_mm'])[0])
        return (torch.stack(terms)*self.frame_weights).sum()


def adapt_support(model,windows,folders,confidence_data,tau,steps,rng,device):
    """Online ordinary transitions; retain labeled training interactions for heads.

    Every new folder is a support rollout. Query data has no path to this loss.
    Accepted and rejected valid candidate experiences are both retained.
    """
    records=[]
    for e,folder in folders:
        receipt=read_json(folder/'state.json')
        if e.episode!=receipt['episode'] or e.obj!=receipt['object']:raise ValueError('Wrong support experience')
        a=load_npz(folder/'state.npz');raw=a['state'].copy()
        raw[1:,:,3:6]=np.diff(raw[:,:,:3],axis=0)/np.diff(a['times'])[:,None,None]
        records.append((e,normalize_state(raw,e.real.origin,e.real.scale),a))
    model.requires_grad_(True);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=1e-5)
    for step in range(steps):
        e,state,a=records[int(rng.integers(len(records)))];horizon=8
        start=int(rng.integers(max(1,len(state)-horizon)))
        x=torch.tensor(state[start:start+horizon+1][None],device=device)
        pads=torch.tensor(a['pads'][start:start+horizon+1][None],device=device)
        times=torch.tensor(a['times'][start:start+horizon+1][None],dtype=torch.float32,device=device)
        theta=torch.tensor(a['theta'][None],device=device)
        prediction=rollout(model.world,x[:,0],pads,theta,e.real.geometry,times)
        loss=dynamics_loss(prediction,x[:,1:],model.world,e.real.geometry['scale_m'])
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if step%4==0:
            c=confidence_loss(model,confidence_data,confidence_data.sample(1,rng),device,tau);c.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    model.eval();model.requires_grad_(False)


def run_baseline(run,obj,device):
    spec=read_json(Path(run)/'physics/plan.json');item=next(o for o in spec['objects'] if o['object']==obj)
    losses,records=physical_result(run,obj,item['support'],[0.]*4,device)
    write_json(Path(run)/'baselines'/f'{obj}.json',dict(object=obj,support=item['support'],query=item['query'],
        losses_mm=losses,valid=bool(np.isfinite(losses).all()),records=records))
    if not np.isfinite(losses).all():raise ValueError('Baseline physics failed; retain failure for this object')


def run_inverse(run,obj,method,device):
    run=Path(run);p=read_json(run/'protocol.json');out=run/'inverse'/obj/method;out.mkdir(parents=True,exist_ok=True)
    baseline=read_json(run/'baselines'/f'{obj}.json')
    if not baseline['valid']:raise ValueError('Invalid baseline')
    if str(device).startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.35,torch.device(device))
    checkpoint_path=run/'training/best.pt';model,bundle=load_joint(checkpoint_path,device)
    indices=model.target_indices
    if tuple(p.get('target_parameters',PARAMETERS))!=model.confidence_parameters:
        raise ValueError('Optimizer targets differ from the confidence checkpoint')
    if obj in bundle['training_objects']:raise ValueError('Test object in joint pretraining')
    model.requires_grad_(False);space=ParameterSpace.from_config(read_json(run/'physics/plan.json')['config'])
    base=torch.tensor([space.base[k] for k in PARAMETERS],dtype=torch.float32,device=device)
    scales=torch.tensor(space.scales,dtype=torch.float32,device=device)
    windows=[EndpointWindows(run,obj,e,device) for e in baseline['support']]
    data=ConfidenceData(p['pretrain'],splits=('train',),allowed_objects=set(bundle['training_objects']))
    tau=np.asarray(bundle['protocol']['tau']);rng=np.random.default_rng(731)
    current=np.zeros(4);losses=np.array(baseline['losses_mm']);history=[];selections=[]
    radius=p['inverse']['trust_radius'];started=time.monotonic();model_seconds=0.;simulation_seconds=0.
    replay=[]
    for round_index in range(1,p['inverse']['physical_rounds']+1):
        proposal_started=time.monotonic()
        theta=physical_parameters(torch.tensor(current,dtype=torch.float32,device=device),base,scales)
        for e in windows:e.set_anchor(model.world,current.tolist(),theta)
        with torch.no_grad():
            raw_weights=model.confidence(context_batch([e.confidence_context for e in windows],device),theta[None].expand(len(windows),4))
        if method=='joint_uniform_adam':raw_weights=torch.ones_like(raw_weights)
        weights=normalized_weights(raw_weights)
        q=torch.tensor(current[indices],dtype=torch.float32,device=device,requires_grad=True)
        optimizer=torch.optim.Adam([q],lr=p['inverse']['adam_lr']);inner=[]
        lo=torch.tensor(np.maximum(-.025,current[indices]-radius),dtype=torch.float32,device=device)
        hi=torch.tensor(np.minimum(.025,current[indices]+radius),dtype=torch.float32,device=device)
        for step in range(p['inverse']['inner_steps']):
            gradient=torch.zeros_like(q);proxy=[]
            for i,e in enumerate(windows):
                loss=e.loss(model.world,physical_parameters(expand_active_coordinates(q,indices),base,scales))
                g=torch.autograd.grad(loss,q)[0]
                if not torch.isfinite(g).all() or not torch.isfinite(loss):raise FloatingPointError('Invalid imagined parameter gradient')
                gradient+=weights[i]*g;proxy.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True);q.grad=gradient
            torch.nn.utils.clip_grad_norm_([q],100.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():q.copy_(torch.maximum(lo,torch.minimum(hi,q)))
            inner.append(dict(step=step+1,q=expand_active_coordinates(q,indices).detach().cpu().tolist(),
                target_parameters=list(model.confidence_parameters),proxy_losses_mm=proxy,gradient=gradient.cpu().tolist()))
            write_json(out/'status.json',dict(pid=os.getpid(),round=round_index,inner_step=step+1,
                inner_total=p['inverse']['inner_steps'],phase='imagined_gradient_updates',updated_at=time.time()))
        candidate=np.clip(expand_active_coordinates(q,indices).detach().cpu().numpy().astype(float),-.025,.025)
        proposal_seconds=time.monotonic()-proposal_started;model_seconds+=proposal_seconds
        del q;gc.collect();torch.cuda.empty_cache()
        write_json(out/'status.json',dict(pid=os.getpid(),round=round_index,phase='full_support_physics',candidate_q=candidate,updated_at=time.time()))
        actual,records=physical_result(run,obj,baseline['support'],candidate.tolist(),device)
        simulation_seconds+=sum(r['isolated_execution_s'] for r in records)
        accepted=bool(np.isfinite(actual).all() and actual.mean()<losses.mean()-1e-9)
        before=current.copy()
        if accepted:current=candidate.copy();losses=actual.copy();radius=min(p['inverse']['trust_radius'],radius*1.2)
        else:radius=max(.0005,radius*.5)
        for e,r in zip(windows,records):
            if r['valid']:replay.append((e,Path(r['folder'])))
        adaptation_start=time.monotonic()
        if replay:adapt_support(model,windows,replay,data,tau,p['inverse']['online_update_steps'],rng,device)
        adaptation_seconds=time.monotonic()-adaptation_start;model_seconds+=adaptation_seconds
        row=dict(round=round_index,previous_q=before,candidate_q=candidate,selected_q=current,
            accepted=accepted,candidate_support_losses_mm=actual,selected_support_losses_mm=losses,
            weights=weights.cpu().tolist(),inner=inner,proposal_seconds=proposal_seconds,adaptation_seconds=adaptation_seconds,
            cumulative_model_seconds=model_seconds,cumulative_simulation_seconds=simulation_seconds,
            support_candidate_validations=round_index,support_episode_rollouts=round_index*len(windows),
            baseline_episode_rollouts=len(windows),records=records,wall_seconds=time.monotonic()-started)
        row['actual_new_episode_rollouts']=sum(not r['cached'] for h in history for r in h['records'])+sum(not r['cached'] for r in records)
        history.append(row);write_json(out/'history.json',history)
        if round_index in p['inverse']['report_budgets']:
            selection=dict(object=obj,method=method,budget=round_index,q=current,support_mean_mm=float(losses.mean()),
                model_seconds=model_seconds,simulation_seconds=simulation_seconds,wall_seconds=row['wall_seconds'],
                baseline_seconds=sum(r['isolated_execution_s'] for r in baseline['records']),
                query_observations_used=False,frozen_at=time.time(),checkpoint_sha256=sha256(checkpoint_path))
            path=out/f'selection_{round_index}.json';write_json(path,selection);selections.append(path)
        print('INVERSE_ROUND',obj,method,round_index,'accepted',accepted,'support_mm',float(losses.mean()),flush=True)
    # Freeze every support choice before reading query observations.
    del windows,model,data;gc.collect();torch.cuda.empty_cache()
    results=[]
    initial_query,_=physical_result(run,obj,baseline['query'],[0.]*4,device,capture=False)
    for path in selections:
        selection=read_json(path)
        query,records=physical_result(run,obj,baseline['query'],selection['q'],device,capture=False)
        results.append(dict(**selection,query_losses_mm=query,query_mean_mm=float(query.mean()),initial_query_mean_mm=float(initial_query.mean()),
            query_records=records,selection_sha256=sha256(path)))
    write_json(out/'complete.json',dict(status='complete',object=obj,method=method,results=results,
        query_excluded_from_fitting=True,online_adaptation='support-only private copy of one shared pretrained model'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','physics','baseline','inverse','cma'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--object');p.add_argument('--episode',type=int)
    p.add_argument('--q',type=float,nargs=4);p.add_argument('--device',default='cuda:0');p.add_argument('--no-capture',action='store_true')
    p.add_argument('--method',choices=['joint_learned_adam','joint_uniform_adam'],default='joint_learned_adam')
    a=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode=='prepare':prepare_physics(a.run)
    elif a.mode=='physics':collect_physics(a.run,a.object,a.episode,a.q,a.device,not a.no_capture)
    elif a.mode=='baseline':run_baseline(a.run,a.object,a.device)
    elif a.mode=='inverse':run_inverse(a.run,a.object,a.method,a.device)
    else:
        from .joint_acceleration_benchmark import run_cma
        run_cma(a.run,a.object,a.device)


if __name__=='__main__':main()
