"""Shared visual/action dynamics encoder with four independent confidence outputs.

Confidence uses observed interaction summaries. Dynamics consumes only the
current state and commanded pads. Offline sensitivity targets are reused with
their masks; no response loss, tactile input, or test-object pretraining.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import os
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from .io import load_npz, read_json, sha256, write_json
from .parameters import PARAMETERS, ParameterSpace
from .shared_world_model import SharedWorldModel, DT_REF, mlp, rollout, dynamics_loss
from .shared_world_training import Replay, validate


class JointWorld(nn.Module):
    def __init__(self, architecture, confidence_parameters=PARAMETERS):
        super().__init__()
        self.confidence_parameters=tuple(confidence_parameters)
        if not self.confidence_parameters or len(set(self.confidence_parameters))!=len(self.confidence_parameters):
            raise ValueError('Confidence parameters must be nonempty and unique')
        self.target_indices=[PARAMETERS.index(p) for p in self.confidence_parameters]
        self.world = SharedWorldModel(**architecture)
        hidden = architecture['hidden']
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(4*hidden), mlp(4*hidden, 128, 1)) for _ in self.confidence_parameters])

    def confidence(self, context, theta):
        B, T, N, _ = context['states'].shape
        def flatten(x):
            return x.reshape(B*T, *x.shape[2:])
        geometry = {k: flatten(v) for k, v in context['geometry'].items()}
        geometry['node_valid'] = flatten(context['valid'])
        h = self.world.encode(flatten(context['states']), flatten(context['pads0']),
            flatten(context['pads1']), theta[:, None].expand(B,T,self.world.material_dim).reshape(B*T,self.world.material_dim),
            geometry, context['dt'].reshape(-1)).reshape(B,T,N,-1)
        mask = context['valid'][...,None]
        mean = (h*mask).sum(2)/mask.sum(2).clamp_min(1)
        maximum = h.masked_fill(~mask, -torch.inf).amax(2)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.)
        pooled = torch.cat((mean, maximum), -1)
        frame_valid = context['valid'].any(2)
        if not frame_valid.any(1).all():
            raise ValueError('Confidence requires at least one valid observed frame per interaction')
        temporal_mean = (pooled*frame_valid[...,None]).sum(1)/frame_valid.sum(1,keepdim=True)
        temporal_max = pooled.masked_fill(~frame_valid[...,None], -torch.inf).amax(1)
        features = torch.cat((temporal_mean, temporal_max), -1)
        return torch.cat([head(features).sigmoid() for head in self.confidence_heads], -1)


def interaction_context(data, meta, points=128, samples_per_phase=2):
    """Phase-stratified observed frames; backward velocities, explicit masks.

    This representation is only passed to the confidence head. No future
    observed point positions are arguments of the dynamics transition.
    """
    times = np.asarray(data['times']); x = np.asarray(data['positions'])
    valid = np.asarray(data['valid'], bool) & np.isfinite(x).all(-1)
    if times.shape != (len(x),) or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('Confidence context requires finite increasing times')
    candidates = np.arange(1, len(times)-1)
    phases = np.where(candidates < meta['clip_start'], 0,
        np.where(candidates <= meta['clip_end'], 1, 2))
    frames = []; present = []
    for phase in range(3):
        available = candidates[phases == phase]
        if not len(available):
            if not meta.get('allow_missing_phases'):
                raise ValueError('Confidence context requires all three observed phases')
            if not len(candidates):raise ValueError('Confidence context requires at least three frames')
            frames.extend([int(candidates[0])]*samples_per_phase)
            present.extend([False]*samples_per_phase)
            continue
        frames.extend(available[(np.linspace(.25, .75, samples_per_phase)*(len(available)-1)).astype(int)])
        present.extend([True]*samples_per_phase)
    frames = np.asarray(frames)
    # Fixed initial geometry determines spatial samples, never future errors.
    from .data import fps
    initial_ids = np.flatnonzero(np.isfinite(x[0]).all(-1))
    ids = initial_ids[fps(x[0,initial_ids], min(points, len(initial_ids)))]
    canonical = x[0,ids].astype('f')
    neighbors = cKDTree(canonical).query(canonical,k=min(25,len(ids)))[1][:,1:]
    velocity = (x[frames][:,ids]-x[frames-1][:,ids])/(times[frames]-times[frames-1])[:,None,None]*DT_REF
    mask = valid[frames][:,ids] & valid[frames-1][:,ids] & np.asarray(present)[:,None]
    states = np.zeros((len(frames),len(ids),24),np.float32)
    states[...,:3] = np.where(mask[...,None], x[frames][:,ids], canonical[None])
    states[...,3:6] = np.where(mask[...,None],velocity,0.)
    scale = float(data['scale_m'])
    # The first module stores a flattened 3x4 matrix; dynamics stores t,R,valid.
    source_pads = data['actions'].reshape(len(times),4,13)
    matrices = source_pads[...,:12].reshape(len(times),4,3,4)
    pads = np.concatenate((matrices[...,3],matrices[...,:3].reshape(len(times),4,9),source_pads[...,12:]),-1)
    geo = dict(canonical=canonical,neighbors=neighbors,surface_mask=np.ones(len(ids),bool),
               scale_m=np.float32(scale),ground_y=np.float32((-.01-data['origin_m'][1])/scale))
    return dict(states=states,pads0=pads[frames],pads1=pads[frames+1],
        valid=mask,dt=(times[frames+1]-times[frames]).astype('f'),
        geometry={k:np.broadcast_to(v,(len(frames),*np.shape(v))).copy() for k,v in geo.items()})


def context_batch(contexts, device):
    result = {k:torch.as_tensor(np.stack([c[k] for c in contexts]),device=device)
              for k in ('states','pads0','pads1','valid','dt')}
    result['geometry'] = {k:torch.as_tensor(np.stack([c['geometry'][k] for c in contexts]),device=device)
                          for k in contexts[0]['geometry']}
    return result


class ConfidenceData:
    def __init__(self, pretrain, splits=('train','validation'), allowed_objects=None):
        self.root = Path(pretrain); self.config = read_json(self.root/'config.json')
        self.rows = []; self.contexts = {}; self.sources = {}
        manifest = self.root/'full_training_manifest.json'
        self.sources[str(manifest)] = sha256(manifest)
        for filename in read_json(manifest)['teachers']:
            row = read_json(filename)
            if row['split'] not in splits or (allowed_objects is not None and row['object'] not in allowed_objects):
                continue
            prepared = read_json(self.root/'objects'/row['object']/'prepared'/f"episode_{row['episode']}"/'complete.json')
            meta = read_json(prepared['metadata'])
            key = row['input']
            if key not in self.contexts:
                self.contexts[key] = interaction_context(load_npz(key),meta)
                self.sources[key] = sha256(key)
            self.sources[filename] = sha256(filename)
            for record in row['records']:
                path = Path(key).parent/record['path']
                if sha256(path) != record['sha256']:
                    raise ValueError('Changed offline sensitivity payload')
                a = load_npz(path)
                self.rows.append(dict(object=row['object'],episode=row['episode'],split=row['split'],
                    context=key,q=a['q'],S=a['S'],valid=a['valid'] & np.isfinite(a['S']) & (a['S']>=0)))
        self.indices = {s:np.array([i for i,r in enumerate(self.rows) if r['split']==s],int) for s in splits}
        train = [r for r in self.rows if r['split']=='train']
        self.tau = np.array([max(np.median([r['S'][j] for r in train if r['valid'][j]]),1e-8)
                            for j in range(4)]) if train else None
        self.space = ParameterSpace.from_config(self.config)

    def batch(self, ids, device, tau):
        rows = [self.rows[i] for i in ids]
        context = context_batch([self.contexts[r['context']] for r in rows],device)
        theta = torch.tensor([[self.space.parameters(r['q'])[p] for p in PARAMETERS] for r in rows],device=device,dtype=torch.float32)
        valid = np.stack([r['valid'] for r in rows])
        S = np.stack([r['S'] for r in rows]); target = np.where(valid,S/(S+tau),0.)
        return context,theta,torch.tensor(target,device=device,dtype=torch.float32),torch.tensor(valid,device=device)

    def sample(self,count,rng):
        groups = defaultdict(list)
        for i in self.indices['train']: groups[self.rows[i]['object']].append(i)
        objects = sorted(groups)
        return [int(rng.choice(groups[str(rng.choice(objects))])) for _ in range(count)]


def confidence_loss(model, data, ids, device, tau):
    c,theta,target,mask = data.batch(ids,device,tau)
    target,mask=target[:,model.target_indices],mask[:,model.target_indices]
    prediction = model.confidence(c,theta)
    if not mask.any(): raise ValueError('Confidence batch contains no reliable labels')
    return (prediction-target).square()[mask].mean()


@torch.no_grad()
def confidence_evaluation(model,data,split,device,tau,batch_size=4):
    errors = []; masks = []; targets = []
    for ids in np.array_split(data.indices[split],max(1,int(np.ceil(len(data.indices[split])/batch_size)))):
        if not len(ids): continue
        c,theta,target,mask = data.batch(ids,device,tau)
        target,mask=target[:,model.target_indices],mask[:,model.target_indices]
        errors.append((model.confidence(c,theta)-target).abs().cpu().numpy())
        masks.append(mask.cpu().numpy()); targets.append(target.cpu().numpy())
    error,mask,target = map(np.concatenate,(errors,masks,targets))
    mae = [float(error[mask[:,j],j].mean()) if mask[:,j].any() else None for j in range(len(model.target_indices))]
    return dict(parameters=list(model.confidence_parameters),mae=mae,mean_mae=float(np.mean([v for v in mae if v is not None])),counts=mask.sum(0).tolist())


def load_joint(path, device):
    bundle = torch.load(path,map_location='cpu',weights_only=False)
    model = JointWorld(bundle['architecture'],bundle.get('confidence_parameters',PARAMETERS)).to(device)
    model.load_state_dict(bundle['model']); model.eval()
    return model,bundle


def save_joint(path,model,protocol,step,metrics,optimizer=None):
    payload = dict(schema='softdynamics.joint-world.v1',model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        architecture=model.world.architecture,protocol=protocol,step=step,metrics=metrics,
        confidence_parameters=list(model.confidence_parameters),
        training_objects=protocol['training_objects'],validation_objects=protocol['validation_objects'])
    if optimizer is not None: payload['optimizer'] = optimizer.state_dict()
    path=Path(path); temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)


def train(run, device='cuda:0', steps=1200, warmup=120, seed=0):
    run=Path(run);p=read_json(run/'protocol.json');out=run/'training';out.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    if str(device).startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.35,torch.device(device))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    parent=torch.load(p['initialize_from'],map_location='cpu',weights_only=False)
    if sha256(p['initialize_from'])!=p['initialization_sha256']:raise ValueError('Changed parent world checkpoint')
    if set(parent['training_objects']) & set(p['test_objects']):raise ValueError('Test object in initialization')
    model=JointWorld(parent['architecture'],p.get('target_parameters',PARAMETERS)).to(device);model.world.load_state_dict(parent['model'])
    data=ConfidenceData(p['pretrain'],allowed_objects=set(p['training_objects']+p['validation_objects']))
    replay=Replay(p['data_root'],['train'],True);replay.refresh()
    val=Replay(p['data_root'],['validation'],True);val.refresh()
    frozen=read_json(p['replay_manifest'])
    for split,buffer in [('train',replay),('validation',val)]:
        buffer.trajectories={i:t for i,t in buffer.trajectories.items() if i in frozen[split]}
        for i,t in buffer.trajectories.items():
            if t['record']['trajectory_sha256']!=frozen[split][i]:
                raise ValueError('Changed frozen replay trajectory')
        if not buffer.trajectories:raise ValueError('Empty frozen replay')
    if set(replay.object_names()) & set(p['test_objects']):raise ValueError('Test object in replay')
    p.update(tau=data.tau.tolist(),training_ids=sorted(replay.trajectories),
             confidence_sources=data.sources,training_steps=steps,warmup_steps=warmup,seed=seed)
    write_json(out/'training_protocol.json',p)
    rng=np.random.default_rng(seed+20261030)
    validation_windows=val.sample(12,32,np.random.default_rng(1927))
    optimizer=torch.optim.AdamW([
        dict(params=model.world.parameters(),lr=1e-5),
        dict(params=model.confidence_heads.parameters(),lr=3e-4)],weight_decay=1e-4)
    model.world.requires_grad_(False)
    best=float('inf');history=[]
    initial_world=validate(model.world,val,validation_windows,32,2,device)
    write_json(out/'initial_validation.json',initial_world)
    for step in range(steps+warmup):
        model.train();optimizer.zero_grad(set_to_none=True)
        if step==warmup:model.world.requires_grad_(True)
        conf=confidence_loss(model,data,data.sample(2,rng),device,data.tau)
        conf.backward();dynamic_value=None
        if step>=warmup:
            choices=replay.sample(2,8,rng)
            batch,geo=replay.batch(choices,8,device)
            prediction=rollout(model.world,batch['states'][:,0],batch['pads'],batch['theta'],geo,batch['times'])
            dynamic=dynamics_loss(prediction,batch['states'][:,1:],model.world,geo['scale_m'])
            dynamic.backward();dynamic_value=float(dynamic.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%20==0:
            write_json(out/'status.json',dict(pid=os.getpid(),step=step+1,total=steps+warmup,
                phase='confidence_warmup' if step<warmup else 'joint_training',elapsed_s=time.monotonic()-started,updated_at=time.time()))
        if (step+1)%120==0 or step==steps+warmup-1:
            model.eval();cm=confidence_evaluation(model,data,'validation',device,data.tau)
            wm=validate(model.world,val,validation_windows,32,2,device)
            metric=cm['mean_mae']+0.1*wm['mean_mm']/max(initial_world['mean_mm'],.001)
            row=dict(step=step+1,confidence=cm,dynamics=wm,selection_metric=metric,
                training_confidence_loss=float(conf.detach()),training_dynamics_loss=dynamic_value,elapsed_s=time.monotonic()-started)
            history.append(row);save_joint(out/'last.pt',model,p,step+1,row,optimizer)
            # A warmup checkpoint is diagnostic; select only joint updates.
            if step>=warmup and metric<best:
                best=metric;save_joint(out/'best.pt',model,p,step+1,row)
            write_json(out/'history.json',history);print('JOINT_TRAIN',row,flush=True)
    write_json(out/'complete.json',dict(status='complete',checkpoint=str(out/'best.pt'),
        checkpoint_sha256=sha256(out/'best.pt'),steps=steps+warmup,elapsed_s=time.monotonic()-started,
        test_objects_excluded=True,query_observations_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--warmup',type=int,default=120);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();train(a.run,a.device,a.steps,a.warmup,a.seed)


if __name__=='__main__':main()
