"""Prediction and gradient diagnostics, never training labels."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .io import read_json, sha256, write_json
from .inference import PARAMETERS
from .shared_world_model import DT_REF, rollout, pad_contact_features, physical_parameters, observation_loss
from .shared_world_training import Replay, load_model
from .shared_world_inverse import RealEpisode, project_observations


@torch.no_grad()
def prediction(root, replay_manifest, checkpoints, output, device='cuda:0'):
    root, output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise ValueError('Diagnostics are immutable')
    torch.set_num_threads(2);frozen=read_json(replay_manifest)
    replay=Replay(root,['train','validation'],True);replay.refresh()
    datasets={split:list(frozen[split]) for split in ['train','validation']}
    datasets['new_exploration']=[k for k in replay.trajectories if k.startswith('experience/exploration_')]
    for split in ['train','validation']:
        for key,digest in frozen[split].items():
            if replay.trajectories[key]['record']['trajectory_sha256']!=digest:raise ValueError('Changed frozen data')
    cases=[]
    for group,identities in datasets.items():
        for identity in identities:
            t=replay.trajectories[identity]
            for horizon in [1,8,32]:
                last=len(t['state'])-horizon-1
                if last<t['minimum_start']:continue
                for start in np.unique(np.linspace(t['minimum_start'],last,3).astype(int)):
                    cases.append(dict(group=group,id=identity,start=int(start),horizon=horizon))
    write_json(output/'cases.json',dict(cases=cases,trajectories={k:replay.trajectories[k]['record']['trajectory_sha256']
        for ids in datasets.values() for k in ids},created_at=time.time()))
    rows=[];started=time.monotonic()
    for checkpoint_path in checkpoints:
        model,bundle=load_model(checkpoint_path,device);name=Path(checkpoint_path).parent.name
        if not model.direct_displacement:raise ValueError('This comparison requires observed-velocity models')
        for index,case in enumerate(cases):
            b,g=replay.batch([(case['id'],case['start'])],case['horizon'],device)
            truth=b['states'][:,1:,:,:3];initial=b['states'][:,0]
            free=rollout(model,initial,b['pads'],b['theta'],g,b['times'])[...,:3]
            forced=[];contact=[]
            for t in range(case['horizon']):
                dt=b['times'][:,t+1]-b['times'][:,t]
                forced.append(model(b['states'][:,t],b['pads'][:,t],b['pads'][:,t+1],b['theta'],g,dt)[...,:3])
                features=pad_contact_features(b['states'][:,t],b['pads'][:,t],b['pads'][:,t+1],g['scale_m'],dt)
                contact.append(features[...,15:17].amax((-1,-2))>.02)
            contact=torch.stack(contact,1);surface=g['surface_mask'][:,None].expand_as(contact)
            persistence=initial[:,None,:,:3].expand_as(truth)
            velocity=initial[:,None,:,:3]+initial[:,None,:,3:6]*(b['times'][:,1:,None,None]/DT_REF)
            for method,pred in [('rollout',free),('teacher_forced',torch.stack(forced,1)),('persistence',persistence),('constant_velocity',velocity)]:
                error=torch.linalg.vector_norm(pred-truth,dim=-1)*g['scale_m'][:,None,None]*1000
                rows.append(dict(**case,model=name,checkpoint_step=bundle['step'],method=method,
                    object=replay.trajectories[case['id']]['task']['object'],mean_mm=float(error.mean()),
                    endpoint_mm=float(error[:,-1].mean()),surface_mm=float(error[surface].mean()),
                    contact_mm=float(error[contact].mean()) if bool(contact.any()) else None,
                    contact_node_frames=int(contact.sum())))
            if index%20==0:write_json(output/'progress.json',dict(model=name,completed=index+1,total=len(cases),elapsed_s=time.monotonic()-started))
        del model,bundle
    groups=defaultdict(list)
    for row in rows:groups[(row['model'],row['group'],row['horizon'],row['method'])].append(row)
    summary=[]
    for key,values in groups.items():
        per_object=defaultdict(list)
        for row in values:per_object[row['object']].append(row)
        means={obj:{field:float(np.mean([r[field] for r in entries if r[field] is not None]))
            if any(r[field] is not None for r in entries) else None for field in ['mean_mm','surface_mm','contact_mm']}
            for obj,entries in per_object.items()}
        summary.append(dict(model=key[0],group=key[1],horizon=key[2],method=key[3],per_object=means,
            object_macro_mean_mm=float(np.mean([v['mean_mm'] for v in means.values()])),
            contact_objects=sum(v['contact_mm'] is not None for v in means.values())))
    write_json(output/'result.json',dict(rows=rows,summary=summary,
        checkpoint_hashes={str(p):sha256(p) for p in checkpoints},teacher_forcing_is_diagnostic_only=True,
        contact_region='Known signed-pad support feature >0.02 at either interval endpoint; fixed across compared predictions',
        elapsed_s=time.monotonic()-started))


def gradient(root,physical_root,checkpoint_path,output,obj='048-butter-sponge',episode=0,device='cuda:0',double_precision=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise ValueError('Gradient audit is immutable')
    torch.set_num_threads(2);model,bundle=load_model(checkpoint_path,device);model.requires_grad_(False)
    plan=read_json(Path(physical_root)/'plan.json');e=RealEpisode(root,obj,episode,device)
    dtype=torch.float64 if double_precision else torch.float32
    if double_precision:
        model.double()
        for key in ['initial','pads','times','surface','tracks']:setattr(e,key,getattr(e,key).double())
        e.geometry={k:v.double() if v.is_floating_point() else v for k,v in e.geometry.items()}
        e.projection={k:v.double() for k,v in e.projection.items()}
    base=torch.tensor([plan['config']['baseline'][k] for k in PARAMETERS],device=device,dtype=dtype)
    scales=torch.tensor(plan['config']['coordinate_scales'],device=device,dtype=dtype)
    q0=torch.tensor(plan['incumbent_q'],device=device,dtype=dtype)
    e.set_physical_anchor(model,physical_parameters(q0,base,scales),physical_root,plan['incumbent_q'])
    horizons=sorted({h for h in [1,8,16,32,64,128,256,int(e.frames.max())] if h in e.frames.tolist()})
    lookup={int(v):i for i,v in enumerate(e.frames.tolist())};direction=torch.tensor([.5,-.5,.5,-.5],device=device,dtype=dtype)
    def frame_loss(state,frame):
        index=lookup[frame]
        pos=state[0,:,:3][None]*e.scale+torch.as_tensor(e.origin,device=device)
        surface,tracks=project_observations(pos,e.projection)
        surface=e.anchor['physical_surface'][index:index+1]+(surface-e.anchor['world_surface'][index:index+1])
        tracks=e.anchor['physical_tracks'][index:index+1]+(tracks-e.anchor['world_tracks'][index:index+1])
        return observation_loss(surface,e.surface[index:index+1,:e.counts[index]],tracks,
            e.tracks[index:index+1],e.visible[index:index+1],surface_weight=e.config['surface_weight'],
            track_weight=e.config['track_weight'],huber_delta_mm=e.config['huber_delta_mm']).sum()
    q=q0.clone().requires_grad_(True);theta=physical_parameters(q,base,scales);state=e.initial;rows={}
    for t in range(max(horizons)):
        state=checkpoint(model,state,e.pads[:,t],e.pads[:,t+1],theta[None],e.geometry,e.times[:,t+1]-e.times[:,t],
            use_reentrant=False,preserve_rng_state=False)
        if t+1 in horizons:
            loss=frame_loss(state,t+1);grad=torch.autograd.grad(loss,q,retain_graph=True)[0]
            rows[t+1]=dict(horizon=t+1,endpoint_observation_loss_mm=float(loss.detach()),
                gradient=grad.detach().cpu().tolist(),gradient_norm=float(grad.norm()),
                directional_gradient=float(grad@direction),finite_difference={})
            write_json(output/'progress.json',dict(rows=list(rows.values())))
    del state,theta,loss,grad
    with torch.no_grad():
        for epsilon in ([.001,.0001,.000001] if double_precision else [.001,.0001]):
            values=[]
            for sign in [-1,1]:
                theta=physical_parameters(q0+sign*epsilon*direction,base,scales);state=e.initial;sample={}
                for t in range(max(horizons)):
                    state=model(state,e.pads[:,t],e.pads[:,t+1],theta[None],e.geometry,e.times[:,t+1]-e.times[:,t])
                    if t+1 in horizons:sample[t+1]=float(frame_loss(state,t+1))
                values.append(sample)
            for h in horizons:rows[h]['finite_difference'][str(epsilon)]=(values[1][h]-values[0][h])/(2*epsilon)
    write_json(output/'result.json',dict(object=obj,episode=episode,checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256(checkpoint_path),checkpoint_step=bundle['step'],rows=list(rows.values()),
        double_precision=double_precision,
        purpose='Diagnose horizon-dependent gradient conditioning; these values never enter training replay',
        material_coordinates='Same normalized four-vector as the first-module parameter optimizer'))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prediction');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--replay-manifest',type=Path,required=True);a.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--device',default='cuda:0')
    a=sub.add_parser('gradient');a.add_argument('--root',type=Path,required=True);a.add_argument('--physical-root',type=Path,required=True)
    a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--object',default='048-butter-sponge');a.add_argument('--episode',type=int,default=0)
    a.add_argument('--device',default='cuda:0');a.add_argument('--double-precision',action='store_true');a=p.parse_args()
    if a.mode=='prediction':prediction(a.root,a.replay_manifest,a.checkpoints,a.output,a.device)
    else:gradient(a.root,a.physical_root,a.checkpoint,a.output,a.object,a.episode,a.device,a.double_precision)


if __name__=='__main__':main()
