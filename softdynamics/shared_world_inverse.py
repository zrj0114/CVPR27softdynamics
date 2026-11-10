"""Material fitting through a frozen shared world model and the original observations.

The confidence network supplies four independent frozen interaction columns.
Recorded actions, geometry, observation bindings and world-model weights stay
fixed. Only material coordinates are updated by the real-observation objective.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .io import load_npz,read_json,sha256,write_json
from .inference import ConfidenceModel,PARAMETERS
from .shared_world_model import observation_loss,physical_parameters,recurrent_channels
from .shared_world_training import load_model
from .world_model import normalize_state
from .teacher_contact import load_completed_actions,pack_pads,require_selected_architecture


def export_confidences(root):
    root=Path(root);plan=read_json(root/'plan.json')
    if sha256(plan['confidence_checkpoint'])!=plan['confidence_sha256']:
        raise ValueError('Changed frozen first-module checkpoint')
    torch.set_num_threads(2);model=None;results={}
    for obj,handoff in plan['first_module_handoffs'].items():
        path=root/'first_module'/f'{obj}.json'
        if path.exists():
            result=read_json(path)
            if result['model_sha256']!=plan['confidence_sha256']:raise ValueError('Changed confidence export')
            results[obj]=result;continue
        if handoff['weights'] is not None:
            source=Path(handoff['weights'])
            if sha256(source)!=plan['source_sha256'][str(source)]:raise ValueError('Changed old confidence weights')
            record=read_json(source);context=next(c for c in record['contexts'] if c['id']=='real_start0')
            values=np.asarray(context['weights']['learned']);initial_q=context['start']
            provenance=dict(kind='existing_first_module_export',path=str(source),sha256=sha256(source))
        else:
            if model is None:model=ConfidenceModel(plan['confidence_checkpoint'],'cpu')
            if obj in model.bundle['training_objects']:raise ValueError('Downstream object leaked into first-module training')
            values=[];sources={};initial_q=[0.,0.,0.,0.]
            for episode in handoff['support']:
                row_path=Path(plan['pretrain'])/'objects'/obj/'prepared'/f'episode_{episode}'/'complete.json'
                row=read_json(row_path);data=load_npz(row['real']);obs=load_npz(row['observation'])
                data['positions']=((obs['tracks']-data['origin_m'])/float(data['scale_m'])).astype('f')
                prediction=model.predict(data,model.bundle['base'].numpy())
                if not prediction['within_training_query_box']:raise ValueError('Outside confidence query domain')
                values.append(prediction['confidence']);sources[row['real']]=sha256(row['real'])
                sources[row['observation']]=sha256(row['observation'])
            values=np.stack(values)
            provenance=dict(kind='new_prediction_from_frozen_checkpoint',sources=sources)
        if values.shape!=(len(handoff['support']),4) or not np.isfinite(values).all() or np.any(values<0):
            raise ValueError('Invalid independent confidence columns')
        result=dict(object=obj,support=handoff['support'],query=handoff['query'],
            confidence_weights=values,initial_q=initial_q,selected_q=handoff.get('selected_q'),
            model_sha256=plan['confidence_sha256'],provenance=provenance,
            teacher_labels_used=False,query_observations_used=False,exported_at=time.time())
        write_json(path,result);results[obj]=result
    return results


def phase_objective(per_frame,frames,start,end,weights):
    masks=dict(before=frames<start,interaction=(frames>=start)&(frames<=end),release=frames>end)
    if not all(bool(mask.any()) for mask in masks.values()):raise ValueError('Missing scored phase')
    return sum(weights[key]*per_frame[mask].mean() for key,mask in masks.items())


def sparse_projection(nodes,observation,initial_full,surface_ids):
    """Compose the existing fixed 8-particle track binding with sparse-node motion.

    The only approximation is spatial interpolation of displacement from 512
    nodes. This can be measured separately using exact simulated node states.
    """
    ids=nodes['material_indices'];N=len(ids);saved=observation['saved_ids']
    dense_ids=nodes['dense_binding'];dense_w=nodes['dense_weights']
    surface_full=saved[surface_ids]
    surface_matrix=np.zeros((len(surface_ids),N),np.float32)
    np.add.at(surface_matrix,(np.arange(len(surface_ids))[:,None],dense_ids[surface_full]),dense_w[surface_full])
    tracks_full=saved[observation['binding']]
    track_matrix=np.zeros((len(tracks_full),N),np.float32)
    np.add.at(track_matrix,(np.arange(len(tracks_full))[:,None,None],dense_ids[tracks_full]),
              dense_w[tracks_full]*observation['weights'][...,None])
    return dict(surface_matrix=surface_matrix,track_matrix=track_matrix,
        surface_initial=initial_full[surface_full],track_initial=observation['tracks'][0],
        nodes_initial=initial_full[ids])


def project_observations(node_positions,projection):
    displacement=node_positions-projection['nodes_initial']
    surface=projection['surface_initial']+torch.einsum('tnc,sn->tsc',displacement,projection['surface_matrix'])
    tracks=projection['track_initial']+torch.einsum('tnc,pn->tpc',displacement,projection['track_matrix'])
    return surface,tracks


class RealEpisode:
    """Initial state and recorded inputs; future observations are loss targets only."""
    def __init__(self,root,obj,episode,device):
        self.root=Path(root);plan=read_json(self.root/'plan.json');self.episode=episode;self.obj=obj;self.anchor=None
        info=next(o for o in plan['objects'] if o['object']==obj)
        fit=read_json(Path(plan['fit_run'])/'plan.json')
        entry=next(e for o in fit['objects'] if o['object']==obj for e in o['episodes'] if e['episode']==episode)
        row=read_json(entry['row']);workspace=Path(row['workspace']);self.meta=read_json(row['metadata'])
        for path in (row['observation'],row['metadata'],entry['row'],str(workspace/f'initial_states/episode_{episode}.npz'),
                     str(workspace/f'queries/episode_{episode}/actions.npz')):
            if sha256(path)!=plan['source_sha256'][path]:raise ValueError('Changed real episode input: '+path)
        if sha256(info['nodes'])!=info['nodes_sha256']:raise ValueError('Changed sparse nodes')
        nodes=load_npz(info['nodes']);initial=load_npz(workspace/f'initial_states/episode_{episode}.npz')
        actions,self.action_provenance=load_completed_actions(obj,episode,workspace/f'queries/episode_{episode}/actions.npz')
        self.actions=actions;observation=load_npz(row['observation'])
        self.scale=float(nodes['scale_m']);self.origin=initial['positions_world_m'].mean(0)
        times=(actions['source_timestamps_us']-actions['source_timestamps_us'][0])*1e-6
        pads=pack_pads(actions,self.origin,self.scale)
        ids=nodes['material_indices'];state=np.concatenate((initial['positions_world_m'][ids],
            initial['velocities_world_m_s'][ids],initial['deformation_gradient'][ids].reshape(-1,9),np.zeros((len(ids),9))),-1).astype('f')
        self.initial=torch.as_tensor(normalize_state(state,self.origin,self.scale)[None],device=device)
        self.pads=torch.as_tensor(pads[None],device=device);self.times=torch.as_tensor(times[None],dtype=torch.float32,device=device)
        self.geometry={k:torch.as_tensor(nodes[k][None],device=device) for k in ('canonical','neighbors','surface_mask')}
        self.geometry['scale_m']=torch.tensor([self.scale],device=device)
        self.geometry['ground_y']=torch.tensor([(-.01-self.origin[1])/self.scale],device=device,dtype=torch.float32)
        inp=Path(entry['input']);surface_ids=np.load(inp/'surface_ids.npy')
        if sha256(inp/'surface_ids.npy')!=entry['surface_ids_sha256']:raise ValueError('Changed observation surface binding')
        self.projection={k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in
            sparse_projection(nodes,observation,initial['positions_world_m'],surface_ids).items()}
        frames=np.load(inp/'score_frames.npy');self.frames=torch.tensor(frames,device=device)
        counts=np.load(inp/'real_surface_counts.npy');self.counts=counts[frames]
        if sha256(inp/'real_surface.npy')!=entry['real_surface_sha256']:raise ValueError('Changed observed surface')
        self.surface=torch.tensor(np.load(inp/'real_surface.npy')[frames],device=device)
        self.tracks=torch.tensor(observation['tracks'][frames],device=device)
        self.visible=torch.tensor(observation['track_visible'][frames],device=device)
        self.config=fit['config']['fitting']

    def predict(self,model,theta,gradients=True):
        state=self.initial;states=[state]
        for t in range(self.pads.shape[1]-1):
            args=(state,self.pads[:,t],self.pads[:,t+1],theta[None],self.geometry,self.times[:,t+1]-self.times[:,t])
            state=activation_checkpoint(model,*args,use_reentrant=False,preserve_rng_state=False) if gradients else model(*args)
            states.append(state)
        return torch.stack(states,1)[0]

    @torch.no_grad()
    def set_physical_anchor(self,model,theta,physical_root,q):
        """Match the exact full-solver incumbent before using local WM changes.

        Only the fixed support simulation is read. The model still trains with
        ordinary trajectories; this is an inference-time observation correction.
        """
        from .recovery_data import trajectory_folder
        physical_root=Path(physical_root);plan=read_json(physical_root/'plan.json')
        if plan['data_plan_sha256']!=sha256(self.root/'plan.json'):raise ValueError('Anchor uses another world-data protocol')
        if list(q)!=plan['incumbent_q']:raise ValueError('Physical anchor must use the exact initial material coordinates')
        item=next(o for o in plan['objects'] if o['object']==self.obj)
        if self.episode not in item['support']:raise ValueError('Query trajectories cannot anchor material fitting')
        folder=trajectory_folder(physical_root,self.obj,self.episode,q);record=read_json(folder/'complete.json')
        if not record['valid'] or record.get('smoke'):raise ValueError('Invalid physical anchor')
        for name,digest in record['payload_sha256'].items():
            if sha256(folder/(name+'.npy'))!=digest:raise ValueError('Changed physical anchor payload')
        states=self.predict(model,theta,gradients=False)
        positions=states[self.frames,:,:3]*self.scale+torch.as_tensor(self.origin,device=states.device)
        baseline_surface,baseline_tracks=project_observations(positions,self.projection)
        indices=self.frames.cpu().numpy()
        self.anchor=dict(world_surface=baseline_surface,world_tracks=baseline_tracks,
            physical_surface=torch.as_tensor(np.load(folder/'surface.npy')[indices],device=states.device),
            physical_tracks=torch.as_tensor(np.load(folder/'tracks.npy')[indices],device=states.device),
            complete_sha256=sha256(folder/'complete.json'),physical_seconds=record['elapsed_s'])

    def loss(self,states):
        positions=states[self.frames,:,:3]*self.scale+torch.as_tensor(self.origin,device=states.device)
        surface,tracks=project_observations(positions,self.projection)
        if self.anchor is not None:
            surface=self.anchor['physical_surface']+(surface-self.anchor['world_surface'])
            tracks=self.anchor['physical_tracks']+(tracks-self.anchor['world_tracks'])
        # Group equal observed point counts while preserving every original frame.
        curves=torch.zeros(len(self.frames),device=states.device,dtype=states.dtype)
        for count in np.unique(self.counts):
            ids=torch.tensor(np.flatnonzero(self.counts==count),device=states.device)
            for chunk in ids.split(8):
                args=(surface[chunk],self.surface[chunk,:count],tracks[chunk],self.tracks[chunk],self.visible[chunk])
                def score_chunk(*values):
                    return observation_loss(*values,surface_weight=self.config['surface_weight'],
                        track_weight=self.config['track_weight'],huber_delta_mm=self.config['huber_delta_mm'])
                part=activation_checkpoint(score_chunk,*args,use_reentrant=False,preserve_rng_state=False) if states.requires_grad else score_chunk(*args)
                curves=curves.index_copy(0,chunk,part)
        return phase_objective(curves,self.frames,self.meta['clip_start'],self.meta['clip_end'],self.config['phase_weights'])


def fit_material(root,checkpoint_path,obj,output,method='learned',steps=40,device='cuda:0',physical_anchor_root=None,state_anchor_root=None,window_horizon=16,full_step_anchor=False):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():raise ValueError('A frozen selection already exists')
    confidence=export_confidences(root)[obj];plan=read_json(root/'plan.json')
    fit=read_json(Path(plan['fit_run'])/'plan.json');model,bundle=load_model(checkpoint_path,device)
    require_selected_architecture(bundle['architecture'])
    if obj in bundle['training_objects']:raise ValueError('World-model training includes the downstream object')
    model.requires_grad_(False)
    if full_step_anchor and (state_anchor_root is None or physical_anchor_root is None):
        raise ValueError('Full step references require the matching physical state and dense anchors')
    if state_anchor_root is not None:
        from .shared_world_material_windows import WindowedRealEpisode,StepAnchoredRealEpisode
        if physical_anchor_root is None:raise ValueError('Physical windows require the matching dense observation anchor')
        episode_class=StepAnchoredRealEpisode if full_step_anchor else WindowedRealEpisode
        episodes=[episode_class(root,obj,e,device,state_anchor_root,physical_anchor_root,window_horizon) for e in confidence['support']]
    else:episodes=[RealEpisode(root,obj,e,device) for e in confidence['support']]
    base=torch.tensor([fit['config']['baseline'][k] for k in PARAMETERS],dtype=torch.float32,device=device)
    scales=torch.tensor(fit['config']['coordinate_scales'],dtype=torch.float32,device=device)
    initial=confidence['selected_q'] if confidence['selected_q'] is not None else confidence['initial_q']
    q=torch.tensor(initial,dtype=torch.float32,device=device,requires_grad=True)
    anchor_started=time.monotonic()
    if physical_anchor_root is not None:
        theta=physical_parameters(q.detach(),base,scales)
        for episode in episodes:episode.set_physical_anchor(model,theta,physical_anchor_root,initial)
    anchor_setup_seconds=time.monotonic()-anchor_started if physical_anchor_root is not None else 0.
    weights=torch.tensor(confidence['confidence_weights'],dtype=torch.float32,device=device)
    if method=='uniform':weights=torch.ones_like(weights)
    elif method!='learned':raise ValueError('Unknown material weighting')
    weights=weights/weights.sum(0,keepdim=True).clamp_min(1e-8)
    optimizer=torch.optim.Adam([q],lr=.0025);history=[];best=None;started=time.monotonic()
    for step in range(steps+1):
        losses=[];columns=[]
        for e in episodes:
            states=e.predict(model,physical_parameters(q,base,scales));loss=e.loss(states)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite shared-world inverse loss')
            gradient=torch.autograd.grad(loss,q)[0]
            if not torch.isfinite(gradient).all():raise FloatingPointError('Non-finite material gradient')
            losses.append(float(loss.detach()));columns.append(gradient)
            del states,loss
        row=dict(step=step,q=np.clip(q.detach().cpu().numpy().astype(np.float64),-.025,.025),support_losses_mm=losses,
                 support_mean_mm=float(np.mean(losses)),elapsed_s=time.monotonic()-started,
                 support_gradient_norms=[float(g.norm()) for g in columns],
                 support_gradients=[g.cpu().tolist() for g in columns],
                 weighted_gradient=(torch.stack(columns)*weights).sum(0).cpu().tolist())
        history.append(row)
        if best is None or row['support_mean_mm']<best['support_mean_mm']:best=row
        write_json(output/'progress.json',dict(object=obj,method=method,**row))
        if step==steps:break
        optimizer.zero_grad(set_to_none=True);q.grad=(torch.stack(columns)*weights).sum(0)
        torch.nn.utils.clip_grad_norm_([q],100.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():q.clamp_(-.025,.025)
    result=dict(object=obj,method=method,initial_q=initial,proposed_q=best['q'],history=history,
        model_checkpoint=str(checkpoint_path),model_sha256=sha256(checkpoint_path),
        confidence_sha256=plan['confidence_sha256'],support=confidence['support'],query=confidence['query'],
        query_observations_used=False,world_weights_frozen=True,recorded_actions_fixed=True,
        bounds_q=[-.025,.025],physical_validation_pending=True,
        physical_anchor_root=str(physical_anchor_root) if physical_anchor_root is not None else None,
        physical_anchor_hashes=[e.anchor['complete_sha256'] for e in episodes] if physical_anchor_root is not None else None,
        anchor_setup_seconds=anchor_setup_seconds,
        physical_anchor_rollouts=len(episodes) if physical_anchor_root is not None else 0,
        physical_anchor_seconds=sum(e.anchor['physical_seconds'] for e in episodes) if physical_anchor_root is not None else 0.,
        state_anchor_root=str(state_anchor_root) if state_anchor_root is not None else None,
        window_horizon=window_horizon if state_anchor_root is not None and not full_step_anchor else None,
        approximate_gradient_with_fixed_window_starts=state_anchor_root is not None and not full_step_anchor,
        full_episode_step_reference=full_step_anchor,
        recurrent_state_channels=recurrent_channels(model),
        deformation_memory=getattr(model,'deformation_memory',False),
        deformation_initialization=('Existing initial asset or declared physical state anchor; no future observation input'
            if getattr(model,'deformation_memory',False) else None),
        physical_step_reference_includes_F=full_step_anchor and getattr(model,'deformation_memory',False),
        state_anchor_capture_seconds=sum(e.state_anchor_record['elapsed_s'] for e in episodes) if state_anchor_root is not None else 0.,
        interpretation='Surrogate proposal only; acceptance requires full physical-engine support/query evaluation')
    write_json(output/'selection.json',result);return result


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('export-confidence');a.add_argument('--root',type=Path,required=True)
    a=s.add_parser('fit');a.add_argument('--root',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--object',required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--method',choices=['uniform','learned'],default='learned');a.add_argument('--steps',type=int,default=40)
    a.add_argument('--physical-anchor-root',type=Path)
    a.add_argument('--state-anchor-root',type=Path);a.add_argument('--window-horizon',type=int,default=16)
    a.add_argument('--full-step-anchor',action='store_true')
    a.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.mode=='export-confidence':
        print({k:len(v['support']) for k,v in export_confidences(args.root).items()})
    else:fit_material(args.root,args.checkpoint,args.object,args.output,args.method,args.steps,args.device,args.physical_anchor_root,args.state_anchor_root,args.window_horizon,args.full_step_anchor)


if __name__=='__main__':main()
