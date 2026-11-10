"""Short-window material gradients anchored to a full physical incumbent.

Physical window starts are fixed during a proposal, so these are approximate
material updates, not exact derivatives of a complete MPM trajectory. Full
support simulation remains the acceptance criterion. No query states or
response/Jacobian labels enter fitting or world-model training.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .io import load_npz,read_json,sha256,write_json
from .recovery_data import find_entry,trajectory_folder
from .parameters import ParameterSpace
from .shared_world_inverse import RealEpisode
from .shared_world_model import recurrent_channels
from .world_model import normalize_state


def window_rollout(model,starts,pads,theta,geometry,times,horizon,gradients=True):
    """Cover every output frame; reset only at declared physical window starts."""
    if horizon<1 or starts.shape[1]!=pads.shape[1] or times.shape[1]!=pads.shape[1]:
        raise ValueError('Invalid full-trajectory window inputs')
    states=[starts[:,0].detach()]
    for start in range(0,pads.shape[1]-1,horizon):
        state=starts[:,start].detach()
        for t in range(start,min(start+horizon,pads.shape[1]-1)):
            args=(state,pads[:,t],pads[:,t+1],theta,geometry,times[:,t+1]-times[:,t])
            state=checkpoint(model,*args,use_reentrant=False,preserve_rng_state=False) if gradients else model(*args)
            states.append(state)
    return torch.stack(states,1)


def step_reference_rollout(model,reference,reference_predictions,pads,theta,geometry,times,gradients=True):
    """Differentiate every step while reproducing the fixed reference at theta0.

    The reference predictions are computed once at theta0 and detached. Their
    subtraction precedes addition to the next physical state, giving an exact
    zero correction at the baseline. Recurrent F is included for the explicit
    deformation-memory model; the original model continues to consume x/v only.
    """
    if (reference.shape[1]!=pads.shape[1] or reference_predictions.shape[1]!=pads.shape[1]-1
            or times.shape[1]!=pads.shape[1]):
        raise ValueError('Incomplete full-episode step reference')
    channels=recurrent_channels(model)
    if reference_predictions.shape[-1]!=channels:raise ValueError('Wrong reference state representation')
    state=reference[:,0].detach();states=[state]
    for t in range(pads.shape[1]-1):
        args=(state,pads[:,t],pads[:,t+1],theta,geometry,times[:,t+1]-times[:,t])
        raw=checkpoint(model,*args,use_reentrant=False,preserve_rng_state=False) if gradients else model(*args)
        dynamic=reference[:,t+1,:,:channels].detach()+(raw[...,:channels]-reference_predictions[:,t].detach())
        state=torch.cat((dynamic,raw[...,channels:]),-1)
        states.append(state)
    return torch.stack(states,1)


def collect_anchor(root,physical_root,obj,episode,output,device='cuda:0'):
    """Read-only state capture; require parity with the existing dense rollout."""
    from .adapters.pad_empm import PadRuntime
    from .shared_world_control import sample_state
    from .shared_world_observation import GPUObservation
    root,physical_root,output=Path(root),Path(physical_root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'complete.json').exists():raise ValueError('Physical state anchor is immutable')
        data=read_json(root/'plan.json');plan=read_json(physical_root/'plan.json')
        if plan['data_plan_sha256']!=sha256(root/'plan.json'):raise ValueError('Changed physical data plan')
        item=next(o for o in plan['objects'] if o['object']==obj)
        if episode not in item['support']:raise ValueError('Only support episodes may provide material anchors')
        q=plan['incumbent_q'];entry=find_entry(plan,obj,episode);row=read_json(entry['row'])
        original=trajectory_folder(physical_root,obj,episode,q);record=read_json(original/'complete.json')
        if not record['valid'] or record.get('smoke'):raise ValueError('A complete valid physical incumbent is required')
        for name,digest in record['payload_sha256'].items():
            if sha256(original/(name+'.npy'))!=digest:raise ValueError('Changed dense incumbent trajectory')
        info=next(o for o in data['objects'] if o['object']==obj)
        if sha256(info['nodes'])!=info['nodes_sha256']:raise ValueError('Changed node binding')
        nodes=load_npz(info['nodes']);ids=nodes['material_indices']
        if sha256(row['observation'])!=entry['observation_sha256']:raise ValueError('Changed observation binding')
        observation=load_npz(row['observation']);surface_ids=np.load(Path(entry['input'])/'surface_ids.npy')
        if sha256(Path(entry['input'])/'surface_ids.npy')!=entry['surface_ids_sha256']:raise ValueError('Changed surface binding')
        parameters=ParameterSpace.from_config(plan['config']).parameters(q)
        started=time.monotonic();sim=PadRuntime(episode,parameters,row['workspace'],plan['runtime'],entry['dtmax'],device,entry['domain_margin'])
        if sim.source_hashes!=record['source_hashes']:raise ValueError('Physical source differs from the dense incumbent')
        observer=GPUObservation(sim,ids) if sim.n>=200000 else None
        if observer is not None:observer.install_validation()
        dense_surface=np.load(original/'surface.npy',mmap_mode='r');dense_tracks=np.load(original/'tracks.npy',mmap_mode='r')
        if len(dense_tracks)!=len(dense_surface) or len(dense_surface)>len(sim.times):
            raise ValueError('Observed frames must be contained in the complete control trajectory')
        states=[];max_surface=max_track=0.
        def capture(frame):
            nonlocal max_surface,max_track
            states.append(observer.capture() if observer is not None else sample_state(sim,ids))
            if frame>=len(dense_surface):return
            x=sim.positions(observation['saved_ids']);binding=observation['binding'];weights=observation['weights']
            tracks=observation['tracks'][0]+np.sum((x[binding]-observation['initial'][binding])*weights[...,None],axis=1)
            # Match the original float32 payload serialization before comparing.
            surface=x[surface_ids].astype(np.float32);tracks=tracks.astype(np.float32)
            max_surface=max(max_surface,float(np.max(np.abs(surface-dense_surface[frame]))))
            max_track=max(max_track,float(np.max(np.abs(tracks-dense_tracks[frame]))))
            if max(max_surface,max_track)>1e-7:raise ValueError('Captured state rollout differs from its dense physical anchor')
        failure=None
        try:
            capture(0)
            for frame in range(1,len(sim.times)):
                check=sim.advance(frame)
                if not check['valid']:raise ValueError('Invalid physical anchor state: '+str(check))
                capture(frame)
                if frame%25==0 or frame==len(sim.times)-1:
                    write_json(output/'status.json',dict(pid=os.getpid(),frame=frame,total=len(sim.times)-1,
                        elapsed_s=time.monotonic()-started,updated_at=time.time()))
                    print('ANCHOR',obj,episode,frame,len(sim.times)-1,flush=True)
        except (ValueError,RuntimeError) as exc:failure=repr(exc)
        result=dict(schema='softdynamics.material-state-anchor.v1',valid=failure is None,error=failure,
            object=obj,episode=episode,q=q,frames=len(states),planned_frames=len(sim.times),
            dense_observation_frames=len(dense_surface),
            data_plan_sha256=sha256(root/'plan.json'),physical_plan_sha256=sha256(physical_root/'plan.json'),
            physical_complete_sha256=sha256(original/'complete.json'),nodes_sha256=info['nodes_sha256'],
            source_hashes=sim.source_hashes,parameters=parameters,dtmax=sim.dtmax,domain=sim.domain,
            max_surface_difference_m=max_surface,max_track_difference_m=max_track,
            elapsed_s=time.monotonic()-started,query_states_used=False,world_training_data=False)
        if failure is None:
            np.savez_compressed(output/'trajectory.npz',state=np.stack(states),times=sim.times)
            result['trajectory_sha256']=sha256(output/'trajectory.npz')
        write_json(output/'complete.json',result)
        if failure is not None:raise ValueError(failure)
        return result


class WindowedRealEpisode(RealEpisode):
    def __init__(self,root,obj,episode,device,state_anchor_root,physical_root,horizon):
        super().__init__(root,obj,episode,device)
        if horizon<1:raise ValueError('Window horizon must be positive')
        folder=Path(state_anchor_root)/obj/f'episode_{episode}';record=read_json(folder/'complete.json')
        plan=read_json(Path(physical_root)/'plan.json')
        item=next(o for o in plan['objects'] if o['object']==obj)
        if episode not in item['support']:raise ValueError('Query states cannot initialize material windows')
        if not record['valid'] or record['frames']!=record['planned_frames']:raise ValueError('Incomplete physical state anchor')
        if record['data_plan_sha256']!=sha256(Path(root)/'plan.json') or record['physical_plan_sha256']!=sha256(Path(physical_root)/'plan.json'):
            raise ValueError('Changed anchor data plan')
        if record['object']!=obj or record['episode']!=episode or record['q']!=plan['incumbent_q']:
            raise ValueError('Wrong physical anchor state')
        original=trajectory_folder(physical_root,obj,episode,plan['incumbent_q'])
        if record['physical_complete_sha256']!=sha256(original/'complete.json'):raise ValueError('Changed dense anchor')
        if record['trajectory_sha256']!=sha256(folder/'trajectory.npz'):raise ValueError('Changed physical state payload')
        trajectory=load_npz(folder/'trajectory.npz');state=trajectory['state'].copy();times=trajectory['times']
        np.testing.assert_allclose(times-times[0],self.times[0].cpu().numpy(),atol=2e-6,rtol=1e-6)
        # The trained input uses backward position differences, available from
        # the physical prefix only; no future frame defines a window velocity.
        state[1:,:,3:6]=np.diff(state[:,:,:3],axis=0)/np.diff(times)[:,None,None]
        self.window_starts=torch.tensor(normalize_state(state,self.origin,self.scale)[None],device=device)
        self.window_horizon=horizon;self.state_anchor_record=record

    def predict(self,model,theta,gradients=True):
        if not model.direct_displacement:raise ValueError('Physical windows currently require the observed-velocity model')
        return window_rollout(model,self.window_starts,self.pads,theta[None],self.geometry,
            self.times,self.window_horizon,gradients)[0]


class StepAnchoredRealEpisode(WindowedRealEpisode):
    """One complete parameter-gradient chain around the full physical incumbent."""
    @torch.no_grad()
    def set_physical_anchor(self,model,theta,physical_root,q):
        if not model.direct_displacement:raise ValueError('Full step references require observed-velocity dynamics')
        channels=recurrent_channels(model)
        values=[]
        for t in range(self.pads.shape[1]-1):
            raw=model(self.window_starts[:,t],self.pads[:,t],self.pads[:,t+1],theta[None],self.geometry,
                self.times[:,t+1]-self.times[:,t])
            values.append(raw[...,:channels])
        self.reference_predictions=torch.stack(values,1).detach()
        reproduced=self.predict(model,theta,gradients=False)
        if not torch.equal(reproduced[...,:channels],self.window_starts[0,...,:channels]):
            raise ValueError('Full step reference does not reproduce the recurrent physical state exactly')
        super().set_physical_anchor(model,theta,physical_root,q)

    def predict(self,model,theta,gradients=True):
        if not hasattr(self,'reference_predictions'):raise ValueError('Initialize the full physical step reference first')
        return step_reference_rollout(model,self.window_starts,self.reference_predictions,self.pads,
            theta[None],self.geometry,self.times,gradients)[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--physical-root',type=Path,required=True);p.add_argument('--object',required=True)
    p.add_argument('--episode',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    collect_anchor(a.root,a.physical_root,a.object,a.episode,a.output,a.device)


if __name__=='__main__':main()
