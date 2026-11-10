"""Project cached material responses through the real observation objective.

These are fixed-start window diagnostics. They never train a model, run a new
simulation, or claim the gradient of an entire physical episode. The original
phase denominators and observation bindings are retained. World trajectories
are corrected at their common baseline before comparing local directions.
"""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from .io import load_npz,read_json,sha256,write_json
from .parameters import PARAMETERS
from .shared_world_inverse import RealEpisode,project_observations
from .shared_world_model import observation_loss
from .shared_world_material_audit import load_case,response_agreement


def phase_coefficients(frames,start,end,weights):
    masks=dict(before=frames<start,interaction=(frames>=start)&(frames<=end),release=frames>end)
    if not all(bool(mask.any()) for mask in masks.values()):raise ValueError('Missing original scored phase')
    values=torch.zeros_like(frames,dtype=torch.float64)
    for name,mask in masks.items():values[mask]=weights[name]/int(mask.sum())
    return values


class WindowObservation:
    def __init__(self,root,case,device):
        source=case['source'];episode=RealEpisode(root,source['object'],source['episode'],device)
        keep=(episode.frames>=source['start'])&(episode.frames<=source['start']+source['horizon'])
        if not bool(keep.any()):raise ValueError('Window contains no original scored observations')
        self.frames=episode.frames[keep];self.local=self.frames-source['start']
        self.coefficients=phase_coefficients(episode.frames,episode.meta['clip_start'],episode.meta['clip_end'],
            episode.config['phase_weights'])[keep]
        self.projection={k:v.double() for k,v in episode.projection.items()}
        self.surface=episode.surface[keep].double();self.tracks=episode.tracks[keep].double()
        self.visible=episode.visible[keep];self.counts=episode.counts[keep.cpu().numpy()]
        self.config=episode.config
        self.provenance=dict(original_scored_frames=len(episode.frames),selected_frames=self.frames.cpu().tolist(),
            original_phase_weight_coefficients=self.coefficients.cpu().tolist(),
            targets='Existing real_surface.npy from cleaned multi-camera point clouds and official processed tracks',
            spatial_projection='Existing 512-node displacement interpolation and fixed surface/track bindings')

    def loss(self,positions):
        surface,tracks=project_observations(positions[self.local],self.projection)
        values=[]
        for frame,count in enumerate(self.counts):
            value=observation_loss(surface[frame:frame+1],self.surface[frame:frame+1,:count],
                tracks[frame:frame+1],self.tracks[frame:frame+1],self.visible[frame:frame+1],
                surface_weight=self.config['surface_weight'],track_weight=self.config['track_weight'],
                huber_delta_mm=self.config['huber_delta_mm'])
            values.append(value[0])
        return (torch.stack(values)*self.coefficients).sum()


def evaluate(root,physical_cache,prediction_caches,output,device='cpu'):
    root,physical_cache,output=Path(root),Path(physical_cache),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'result.json').exists():raise ValueError('Objective audit is immutable')
        torch.set_num_threads(2);started=time.monotonic()
        complete=read_json(physical_cache/'complete.json')
        cases=[load_case(root,physical_cache/f'episode_{episode}') for episode in complete['episodes']]
        plan=read_json(root/'plan.json');objects={c['source']['object'] for c in cases}
        for case in cases:
            source=case['source']
            if source['episode'] not in plan['first_module_handoffs'][source['object']]['support']:
                raise ValueError('Only declared support episodes may inform parameter directions')
        predictions={};provenance={}
        for label,folder in prediction_caches.items():
            folder=Path(folder);record=read_json(folder/'result.json')
            if sha256(folder/'predictions.npz')!=record['predictions_sha256']:raise ValueError('Changed cached model predictions')
            if record['training_labels'] is not False:raise ValueError('Prediction cache is not diagnostic-only')
            for case in cases:
                match=next(c for c in record['cases'] if c['source_result']==case['provenance']['source_result'])
                if match!=case['provenance']:raise ValueError('Model and physical response caches differ')
            arrays=load_npz(folder/'predictions.npz')
            for name,checkpoint in record['checkpoints'].items():
                key=f'{label}/{name}';predictions[key]=(name,arrays)
                provenance[key]=dict(source=str(folder),source_sha256=sha256(folder/'result.json'),checkpoint=checkpoint,
                    reference_step_correction=record.get('reference_step_correction'))
        rows=[]
        for case in cases:
            source=case['source'];observation=WindowObservation(root,case,device)
            positions=torch.tensor(case['arrays']['baseline']['state'][...,:3],device=device,dtype=torch.float64,requires_grad=True)
            baseline_loss=observation.loss(positions)
            derivative=torch.autograd.grad(baseline_loss,positions)[0].detach().cpu().numpy()
            physical_losses={}
            with torch.no_grad():
                for name,data in case['arrays'].items():
                    if not case['records'][name]['valid']:raise ValueError('Invalid physical probe in objective audit')
                    physical_losses[name]=float(observation.loss(torch.tensor(data['state'][...,:3],device=device,dtype=torch.float64)))
            for radius in case['radii']:
                physical_fd=[];physical_projected=[];model_gradients={name:[] for name in predictions}
                for parameter in PARAMETERS:
                    minus,plus=f'h_{radius}_{parameter}_-1',f'h_{radius}_{parameter}_1'
                    delta=(case['arrays'][plus]['state'][...,:3].astype('d')-case['arrays'][minus]['state'][...,:3])/(2*radius)
                    physical_fd.append((physical_losses[plus]-physical_losses[minus])/(2*radius))
                    physical_projected.append(float((derivative*delta).sum()))
                    for label,(name,arrays) in predictions.items():
                        prefix=f'{name}__episode_{source["episode"]}__'
                        a,b=arrays[prefix+plus],arrays[prefix+minus]
                        if a.shape!=derivative.shape or b.shape!=derivative.shape:raise ValueError('Prediction shape differs from physical response')
                        model_gradients[label].append(float((derivative*(a-b)/(2*radius)).sum()))
                rows.append(dict(object=source['object'],episode=source['episode'],start=source['start'],horizon=source['horizon'],
                    radius=radius,window_contribution_to_full_phase_objective_mm=float(baseline_loss.detach()),
                    observation=observation.provenance,physical_objective_fd=physical_fd,
                    physical_linearized_objective=physical_projected,
                    physical_linearization_agreement=response_agreement(physical_projected,physical_fd),
                    models={name:dict(gradient=gradient,agreement=response_agreement(gradient,physical_fd),
                        negative_gradient_directional_derivative=-float(np.dot(gradient,physical_fd))) for name,gradient in model_gradients.items()}))
            write_json(output/'progress.json',dict(completed_cases=len({(r['object'],r['episode']) for r in rows}),rows=rows,elapsed_s=time.monotonic()-started))
        aggregate=[]
        for obj in objects:
            confidence_path=root/'first_module'/f'{obj}.json';confidence=read_json(confidence_path)
            if confidence['model_sha256']!=plan['confidence_sha256']:raise ValueError('Confidence checkpoint changed')
            for radius in sorted({r['radius'] for r in rows if r['object']==obj},reverse=True):
                selected=[r for r in rows if r['object']==obj and r['radius']==radius]
                weights=np.array([confidence['confidence_weights'][confidence['support'].index(r['episode'])] for r in selected])
                if weights.shape!=(len(selected),4) or not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Invalid independent confidence columns')
                weights=weights/weights.sum(0,keepdims=True).clip(min=1e-8)
                physical=np.array([r['physical_objective_fd'] for r in selected]);common=physical.mean(0)
                methods={}
                for name in predictions:
                    gradients=np.array([r['models'][name]['gradient'] for r in selected])
                    for method,w in [('uniform',np.ones_like(weights)/len(selected)),('learned',weights)]:
                        gradient=(gradients*w).sum(0);reference=(physical*w).sum(0)
                        methods[f'{name}/{method}']=dict(gradient=gradient.tolist(),
                            agreement_with_same_weighted_physics=response_agreement(gradient,reference),
                            agreement_with_uniform_physical_objective=response_agreement(gradient,common),
                            negative_gradient_directional_derivative_on_uniform_objective=-float(gradient@common))
                aggregate.append(dict(object=obj,radius=radius,episodes=[r['episode'] for r in selected],
                    support_subset=len(selected)!=len(confidence['support']),confidence_source=str(confidence_path),
                    confidence_sha256=sha256(confidence_path),normalized_confidence_columns=weights.tolist(),methods=methods))
        result=dict(created_at=time.time(),physical_cases=[c['provenance'] for c in cases],prediction_sources=provenance,
            rows=rows,aggregate=aggregate,new_physical_rollouts=0,training_labels=False,query_observations_used=False,
            gradient_units='Millimetres of original phase-weighted observation loss per normalized material coordinate q',
            method='Finite-difference dynamics responses contracted with the observation-loss gradient at the common physical baseline',
            correction='Fixed trajectory offset aligns each model baseline with physical observations; derivatives remain model predictions',
            limitations='Fixed window starts omit parameter effects on the prefix. Spatial projection uses 512 nodes. Only the listed support subset is scored. A negative directional derivative is a local diagnostic, not a verified accepted parameter update.',
            elapsed_s=time.monotonic()-started)
        write_json(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--physical-cache',type=Path,required=True)
    parser.add_argument('--prediction-cache',action='append',nargs=2,required=True,metavar=('NAME','FOLDER'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();caches=dict(args.prediction_cache)
    if len(caches)!=len(args.prediction_cache):parser.error('Prediction cache names must be unique')
    evaluate(args.root,args.physical_cache,caches,args.output,args.device)


if __name__=='__main__':main()
