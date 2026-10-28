"""Frozen corrected-release assets and full trajectories for downstream recovery."""
import argparse
import hashlib
import io
import itertools
import os
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .adapters.pad_empm import PadRuntime
from .data import fps
from .io import load_npz, read_json, sha256, write_json
from .objective import score
from .parameters import ParameterSpace


def qkey(q):
    values=np.array(q,dtype=float,copy=True)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.any(abs(values)>.025+1e-12):
        raise ValueError('Recovery candidate outside the frozen local parameter bounds')
    indices=np.rint(values/.025).astype(int)
    # Preserve existing grid caches only for exact grid coordinates. Nearby
    # continuous samples must never alias to a grid point or to each other.
    if np.array_equal(values,indices*.025):
        return ''.join(str(i+1) for i in indices)
    values[values==0]=0.  # canonicalize signed zero
    return 'c'+hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()


def noise_conditions():
    return [(sigma,seed) for sigma in [0.,.5,2.] for seed in ([19] if sigma==0 else [19,23,29])]


def condition_key(truth,sigma=0.,seed=19):
    return f'{truth}_noise{sigma:g}_seed{seed}' if truth!='real' else 'real'


def case_specs():
    return [dict(key=condition_key(t,s,seed),truth=t,noise_mm=s,seed=seed)
            for t in ['center','mixed'] for s,seed in noise_conditions()]+[
            dict(key='real',truth='real',noise_mm=0.,seed=19)]


def noise_arrays(tracks,surface,sigma_mm,seed):
    """The previous experiment's reproducible independent + smooth drift noise."""
    if sigma_mm==0:
        return tracks.copy(),surface.copy()
    rng=np.random.default_rng(seed)
    shared=gaussian_filter1d(rng.normal(size=(len(tracks),3)),sigma=5,axis=0)
    shared-=shared.mean(0)
    shared/=np.maximum(shared.std(0),1e-10)
    def add(a,offset):
        eps=(np.random.default_rng(seed+offset).normal(size=a.shape)+shared[:,None,:])
        eps*=sigma_mm/1000/np.sqrt(6)
        eps[0]=0
        return (a+eps).astype(np.float32)
    return add(tracks,2),add(surface,1)


def trajectory_folder(root,obj,episode,q):
    return Path(root)/'trajectories'/obj/f'episode_{episode}'/qkey(q)


def prepare(root,pretrain,code_root):
    root,pretrain,code_root=map(lambda p:Path(p).resolve(),[root,pretrain,code_root])
    if (root/'plan.json').exists():
        return read_json(root/'plan.json')
    root.mkdir(parents=True,exist_ok=True)
    cfg=read_json(pretrain/'config.json');space=ParameterSpace.from_config(cfg)
    report=read_json(pretrain/'training/full/report.json')
    objects=[]
    for obj in report['objects']['test']:
        entries=[]
        for path in sorted((pretrain/'objects'/obj/'prepared').glob('*/complete.json')):
            row=read_json(path)
            manifests=[pretrain/'labels_v1'/row['id']/t/'manifest.json' for t in ['center','mixed']]
            if not all(p.exists() for p in manifests):continue
            labels=[read_json(p) for p in manifests]
            entries.append(dict(episode=row['episode'],row=str(path),workspace=row['workspace'],
                         frames=row['frames'],particles=row['particles'],
                         domain_margin=max(m['domain'].get('margin_m',.075) for m in labels),
                         dtmax=min(m['dtmax'] for m in labels)))
        entries.sort(key=lambda r:r['episode'])
        if len(entries)<3:raise ValueError(f'Need two fitting episodes and one query: {obj}')
        objects.append(dict(object=obj,episodes=entries,support=[r['episode'] for r in entries[:-1]],
                            query=[entries[-1]['episode']]))
    plan=dict(schema='softdynamics.model-recovery.v1',created_at=time.time(),pretrain=str(pretrain),
              checkpoint=str(pretrain/'training/full/model.pt'),model_sha256=report['model_sha256'],
              config=cfg,objects=objects,methods=['uniform','learned','shuffled'],conditions=case_specs(),
              initial_guesses=cfg['initial_guesses'],real_initial_guess=[0.,0.,0.,0.],
              grid_levels=cfg['grid_levels'],passes=3,logical_candidates_per_fit=36,
              runtime=str(code_root/'runtime'),source_data=read_json(pretrain/'selection.json')['source'],
              noise='RMS 3D scale: independent points plus smooth common drift; frame zero unchanged',
              fairness='Same solver, initial states, full actions, per-episode fixed grid and dt, objective, search and candidate budget. Only interaction weights change. No teacher reliability masks used by any method.',
              selection='All episodes with completed center and mixed teachers; largest episode index reserved for query. Eligibility uses only numerical rollout completion, never sensitivity scores or downstream outcomes.',
              interpretation='Local discrete coordinate-search comparison against current uniform EMPM pipeline; no physics autodiff, no continuous-convergence claim. Synthetic truths lie on the local grid.',
              real_acceptance='Use selected candidate only if valid and unweighted support objective strictly improves the shared q=0 baseline; otherwise fallback, reporting both raw and accepted candidates.',
              weight_query='Frozen model scores at initial guess, fixed during all coordinate updates; input motion matches the observation and noise condition.',
              query_policy='Query observations never enter weight prediction, parameter selection, or acceptance. They are scored only after support choices are frozen.')
    write_json(root/'config.json',cfg)
    for item in objects:
        obj=item['object'];workspace=Path(item['episodes'][0]['workspace'])
        body=load_npz(workspace/'asset/body.npz')
        boundary=body['surface_particle_ids']
        pool=boundary[np.linspace(0,len(boundary)-1,min(len(boundary),8192)).astype(int)]
        surface_ids=pool[fps(body['rest_positions'][pool],min(1500,len(pool)))]
        for entry in item['episodes']:
            row=read_json(entry['row']);observation=load_npz(row['observation'])
            dest=root/'inputs'/obj/f'episode_{entry["episode"]}';dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/'surface_ids.npy',surface_ids)
            T=len(observation['tracks']);surface=np.zeros((T,1500,3),np.float32);counts=np.zeros(T,np.int32)
            with tarfile.open(Path(plan['source_data'])/obj/f'episode_{entry["episode"]}'/'pcd_clean.tar') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.npz'):continue
                    frame=int(Path(member.name).stem)
                    with np.load(io.BytesIO(archive.extractfile(member).read())) as data:
                        points=data['pts'];good=np.isfinite(points).all(1)&(data['visibility_matrix'].sum(1)>=2)
                        ids=np.flatnonzero(good)
                        if len(ids):
                            sample=ids[np.linspace(0,len(ids)-1,min(1500,len(ids))).astype(int)]
                            counts[frame]=len(sample);surface[frame,:len(sample)]=points[sample]
            frames=np.flatnonzero(observation['track_visible'].any(1)&(counts>0))
            meta=read_json(row['metadata'])
            phase=np.where(frames<meta['clip_start'],0,np.where(frames<=meta['clip_end'],1,2))
            if set(phase)!={0,1,2}:raise ValueError(f'Missing scored phase: {row["id"]}')
            np.save(dest/'real_surface.npy',surface);np.save(dest/'real_surface_counts.npy',counts)
            np.save(dest/'score_frames.npy',frames)
            entry['input']=str(dest);entry['observation_sha256']=sha256(row['observation'])
            entry['initial_sha256']=sha256(workspace/'initial_states'/f'episode_{entry["episode"]}.npz')
            entry['actions_sha256']=meta['source_sha256']['actions']
            entry['asset_sha256']=sha256(workspace/'asset/asset.json')
            entry['surface_ids_sha256']=sha256(dest/'surface_ids.npy')
            entry['real_surface_sha256']=sha256(dest/'real_surface.npy')
    if sha256(plan['checkpoint'])!=plan['model_sha256']:raise ValueError('Model changed')
    write_json(root/'plan.json',plan)
    return plan


def find_entry(plan,obj,episode):
    return next(r for o in plan['objects'] if o['object']==obj for r in o['episodes'] if r['episode']==episode)


def simulate(root,obj,episode,q,device='cuda:0',smoke_frames=None):
    root=Path(root);plan=read_json(root/'plan.json');entry=find_entry(plan,obj,episode)
    space=ParameterSpace.from_config(plan['config']);row=read_json(entry['row'])
    folder=trajectory_folder(root,obj,episode,q)
    if smoke_frames is not None:folder=root/'smoke'/obj/f'episode_{episode}'/qkey(q)
    if (folder/'complete.json').exists():return read_json(folder/'complete.json')
    folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    observation=load_npz(row['observation'])
    if sha256(row['observation'])!=entry['observation_sha256']:raise ValueError('Observation changed')
    sim=PadRuntime(episode,space.parameters(q),row['workspace'],plan['runtime'],entry['dtmax'],device,entry['domain_margin'])
    for k,name in [('initial','initial_sha256'),('actions','actions_sha256'),('asset','asset_sha256')]:
        if sim.source_hashes[k]!=entry[name]:raise ValueError('Simulation source changed: '+k)
    surface_ids=np.load(Path(entry['input'])/'surface_ids.npy');T=len(observation['tracks'])
    tracks=np.empty_like(observation['tracks'],dtype=np.float32)
    surface=np.empty((T,len(surface_ids),3),np.float32)
    binding,weights=observation['binding'],observation['weights'];initial=observation['initial']
    validations=[]
    def capture(frame):
        if frame>=T:return
        x=sim.positions(observation['saved_ids'])
        tracks[frame]=observation['tracks'][0]+np.sum((x[binding]-initial[binding])*weights[...,None],axis=1)
        surface[frame]=x[surface_ids]
    capture(0)
    last=min(smoke_frames,len(sim.times)-1) if smoke_frames is not None else len(sim.times)-1
    for frame in range(1,last+1):
        try:
            check=sim.advance(frame)
        except RuntimeError as exc:
            if 'outside allocated grid' not in str(exc):raise
            check=dict(valid=False,failure_kind='domain_boundary',error=str(exc))
        validations.append(dict(frame=frame,**check))
        if not check['valid']:
            result=dict(valid=False,object=obj,episode=episode,q=q,invalid_frame=frame,check=check,
                        domain=sim.domain,source_hashes=sim.source_hashes,elapsed_s=time.monotonic()-start)
            write_json(folder/'validation.json',validations);write_json(folder/'complete.json',result);return result
        capture(frame)
        if frame%25==0 or frame==last:
            write_json(folder/'status.json',dict(pid=os.getpid(),frame=frame,total=last,updated_at=time.time(),elapsed_s=time.monotonic()-start))
            print('ROLLOUT',obj,episode,qkey(q),frame,last,round(time.monotonic()-start,1),flush=True)
    write_json(folder/'validation.json',validations)
    if smoke_frames is None:
        for name,value in [('tracks',tracks),('surface',surface)]:np.save(folder/(name+'.npy'),value)
    result=dict(valid=True,object=obj,episode=episode,q=q,parameters=space.parameters(q),
                domain=sim.domain,dtmax=sim.dtmax,source_hashes=sim.source_hashes,frames=T,
                elapsed_s=time.monotonic()-start,smoke=smoke_frames is not None,
                payload_sha256={} if smoke_frames is not None else {k:sha256(folder/(k+'.npy')) for k in ['tracks','surface']})
    write_json(folder/'complete.json',result);return result


def observation_for(root,plan,obj,episode,condition):
    entry=find_entry(plan,obj,episode);row=read_json(entry['row']);data=load_npz(row['observation'])
    inp=Path(entry['input']);frames=np.load(inp/'score_frames.npy');meta=read_json(row['metadata'])
    if condition['truth']=='real':
        tracks=data['tracks'];surface=np.load(inp/'real_surface.npy',mmap_mode='r');counts=np.load(inp/'real_surface_counts.npy')
    else:
        truth=next(t for t in plan['config']['truths'] if t['name']==condition['truth'])
        base=trajectory_folder(root,obj,episode,truth['q'])
        if not read_json(base/'complete.json')['valid']:raise ValueError('Invalid synthetic truth')
        tracks=np.load(base/'tracks.npy',mmap_mode='r');surface=np.load(base/'surface.npy',mmap_mode='r')
        seed=condition['seed']+10000*episode+100000*(condition['truth']=='mixed')+int(obj.split('-')[0])*1000000
        tracks,surface=noise_arrays(tracks,surface,condition['noise_mm'],seed)
        counts=np.full(len(tracks),surface.shape[1])
    surfaces=[surface[i,:counts[i]] for i in frames]
    return dict(surface=surfaces,tracks=tracks[frames],visible=data['track_visible'][frames],
                meta={**meta,'frames':frames},trees=[cKDTree(s) for s in surfaces]),tracks


def score_candidate(root,obj,episode,q):
    root=Path(root);plan=read_json(root/'plan.json');folder=trajectory_folder(root,obj,episode,q)
    result=read_json(folder/'complete.json');dest=folder/'scores.json'
    if dest.exists():return read_json(dest)
    query=episode in next(o['query'] for o in plan['objects'] if o['object']==obj)
    conditions=[c for c in plan['conditions'] if not query or c['noise_mm']==0]
    if not result['valid']:
        write_json(dest,dict(valid=False,conditions={c['key']:dict(valid=False) for c in conditions}));return
    for name,digest in result['payload_sha256'].items():
        if sha256(folder/(name+'.npy'))!=digest:raise ValueError('Trajectory changed')
    entry=find_entry(plan,obj,episode);frames=np.load(Path(entry['input'])/'score_frames.npy')
    pred={k:np.load(folder/(k+'.npy'),mmap_mode='r')[frames] for k in ['tracks','surface']}
    scores={}
    for condition in conditions:
        obs,_=observation_for(root,plan,obj,episode,condition)
        scores[condition['key']]=dict(valid=True,**score(pred,obs,plan['config']['fitting']))
    write_json(dest,dict(valid=True,conditions=scores));return scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--object');p.add_argument('--episode',type=int);p.add_argument('--q',type=float,nargs=4)
    p.add_argument('--mode',choices=['prepare','simulate','score'],required=True)
    p.add_argument('--pretrain',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-frames',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.pretrain,Path(__file__).resolve().parents[1])
    elif a.mode=='simulate':simulate(a.root,a.object,a.episode,a.q,a.device,a.smoke_frames)
    else:score_candidate(a.root,a.object,a.episode,a.q)


if __name__=='__main__':main()
