"""Dynamics labels generated from the first contribution's selected shared asset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
from queue import Queue, Empty
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from .data import fps
from .fit_handoff import check_files, load_handoff
from .io import load_npz, read_json, sha256, write_json
from .recovery_data import qkey

STATE_DIM = 24
MAX_PADS = 4


def validate_splits(splits):
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Require train, validation and test episode lists')
    seen = set()
    for name, episodes in splits.items():
        if not episodes or len(set(episodes)) != len(episodes) or seen.intersection(episodes):
            raise ValueError(f'Overlapping or empty episode split: {name}')
        seen.update(episodes)


def augmented_actions(poses, valid, times, start, end, variant, amplitude):
    """Smooth common translation per gripper, zero at/before contact reference.

    Both fingers translate together: the aperture and orientations are retained.
    No arbitrary particle forces, reference geometry updates or missing-pose fill.
    """
    result = np.array(poses, copy=True)
    if variant == 'nominal':
        return result
    if variant not in ('translate_positive', 'translate_negative'):
        raise ValueError(f'Unknown action variant: {variant}')
    sign = 1. if variant == 'translate_positive' else -1.
    if end <= start:
        raise ValueError('Empty operation interval')
    u = np.clip((times-times[start])/(times[end]-times[start]), 0, 1)
    envelope = np.sin(np.pi*u)**2
    direction = np.array([1., .35, .5]); direction /= np.linalg.norm(direction)
    delta = sign*amplitude*envelope[:, None]*direction
    for g in range(poses.shape[1]):
        for side in range(2):
            good = valid[:, g, side]
            result[good, g, side, :3, 3] += delta[good]
    return result


def prepare(root, fit_run, config):
    root = Path(root).resolve(); config_path = Path(config).resolve(); cfg = read_json(config_path)
    if 'parameters' in cfg or 'splits' in cfg:
        raise ValueError('Parameters and episode splits must come from the first contribution')
    if (root/'plan.json').exists():
        plan = read_json(root/'plan.json')
        if plan['requested_config_sha256'] != sha256(config_path) or plan['handoff']['fit_run'] != str(Path(fit_run).resolve()):
            raise ValueError('Existing frozen experiment differs')
        check_files(plan['source_sha256'])
        return plan
    handoff = load_handoff(fit_run, cfg['object'], cfg['method'], cfg['context'], cfg['budget'])
    if cfg['validation_rule'] != 'largest_support_episode':
        raise ValueError('Unsupported validation protocol')
    val = max(handoff['support'])
    cfg['splits'] = dict(train=sorted(e for e in handoff['support'] if e != val), validation=[val], test=handoff['query'])
    cfg['parameters'] = handoff['parameters']; validate_splits(cfg['splits'])
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['base_commit'], 'HEAD'], cwd=repo, check=False).returncode:
        raise ValueError('Run from the requested published first-contribution base commit')
    previous = Path(handoff['source_runtime'])
    # Physics must match the actual first-contribution run, including adapters.
    physics = ['run_episode.py', 'gs-layer/material_contact_kernels.py',
               'gs-layer/pad_contact_kernels.py', 'gs-layer/deterministic_transfer.py']
    physics += [str(p.relative_to(repo/'runtime')) for p in sorted((repo/'runtime/empm').rglob('*.py'))]
    for name in physics:
        if sha256(repo/'runtime'/name) != sha256(previous/name):
            raise ValueError(f'Changed first-contribution physics: {name}')
    for name in ('empm.py','pad_empm.py'):
        if sha256(repo/'softdynamics/adapters'/name) != sha256(previous.parent/'softdynamics/adapters'/name):
            raise ValueError(f'Changed first-contribution adapter: {name}')
    root.mkdir(parents=True, exist_ok=True)
    code = root/'code'
    for name in ('runtime','softdynamics'):
        shutil.copytree(repo/name, code/name, ignore=shutil.ignore_patterns('__pycache__','.warp-cache'))
    workspace = Path(handoff['workspace']); body = load_npz(workspace/'asset/body.npz')
    n = cfg['nodes']; surface_n = int(n*cfg['surface_fraction'])
    surface = body['surface_particle_ids']
    surface = surface[fps(body['rest_positions'][surface], surface_n)]
    interior = np.setdiff1d(np.arange(len(body['material_ids'])), body['surface_particle_ids'])
    interior = interior[fps(body['rest_positions'][interior], n-surface_n)]
    ids = np.concatenate([surface, interior]); rest = body['rest_positions'][ids]
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError('Insufficient distinct material nodes')
    scale = float(np.ptp(body['rest_positions'],axis=0).max())
    canonical = (rest-body['rest_positions'].mean(0))/scale
    neighbors = cKDTree(rest).query(rest,k=cfg['neighbors']+1)[1][:,1:]
    distances, binding = cKDTree(rest).query(body['rest_positions'],k=4)
    weights = 1/np.maximum(distances,1e-8)**2; weights /= weights.sum(-1,keepdims=True)
    np.savez_compressed(root/'nodes.npz',material_indices=ids,canonical=canonical.astype('f'),
        surface_mask=np.arange(n)<surface_n,neighbors=neighbors,scale_m=scale,
        masses_kg=body['masses_kg'][ids],dense_binding=binding,dense_weights=weights.astype('f'))
    tasks = []
    sources = dict(handoff['source_files']); sources[str(config_path)] = sha256(config_path)
    references = {}
    for split, episodes in cfg['splits'].items():
        for e in episodes:
            entry = handoff['entries'][str(e)]
            cache = Path(fit_run)/'trajectories'/cfg['object']/f'episode_{e}'/qkey(handoff['selected_q'])
            record = read_json(cache/'complete.json')
            if not record['valid'] or record.get('smoke') or record['parameters'] != handoff['parameters']:
                raise ValueError(f'No matching complete first-contribution trajectory: {cache}')
            sources[str(cache/'complete.json')] = sha256(cache/'complete.json')
            for name in ('surface','tracks'):
                sources[str(cache/(name+'.npy'))] = record['payload_sha256'][name]
            obs = read_json(entry['row'])['observation']
            sources[obs] = entry['observation_sha256']
            sources[str(Path(entry['input'])/'surface_ids.npy')] = entry['surface_ids_sha256']
            references[str(e)] = str(cache)
            variants = cfg['train_variants'] if split == 'train' else cfg['evaluation_variants']
            for variant in variants:
                tasks.append(dict(id=f'episode_{e}/{variant}',episode=e,split=split,variant=variant,
                    dtmax=entry['dtmax'],domain_margin=entry['domain_margin']))
    # Establish replay parity for every nominal interaction early in collection.
    tasks.sort(key=lambda t:(t['variant'] != 'nominal', t['episode'], t['variant']))
    check_files(sources)
    frozen = {str(p):sha256(p) for p in sorted(code.rglob('*.py'))}
    sources.update(frozen)
    plan = dict(schema=2,created_at=time.time(),config=cfg,handoff=handoff,tasks=tasks,
        runtime=str(code/'runtime'),workspace=str(workspace),references=references,
        source_sha256=sources,nodes_sha256=sha256(root/'nodes.npz'),
        requested_config_sha256=sha256(config_path),base_commit=cfg['base_commit'],
        teacher='Actual learned CMA-ES selection with unchanged corrected-pad FCR/MPM',
        tactile_input=False,state_layout=['x_m:3','v_m_s:3','F_trial:9','APIC_C_s_inv:9'])
    write_json(root/'first_contribution.json',handoff)
    write_json(root/'plan.json',plan)
    return plan


def controller_features(sim, origin, scale):
    """Corrected actions already describe the actual pad centers; no marker offset."""
    from .teacher_contact import pack_pads
    return pack_pads(dict(poses=sim.poses,valid=sim.valid),origin,scale)


def collect(root, task_id, device='cuda:0', smoke_frames=None):
    root=Path(root).resolve();plan=read_json(root/'plan.json')
    task=next(t for t in plan['tasks'] if t['id']==task_id)
    out=root/('smoke' if smoke_frames else 'trajectories')/task_id
    out.mkdir(parents=True,exist_ok=True)
    with (out/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'complete.json').exists():
            result=read_json(out/'complete.json')
            if result['valid'] and sha256(out/'trajectory.npz')!=result['payload_sha256']:
                raise ValueError('Changed trajectory payload')
            return result
        return _collect(root,plan,task,out,device,smoke_frames)


def _collect(root,plan,task,out,device,smoke_frames):
    from .adapters.pad_empm import PadRuntime
    cfg=plan['config'];h=plan['handoff'];e=task['episode'];entry=h['entries'][str(e)]
    sources={p:v for p,v in plan['source_sha256'].items() if p.endswith('.py') or
             '/asset/' in p or f'episode_{e}' in p or p in h['source_files'] and p.endswith('.json')}
    check_files(sources)
    if sha256(root/'nodes.npz')!=plan['nodes_sha256']:raise ValueError('Changed nodes')
    started=time.monotonic()
    sim=PadRuntime(e,h['parameters'],h['workspace'],plan['runtime'],task['dtmax'],device,task['domain_margin'])
    for key in ('initial','actions','asset'):
        if sim.source_hashes[key]!=entry[key+'_sha256']:raise ValueError('Teacher handoff changed: '+key)
    cache=Path(plan['references'][str(e)]);cached=read_json(cache/'complete.json')
    if sim.domain!=cached['domain'] or sim.dtmax!=cached['dtmax']:
        raise ValueError('Teacher domain/time step differs from first contribution')
    sim.poses=augmented_actions(sim.poses,sim.valid,sim.times,sim.meta['clip_start'],
                               sim.meta['clip_end'],task['variant'],cfg['translation_amplitude_m'])
    nodes=load_npz(root/'nodes.npz');ids=nodes['material_indices'];T=len(sim.times)
    if smoke_frames is not None:T=min(T,smoke_frames+1)
    state=np.empty((T,len(ids),STATE_DIM),np.float32)
    origin=sim.positions(np.arange(sim.n)).mean(0);scale=float(nodes['scale_m'])
    pads=controller_features(sim,origin,scale)[:T]
    # The corrected-pad kernel allows all materials to contact either side.
    eligible=np.zeros((len(ids),MAX_PADS),np.float32);eligible[:,:sim.poses.shape[1]*2]=1
    observation=load_npz(read_json(entry['row'])['observation'])
    surf_ids=np.load(Path(entry['input'])/'surface_ids.npy')
    reference={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ('surface','tracks')}
    checked=0;difference=0.;physics_s=0.;checks=[]
    def capture(frame):
        nonlocal checked,difference
        s=sim.states[sim.current]
        x=s.particle_x.numpy()-sim.shift
        state[frame]=np.concatenate([x[ids],s.particle_v.numpy()[ids],
            s.particle_F_trial.numpy()[ids].reshape(len(ids),9),s.particle_C.numpy()[ids].reshape(len(ids),9)],-1)
        if task['variant']=='nominal' and frame<len(reference['surface']):
            saved=x[observation['saved_ids']].astype(np.float32)
            tracks=observation['tracks'][0]+np.sum((saved[observation['binding']]-observation['initial'][observation['binding']])*observation['weights'][...,None],axis=1)
            difference=max(difference,float(np.max(np.abs(saved[surf_ids]-reference['surface'][frame]))),
                           float(np.max(np.abs(tracks-reference['tracks'][frame]))))
            checked+=1
    capture(0)
    for frame in range(1,T):
        before=time.monotonic()
        try:check=sim.advance(frame)
        except RuntimeError as exc:
            if 'outside allocated grid' not in str(exc):raise
            check=dict(valid=False,error=str(exc),failure_kind='domain_boundary')
        physics_s+=time.monotonic()-before
        if not check['valid']:
            result=dict(valid=False,task=task,frame=frame,reason=check,first_contribution_sha256=sha256(root/'first_contribution.json'))
            write_json(out/'complete.json',result);return result
        capture(frame)
        if frame%25==0 or frame==T-1:
            row=dict(frame=frame,total=T,elapsed_s=time.monotonic()-started,**check)
            checks.append(row);write_json(out/'progress.json',row)
            print('TEACHER',task['id'],frame,T,round(row['elapsed_s'],1),flush=True)
    if checked and difference>1e-6:raise ValueError(f'First-contribution replay differs by {difference} m')
    np.savez_compressed(out/'trajectory.tmp.npz',state=state,pads=pads,eligibility=eligible,
        times=sim.times[:T],origin_m=origin,scale_m=scale,phase=np.where(np.arange(T)<sim.meta['clip_start'],0,
            np.where(np.arange(T)<=sim.meta['clip_end'],1,2)).astype('int8'))
    (out/'trajectory.tmp.npz').replace(out/'trajectory.npz')
    result=dict(valid=True,task=task,frames=T,particles=sim.n,nodes=len(ids),dtmax=sim.dtmax,
        domain=sim.domain,physics_s=physics_s,total_s=time.monotonic()-started,
        payload_sha256=sha256(out/'trajectory.npz'),source_hashes=sim.source_hashes,parameters=h['parameters'],
        first_contribution_sha256=sha256(root/'first_contribution.json'),
        nominal_parity=dict(checked_frames=checked,max_abs_m=difference),validation=checks,
        completed_at=time.time(),diagnostic_truncation=smoke_frames is not None,
        future_observed_geometry_used=False,mid_rollout_resets=False,
        contact_model='corrected independent pads, all-material eligibility, 64x25 mm')
    write_json(out/'complete.json',result);return result


def farm(root,gpus):
    root=Path(root).resolve();plan=read_json(root/'plan.json')
    if not gpus or len(set(gpus))!=len(gpus):raise ValueError('Distinct GPU IDs required')
    pending=[t for t in plan['tasks'] if not (root/'trajectories'/t['id']/'complete.json').exists()]
    queue=Queue()
    for task in pending:queue.put(task)
    (root/'logs').mkdir(exist_ok=True)
    def worker(gpu):
        while True:
            try:task=queue.get_nowait()
            except Empty:return
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free<26000 or shutil.disk_usage(root).free<8*1024**3:
                raise RuntimeError('Insufficient GPU memory or disk for teacher')
            log=root/'logs'/(task['id'].replace('/','_')+'.log')
            command=[sys.executable,'-u','-m','softdynamics.world_data','collect','--root',str(root),'--task',task['id']]
            with log.open('a') as stream:
                p=subprocess.run(command,cwd=root/'code',stdout=stream,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(root/'code'),
                         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            if p.returncode:raise RuntimeError(f'Teacher failed: {task["id"]}, see {log}')
            print('DONE',task['id'],flush=True)
    write_json(root/'collection_status.json',dict(status='running',pid=os.getpid(),gpus=gpus,pending=len(pending),started_at=time.time()))
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures=[pool.submit(worker,g) for g in gpus]
            for f in futures:f.result()
        records=[read_json(root/'trajectories'/t['id']/'complete.json') for t in plan['tasks']]
        write_json(root/'collection_status.json',dict(status='complete',valid=sum(r['valid'] for r in records),
                   total=len(records),unavailable=[r for r in records if not r['valid']],completed_at=time.time()))
    except Exception as exc:
        write_json(root/'collection_status.json',dict(status='failed',error=repr(exc),updated_at=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','collect','farm']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--fit-run',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--task');p.add_argument('--gpus',type=int,nargs='+');p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke-frames',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.fit_run,a.config)
    elif a.mode=='farm':farm(a.root,a.gpus)
    else:collect(a.root,a.task,a.device,a.smoke_frames)


if __name__=='__main__':main()
