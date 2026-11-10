"""All-50-object teacher collection, with immutable tasks and resumable workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import io
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .io import read_json,write_json,load_npz,sha256
from .teacher_contact import contract,contract_sha256,load_completed_actions,selected_architecture,verify_selected_kernels
from .shared_data import build_canonical,runtime_imports,normalize_pads
from .shared_world_data import build_nodes,stable_seed
from .data import fps

OLD=Path('/data/guangqi/empm-experiments/deform360_corrected_pretrain_20261020')
PREP=Path('/home/guangqi/data/deform360_prepared_20261103')
SHARED=Path('/data/shared/deform360_pad50_20261102_v1')
SPLIT=Path('/data/guangqi/empm-experiments/deform360_validation_4x10_20261103/split.json')

def link(source,destination):
    source,destination=Path(source),Path(destination)
    if not destination.exists() and not destination.is_symlink():
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.symlink_to(source.resolve(),target_is_directory=source.is_dir())

def candidates(obj):
    return [PREP/'physics_objects'/obj,PREP/'physics_new_objects'/obj,OLD/'objects'/obj]

def source_view(root,identity,source_root):
    """Resolve surviving raw packages into a local view without changing shared data."""
    directory=root/'source_views/processed'/identity;directory.mkdir(parents=True,exist_ok=True)
    roots=[source_root,SHARED/'processed']
    recovered=root/'source_recovery/source/processed'
    if (recovered/identity/'SOURCE_COMPLETE.json').exists():roots.insert(0,recovered)
    for name in ('metadata.json','cameras.tar','pcd_clean.tar','splatfacto.tar','splat_0.ply','start_obj_pcd.ply'):
        source=next((r/identity/name for r in roots if (r/identity/name).exists()),None)
        if source is not None:link(source,directory/name)
    if not (directory/'metadata.json').exists():
        cached=read_json(SHARED/'dataset/episodes'/identity/'episode.json')['processed_metadata']
        write_json(directory/'metadata.json',cached)
    if not (directory/'splat_0.ply').exists() and not (directory/'splatfacto.tar').exists():
        obj,name=identity.split('/')
        cached=next((w/'prepared'/name/'splat_0.ply' for w in candidates(obj) if (w/'prepared'/name/'splat_0.ply').exists()),None)
        if cached is not None:link(cached,directory/'splat_0.ply')
    return root/'source_views/processed'

def plan(root):
    root=Path(root).resolve();path=root/'plan.json'
    if path.exists():
        result=read_json(path)
        if result['contact_contract']!=contract():raise ValueError('Existing collection contact differs')
        return result
    package=read_json(SHARED/'plan.json');split=read_json(SPLIT);old=read_json(OLD/'config.json')
    code=Path(__file__).resolve().parents[1];verify_selected_kernels(code/'runtime')
    cfg=dict(baseline=old['baseline'],reference_E_pa=3500.,reference_nu=.27,nu_scale=.2,
        E_range_pa=[1000.,12000.],nu_range=[.08,.42],variants_per_episode=8,
        seed=0,nodes=512,neighbors=24,surface_fraction=.75,fps_candidates=8192,
        phase_weights=old['fitting']['phase_weights'],
        teacher=dict(radii=[.05,.025],refinement_radii=[.0125,.00625],relative_column_tolerance=.25,
            windows_per_phase=2,points=256),
        model=selected_architecture(dict(hidden=256,layers=4,heads=4,kinematic_only=True,
            direct_displacement=True,contact_geometry=True,material_dim=2)))
    from .elastic_data import ElasticSpace
    space=ElasticSpace(cfg);max_E=12000*math.exp(.1);max_nu=.44
    dtmax=min(.0002,.35*.003125/math.sqrt(max_E*(1-max_nu)/(100*(1+max_nu)*(1-2*max_nu))))
    objects=[];episodes=[];tasks=[]
    for obj in package['objects']:
        group='test' if obj in split['test_objects'] else 'validation' if obj in split['validation_objects'] else 'train'
        prior=next((w for w in candidates(obj) if (w/'asset/asset.json').exists()),None)
        asset=read_json(prior/'asset/asset.json') if prior else {}
        objects.append(dict(object=obj,split=group,episodes=list(range(10)),particles=asset.get('particles',0),
            geometry_scope='episode preparation selects matching asset and nodes'))
        for ep in range(10):
            identity=f'{obj}/episode_{ep}';directory=root/'episodes'/identity
            initial=next((w for w in candidates(obj) if (w/'initial_states'/f'episode_{ep}.npz').exists()),None)
            e=dict(id=identity,object=obj,episode=ep,split=group,preparation=str(directory/'complete.json'),
                reusable_initial_workspace=str(initial) if initial else None,
                particles_estimate=asset.get('particles',0),source_kind='new' if obj in package['new_objects'] else 'old')
            episodes.append(e)
            cameras=SHARED/'dataset/episodes'/identity/'cameras.npz'
            link(cameras,root/'corrected/finger_data/cameras'/identity/'cameras.npz')
            unit=qmc.LatinHypercube(2,seed=stable_seed(0,identity)).random(7)
            nominal=[[3500.,.27]]+[[math.exp(math.log(1000.)+u[0]*math.log(12.)),.08+.34*u[1]] for u in unit]
            for variant,(E,nu) in enumerate(nominal):
                parameters=dict(cfg['baseline'],E_pa=E,nu=nu)
                tasks.append(dict(id=f'{identity}/sample_{variant}',object=obj,episode=ep,split=group,
                    variant=variant,episode_index=ep,parameters=parameters,q=space.coordinate(parameters).tolist(),
                    dtmax=dtmax,domain_margin=.075,randomized_actions=False,
                    episode_preparation=e['preparation'],particles=e['particles_estimate']))
    tasks.sort(key=lambda t:(t['variant']>0,t['episode'],t['variant'],t['particles'],t['object']))
    result=dict(schema='softdynamics.teacher50.v1',created_at=time.time(),runtime=str(code/'runtime'),
        code_root=str(code),config=cfg,objects=objects,episodes=episodes,tasks=tasks,parameters=['E_pa','nu'],
        test_objects=split['test_objects'],excluded_objects=split['test_objects'],
        collection_includes_test=True,training_started=False,old_weights_used=False,old_simulation_targets_used=False,
        contact_contract=contract(),contact_sha256=contract_sha256(),expected_objects=50,expected_episodes=500,
        expected_trajectories=4000,expected_sensitivity_workpoints=12000,
        source_split=str(SPLIT),source_split_sha256=sha256(SPLIT),
        split_policy='Preserve four validation and four test objects; add the 28 new objects to the training candidates. Collection covers all splits.',
        initialization_policy='Reuse compatible existing initial states; otherwise reconstruct the episode first frame as its own stress-free body.',
        observation_policy='Use recorded tracks where available; otherwise explicitly tagged simulated surface probes, never fabricated observations.',
        code_sha256={str(p.relative_to(code)):sha256(p) for p in sorted(code.rglob('*.py')) if '__pycache__' not in p.parts})
    assert len(objects)==50 and len(episodes)==500 and len(tasks)==4000
    write_json(path,result);write_json(root/'config.json',cfg)
    return result

def raw_tracks(source,frames,allowed_ids=None):
    tracks=[];visible=[];ids=None
    with tarfile.open(source/'pcd_clean.tar') as archive:
        members=sorted((m for m in archive if m.isfile() and m.name.endswith('.npz')),key=lambda m:int(Path(m.name).stem))
        if len(members)>frames:raise ValueError('More object frames than action frames')
        for frame,member in enumerate(members):
            if int(Path(member.name).stem)!=frame:raise ValueError('Non-contiguous source track indices')
            with np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False) as z:
                x=z['pts'];v=(z['visibility_matrix'].sum(1)>=2)&np.isfinite(x).all(-1)
                if ids is None:
                    eligible=np.flatnonzero(v)
                    if allowed_ids is not None:eligible=eligible[np.isin(eligible,allowed_ids)]
                    if len(eligible)<128:raise ValueError(f'Only {len(eligible)} observed initial points')
                    ids=eligible[fps(x[eligible],min(512,len(eligible)))]
                if len(x)<=int(ids.max()):raise ValueError('Source point identities changed')
                tracks.append(x[ids]);visible.append(v[ids])
    return np.asarray(tracks),np.asarray(visible),ids

def teacher_windows(observation,metadata,weights):
    # Visibility here describes object probes, never a missing controller pose.
    eligible=np.flatnonzero(observation['track_visible'][0]&np.isfinite(observation['tracks'][0]).all(-1))
    if not len(eligible):raise ValueError('No finite initial probes')
    ids=eligible[fps(observation['tracks'][0,eligible],256)]
    rows=[];available=[]
    for phase in weights:
        frames=[]
        for f in range(1,len(observation['tracks'])):
            label='before' if f<metadata['clip_start'] else 'interaction' if f<=metadata['clip_end'] else 'release'
            if label==phase and observation['track_visible'][f,ids].any():frames.append(f)
        if frames:
            chosen=np.unique(np.quantile(frames,[.25,.75],method='nearest').astype(int))
            rows.extend(dict(index=int(f),start=int(f)-1,end=int(f),phase=phase,weight=weights[phase]/len(chosen)) for f in chosen)
            available.append(phase)
    total=sum(weights[p] for p in available)
    if total<=0:raise ValueError('No teacher transition is available')
    for row in rows:row['weight']/=total
    return sorted(rows,key=lambda w:w['end']),{p:weights[p]/total for p in available}

def prepare_episode(root,identity):
    root=Path(root);p=read_json(root/'plan.json');e=next(e for e in p['episodes'] if e['id']==identity)
    directory=root/'episodes'/identity;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'complete.json').exists():return read_json(directory/'complete.json')
        started=time.monotonic();obj,ep=e['object'],e['episode'];name=f'episode_{ep}'
        workspace=directory/e.get('workspace_subdir','workspace');workspace.mkdir(exist_ok=True)
        action,provenance=load_completed_actions(obj,ep)
        old_selection=read_json(OLD/'selection.json')
        source_root=PREP/'source/processed' if e['source_kind']=='new' else Path(old_selection['source'])
        source_root=source_view(root,identity,source_root)
        source=source_root/identity;metadata=read_json(source/'metadata.json')
        for d in ('initial_states','prepared','queries','observations','checks'):(workspace/d).mkdir(exist_ok=True)
        _,_,_,PhysicalAsset,_=runtime_imports(Path(p['runtime']))
        if e['reusable_initial_workspace']:
            prior=Path(e['reusable_initial_workspace']);link(prior/'asset',workspace/'asset')
            for suffix in ('.npz','.json'):link(prior/'initial_states'/(name+suffix),workspace/'initial_states'/(name+suffix))
            asset=PhysicalAsset(workspace/'asset');geometry_policy='reused_existing_initial_state'
        else:
            asset=build_canonical(workspace,obj,ep,source_root,root/'corrected',Path(p['runtime']),
                                  reconstruction_method=e.get('initialization_reconstruction'))
            geometry_policy='episode_first_frame_stress_free_asset'
        state,state_meta=asset.load_initial_state(workspace/'initial_states'/(name+'.npz'))
        body=asset.body;node_path=root/'node_maps'/(asset.fingerprint+'.npz')
        with node_path.with_suffix('.lock').open('a') as node_lock:
            fcntl.flock(node_lock,fcntl.LOCK_EX)
            if not node_path.exists():
                nodes=build_nodes(body,p['config'],stable_seed(0,obj));np.savez_compressed(node_path,**nodes)
        nodes=load_npz(node_path)
        T=len(action['poses']);observation_source=None;observation_error=None
        existing=next((read_json(w/'prepared'/name/'complete.json') for w in candidates(obj)
            if (w/'prepared'/name/'complete.json').exists() and read_json(w/'prepared'/name/'complete.json').get('status')=='ready'),None)
        packet=PREP/'observations'/identity/'observation.npz'
        try:
            support_path=workspace/'prepared'/name/'first-frame-surface-support.npz'
            if support_path.exists():
                support=load_npz(support_path)
                tracks,visible,ids=raw_tracks(source,T,allowed_ids=support['point_ids'])
                observation_source=str(source/'pcd_clean.tar')
            elif existing:
                z=load_npz(existing['observation']);tracks,visible,ids=z['tracks'],z['track_visible'],z['point_ids']
                observation_source=existing['observation']
            elif packet.exists():
                z=load_npz(packet);tracks,visible,ids=z['positions_m'],z['track_visible'],z['point_ids']
                observation_source=str(packet)
            else:
                tracks,visible,ids=raw_tracks(source,T);observation_source=str(source/'pcd_clean.tar')
            if not np.isfinite(tracks[0]).all() or len(tracks)<3:raise ValueError('No finite initial observation packet')
            # Unobserved object coordinates are placeholders with an explicit mask.
            finite=np.isfinite(tracks).all(-1);visible=visible&finite
            tracks=np.where(finite[...,None],tracks,tracks[0][None])
            observation_kind='recorded_visual_track_sampling'
        except (ValueError,KeyError,FileNotFoundError) as exc:
            observation_error=str(exc)
            surface=body['surface_particle_ids'];x=state['positions_world_m'][surface]
            chosen=fps(x,512);ids=surface[chosen];initial=x[chosen]
            tracks=np.broadcast_to(initial,(T,len(ids),3)).copy();visible=np.ones((T,len(ids)),bool)
            observation_source=str(workspace/'initial_states'/(name+'.npz'))
            observation_kind='simulator_surface_probes_no_recorded_track_claim'
        distance,binding=cKDTree(state['positions_world_m']).query(tracks[0],k=8)
        weights=1/np.maximum(distance,1e-6)**2;weights/=weights.sum(1,keepdims=True)
        observation=dict(saved_ids=body['material_ids'],initial=state['positions_world_m'],
            surface_ids=body['surface_particle_ids'],tracks=tracks,track_visible=visible,
            binding=binding,weights=weights,frames=np.arange(len(tracks)),point_ids=ids)
        meta=dict(episode=ep,object=obj,clip_start=int(metadata['start_frame']),clip_end=int(metadata['end_frame']),
            action_schema='pad_center',face_size_m=[.064,.025],ground_y_world_m=-.01,
            observation_kind=observation_kind,allow_missing_phases=True)
        windows,phase_weights=teacher_windows(observation,meta,p['config']['phase_weights'])
        meta['teacher_windows']=windows
        qdir=workspace/'queries'/name;qdir.mkdir(exist_ok=True)
        np.savez_compressed(qdir/'actions.npz',**action)
        cameras=SHARED/'dataset/episodes'/identity/'cameras.npz';link(cameras,qdir/'cameras.npz')
        meta['source_sha256']=dict(actions=sha256(qdir/'actions.npz'),camera=sha256(cameras))
        write_json(qdir/'episode.json',meta)
        observation_path=workspace/'observations'/(name+'.npz');np.savez_compressed(observation_path,**observation)
        normalized=normalize_pads(tracks,visible,tracks[0],body,state,action['poses'][:len(tracks)],
            action['valid'][:len(tracks)],action['source_timestamps_us'][:len(tracks)]*1e-6,ids)
        prepared=workspace/'prepared'/name;prepared.mkdir(exist_ok=True)
        np.savez_compressed(prepared/'real.npz',**normalized)
        row=dict(id=identity,object=obj,episode=ep,split=e['split'],status='ready',workspace=str(workspace),
            observation=str(observation_path),metadata=str(qdir/'episode.json'),real=str(prepared/'real.npz'),
            particles=len(body['material_ids']),frames=len(tracks),teacher_windows=windows,phase_weights=phase_weights,
            geometry_policy=geometry_policy,observation_kind=observation_kind,observation_source=observation_source,
            observation_fallback_reason=observation_error,action_provenance=provenance,
            initial_state_provenance=state_meta,pose_quality_filter=False)
        row_path=prepared/'complete.json';write_json(row_path,row)
        result=dict(status='ready',id=identity,object=obj,episode=ep,workspace=str(workspace),row=str(row_path),
            row_sha256=sha256(row_path),nodes=str(node_path),nodes_sha256=sha256(node_path),
            particles=len(body['material_ids']),geometry_policy=geometry_policy,observation_kind=observation_kind,
            source_hashes={str(x):sha256(x) for x in (observation_path,prepared/'real.npz',qdir/'episode.json',workspace/'initial_states'/(name+'.npz'))},
            elapsed_s=time.monotonic()-started,completed_at=time.time())
        write_json(directory/'complete.json',result)
        print('EPISODE_READY',identity,result['particles'],geometry_policy,observation_kind,flush=True)
        return result

def subprocess_env(gpu=None):
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env

def prepare_farm(root,workers=4):
    root=Path(root);p=read_json(root/'plan.json')
    episodes=sorted(p['episodes'],key=lambda e:(e['episode']!=0,not bool(e['reusable_initial_workspace']),e['episode'],e['particles_estimate']))
    def worker(e):
        directory=Path(e['preparation']).parent
        if Path(e['preparation']).exists():return
        log=root/'logs'/('prepare_'+e['id'].replace('/','_')+'.log')
        cmd=[sys.executable,'-u','-m','softdynamics.teacher50','prepare-episode','--root',str(root),'--identity',e['id']]
        with log.open('a') as stream:
            result=subprocess.run(cmd,cwd=p['code_root'],env=subprocess_env(),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(directory/'failure.json',dict(id=e['id'],returncode=result.returncode,log=str(log),time=time.time()))
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(worker,episodes))
    ready=[e['id'] for e in episodes if Path(e['preparation']).exists()]
    write_json(root/'preparation_status.json',dict(status='complete' if len(ready)==len(episodes) else 'incomplete',
        expected=len(episodes),ready=len(ready),failed=[e['id'] for e in episodes if e['id'] not in ready],updated_at=time.time()))

def farm(root,gpus):
    root=Path(root);p=read_json(root/'plan.json');active={};attempted=set()
    control=root/'run_control.json'
    if control.exists() and not read_json(control).get('collection_enabled',True):
        raise RuntimeError('Teacher collection is stopped by the user; initialization repair is independent.')
    from .teacher50_schedule import workload,choose_task,result_path,size_tier,required_free_mib
    work=workload(p)
    lock=(root/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    started=time.time()
    while True:
        if stopping or (control.exists() and not read_json(control).get('collection_enabled',True)):
            for process,task,stream in active.values():process.terminate()
            for process,task,stream in active.values():
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                stream.close()
            active.clear()
            status=read_json(root/'collection_status.json') if (root/'collection_status.json').exists() else {}
            status.update(status='stopped_by_user',active=[],stopped_at=time.time(),updated_at=time.time())
            write_json(root/'collection_status.json',status)
            if control.exists():
                settings=read_json(control);settings['collection_enabled']=False;write_json(control,settings)
            break
        for gpu,(process,task,stream) in list(active.items()):
            if process.poll() is not None:
                stream.close();active.pop(gpu)
                if not (root/'trajectories'/task['id']/'complete.json').exists():
                    write_json(root/'trajectories'/task['id']/'worker_failure.json',dict(returncode=process.returncode,updated_at=time.time()))
        valid=[];failed=[];pending=[];objects=set()
        retries=read_json(root/'retry_index.json') if (root/'retry_index.json').exists() else {}
        for task in p['tasks']:
            receipt=result_path(root,task,retries)
            if receipt.exists():
                r=read_json(receipt)
                if r.get('valid') and r.get('full_rollout'):valid.append(task['id']);objects.add(task['object'])
                else:failed.append(task['id'])
            elif task['id'] not in attempted and Path(task['episode_preparation']).exists():pending.append(task)
        gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        resources={}
        for row in gpu_rows.strip().splitlines():
            gpu,free,util=map(int,row.split(','));resources[gpu]=dict(free_mib=free,utilization=util)
        # Shared GPUs may be busy but still have enough memory. Fill available
        # cards starting with the least busy; never stop another user's job.
        for gpu in sorted(gpus,key=lambda g:(resources[g]['utilization'],-resources[g]['free_mib'])):
            if gpu in active or not pending:continue
            if shutil.disk_usage(root).free<35*1024**3:continue
            task=choose_task(pending,[t['object'] for _,t,_ in active.values()],work,resources[gpu]['free_mib'])
            if task is None:continue
            pending.remove(task);attempted.add(task['id'])
            log=root/'logs'/('collect_'+task['id'].replace('/','_')+'.log');stream=log.open('a')
            command=[sys.executable,'-u','-m','softdynamics.elastic_data','collect','--root',str(root),'--task',task['id']]
            process=subprocess.Popen(command,cwd=p['code_root'],env=subprocess_env(gpu),stdout=stream,stderr=subprocess.STDOUT)
            active[gpu]=(process,task,stream)
        ready=sum(Path(e['preparation']).exists() for e in p['episodes'])
        prep_done=(root/'preparation_status.json').exists()
        running=[dict(gpu=g,pid=v[0].pid,id=v[1]['id'],
            particles=work['episodes'][f"{v[1]['object']}/episode_{v[1]['episode']}"]['particles'],
            size_tier=size_tier(work['object_particles'][v[1]['object']])) for g,v in active.items()]
        status=dict(status='running',pid=os.getpid(),started_at=started,updated_at=time.time(),
            expected_objects=len(p['objects']),expected_episodes=len(p['episodes']),expected_trajectories=len(p['tasks']),
            prepared_episodes=ready,valid_trajectories=len(valid),failed_trajectories=len(failed),
            objects_with_valid_trajectory=len(objects),completed_objects=sorted(objects),active=running,
            pending_prepared_tasks=len(pending),gpus=gpus,training_started=False,
            gpu_resources=resources,scheduling='small_then_medium_then_large_with_object_diversity',
            registered_successful_retries=len(retries),
            disk_free_bytes=shutil.disk_usage(root).free)
        exhausted=not active and not pending and prep_done
        if exhausted:status['status']='complete' if len(valid)==len(p['tasks']) else 'incomplete'
        write_json(root/'collection_status.json',status)
        if exhausted:break
        time.sleep(10)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','prepare-episode','prepare-farm','farm'])
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--identity')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--gpus',type=int,nargs='+',default=[0,2,3,4,6,7])
    a=parser.parse_args()
    if a.mode=='plan':plan(a.root)
    elif a.mode=='prepare-episode':prepare_episode(a.root,a.identity)
    elif a.mode=='prepare-farm':prepare_farm(a.root,a.workers)
    else:farm(a.root,a.gpus)

if __name__=='__main__':main()
