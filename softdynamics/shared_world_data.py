"""Cross-object ordinary dynamics data from the established corrected-pad assets.

All split decisions precede simulation. A failed rollout is recorded as failed,
never silently shortened. The physical engine is the original first-module
runtime; only material values and continuous controller actions are randomized.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .data import fps
from .io import load_npz, read_json, sha256, write_json
from .parameters import PARAMETERS
from .world_data import controller_features


def stable_seed(seed, identity):
    return int.from_bytes(hashlib.sha256(f'{seed}/{identity}'.encode()).digest()[:4], 'little')


def object_split(obj, config):
    sets = {k: set(config['dataset'][k+'_objects']) for k in ('validation', 'test')}
    if sets['validation'] & sets['test']:
        raise ValueError('Object split overlap')
    return next((k for k, values in sets.items() if obj in values), 'train')


def randomize_actions(poses, valid, times, start, end, seed, config):
    """Smooth physically coherent gripper motion; keep two fingers in one frame.

    Invalid observed poses stay invalid. Initial and operation endpoint controls
    are unchanged. No particle forces, observed future shapes, or response labels.
    """
    result = np.array(poses, copy=True)
    if end <= start:
        return result
    rng = np.random.default_rng(seed)
    u = np.clip((times-times[start])/(times[end]-times[start]), 0, 1)
    envelope = np.sin(np.pi*u)**2
    for g in range(poses.shape[1]):
        translation = rng.uniform(-1, 1, 3)*config['action_translation_m']
        rotation = rng.uniform(-1, 1, 3)*config['action_rotation_rad']
        aperture = rng.uniform(-1, 1)*config['action_aperture_m']
        for t in np.flatnonzero(valid[:, g].all(-1)):
            if envelope[t] == 0:
                continue
            center = poses[t, g, :, :3, 3].mean(0)
            axis = poses[t, g, 1, :3, 3]-poses[t, g, 0, :3, 3]
            width = np.linalg.norm(axis)
            if width < 1e-6:
                continue
            R = Rotation.from_rotvec(rotation*envelope[t]).as_matrix()
            dw = max(-.4*width, aperture*envelope[t])
            for side, sign in ((0, -1), (1, 1)):
                offset = poses[t, g, side, :3, 3]-center+sign*.5*dw*axis/width
                result[t, g, side, :3, 3] = center+R@offset+translation*envelope[t]
                result[t, g, side, :3, :3] = R@poses[t, g, side, :3, :3]
    return result


def build_nodes(body, config, seed):
    rng = np.random.default_rng(seed)
    n = config['nodes']; ns = int(n*config['surface_fraction'])
    surface = body['surface_particle_ids']
    interior = np.setdiff1d(np.arange(len(body['rest_positions'])), surface)
    chosen = []
    for candidates, count in ((surface, ns), (interior, n-ns)):
        if len(candidates) < count:
            raise ValueError('Asset has insufficient distinct surface/interior nodes')
        if len(candidates) > config['fps_candidates']:
            candidates = np.sort(rng.choice(candidates, config['fps_candidates'], replace=False))
        chosen.append(candidates[fps(body['rest_positions'][candidates], count)])
    ids = np.concatenate(chosen)
    rest = body['rest_positions'][ids]
    scale = float(np.ptp(body['rest_positions'], axis=0).max())
    center = body['rest_positions'].mean(0)
    tree = cKDTree(rest)
    distance, binding = tree.query(body['rest_positions'], k=4)
    weights = 1/np.maximum(distance, 1e-8)**2
    weights /= weights.sum(-1, keepdims=True)
    return dict(material_indices=ids, canonical=((rest-center)/scale).astype('f'),
                rest_positions=rest.astype('f'), scale_m=scale, rest_center_m=center,
                neighbors=tree.query(rest, k=config['neighbors']+1)[1][:, 1:],
                surface_mask=np.arange(n)<ns, dense_binding=binding.astype('i'),
                dense_weights=weights.astype('f'), total_mass_kg=body['masses_kg'].sum(),
                density_kg_m3=float(body['masses_kg'].sum()/body['rest_volumes_m3'].sum()))


def prepare(root, pretrain, fit_run, config_path):
    root, pretrain, fit_run = (Path(p).resolve() for p in (root, pretrain, fit_run))
    config_path = Path(config_path).resolve(); cfg = read_json(config_path)
    if cfg.get('response_supervision') or cfg.get('tactile_input'):
        raise ValueError('The authorized protocol excludes response and tactile supervision')
    if (root/'plan.json').exists():
        plan = read_json(root/'plan.json')
        if plan['config_sha256'] != sha256(config_path):
            raise ValueError('Existing immutable data configuration differs')
        return plan
    source_cfg = read_json(pretrain/'config.json')
    fit = read_json(fit_run/'plan.json')
    repo = Path(__file__).resolve().parents[1]
    original_runtime = Path(fit['runtime'])
    physics_paths = ['run_episode.py', 'gs-layer/material_contact_kernels.py',
                     'gs-layer/pad_contact_kernels.py', 'gs-layer/deterministic_transfer.py']
    physics_paths += [str(p.relative_to(repo/'runtime')) for p in (repo/'runtime/empm').rglob('*.py')]
    for name in physics_paths:
        p = repo/'runtime'/name
        ref = original_runtime/name
        if not ref.exists() or sha256(p) != sha256(ref):
            raise ValueError('The shared model must inherit the first-module physical engine: '+str(p))
    for name in ('empm.py', 'pad_empm.py'):
        if sha256(repo/'softdynamics/adapters'/name) != sha256(original_runtime.parent/'softdynamics/adapters'/name):
            raise ValueError('Changed first-module physics adapter: '+name)
    root.mkdir(parents=True, exist_ok=True)
    for name in ('runtime', 'softdynamics'):
        shutil.copytree(repo/name, root/'code'/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    runtime = root/'code/runtime'
    sources = {str(pretrain/'selection.json'):sha256(pretrain/'selection.json'),
               str(pretrain/'config.json'):sha256(pretrain/'config.json'),
               str(fit_run/'plan.json'):sha256(fit_run/'plan.json'),
               fit['checkpoint']:fit['model_sha256'], str(config_path):sha256(config_path)}
    # Freeze exact first-module selections/weights for subsequent inverse tests.
    handoffs = {}
    for obj in fit['objects']:
        name = obj['object']; selection = fit_run/'selections'/name/'real_start0/96/learned.json'
        weight = fit_run/'weights'/f'{name}.json'
        from .fit_handoff import audit_weighted_trace
        sel = read_json(selection) if selection.exists() else None
        if not weight.exists():
            handoffs[name] = dict(selection=None, weights=None, selected_q=None,
                support=obj['support'], query=obj['query'],
                status='requires_confidence_prediction_from_frozen_checkpoint')
            continue
        w = read_json(weight)
        ctx = next(c for c in w['contexts'] if c['id']=='real_start0')
        if (sel is not None and sel['query_observations_used']) or w['model_sha256'] != fit['model_sha256']:
            raise ValueError('Invalid first-module handoff')
        audit = audit_weighted_trace(sel, ctx['weights']['learned']) if sel is not None else None
        handoffs[name] = dict(selection=str(selection) if sel is not None else None, weights=str(weight),
            selected_q=sel['selected_q'] if sel is not None else None,
            status='accepted_fit_available' if sel is not None else 'weights_only_no_previous_fit',
            initial_q=ctx['start'], support=obj['support'], query=obj['query'],
            confidence_weights=ctx['weights']['learned'], audit=audit)
        if sel is not None:sources[str(selection)] = sha256(selection)
        sources[str(weight)] = sha256(weight)
    objects = []; tasks = []
    margins = {(o['object'], e['episode']):e['domain_margin'] for o in fit['objects'] for e in o['episodes']}
    # Prior successful nominal simulations inform numerical domains, not selection
    # by test error. They do not supply any world-model training targets.
    old_manifests = [read_json(p) for p in sorted((pretrain/'labels_v1').glob('*/*/*/manifest.json'))]
    for m in old_manifests:
        key = (m['object'], m['episode'])
        margins[key] = max(margins.get(key, .075), m['domain'].get('margin_m', .075))
    upper = cfg['material_upper']; rho = source_cfg['physics']['density_kg_m3']
    dtmax = min(source_cfg['physics']['max_dt_s'], .35*.003125/np.sqrt(
        upper[0]*(1-upper[1])/(rho*(1+upper[1])*(1-2*upper[1]))))
    for workspace in sorted((pretrain/'objects').iterdir()):
        rows = sorted((workspace/'prepared').glob('episode_*/complete.json'))
        if not rows or not (workspace/'asset/asset.json').exists():
            continue
        obj = workspace.name; split = object_split(obj, source_cfg)
        out = root/'objects'/obj; out.mkdir(parents=True, exist_ok=True)
        asset = read_json(workspace/'asset/asset.json')
        if asset['material']['model'] != 'fixed corotated elastic':
            raise ValueError('Unsupported material law')
        for name, digest in asset['payload_sha256'].items():
            path = workspace/'asset'/name
            if sha256(path) != digest:
                raise ValueError('Changed asset payload')
            sources[str(path)] = digest
        nodes = build_nodes(load_npz(workspace/'asset/body.npz'), cfg, stable_seed(cfg['seed'], obj))
        np.savez_compressed(out/'nodes.npz', **nodes)
        item = dict(object=obj, split=split, workspace=str(workspace), particles=asset['particles'],
                    nodes=str(out/'nodes.npz'), nodes_sha256=sha256(out/'nodes.npz'),
                    asset_sha256=sha256(workspace/'asset/asset.json'), episodes=[])
        sources[str(workspace/'asset/asset.json')] = item['asset_sha256']
        for row_path in rows:
            row = read_json(row_path)
            if row['status'] != 'ready' or row['split'] != split:
                raise ValueError('Prepared split/status mismatch')
            episode = row['episode']; item['episodes'].append(episode)
            files = [row_path, Path(row['metadata']), Path(row['observation']),
                     workspace/f'initial_states/episode_{episode}.npz',
                     workspace/f'queries/episode_{episode}/actions.npz']
            for p in files:
                sources[str(p)] = sha256(p)
            for variant in range(cfg['variants_per_episode']):
                identity = f'{obj}/episode_{episode}/sample_{variant}'
                seed = stable_seed(cfg['seed'], identity)
                rng = np.random.default_rng(seed)
                theta = np.array([source_cfg['baseline'][k] for k in PARAMETERS]) if variant==0 else rng.uniform(cfg['material_lower'],cfg['material_upper'])
                parameters = dict(source_cfg['baseline']); parameters.update(zip(PARAMETERS,theta.tolist()))
                tasks.append(dict(id=identity, object=obj, episode=episode, row=str(row_path),
                    split=split, variant=variant, seed=seed, parameters=parameters,
                    randomized_actions=variant>0, dtmax=dtmax,
                    domain_margin=margins.get((obj, episode), .075), particles=asset['particles'],
                    input_hashes={str(p):sources[str(p)] for p in files}))
        objects.append(item)
    # First cover every object; then every episode; then add material/action diversity.
    objects_by_name = {o['object']:o for o in objects}
    for o in objects:
        o['episodes'].sort()
    tasks.sort(key=lambda t:(t['variant'], objects_by_name[t['object']]['episodes'].index(t['episode']),
                             t['particles'],t['object']))
    plan = dict(schema='softdynamics.shared-world-data.v1', created_at=time.time(),config=cfg,
                config_sha256=sha256(config_path), pretrain=str(pretrain), fit_run=str(fit_run),
                runtime=str(runtime), objects=objects, tasks=tasks,source_sha256=sources,
                first_module_handoffs=handoffs, confidence_checkpoint=fit['checkpoint'],
                confidence_sha256=fit['model_sha256'], response_supervision=False,
                split_rule='inherited object-disjoint first-module split',
                numerical_protocol=dict(dtmax=dtmax,parameter_independent_timestep=True,
                    same_physics_source=True,same_full_assets=True,
                    note='Fixed stable timestep covers the entire new material box; may differ from old CMA timestep.'))
    write_json(root/'plan.json',plan)
    return plan


def collect(root, task_id, device='cuda:0', smoke_frames=None,gpu_observation=False,domain_padding_cells=0,lower_padding_cells=None):
    from .adapters.pad_empm import PadRuntime
    root = Path(root).resolve(); plan = read_json(root/'plan.json')
    task = next(t for t in plan['tasks'] if t['id']==task_id)
    obj = next(o for o in plan['objects'] if o['object']==task['object'])
    out = root/('smoke' if smoke_frames else 'trajectories')/task_id
    original_failure=None
    if lower_padding_cells is not None and not domain_padding_cells:raise ValueError('Directed padding requires an existing boundary failure')
    if domain_padding_cells:
        original_path=root/'trajectories'/task_id/'complete.json';original_failure=read_json(original_path)
        if original_failure['valid'] or 'domain_boundary' not in original_failure.get('error',''):
            raise ValueError('Grid retry requires an existing recorded domain-boundary failure')
        out=root/'retries'/task_id/f'padding_{domain_padding_cells}'
        if lower_padding_cells is not None:out=out.with_name(out.name+'_lower_'+'_'.join(map(str,lower_padding_cells)))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'task.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():
            return read_json(out/'complete.json')
        started = time.monotonic()
        try:
            for path,digest in task['input_hashes'].items():
                if sha256(path)!=digest:raise ValueError('Changed frozen input: '+path)
            if sha256(obj['nodes'])!=obj['nodes_sha256']:raise ValueError('Changed nodes')
            from .shared_world_observation import padded_simulation_grid
            with padded_simulation_grid(domain_padding_cells,lower_padding_cells):
                sim=PadRuntime(task['episode'],task['parameters'],obj['workspace'],plan['runtime'],
                               task['dtmax'],device,task['domain_margin'])
            nodes=load_npz(obj['nodes']); ids=nodes['material_indices']
            observer=None
            if gpu_observation and sim.n>=200000:
                from .shared_world_observation import GPUObservation
                observer=GPUObservation(sim,ids);observer.install_validation()
            origin=sim.positions(np.arange(sim.n)).mean(0);scale=float(nodes['scale_m'])
            if task['randomized_actions']:
                sim.poses=randomize_actions(sim.poses,sim.valid,sim.times,sim.meta['clip_start'],
                    sim.meta['clip_end'],task['seed'],plan['config'])
            T=min(len(sim.times),smoke_frames+1) if smoke_frames else len(sim.times)
            state=np.empty((T,len(ids),24),np.float32)
            pads=controller_features(sim,origin,scale)[:T]
            def capture(frame):
                if observer is not None:
                    state[frame]=observer.capture();return
                s=sim.states[sim.current]
                state[frame]=np.concatenate([s.particle_x.numpy()[ids]-sim.shift,
                    s.particle_v.numpy()[ids],s.particle_F_trial.numpy()[ids].reshape(-1,9),
                    s.particle_C.numpy()[ids].reshape(-1,9)],-1)
            capture(0); minimum_j=float('inf')
            for frame in range(1,T):
                check=sim.advance(frame)
                if not check['valid']:raise ValueError('Invalid physical rollout: '+str(check))
                minimum_j=min(minimum_j,check['min_J']);capture(frame)
                if frame%25==0 or frame==T-1:
                    progress=dict(pid=os.getpid(),task=task_id,frame=frame,total=T,
                                  elapsed_s=time.monotonic()-started,updated_at=time.time())
                    write_json(out/'progress.json',progress)
                    print('COLLECT',task_id,frame,T,round(progress['elapsed_s'],1),flush=True)
            np.savez_compressed(out/'trajectory.tmp.npz',state=state,pads=pads,times=sim.times[:T],
                origin_m=origin,scale_m=scale,theta=np.array([task['parameters'][k] for k in PARAMETERS]),
                phase=np.where(np.arange(T)<sim.meta['clip_start'],0,np.where(np.arange(T)<=sim.meta['clip_end'],1,2)),
                source_poses=sim.poses[:T],source_valid=sim.valid[:T])
            (out/'trajectory.tmp.npz').replace(out/'trajectory.npz')
            result=dict(valid=True,task=task,frames=T,elapsed_s=time.monotonic()-started,
                trajectory_sha256=sha256(out/'trajectory.npz'),source_hashes=sim.source_hashes,
                nodes_sha256=obj['nodes_sha256'],domain=sim.domain,min_J=minimum_j,
                full_rollout=not bool(smoke_frames),future_observed_geometry_used=False,
                recorded_initial_state_preserved=True,response_labels_used=False,
                observation_backend='gpu_read_only' if observer is not None else 'original_cpu',
                observation_shadow_checks=observer.shadow_checks if observer is not None else [],
                observation_source_sha256=sha256(Path(__file__).with_name('shared_world_observation.py')) if observer is not None else None,
                numerical_grid_padding_cells=domain_padding_cells,
                numerical_lower_padding_cells=lower_padding_cells,
                original_failure_sha256=sha256(root/'trajectories'/task_id/'complete.json') if original_failure is not None else None)
        except Exception as exc:
            traceback.print_exc()
            result=dict(valid=False,task=task,error=repr(exc),traceback=traceback.format_exc(),
                        elapsed_s=time.monotonic()-started,full_rollout=False)
            if 'sim' in locals():
                result.update(failure_frame=sim.frame,domain=sim.domain)
                try:
                    failed=sim.states[sim.current]
                    world=failed.particle_x.numpy()-sim.shift
                    speed=np.linalg.norm(failed.particle_v.numpy(),axis=-1)
                    result['failure_state_summary']=dict(world_min=world.min(0),world_max=world.max(0),
                        speed_quantiles_m_s=np.quantile(speed,[0,.5,.95,.99,1]),
                        grid_world_min=-sim.shift,grid_world_max=sim.model.grid_lim-sim.shift)
                except Exception as diagnostic_error:
                    result['failure_diagnostic_error']=repr(diagnostic_error)
        write_json(out/'complete.json',result)
        if result['valid'] and domain_padding_cells:
            with (root/'retry_index.lock').open('w') as index_lock:
                fcntl.flock(index_lock,fcntl.LOCK_EX)
                path=root/'retry_index.json';index=read_json(path) if path.exists() else {}
                index[task_id]=dict(folder=str(out.relative_to(root)),complete_sha256=sha256(out/'complete.json'),
                    original_failure_sha256=result['original_failure_sha256'],domain_padding_cells=domain_padding_cells,
                    lower_padding_cells=lower_padding_cells)
                write_json(path,index)
        return result


def task_available(root,identity):
    """A busy collector is not a failed simulation and must not be overwritten."""
    folder=Path(root)/'trajectories'/identity
    if (folder/'complete.json').exists():return False
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'task.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def farm(root,gpus,task_ids=None,worker_code=None,gpu_observation=False,drain_pids=None):
    root=Path(root).resolve();plan=read_json(root/'plan.json');(root/'logs').mkdir(exist_ok=True)
    if task_ids is None:
        tasks=plan['tasks'];label='farm'
    else:
        if len(set(task_ids))!=len(task_ids):raise ValueError('Repeated task IDs')
        lookup={t['id']:t for t in plan['tasks']}
        tasks=[lookup[identity] for identity in task_ids]
        label='priority_'+hashlib.sha256('\n'.join(task_ids).encode()).hexdigest()[:12]
    with (root/(label+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import threading
        claim=threading.Lock();queue=list(tasks);active={}
        def worker(gpu):
            for pid in (drain_pids or {}).get(str(gpu),[]):
                path=Path('/proc')/str(pid)/'stat'
                while path.exists():
                    try:running=path.read_text().split()[2] not in ('Z','X')
                    except FileNotFoundError:break
                    if not running:break
                    time.sleep(1)
            while True:
                with claim:
                    queue[:]=[t for t in queue if not (root/'trajectories'/t['id']/'complete.json').exists()]
                    if not queue:return
                    pending=next((t for t in queue if task_available(root,t['id'])),None)
                    if pending is not None:queue.remove(pending)
                if pending is None:
                    time.sleep(1);continue
                name=pending['id'].replace('/','__');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                with (root/'logs'/f'{name}.log').open('a') as log:
                    command=[sys.executable,'-u','-m','softdynamics.shared_world_data','collect',
                             '--root',str(root),'--task',pending['id']]
                    if gpu_observation:command.append('--gpu-observation')
                    p=subprocess.Popen(command,cwd=worker_code or root/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
                    with claim:
                        active[str(gpu)]=dict(pid=p.pid,task=pending['id'],started_at=time.time())
                        write_json(root/(label+'_status.json'),dict(pid=os.getpid(),active=active,updated_at=time.time()))
                    code=p.wait()
                    path=root/'trajectories'/pending['id']/'complete.json'
                    if not path.exists():
                        if code==75 or not task_available(root,pending['id']):
                            with claim:queue.append(pending)
                        else:
                            # Hold the actual output lock while writing a crash
                            # record, closing the scheduler/collector race.
                            with (path.parent/'task.lock').open('a') as task_lock:
                                try:fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                                except BlockingIOError:
                                    with claim:queue.append(pending)
                                else:
                                    if not path.exists():write_json(path,dict(valid=False,task=pending,error='Worker exit '+str(code)))
                    with claim:
                        active.pop(str(gpu),None)
                        write_json(root/(label+'_status.json'),dict(pid=os.getpid(),active=active,updated_at=time.time()))
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker,gpus))
        records=[read_json(root/'trajectories'/t['id']/'complete.json') for t in tasks]
        full=task_ids is None or set(task_ids)=={t['id'] for t in plan['tasks']}
        final='collection_complete.json' if full else label+'_complete.json'
        write_json(root/final,dict(tasks=len(records),valid=sum(bool(r['valid']) for r in records),
                   failed=[r['task']['id'] for r in records if not r['valid']],completed_at=time.time()))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--pretrain',type=Path,required=True);a.add_argument('--fit-run',type=Path,required=True)
    a.add_argument('--config',type=Path,required=True)
    a=sub.add_parser('collect');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--task',required=True);a.add_argument('--device',default='cuda:0');a.add_argument('--smoke-frames',type=int)
    a.add_argument('--gpu-observation',action='store_true')
    a.add_argument('--domain-padding-cells',type=int,default=0)
    a.add_argument('--lower-padding-cells',type=int,nargs=3)
    a=sub.add_parser('farm');a.add_argument('--root',type=Path,required=True);a.add_argument('--gpus',type=int,nargs='+',required=True)
    a.add_argument('--task-list',type=Path,help='Ordered subset of existing tasks, for early replay diversity')
    a.add_argument('--worker-code',type=Path);a.add_argument('--gpu-observation',action='store_true')
    a.add_argument('--drain-pids',type=Path,help='GPU-to-PIDs mapping; let existing physical rollouts finish before using each GPU')
    a=p.parse_args()
    if a.mode=='prepare':
        plan=prepare(a.root,a.pretrain,a.fit_run,a.config)
        print('PREPARED',len(plan['objects']),'objects',len(plan['tasks']),'tasks',flush=True)
    elif a.mode=='collect':
        try:result=collect(a.root,a.task,a.device,a.smoke_frames,a.gpu_observation,a.domain_padding_cells,a.lower_padding_cells)
        except BlockingIOError:raise SystemExit(75)
        if not result['valid']:raise SystemExit(1)
    else:farm(a.root,a.gpus,read_json(a.task_list) if a.task_list else None,a.worker_code,a.gpu_observation,
        read_json(a.drain_pids) if a.drain_pids else None)


if __name__=='__main__':main()
