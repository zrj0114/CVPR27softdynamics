"""Fresh E/nu trajectories and fixed-state sensitivity labels from existing assets."""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.stats import qmc

from .io import load_npz, read_json, sha256, write_json
from .shared_world_data import build_nodes, stable_seed
from .bound_observation import BoundObservation
from .teacher_contact import (contract,selected_architecture,require_selected_architecture,
    validate_contact_parameters,verify_selected_kernels)

PARAMETERS = ('E_pa', 'nu')


class ElasticSpace:
    """Two coordinates; all other physical values remain fixed constants."""
    def __init__(self, config):
        self.base = dict(config['baseline'])
        self.reference = float(config['reference_E_pa'])
        self.nu_reference = float(config['reference_nu'])
        self.nu_scale = float(config['nu_scale'])

    def parameters(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError('Expected exactly two finite coordinates')
        p = dict(self.base)
        p['E_pa'] = self.reference * math.exp(float(q[0]))
        p['nu'] = self.nu_reference + self.nu_scale * float(q[1])
        if not p['E_pa'] > 0 or not 0 < p['nu'] < .49:
            raise ValueError('Unsupported elastic material')
        return p

    def coordinate(self, p):
        return np.array([math.log(p['E_pa']/self.reference),
                         (p['nu']-self.nu_reference)/self.nu_scale])


def prepare(root, source, code_root):
    root, source, code_root = map(lambda p: Path(p).resolve(), (root, source, code_root))
    if (root/'plan.json').exists():
        plan=read_json(root/'plan.json')
        if plan.get('contact_contract')!=contract():
            raise ValueError('Existing teacher plan predates the selected contact/actions; prepare a fresh data directory')
        require_selected_architecture(plan['config']['model'])
        return plan
    root.mkdir(parents=True, exist_ok=True)
    old = read_json(source/'config.json')
    config = dict(baseline=old['baseline'], reference_E_pa=3500., reference_nu=.27,
                  nu_scale=.2, E_range_pa=[1000.,12000.], nu_range=[.08,.42],
                  variants_per_episode=8, seed=0, nodes=512, neighbors=24,
                  surface_fraction=.75, fps_candidates=8192,
                  phase_weights=old['fitting']['phase_weights'],
                  teacher=dict(radii=[.05,.025], refinement_radii=[.0125,.00625],
                      relative_column_tolerance=.25, windows_per_phase=2, points=256),
                  model=dict(hidden=256,layers=4,heads=4,kinematic_only=True,
                      direct_displacement=True,contact_geometry=True,material_dim=2))
    validate_contact_parameters(config['baseline'])
    verify_selected_kernels(code_root/'runtime')
    config['model']=selected_architecture(config['model'])
    space = ElasticSpace(config)
    # Cover all nominal samples, query offsets, and teacher probes with one dt.
    max_E = config['E_range_pa'][1]*math.exp(.1)
    max_nu = config['nu_range'][1]+.1*config['nu_scale']
    dtmax = min(.0002,.35*.003125/math.sqrt(max_E*(1-max_nu)/(100*(1+max_nu)*(1-2*max_nu))))
    objects, tasks = [], []
    exclusions = []
    for workspace in sorted((source/'objects').iterdir()):
        ready = [read_json(p) for p in sorted((workspace/'prepared').glob('episode_*/complete.json'))]
        ready = sorted([r for r in ready if r.get('status')=='ready'], key=lambda r:r['episode'])
        if not ready:
            continue
        obj, split = workspace.name, ready[0]['split']
        if split == 'test':
            exclusions.append(obj)
            continue
        body = load_npz(workspace/'asset/body.npz')
        nodes = build_nodes(body, config, stable_seed(0,obj))
        folder = root/'objects'/obj
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'nodes.npz', **nodes)
        item = dict(object=obj,split=split,workspace=str(workspace),
            nodes=str(folder/'nodes.npz'),nodes_sha256=sha256(folder/'nodes.npz'),
            episodes=[r['episode'] for r in ready],particles=len(body['rest_positions']))
        objects.append(item)
        for episode_index,row in enumerate(ready):
            unit = qmc.LatinHypercube(2,seed=stable_seed(0,row['id'])).random(7)
            nominal = [[3500.,.27]] + [[math.exp(math.log(1000.)+u[0]*math.log(12.)),
                                       .08+.34*u[1]] for u in unit]
            row_path=workspace/'prepared'/f"episode_{row['episode']}"/'complete.json'
            for variant,theta in enumerate(nominal):
                params = dict(config['baseline'],E_pa=theta[0],nu=theta[1])
                tasks.append(dict(id=f"{obj}/episode_{row['episode']}/sample_{variant}",
                    object=obj,episode=row['episode'],split=split,variant=variant,
                    episode_index=episode_index,row=str(row_path),row_sha256=sha256(row_path),
                    parameters=params,q=space.coordinate(params).tolist(),dtmax=dtmax,
                    domain_margin=.075,randomized_actions=False,particles=item['particles']))
    # Ensure material diversity on all objects before filling every interaction.
    tasks.sort(key=lambda t:(t['episode_index']>0,t['variant'],t['episode_index'],t['particles'],t['object']))
    plan = dict(schema='softdynamics.elastic-scratch-data.v1',created_at=time.time(),
        source=str(source),runtime=str(code_root/'runtime'),config=config,objects=objects,tasks=tasks,
        parameters=list(PARAMETERS),test_objects=['048-butter-sponge','125-rabbit','187-white-bear'],
        excluded_objects=exclusions,old_weights_used=False,old_simulation_targets_used=False,
        split_policy='Reuse object-disjoint train/validation assignments; exclude every historical test object',
        numerical_protocol=dict(dtmax=dtmax,parameter_independent_timestep=True),
        contact_contract=contract(),
        physics_source_sha256={str(p.relative_to(code_root)):sha256(p)
            for p in (code_root/'runtime').rglob('*.py')})
    write_json(root/'plan.json',plan)
    return plan


def collect(root, identity, device='cuda:0', smoke_frames=None):
    from .adapters.pad_empm import PadRuntime
    from .shared_world_observation import GPUObservation, padded_simulation_grid
    from .world_data import controller_features
    from .shared_data import normalize_pads
    from .teacher import generate
    root=Path(root);plan=read_json(root/'plan.json');cfg=plan['config']
    acceleration_path=root/'acceleration.json'
    acceleration_settings=read_json(acceleration_path) if acceleration_path.exists() else {}
    if acceleration_settings.get('execution_code_root'):
        execution_root=Path(acceleration_settings['execution_code_root']).resolve()
        if execution_root!=Path(__file__).resolve().parents[1]:
            raise ValueError('Use the configured collection execution revision')
        plan=dict(plan,runtime=str(execution_root/'runtime'))
    if acceleration_settings.get('enabled'):
        for option,variable in [('active_grid','SOFTDYNAMICS_ACTIVE_GRID'),
                                ('snapshot_cache','SOFTDYNAMICS_SNAPSHOT_CACHE'),
                                ('branch_cache','SOFTDYNAMICS_TEACHER_CACHE'),
                                ('compact_observation','SOFTDYNAMICS_COMPACT_OBSERVATION'),
                                ('sparse_index_reset','SOFTDYNAMICS_SPARSE_INDEX_RESET'),
                                ('active_p2g','SOFTDYNAMICS_ACTIVE_P2G')]:
            os.environ[variable]='1' if acceleration_settings.get(option,False) else '0'
    control=root/'run_control.json'
    if control.exists() and not read_json(control).get('collection_enabled',True):
        raise RuntimeError('Teacher collection is stopped by the user; initialization repair is independent.')
    task=next(t for t in plan['tasks'] if t['id']==identity)
    execution_sources={relative:sha256(Path(__file__).parents[1]/relative)
        for relative in ['softdynamics/teacher.py','softdynamics/adapters/empm.py',
                         'softdynamics/bound_observation.py','runtime/gs-layer/active_grid_kernels.py']}
    obj=next(o for o in plan['objects'] if o['object']==task['object'])
    if task.get('episode_preparation'):
        preparation=read_json(task['episode_preparation'])
        if preparation['status']!='ready' or preparation['object']!=task['object'] or preparation['episode']!=task['episode']:
            raise ValueError('Teacher task and prepared episode differ')
        for path,digest in preparation['source_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Prepared episode source changed: '+path)
        task=dict(task,row=preparation['row'],row_sha256=preparation['row_sha256'])
        obj=dict(obj,workspace=preparation['workspace'],nodes=preparation['nodes'],nodes_sha256=preparation['nodes_sha256'])
    out=root/('preflight' if smoke_frames else 'trajectories')/identity
    out.mkdir(parents=True,exist_ok=True)
    with (out/'task.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'complete.json').exists():return read_json(out/'complete.json')
        from .teacher50_schedule import archive_incomplete
        archive_incomplete(out)
        started=time.monotonic()
        if sha256(task['row']) != task['row_sha256']:raise ValueError('Prepared input changed')
        if sha256(obj['nodes']) != obj['nodes_sha256']:raise ValueError('Node map changed')
        row=read_json(task['row']);nodes=load_npz(obj['nodes']);workspace=Path(obj['workspace'])
        # Medium repair policies fix the episode lattice. Implicit padding would
        # translate float32 coordinates and break consistency between variants.
        for padding in (0,):
            attempt=out/f'attempt_{padding}';attempt.mkdir(exist_ok=True)
            try:
                with padded_simulation_grid(padding):
                    sim=PadRuntime(task['episode'],task['parameters'],workspace,plan['runtime'],
                                   task['dtmax'],device,task['domain_margin'])
                observer=GPUObservation(sim,nodes['material_indices']);observer.install_validation()
                equivalence=None
                if acceleration_settings.get('transfer_equivalence'):
                    alias=acceleration_settings['transfer_equivalence']
                    if sim.source_hashes['transfer']!=alias['executed']:
                        raise ValueError('Accelerated transfer does not match validated revision')
                    equivalence=dict(schema='softdynamics.transfer-equivalence.v1',
                        aliases={'transfer':alias},validation=acceleration_settings['validation'])
                if smoke_frames:
                    for f in range(1,min(smoke_frames+1,len(sim.times))):
                        check=sim.advance(f)
                        if not check['valid']:raise ValueError(str(check))
                    result=dict(valid=True,frames=sim.frame,task=task,domain=sim.domain,
                                elapsed_s=time.monotonic()-started,full_rollout=False)
                    write_json(out/'complete.json',result);return result
                T=len(sim.times);states=np.empty((T,len(nodes['material_indices']),24),np.float32)
                observation=load_npz(row['observation']);meta=read_json(row['metadata'])
                bound=BoundObservation(observation,compact=os.environ.get('SOFTDYNAMICS_COMPACT_OBSERVATION','0')=='1')
                meta['teacher_windows']=row['teacher_windows']
                synthetic=np.empty_like(observation['tracks']);minimum_j=float('inf')
                origin=sim.positions(np.arange(sim.n)).mean(0);scale=float(nodes['scale_m'])
                pads=controller_features(sim,origin,scale)
                def capture(runtime):
                    f=runtime.frame;states[f]=observer.capture()
                    if f<len(synthetic):
                        synthetic[f]=bound.capture(runtime)
                    if f%25==0 or f==T-1:
                        progress=dict(pid=os.getpid(),frame=f,total=T,phase='nominal',
                            task=identity,updated_at=time.time(),elapsed_s=time.monotonic()-started,
                            active_grid=runtime.transfer.active_grid,
                            sparse_index_reset=runtime.transfer.sparse_index_reset,
                            active_p2g=runtime.transfer.active_p2g,
                            observation_particles=len(bound.particle_ids),
                            execution_revision=acceleration_settings.get('revision','explicit_environment'))
                        write_json(out/'progress.json',progress);print('ELASTIC_COLLECT',progress,flush=True)
                space=ElasticSpace(cfg);spec=copy.deepcopy(cfg['teacher']);q=np.array(task['q'])
                spec['workpoints']=[q.tolist(),(q+np.array([.05,-.05])).tolist(),
                                    (q+np.array([-.05,.05])).tolist()]
                timings={}
                records=generate(sim,observation,meta,space,spec,row.get('phase_weights',cfg['phase_weights']),
                                 attempt/'teacher',nominal_observer=capture,timings=timings)
                packing_started=time.perf_counter()
                real=load_npz(row['real']);body=load_npz(workspace/'asset/body.npz')
                initial=load_npz(workspace/'initial_states'/f"episode_{task['episode']}.npz")
                context=normalize_pads(synthetic,observation['track_visible'],observation['tracks'][0],
                    body,initial,sim.poses[:len(synthetic)],sim.valid[:len(synthetic)],
                    sim.times[:len(synthetic)],observation['point_ids'],real['origin_m'],float(real['scale_m']))
                timings['packing_seconds']=time.perf_counter()-packing_started
                saving_started=time.perf_counter()
                np.savez_compressed(out/'confidence_input.npz',**context)
                np.savez_compressed(out/'trajectory.tmp.npz',state=states,pads=pads,times=sim.times,
                    origin_m=origin,scale_m=scale,theta=np.array([task['parameters'][k] for k in PARAMETERS]),
                    phase=np.where(np.arange(T)<meta['clip_start'],0,np.where(np.arange(T)<=meta['clip_end'],1,2)))
                (out/'trajectory.tmp.npz').replace(out/'trajectory.npz')
                manifest=dict(object=task['object'],episode=task['episode'],split=task['split'],
                    input=str(out/'confidence_input.npz'),input_sha256=sha256(out/'confidence_input.npz'),
                    metadata=row['metadata'],records=records,teacher_directory=str(attempt/'teacher'),
                    parameters=list(PARAMETERS),fixed_parameters={k:v for k,v in cfg['baseline'].items() if k not in PARAMETERS})
                write_json(out/'teacher_manifest.json',manifest)
                timings['saving_seconds']=time.perf_counter()-saving_started
                result=dict(valid=True,full_rollout=True,task=task,frames=T,
                    trajectory_sha256=sha256(out/'trajectory.npz'),nodes_sha256=obj['nodes_sha256'],
                    teacher_manifest=str(out/'teacher_manifest.json'),domain=sim.domain,nodes=obj['nodes'],
                    episode_preparation=task.get('episode_preparation'),
                    geometry_policy=row.get('geometry_policy','shared_object_asset'),
                    observation_kind=row.get('observation_kind','recorded_visual_track_sampling'),
                    numerical_grid_padding_cells=padding,source_hashes=sim.source_hashes,
                    elapsed_s=time.monotonic()-started,old_targets_used=False,created_at=time.time(),
                    timings=timings,acceleration=dict(active_grid=sim.transfer.active_grid,
                        snapshot_cache=os.environ.get('SOFTDYNAMICS_SNAPSHOT_CACHE','0')=='1',
                        branch_cache=os.environ.get('SOFTDYNAMICS_TEACHER_CACHE','0')=='1',
                        compact_observation=os.environ.get('SOFTDYNAMICS_COMPACT_OBSERVATION','0')=='1',
                        sparse_index_reset=sim.transfer.sparse_index_reset,
                        active_p2g=sim.transfer.active_p2g,
                        revision=acceleration_settings.get('revision','explicit_environment')))
                if equivalence is not None:result['source_equivalence']=equivalence
                result['execution_source_sha256']=execution_sources
                result['execution_code_root']=str(Path(__file__).resolve().parents[1])
                write_json(out/'complete.json',result);return result
            except Exception as exc:
                error=repr(exc);traceback.print_exc()
                failure=dict(valid=False,full_rollout=False,task=task,error=error,
                    elapsed_s=time.monotonic()-started,padding=padding,created_at=time.time())
                write_json(attempt/'failure.json',failure)
                if 'sim' in locals():del sim
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():torch.cuda.empty_cache()
                if not any(s in error for s in ('domain_boundary','outside allocated grid')):break
        write_json(out/'complete.json',failure);return failure


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--task');p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-frames',type=int)
    a=p.parse_args()
    if a.mode=='collect':
        from .collection_dispatch import redirect_collector
        redirect_collector(a.root)
    if a.mode=='prepare':prepare(a.root,a.source,Path(__file__).resolve().parents[1])
    elif not collect(a.root,a.task,a.device,a.smoke_frames)['valid']:raise SystemExit(2)


if __name__=='__main__':main()
