"""Freeze and launch a three-object joint-world acceleration experiment."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .io import read_json,write_json,sha256
from .parameters import PARAMETERS


def prepare(run,data_root,initialize_from,replay_manifest,test_objects,target_parameters=PARAMETERS):
    run,data_root,initialize_from,replay_manifest=map(lambda p:Path(p).resolve(),
        (run,data_root,initialize_from,replay_manifest))
    if (run/'protocol.json').exists():raise ValueError('Choose a new experiment directory')
    data=read_json(data_root/'plan.json');parent=torch.load(initialize_from,map_location='cpu',weights_only=False)
    split={o['object']:o['split'] for o in data['objects']}
    if len(set(test_objects))!=3 or any(split.get(o)!='test' for o in test_objects):
        raise ValueError('Choose exactly three existing object-disjoint test objects')
    if set(parent['training_objects'])&set(test_objects):raise ValueError('Initialization includes test objects')
    if not target_parameters or len(set(target_parameters))!=len(target_parameters) or any(p not in PARAMETERS for p in target_parameters):
        raise ValueError('Choose unique supported target parameters')
    if parent['data_plan_sha256']!=sha256(data_root/'plan.json'):raise ValueError('Initialization uses another data plan')
    run.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    for name in ('softdynamics','runtime'):
        shutil.copytree(repo/name,run/'code'/name,ignore=shutil.ignore_patterns('__pycache__','.warp-cache'))
    source=read_json(Path(data['pretrain'])/'selection.json')
    protocol=dict(schema='softdynamics.joint-acceleration.v1',created_at=time.time(),data_root=str(data_root),
        pretrain=data['pretrain'],fit_run=data['fit_run'],source_processed=source['source'],
        initialize_from=str(initialize_from),initialization_sha256=sha256(initialize_from),
        replay_manifest=str(replay_manifest),replay_sha256=sha256(replay_manifest),
        training_objects=parent['training_objects'],validation_objects=parent['validation_objects'],
        test_objects=list(test_objects),excluded_other_test_objects=[o for o,s in split.items() if s=='test' and o not in test_objects],
        methods=['joint_learned_adam','joint_uniform_adam','physical_uniform_cma'],
        target_parameters=list(target_parameters),
        inverse=dict(initial_q=[0.]*4,bounds=[-.025,.025],inner_steps=24,window_horizon=8,windows_per_phase=2,
            physical_rounds=12,report_budgets=[4,8,12],adam_lr=.002,trust_radius=.008,online_update_steps=16,
            cma_candidates=24*len(target_parameters),
            cma_report_budgets=([4,8,12,24,48] if len(target_parameters)==2 else
                sorted(set([b for b in [4,8,12,32,64,96] if b<=24*len(target_parameters)]+[24*len(target_parameters)])))),
        query_policy='Largest eligible episode index; no query observations in fitting or adaptation.',
        confidence_protocol='Shared dynamics encoder and one independent masked offline confidence head per target parameter.',
        dynamics_conditioning='Retain four known physical-condition inputs in the pretrained backbone; only target parameters may be optimized.',
        online_adaptation='Support-only private runtime copies initialized from the same shared pretrained checkpoint.',
        development_scope='Historical held-out objects; earlier project inspection is disclosed.',
        acceleration_claim='Report full-physics candidate/episode counts and total model+physics time to matched error.')
    baseline=read_json(Path(data['fit_run'])/'plan.json')['config']['baseline']
    protocol['fixed_parameters']={k:baseline[k] for k in PARAMETERS if k not in target_parameters}
    protocol['simulation_grid_padding']={'187-white-bear':{'1':32}}
    write_json(run/'protocol.json',protocol)
    from .joint_acceleration import prepare_physics
    prepare_physics(run)
    jobs=[dict(id='joint_train',after=[],command=[sys.executable,'-m','softdynamics.joint_world','--run',str(run),'--device','{device}'])]
    for obj in test_objects:
        jobs.append(dict(id='baseline_'+obj,after=[],command=[sys.executable,'-m','softdynamics.joint_acceleration',
            'baseline','--run',str(run),'--object',obj,'--device','{device}']))
    jobs.append(dict(id='model_evaluation',after=['joint_train'],command=[sys.executable,'-m','softdynamics.joint_world_evaluation',
        '--run',str(run),'--device','{device}']))
    for method in ['joint_learned_adam','joint_uniform_adam']:
        for obj in test_objects:
            jobs.append(dict(id=method+'_'+obj,after=['joint_train','baseline_'+obj],command=[sys.executable,'-m',
                'softdynamics.joint_acceleration','inverse','--run',str(run),'--object',obj,'--method',method,'--device','{device}']))
    for obj in test_objects:
        jobs.append(dict(id='cma_'+obj,after=['joint_learned_adam_'+obj],command=[sys.executable,'-m',
            'softdynamics.joint_acceleration_benchmark','cma','--run',str(run),'--object',obj,'--device','{device}']))
    jobs.append(dict(id='summary',after=[j['id'] for j in jobs],command=[sys.executable,'-m',
        'softdynamics.joint_acceleration_benchmark','summary','--run',str(run)]))
    write_json(run/'jobs.json',dict(code_root=str(run/'code'),gpus=list(range(8)),max_active=3,
        shared_memory_limit_mb=20000,sealed=True,jobs=jobs))
    write_json(run/'code_manifest.json',{str(p.relative_to(run/'code')):sha256(p) for p in (run/'code').rglob('*.py')})
    return protocol


def launch(run):
    run=Path(run).resolve()
    env=dict(os.environ,PYTHONPATH=str(run/'code'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with (run/'farm.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-m','softdynamics.experiment_farm','--root',str(run)],
            cwd=run/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(run/'launch.json',dict(pid=process.pid,started_at=time.time(),root=str(run)))
    return process.pid


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--initialize-from',type=Path,required=True)
    p.add_argument('--replay-manifest',type=Path,required=True)
    p.add_argument('--test-objects',nargs=3,default=['048-butter-sponge','125-rabbit','187-white-bear'])
    p.add_argument('--target-parameters',nargs='+',choices=PARAMETERS,default=list(PARAMETERS))
    a=p.parse_args();prepare(a.run,a.data_root,a.initialize_from,a.replay_manifest,a.test_objects,a.target_parameters)
    print('FARM_PID',launch(a.run),flush=True)
