"""Frozen MPM/sensitivity backbone + real-GT residual training and joint CMA.

Prepare is read-only with respect to the old run. Training must be explicitly
started in a new output directory. Support-only fitting never sees query GT.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .coupled_world_model import CoupledWorld, load_coupled
from .coupled_world_train import source_hashes
from .elastic_train import atomic_checkpoint
from .io import read_json, write_json, sha256, json_value
from .real_residual_data import RealReplay
from .real_residual_objective import real_rollout, observation_objective, residual_regularizers


def freeze_base(model):
    model.requires_grad_(False)
    if model.residual_config is None:
        raise ValueError('Residual branch missing')
    model.world.velocity_residual.requires_grad_(True)
    return list(model.world.velocity_residual.parameters())


def initialize(base_checkpoint, residual, device='cpu'):
    base, bundle = load_coupled(base_checkpoint,'cpu')
    if base.residual_config is not None:
        raise ValueError('Initialize from an original MPM/sensitivity checkpoint')
    c = bundle['material_coordinates']
    model = CoupledWorld(bundle['architecture'],bundle['tau'],c['reference_E_pa'],
        c['reference_nu'],c['nu_scale'],residual=residual)
    missing, unexpected = model.load_state_dict(base.state_dict(),strict=False)
    expected = {k for k in model.state_dict() if k.startswith('world.velocity_residual.')}
    if set(missing) != expected or unexpected:
        raise ValueError('Base checkpoint conversion changed existing parameters')
    freeze_base(model)
    return model.to(device), bundle


def q_to_theta(model, q):
    return model.theta(q)


def evaluate_episode(model, data, q, device, surface_points):
    positions, _, _ = real_rollout(model,data,q_to_theta(model,q),device)
    if not torch.isfinite(positions).all():
        raise FloatingPointError('Nonfinite corrected trajectory')
    _, metrics = observation_objective(positions,data,surface_points=surface_points)
    if not math.isfinite(metrics['objective_mm']):
        raise FloatingPointError('Nonfinite real objective')
    return metrics, positions


def fit_object(model, replay, obj, device, generations=4, population=6, seed=20261111,
               surface_points=128, initial_q=None):
    import cma
    if generations < 1 or population < 4:
        raise ValueError('Positive generation budget and population >= 4 required')
    rows = replay.records(role='fit',obj=obj)
    if not rows:
        raise ValueError('No support interactions: '+obj)
    low = np.array([math.log(1000/model.reference_E),(.08-model.reference_nu)/model.nu_scale])
    high = np.array([math.log(12000/model.reference_E),(.42-model.reference_nu)/model.nu_scale])
    q0 = np.zeros(2) if initial_q is None else np.asarray(initial_q,float)
    if q0.shape != (2,) or not np.isfinite(q0).all():
        raise ValueError('Finite two-dimensional material initial value required')
    unit0 = np.clip((q0-low)/(high-low),0,1)
    history, failures = [], []
    def objective(unit):
        q = low+(high-low)*np.asarray(unit)
        values, valid = [], True
        for row in rows:
            try:
                metrics,_ = evaluate_episode(model,replay.data[(obj,row['episode'])],
                    torch.as_tensor(q,dtype=torch.float32,device=device),device,surface_points)
                values.append(metrics['objective_mm'])
            except (FloatingPointError, ValueError) as error:
                valid = False
                values.append(1e6)
                failures.append(dict(episode=row['episode'],q=q.tolist(),error=repr(error)))
        value = float(np.mean(values)) if valid else 1e6
        history.append(dict(q=q.tolist(),objective_mm=value,episode_objectives_mm=values,valid=valid))
        return value
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            best_unit, best_cost = unit0.copy(), objective(unit0)
            initial_cost = best_cost
            # The immutable q-bound box normalizes axes; both parameters are
            # searched jointly. Original sensitivity is not a calibrated GT weight.
            es = cma.CMAEvolutionStrategy(unit0.tolist(),.25,
                dict(bounds=[0.,1.],popsize=population,seed=seed,verbose=-9,maxiter=generations))
            for _ in range(generations):
                population_units = es.ask()
                costs = [objective(u) for u in population_units]
                es.tell(population_units,costs)
                index = int(np.argmin(costs))
                if costs[index] < best_cost:
                    best_cost, best_unit = costs[index], np.asarray(population_units[index]).copy()
    finally:
        model.train(was_training)
    if best_cost >= 1e6:
        raise FloatingPointError('No valid material candidate for '+obj)
    q = low+(high-low)*best_unit
    return dict(object=obj,q=q.tolist(),E_pa=float(model.reference_E*np.exp(q[0])),
        nu=float(model.reference_nu+model.nu_scale*q[1]),objective_mm=best_cost,
        initial_objective_mm=initial_cost,generations=generations,population=population,
        candidates=generations*population,seed=seed,history=history,failures=failures,
        support_episodes=[r['episode'] for r in rows],query_used=False,
        support_sha256=[r['sha256'] for r in rows],sensitivity_weighting='uniform',
        parameter_meaning='Model-calibrated estimate, not measured material truth')


def prepare_training(data_root, output, steps=2000, maximum=.02, hidden=128):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new residual run directory')
    if steps < 1:
        raise ValueError('Positive training budget required')
    replay = RealReplay(data_root)
    if not replay.records(split='validation',role='query'):
        raise ValueError('Validation-object query observations are required for best selection')
    model, _ = initialize(replay.root/'base.pt',dict(hidden=hidden,max_velocity_m_s=maximum))
    output.mkdir(parents=True)
    config = dict(schema='softdynamics.real-residual-training.v1',data_root=str(replay.root),
        manifest_sha256=sha256(replay.root/'manifest.json'),base_sha256=replay.manifest['base_sha256'],
        residual=model.residual_config,steps=steps,seed=20261111,learning_rate=1e-4,
        source_sha256=source_hashes(),surface_points=128,maximum_gradient_tail=32,
        amplitude_weight=.1,smoothness_weight=.05,fit_generations=4,fit_population=6,
        material_refit_interval=500,validation_interval=100,save_interval=20,
        base_frozen=True,sensitivity_frozen=True,sensitivity_semantics='MPM/base, not corrected-GT sensitivity',
        teacher_losses='Original CoupledWorld.forward remains on the uncorrected branch',
        real_targets='Visible measured tracks and one-sided GT-to-material-surface distance',
        state_feedback='Corrected state at every step; no intermediate observed-state resets',
        parameter_fit='Support-only joint log-E/nu CMA; network weights fixed during search',
        selection='Equal-object full query objective on validation objects only')
    write_json(output/'config.json',config)
    return config


def checkpoint_payload(model, base, config, step, optimizer, materials, rng, best, metrics):
    device = next(model.parameters()).device
    return dict(schema='softdynamics.coupled-residual-checkpoint.v1',model=model.state_dict(),
        architecture=base['architecture'],residual=model.residual_config,tau=base['tau'],
        material_coordinates=base['material_coordinates'],statistics=base['statistics'],
        base_step=base['step'],base_sha256=config['base_sha256'],step=step,optimizer=optimizer.state_dict(),
        materials=materials,best_score=best,metrics=metrics,protocol=config,base_protocol=base['protocol'],
        config_sha256=sha256(Path(config['output'])/'config.json'),
        rng=dict(numpy=rng.bit_generator.state,torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))


def training_tail(step, total, maximum=32):
    fraction = step/total
    return min(maximum,4 if fraction < .1 else 8 if fraction < .3 else 16 if fraction < .6 else 32)


def choose_endpoint(data, rng):
    valid = data['visible'].any(-1) | data['reference_valid'].any(-1)
    eligible = np.flatnonzero(valid & (np.arange(len(valid))>0))
    if rng.random() < .25:
        return len(valid)-1  # Includes unobserved withdrawal; these frames aren't targets.
    phases = np.unique(data['phase'][eligible])
    phase = int(rng.choice(phases))
    return int(rng.choice(eligible[data['phase'][eligible]==phase]))


def validate(model,replay,materials,device,surface_points):
    records = []
    for row in replay.records(split='validation',role='query'):
        obj = row['object']
        try:
            metrics,_ = evaluate_episode(model,replay.data[(obj,row['episode'])],
                torch.tensor(materials[obj]['q'],device=device),device,surface_points)
            records.append(dict(object=obj,episode=row['episode'],valid=True,**metrics))
        except (FloatingPointError,ValueError) as error:
            records.append(dict(object=obj,episode=row['episode'],valid=False,
                objective_mm=float('inf'),error=repr(error)))
    per_object = {o:float(np.mean([r['objective_mm'] for r in records if r['object']==o]))
                  for o in {r['object'] for r in records}}
    return dict(score=float(np.mean(list(per_object.values()))),per_object=per_object,records=records,
                failures=sum(not r['valid'] for r in records))


def train(output,resume=False,device='cuda:0'):
    output = Path(output).resolve()
    cfg = read_json(output/'config.json')
    if cfg['source_sha256'] != source_hashes():
        raise ValueError('Residual training source changed since prepare')
    replay = RealReplay(cfg['data_root'])
    if cfg['manifest_sha256'] != sha256(replay.root/'manifest.json') or cfg['base_sha256'] != replay.manifest['base_sha256']:
        raise ValueError('Frozen residual data/base changed')
    device = torch.device(device)
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type=='cuda':
        torch.cuda.set_device(device)
    out = output/'training'; out.mkdir(exist_ok=True)
    lock = (out/'training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'latest.pt').exists() and not resume:
        raise ValueError('Existing residual checkpoint requires explicit resume')
    old = torch.load(out/'latest.pt',map_location='cpu',weights_only=False) if resume else None
    if old and old['config_sha256'] != sha256(output/'config.json'):
        raise ValueError('Residual resume protocol changed')
    model,base = initialize(replay.root/'base.pt',cfg['residual'],device)
    optimizer = torch.optim.AdamW(freeze_base(model),lr=cfg['learning_rate'],weight_decay=1e-4)
    rng = np.random.default_rng(cfg['seed'])
    materials, best, metrics, start = {},float('inf'),None,0
    if old:
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        materials,best,metrics,start = old['materials'],old['best_score'],old['metrics'],old['step']
        rng.bit_generator.state = old['rng']['numpy']; torch.set_rng_state(old['rng']['torch'])
        if device.type=='cuda':
            torch.cuda.set_rng_state(old['rng']['cuda'],device)
    payload_config = dict(cfg,output=str(output))
    def status(stage,**values):
        row = dict(status=stage,pid=os.getpid(),updated_at=time.time(),**values)
        write_json(out/'status.json',row)
        print('RESIDUAL_STATUS',row,flush=True)
    def save(step,improved=False):
        payload = checkpoint_payload(model,base,payload_config,step,optimizer,materials,rng,best,metrics)
        atomic_checkpoint(out/'latest.pt',payload)
        if improved:
            atomic_checkpoint(out/'best.pt',payload)
    def refit(objects,step):
        for index,obj in enumerate(sorted(objects)):
            status('material_fit',step=step,object=obj)
            result = fit_object(model,replay,obj,device,cfg['fit_generations'],cfg['fit_population'],
                cfg['seed']+step*100+index,cfg['surface_points'],materials.get(obj,{}).get('q'))
            result['residual_step'] = step
            result['base_sha256'] = cfg['base_sha256']
            materials[obj] = result
            write_json(out/'materials'/obj/f'step_{step}.json',result)
    training_objects = {r['object'] for r in replay.records(split='train',role='fit')}
    validation_objects = {r['object'] for r in replay.records(split='validation',role='fit')}
    objects = sorted(training_objects)
    step = start
    try:
        if (output/'pause.request.json').exists():
            save(start); status('paused',step=start)
            os.replace(output/'pause.request.json',output/'pause.fulfilled.json'); return
        if not materials:
            refit(training_objects|validation_objects,start)
        save(start)
        for step in range(start,cfg['steps']):
            if (output/'pause.request.json').exists():
                save(step); status('paused',step=step)
                os.replace(output/'pause.request.json',output/'pause.fulfilled.json'); return
            if step%cfg['material_refit_interval']==0:
                due = {obj for obj in training_objects if materials[obj]['residual_step'] != step}
                refit(due,step)
            obj = str(rng.choice(objects)); row = rng.choice(replay.records(split='train',role='fit',obj=obj))
            data = replay.data[(obj,row['episode'])]
            endpoint = choose_endpoint(data,rng)
            tail = training_tail(step,cfg['steps'],cfg['maximum_gradient_tail'])
            model.train(); optimizer.zero_grad(set_to_none=True)
            theta = model.theta(torch.tensor(materials[obj]['q'],device=device))
            positions,corrections,first = real_rollout(model,data,theta,device,endpoint,True,tail)
            # An early short tail can contain only unobserved withdrawal.
            # Use the last observed endpoint, still reached from frame zero.
            try:
                obs,detail = observation_objective(positions,data,first,cfg['surface_points'])
            except ValueError as error:
                if str(error) != 'No valid real GT in scored rollout segment':
                    raise
                endpoint = int(np.flatnonzero(data['visible'].any(-1) | data['reference_valid'].any(-1))[-1])
                positions,corrections,first = real_rollout(model,data,theta,device,endpoint,True,tail)
                obs,detail = observation_objective(positions,data,first,cfg['surface_points'])
            amp,smooth = residual_regularizers(corrections,data['neighbors'],
                model.world.max_residual_velocity_m_s,first-1)
            loss = obs+cfg['amplitude_weight']*amp+cfg['smoothness_weight']*smooth
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite real residual training loss')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.world.velocity_residual.parameters(),1.,error_if_nonfinite=True))
            optimizer.step()
            if step==start or (step+1)%20==0:
                record = dict(step=step+1,object=obj,episode=row['episode'],endpoint=endpoint,gradient_tail=tail,
                    loss=float(loss.detach()),amplitude=float(amp.detach()),smoothness=float(smooth.detach()),
                    gradient_norm=norm,**detail)
                status('running',**record)
                with (out/'train.jsonl').open('a') as stream:
                    stream.write(json.dumps(json_value(record))+'\n')
            if (step+1)%cfg['save_interval']==0:
                save(step+1)
            if (step+1)%cfg['validation_interval']==0 or step+1==cfg['steps']:
                refit(validation_objects,step+1)
                model.eval()
                with torch.no_grad():
                    metrics = validate(model,replay,materials,device,cfg['surface_points'])
                improved = math.isfinite(metrics['score']) and metrics['score'] < best
                if improved:
                    best = metrics['score']
                write_json(out/'validation'/f'step_{step+1}.json',metrics)
                save(step+1,improved)
        status('complete',step=cfg['steps'],best_score=best)
    except Exception as error:
        status('failed',step=step+1,error=repr(error))
        raise


def fit_checkpoint(checkpoint,data_root,obj,output,device='cpu',generations=4,population=6):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new fitting output directory')
    model,bundle = load_coupled(checkpoint,device)
    model.eval()
    replay = RealReplay(data_root)
    if bundle.get('base_sha256') != replay.manifest['base_sha256']:
        raise ValueError('Residual checkpoint belongs to another frozen base')
    result = fit_object(model,replay,obj,device,generations,population)
    result.update(checkpoint_sha256=sha256(checkpoint),residual_step=bundle['step'])
    output.mkdir(parents=True)
    write_json(output/'selection.json',result)  # Selection committed before query is read.
    query = []
    with torch.no_grad():
        for row in replay.records(role='query',obj=obj):
            try:
                metrics,positions = evaluate_episode(model,replay.data[(obj,row['episode'])],
                    torch.tensor(result['q'],device=device),device,128)
                np.savez_compressed(output/f'episode_{row["episode"]}.npz',positions=positions.cpu().numpy())
                query.append(dict(episode=row['episode'],valid=True,**metrics))
            except (FloatingPointError,ValueError) as error:
                query.append(dict(episode=row['episode'],valid=False,error=repr(error)))
    write_json(output/'query.json',dict(records=query,query_used_for_selection=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode',required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--data-root',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True); prep.add_argument('--steps',type=int,default=2000)
    prep.add_argument('--max-velocity-m-s',type=float,default=.02); prep.add_argument('--hidden',type=int,default=128)
    run = sub.add_parser('train'); run.add_argument('--output',type=Path,required=True)
    run.add_argument('--resume',action='store_true'); run.add_argument('--device',default='cuda:0')
    fit = sub.add_parser('fit'); fit.add_argument('--checkpoint',type=Path,required=True)
    fit.add_argument('--data-root',type=Path,required=True); fit.add_argument('--object',required=True)
    fit.add_argument('--output',type=Path,required=True); fit.add_argument('--device',default='cpu')
    fit.add_argument('--generations',type=int,default=4); fit.add_argument('--population',type=int,default=6)
    a = p.parse_args()
    if a.mode=='prepare':
        prepare_training(a.data_root,a.output,a.steps,a.max_velocity_m_s,a.hidden)
    elif a.mode=='train':
        train(a.output,a.resume,a.device)
    else:
        print(fit_checkpoint(a.checkpoint,a.data_root,a.object,a.output,a.device,a.generations,a.population))


if __name__ == '__main__':
    main()
