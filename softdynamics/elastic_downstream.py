"""Frozen two-input best/latest material identification and original-MPM audit.

Only log(E/reference_E) and (nu-reference_nu)/nu_scale are optimized. Four-value
coordinates occur solely in the legacy MPM/render serialization adapter; the
last two entries are asserted zero. No historical trajectories, labels, or
network weights are reused. Static assets and measured observations are reused.
"""
import argparse
import copy
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .io import load_npz, read_json, sha256, write_json
from .elastic_data import ElasticSpace, PARAMETERS
from .parameters import ParameterSpace
from . import recovery_data as rd
from . import joint_acceleration as ja
from .joint_world import load_joint, context_batch
from .physics_protocol import read_protocol,cache_compatible


def wide_key(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.any(q[2:] != 0):
        raise ValueError('The MPM adapter accepts only E/nu; other offsets must be zero')
    return 'elastic_' + hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()


def install_adapter():
    # Isolated to this downstream process, never to another experiment.
    rd.qkey = wide_key
    ja.trajectory_folder = rd.trajectory_folder


def bounds(config):
    return (np.array([np.log(config['E_range_pa'][0]/config['reference_E_pa']),
                      (config['nu_range'][0]-config['reference_nu'])/config['nu_scale']]),
            np.array([np.log(config['E_range_pa'][1]/config['reference_E_pa']),
                      (config['nu_range'][1]-config['reference_nu'])/config['nu_scale']]))


def tensor_parameters(q, config):
    if q.shape != (2,):
        raise ValueError('Expected exactly two differentiable material coordinates')
    return torch.stack((config['reference_E_pa']*q[0].exp(),
                        config['reference_nu']+config['nu_scale']*q[1]))


def simulation_coordinates(q, config, physics_config):
    q = np.asarray(q, float)
    lo, hi = bounds(config)
    if q.shape != (2,) or not np.isfinite(q).all() or np.any(q < lo-1e-7) or np.any(q > hi+1e-7):
        raise ValueError('Material outside the frozen pretrained parameter domain')
    parameters = ElasticSpace(config).parameters(q)
    result = ParameterSpace.from_config(physics_config).coordinate(parameters)
    assert np.all(result[2:] == 0)
    return result


def config_for(run):
    return read_protocol(run)


def folder_for(run, obj, episode, q):
    spec, plan = config_for(run)
    return rd.trajectory_folder(Path(run)/'physics', obj, episode,
                               simulation_coordinates(q, spec['elastic_config'], plan['config']))


def prepare(run, training, data_root, fit_run):
    run, training, data_root, fit_run = [Path(x).resolve() for x in (run, training, data_root, fit_run)]
    if (run/'protocol.json').exists():
        return read_json(run/'protocol.json')
    run.mkdir(parents=True, exist_ok=True)
    train = read_json(training/'plan.json')
    config = train['config']
    checkpoints = {}
    for name in ('best', 'latest'):
        path = training/'training'/f'{name}.pt'
        b = torch.load(path, map_location='cpu', weights_only=False)
        if b['architecture']['material_dim'] != 2 or b['confidence_parameters'] != list(PARAMETERS):
            raise ValueError('Expected the independently trained two-input, two-head checkpoints')
        if set(b['protocol']['training_objects']) & set(train['test_objects']):
            raise ValueError('Test object in pretraining')
        checkpoints[name] = dict(path=str(path), sha256=sha256(path), step=b['step'])
    original = read_json(fit_run/'plan.json')
    physics = dict(objects=copy.deepcopy([o for o in original['objects'] if o['object'] in train['test_objects']]),
                   config={k:copy.deepcopy(original['config'][k]) for k in ('fitting','physics')},
                   pretrain=train['source'],created_at=time.time())
    if len(physics['objects']) != 3:
        raise ValueError('Missing requested test objects')
    for item in physics['objects']:
        for entry in item['episodes']:
            entry['dtmax'] = train['numerical_protocol']['dtmax']
            entry['grid_padding_cells'] = 32 if item['object'] == '187-white-bear' else 0
    # Physical serialization retains existing render support, but has only two
    # nonzero coordinates. Fixed contact/friction/decay values equal training.
    physics['config']['baseline'] = dict(config['baseline'], E_pa=config['reference_E_pa'], nu=config['reference_nu'])
    physics['config']['coordinate_scales'] = [config['reference_E_pa'], .4, config['baseline']['ground_friction'], 1.]
    code_root = Path(__file__).resolve().parents[1]
    physics.update(schema='softdynamics.elastic-downstream-physics.v1', runtime=str(code_root/'runtime'),
        conditions=[dict(key='real', truth='real', noise_mm=0., seed=19)],
        source_plan_sha256=sha256(fit_run/'plan.json'), data_plan_sha256=sha256(data_root/'plan.json'),
        optimized_parameters=list(PARAMETERS), new_trajectories_only=True)
    write_json(run/'physics/plan.json', physics)
    methods = [f'{c}_{w}' for c in ('best','latest') for w in ('uniform','confidence')] + ['physical_cma']
    spec = dict(schema='softdynamics.elastic-downstream.v1', created_at=time.time(),
        training=str(training), training_plan_sha256=sha256(training/'plan.json'), checkpoints=checkpoints,
        elastic_config=config, data_root=str(data_root), fit_run=str(fit_run), pretrain=train['source'],
        test_objects=train['test_objects'], methods=methods, optimized_parameters=list(PARAMETERS),
        physics_plan_sha256=sha256(run/'physics/plan.json'),
        inverse=dict(initial_q=[0.,0.], inner_steps=24, window_horizon=8, windows_per_phase=2,
            physical_rounds=12, report_budgets=[4,8,12], trust_radius=.25, minimum_radius=.00001,
            cma_candidates=48, cma_report_budgets=[4,8,12,24,48], cma_population=4, cma_sigma=.2, seed=9101),
        weights='Independent E/nu columns, nonnegative, detached, normalized across support interactions',
        optimizer='Projected gradient descent in an L2 trust region; one common gradient norm scale preserves relative parameter components',
        acceptance='Strictly lower equal-episode mean full original-physics support objective; no query observation',
        query_policy='All pre-existing quality-ready episodes; largest episode index held out; frozen split unchanged',
        online_adaptation=False, network_weights_frozen=True, historical_simulation_targets_used=False,
        counters='Baseline, support candidate validations, actual new episode rollouts, cached reuse, full solver seconds, and method wall time are separate',
        development_scope='These held-out objects were inspected in historical project experiments; no claim of a blind final test',
        downstream_scope='Material identification, full-solver verification, and six held-out replay metrics; no MPC or RL',
        source_sha256={str(p):sha256(p) for p in [Path(__file__), Path(ja.__file__), Path(rd.__file__)]})
    write_json(run/'protocol.json', spec)
    return spec


def collect(run, obj, episode, q, device, capture=True, requester=None):
    install_adapter()
    run = Path(run); spec, plan = config_for(run)
    legacy = simulation_coordinates(q, spec['elastic_config'], plan['config'])
    folder = rd.trajectory_folder(run/'physics', obj, episode, legacy)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'elastic.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result = ja.collect_physics(run, obj, episode, legacy.tolist(), device, capture)
        receipt = folder/'elastic_receipt.json'
        provenance = dict(elastic_q=list(q), simulation_adapter_coordinates=legacy,
            optimized_parameters=list(PARAMETERS), physics_plan_sha256=spec['physics_plan_sha256'],
            complete_sha256=sha256(folder/'complete.json'), requester=requester,
            execution_pid=read_json(folder/'execution.json')['pid'])
        if receipt.exists():
            old = read_json(receipt)
            if old['complete_sha256']!=provenance['complete_sha256'] or not cache_compatible(
                    run,old['physics_plan_sha256'],obj,episode,spec,plan):
                raise ValueError('Changed downstream physical cache')
        else:
            write_json(receipt, provenance)
    return result


def physical_result(run, obj, episodes, q, device, capture=True, requester=None):
    """Each subprocess performs one isolated solver rollout; cache ownership is explicit."""
    install_adapter(); run = Path(run); values = []; records = []
    for episode in episodes:
        folder = folder_for(run, obj, episode, q)
        cached = (folder/'complete.json').exists()
        started = time.monotonic()
        command = [sys.executable, '-m', 'softdynamics.elastic_downstream', '--run', str(run),
                   '--mode', 'physics', '--object', obj, '--episode', str(episode), '--q',
                   *[repr(float(v)) for v in q], '--device', device]
        if not capture:
            command.append('--no-capture')
        if requester:
            command.extend(['--requester',requester])
        child = subprocess.Popen(command)
        code = child.wait()
        if code:
            raise RuntimeError(f'Physics worker {child.pid} failed with exit {code}: {folder}')
        result = read_json(folder/'complete.json')
        scores = read_json(folder/'scores.json')
        execution = read_json(folder/'execution.json')
        receipt=read_json(folder/'elastic_receipt.json')
        owned = receipt.get('requester')==requester if requester else execution['pid']==child.pid
        value = scores['conditions']['real']['objective_mm'] if result['valid'] else np.inf
        values.append(value)
        records.append(dict(episode=episode, folder=str(folder), valid=result['valid'], cached=cached,
            actual_new_rollout=owned, worker_pid=child.pid, elapsed_s=time.monotonic()-started,
            physics_s=result['elapsed_s'], isolated_execution_s=execution['elapsed_s']))
    return np.asarray(values), records


def baseline(run, obj):
    run = Path(run); spec, plan = config_for(run); install_adapter()
    item = next(o for o in plan['objects'] if o['object'] == obj)
    losses=[]; records=[]
    for episode in item['support']:
        folder=folder_for(run,obj,episode,[0.,0.]); result=read_json(folder/'complete.json')
        if not result['valid']:
            raise ValueError(f'Initial original-physics rollout is invalid: {folder}')
        losses.append(read_json(folder/'scores.json')['conditions']['real']['objective_mm'])
        execution=read_json(folder/'execution.json')
        records.append(dict(episode=episode,folder=str(folder),physics_s=result['elapsed_s'],
                            isolated_execution_s=execution['elapsed_s']))
    result=dict(object=obj,support=item['support'],query=item['query'],q=[0.,0.],
        valid=True,losses_mm=losses,mean_mm=float(np.mean(losses)),records=records,
        physics_plan_sha256=spec['physics_plan_sha256'],query_observations_used=False)
    repair=run/'domain_repair_complete.json'
    if repair.exists():
        fixed=read_json(repair)
        if fixed['object']==obj:
            result.update(numerical_preflight_seconds=fixed['numerical_preflight_seconds'],
                numerical_preflight_episode_rollouts=fixed['numerical_preflight_episode_rollouts'],
                numerical_repair_receipt=str(repair))
    write_json(run/'baselines'/f'{obj}.json',result)
    return result


def project(q, anchor, radius, lo, hi):
    delta=q-anchor
    delta=delta*torch.clamp(torch.as_tensor(radius,device=q.device)/delta.norm().clamp_min(1e-12),max=1.)
    return torch.maximum(lo,torch.minimum(hi,anchor+delta))


def totals(history):
    records=[r for h in history for r in h['records']]
    return dict(support_candidate_validations=len(history), support_episode_requests=len(records),
        actual_new_episode_rollouts=sum(r['actual_new_rollout'] for r in records),
        cached_episode_reuses=sum(not r['actual_new_rollout'] for r in records),
        equivalent_full_solver_seconds=sum(r['isolated_execution_s'] for r in records),
        actual_new_solver_seconds=sum(r['isolated_execution_s'] for r in records if r['actual_new_rollout']),
        setup_seconds=sum(h.get('setup_seconds',0.) for h in history),
        model_seconds=sum(h.get('proposal_seconds',0.) for h in history))


def save_selection(run, obj, method, history, q, losses, baseline_record):
    run=Path(run); spec,_=config_for(run)
    name=method.split('_')[0]
    result=dict(object=obj,method=method,budget=len(history),q=q,
        parameters=ElasticSpace(spec['elastic_config']).parameters(q),
        support_mean_mm=float(np.mean(losses)),support_losses_mm=losses,
        baseline_mean_mm=baseline_record['mean_mm'],
        baseline_episode_rollouts=len(baseline_record['support']),
        baseline_seconds=sum(r['isolated_execution_s'] for r in baseline_record['records'])+baseline_record.get('numerical_preflight_seconds',0.),
        numerical_preflight_episode_rollouts=baseline_record.get('numerical_preflight_episode_rollouts',0),
        **totals(history), method_wall_seconds=sum(h['round_wall_seconds']+h.get('setup_seconds',0.) for h in history),
        frozen_at=time.time(), query_observations_used=False,
        checkpoint_sha256=spec['checkpoints'][name]['sha256'] if name in spec['checkpoints'] else None,
        network_weights_frozen=True)
    write_json(run/'inverse'/obj/method/f'selection_{len(history)}.json',result)


def recover_selections(run,obj,method,history,base,budgets):
    # Recover the tiny crash interval between journaling a completed round and
    # publishing its frozen reporting checkpoint, without repeating a rollout.
    for budget in budgets:
        path=Path(run)/'inverse'/obj/method/f'selection_{budget}.json'
        if budget<=len(history) and not path.exists():
            row=history[budget-1]
            save_selection(run,obj,method,history[:budget],row['selected_q'],row['selected_support_losses_mm'],base)


def inverse(run, obj, method, device):
    setup_started=time.monotonic()
    install_adapter(); torch.set_num_threads(2)
    run=Path(run); spec,physics=config_for(run); config=spec['elastic_config']; options=spec['inverse']
    out=run/'inverse'/obj/method; out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    base=read_json(run/'baselines'/f'{obj}.json')
    checkpoint_name,weighting=method.split('_')
    source=spec['checkpoints'][checkpoint_name]
    if sha256(source['path'])!=source['sha256']:raise ValueError('Frozen checkpoint changed')
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.30,torch.device(device))
    model,bundle=load_joint(source['path'],device); model.eval(); model.requires_grad_(False)
    if obj in bundle['protocol']['training_objects']:raise ValueError('Test object leaked into pretraining')
    if tuple(model.confidence_parameters)!=PARAMETERS:raise ValueError('Unexpected confidence dimensions')
    windows=[ja.EndpointWindows(run,obj,e,device) for e in base['support']]
    contexts=context_batch([w.confidence_context for w in windows],device)
    history=read_json(out/'history.json') if (out/'history.json').exists() else []
    recover_selections(run,obj,method,history,base,options['report_budgets'])
    current=np.array(history[-1]['selected_q'] if history else [0.,0.],float)
    losses=np.array(history[-1]['selected_support_losses_mm'] if history else base['losses_mm'],float)
    radius=history[-1]['next_radius'] if history else options['trust_radius']
    lo,hi=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in bounds(config)]
    seen={wide_key(simulation_coordinates(h['candidate_q'],config,physics['config'])) for h in history}
    seen.add(wide_key(simulation_coordinates([0.,0.],config,physics['config'])))
    setup_seconds=time.monotonic()-setup_started
    for round_index in range(len(history)+1,options['physical_rounds']+1):
        round_started=time.monotonic(); proposal_started=round_started
        anchor=torch.tensor(current,dtype=torch.float32,device=device)
        theta=tensor_parameters(anchor,config)
        legacy=simulation_coordinates(current,config,physics['config'])
        for w in windows:w.set_anchor(model.world,legacy.tolist(),theta)
        with torch.no_grad():
            predicted=model.confidence(contexts,theta[None].expand(len(windows),2))
            raw=torch.ones_like(predicted) if weighting=='uniform' else predicted
            weights=ja.normalized_weights(raw)
        q=anchor.clone().requires_grad_(); inner=[]; normalizer=None
        for inner_step in range(options['inner_steps']):
            gradient=torch.zeros_like(q); proxy=[]; episode_gradients=[]
            for index,w in enumerate(windows):
                loss=w.loss(model.world,tensor_parameters(q,config))
                grad=torch.autograd.grad(loss,q)[0]
                if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite model material gradient')
                gradient+=weights[index]*grad; proxy.append(float(loss.detach()))
                episode_gradients.append(grad.detach().cpu().tolist())
            if normalizer is None:normalizer=gradient.norm().detach().clamp_min(1e-8)
            with torch.no_grad():
                # Common scalar scaling, plus an L2 region, preserves the
                # confidence-dependent relative E/nu direction.
                step=(radius/6.)*gradient/torch.maximum(normalizer,gradient.norm())
                q.copy_(project(q-step,anchor,radius,lo,hi))
            inner.append(dict(step=inner_step+1,q=q.detach().cpu().tolist(),
                proxy_losses_mm=proxy,episode_gradients=episode_gradients,gradient=gradient.cpu().tolist()))
            write_json(out/'status.json',dict(pid=os.getpid(),round=round_index,inner_step=inner_step+1,
                phase='frozen_model_material_gradient',updated_at=time.time()))
        candidate=q.detach().cpu().double().numpy()
        key=wide_key(simulation_coordinates(candidate,config,physics['config']))
        if key in seen or np.linalg.norm(candidate-current)<1e-8:
            raise RuntimeError('Model proposal is repeated/stationary; no new physical budget has been charged')
        proposal_seconds=time.monotonic()-proposal_started
        write_json(out/'proposal.json',dict(candidate_q=candidate,round=round_index,inner=inner,weights=weights.cpu().tolist()))
        del q; gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
        actual,records=physical_result(run,obj,base['support'],candidate,device,requester=f'{obj}/{method}')
        accepted=bool(np.isfinite(actual).all() and actual.mean()<losses.mean()-1e-9)
        previous=current.copy()
        if accepted:current=candidate.copy();losses=actual.copy();radius=min(options['trust_radius'],radius*1.2)
        else:radius=max(options['minimum_radius'],radius*.5)
        row=dict(round=round_index,previous_q=previous,candidate_q=candidate,selected_q=current,
            candidate_support_losses_mm=actual,selected_support_losses_mm=losses,accepted=accepted,
            predicted_confidence=predicted.cpu().tolist(),weights=weights.cpu().tolist(),inner=inner,
            next_radius=radius,proposal_seconds=proposal_seconds,records=records,setup_seconds=setup_seconds,
            round_wall_seconds=time.monotonic()-round_started)
        history.append(row);seen.add(key);write_json(out/'history.json',history)
        setup_seconds=0.
        if round_index in options['report_budgets']:save_selection(run,obj,method,history,current,losses,base)
        print('ELASTIC_INVERSE',obj,method,round_index,'accepted',accepted,'support_mm',float(losses.mean()),flush=True)
    if sha256(source['path'])!=source['sha256']:raise ValueError('Checkpoint changed during identification')
    write_json(out/'complete.json',dict(status='complete',object=obj,method=method,query_observations_used=False,
        checkpoint_sha256=source['sha256'],network_weights_frozen=True,**totals(history)))


def cma(run,obj,device):
    setup_started=time.monotonic()
    import cma as pycma
    install_adapter();run=Path(run);spec,physics=config_for(run);options=spec['inverse'];config=spec['elastic_config']
    out=run/'inverse'/obj/'physical_cma';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    base=read_json(run/'baselines'/f'{obj}.json');lo,hi=bounds(config)
    statepath=out/'optimizer.pkl'
    if statepath.exists():
        state=pickle.loads(statepath.read_bytes());es=state['es'];history=state['history']
        current=state['current'];losses=state['losses'];pending=state['pending']
    else:
        rng=np.random.RandomState(options['seed'])
        es=pycma.CMAEvolutionStrategy((np.zeros(2)-lo)/(hi-lo), options['cma_sigma'],
            dict(bounds=[0.,1.],popsize=options['cma_population'],seed=options['seed'],randn=rng.randn,verbose=-9))
        history=[];current=np.zeros(2);losses=np.array(base['losses_mm']);pending=None
    recover_selections(run,obj,'physical_cma',history,base,options['cma_report_budgets'])
    setup_seconds=time.monotonic()-setup_started
    def checkpoint():
        tmp=statepath.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(dict(es=es,history=history,current=current,losses=losses,pending=pending)));tmp.replace(statepath)
    while len(history)<options['cma_candidates']:
        if pending is None:
            pending=dict(candidates=es.ask(),costs=[]);checkpoint()
        candidates=pending['candidates'];costs=pending['costs']
        for x in candidates[len(costs):]:
            started=time.monotonic();candidate=lo+np.asarray(x)*(hi-lo)
            key=wide_key(simulation_coordinates(candidate,config,physics['config']))
            seen={wide_key(simulation_coordinates(h['candidate_q'],config,physics['config'])) for h in history}
            if key in seen:raise RuntimeError('CMA repeated a candidate; refusing duplicate budget charge')
            actual,records=physical_result(run,obj,base['support'],candidate,device,requester=f'{obj}/physical_cma')
            cost=float(actual.mean());costs.append(cost if np.isfinite(cost) else 1e30)
            accepted=bool(np.isfinite(cost) and cost<losses.mean()-1e-9)
            if accepted:current=candidate.copy();losses=actual.copy()
            history.append(dict(round=len(history)+1,candidate_q=candidate,selected_q=current.copy(),
                candidate_support_losses_mm=actual,selected_support_losses_mm=losses.copy(),accepted=accepted,
                records=records,round_wall_seconds=time.monotonic()-started,setup_seconds=setup_seconds))
            setup_seconds=0.
            checkpoint()
            write_json(out/'history.json',history)
            if len(history) in options['cma_report_budgets']:save_selection(run,obj,'physical_cma',history,current,losses,base)
            print('ELASTIC_CMA',obj,len(history),'support_mm',float(losses.mean()),flush=True)
        es.tell(candidates,costs);pending=None;checkpoint()
    write_json(out/'complete.json',dict(status='complete',object=obj,method='physical_cma',
        dimensions=2,query_observations_used=False,**totals(history)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','physics','baseline','inverse','cma'],required=True)
    p.add_argument('--training',type=Path);p.add_argument('--data-root',type=Path);p.add_argument('--fit-run',type=Path)
    p.add_argument('--object');p.add_argument('--episode',type=int);p.add_argument('--q',type=float,nargs=2)
    p.add_argument('--method');p.add_argument('--device',default='cuda:0');p.add_argument('--no-capture',action='store_true')
    p.add_argument('--requester')
    a=p.parse_args();install_adapter()
    if a.mode=='prepare':prepare(a.run,a.training,a.data_root,a.fit_run)
    elif a.mode=='physics':collect(a.run,a.object,a.episode,a.q,a.device,not a.no_capture,a.requester)
    elif a.mode=='baseline':baseline(a.run,a.object)
    elif a.mode=='cma':cma(a.run,a.object,a.device)
    else:inverse(a.run,a.object,a.method,a.device)


if __name__=='__main__':main()
