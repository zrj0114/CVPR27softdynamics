"""Matched original-physics CMA baseline and three-object acceleration summary."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .cma_search import BlockCMA
from .io import read_json, write_json, sha256
from .joint_acceleration import physical_result


def run_cma(run,obj,device):
    run=Path(run);p=read_json(run/'protocol.json');method='physical_uniform_cma'
    out=run/'inverse'/obj/method;out.mkdir(parents=True,exist_ok=True)
    baseline=read_json(run/'baselines'/f'{obj}.json')
    if not baseline['valid']:raise ValueError('Invalid matched initial simulation')
    state_path=out/'optimizer.pkl'
    from .parameters import PARAMETERS
    targets=p.get('target_parameters',PARAMETERS)
    cma=BlockCMA.load(state_path) if state_path.exists() else BlockCMA(
        [0.]*4,np.ones((len(baseline['support']),len(targets))),seed=9101,cycles=3,block_generations=2,population=4,
        active_parameters=targets)
    if tuple(PARAMETERS[i] for i in cma.parameter_indices)!=tuple(targets):
        raise ValueError('Resumed CMA targets differ from the experiment protocol')
    started=time.monotonic();history=read_json(out/'history.json') if (out/'history.json').exists() else []
    cumulative=history[-1]['simulation_seconds'] if history else 0.
    while not cma.done:
        pending=cma.ask();cma.save(state_path)
        incumbent,_=physical_result(run,obj,baseline['support'],pending['incumbent'].tolist(),device,capture=False)
        losses=[];records=[]
        for q in pending['q']:
            values,receipts=physical_result(run,obj,baseline['support'],q.tolist(),device,capture=False)
            losses.append(values);records.extend(receipts)
        cumulative+=sum(r['isolated_execution_s'] for r in records)
        row=cma.tell(np.array(losses),incumbent);cma.save(state_path)
        row.update(simulation_seconds=cumulative,wall_seconds=time.monotonic()-started,records=records)
        history.append(row);write_json(out/'history.json',history)
        budget=cma.logical_candidates
        if budget in p['inverse']['cma_report_budgets']:
            # Partial blocks expose their best incumbent at the same budget.
            q=cma.best_q.copy();selected,_=physical_result(run,obj,baseline['support'],q.tolist(),device,capture=False)
            if not np.isfinite(selected).all() or selected.mean()>=np.mean(baseline['losses_mm']):
                q=np.zeros(4);selected=np.array(baseline['losses_mm'])
            write_json(out/f'selection_{budget}.json',dict(object=obj,method=method,budget=budget,q=q,
                support_mean_mm=float(selected.mean()),simulation_seconds=cumulative,model_seconds=0.,
                baseline_seconds=sum(r['isolated_execution_s'] for r in baseline['records']),
                wall_seconds=row['wall_seconds'],query_observations_used=False,frozen_at=time.time()))
        write_json(out/'status.json',dict(phase='physical_cma',candidates=budget,total=p['inverse']['cma_candidates'],updated_at=time.time()))
        print('CMA',obj,budget,'of',p['inverse']['cma_candidates'],flush=True)
    results=[]
    for budget in p['inverse']['cma_report_budgets']:
        path=out/f'selection_{budget}.json';selection=read_json(path)
        query,records=physical_result(run,obj,baseline['query'],selection['q'],device,capture=False)
        results.append(dict(**selection,query_losses_mm=query,query_mean_mm=float(query.mean()),
            query_records=records,selection_sha256=sha256(path)))
    write_json(out/'complete.json',dict(status='complete',object=obj,method=method,results=results,
        query_excluded_from_fitting=True))


def summarize(run):
    run=Path(run);p=read_json(run/'protocol.json');rows=[];missing=[]
    for obj in p['test_objects']:
        for method in p['methods']:
            path=run/'inverse'/obj/method/'complete.json'
            if not path.exists():missing.append([obj,method]);continue
            for r in read_json(path)['results']:
                rows.append({k:r[k] for k in ['object','method','budget','support_mean_mm','query_mean_mm',
                                              'model_seconds','simulation_seconds','baseline_seconds','wall_seconds']})
    write_json(run/'summary.json',dict(rows=rows,missing=missing,
        status='complete' if not missing else 'partial',
        metric='Original full-trajectory phase-weighted 0.5 surface distance + 0.5 track Huber, mm',
        timing='Shared-GPU wall time and separately recorded isolated task seconds; report both. Training amortization separate.',
        comparisons='Compare error versus physical candidate budget and total model+physics seconds; a lower error at fixed budget alone does not establish speedup.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['cma','summary'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--object');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(2)
    if a.mode=='cma':run_cma(a.run,a.object,a.device)
    else:summarize(a.run)
