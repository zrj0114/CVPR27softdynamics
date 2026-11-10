"""Fixed support-error thresholds and separate physical-work/time accounting."""
import csv
from pathlib import Path

import numpy as np

from .io import read_json,write_json
from .elastic_downstream import totals
from .physical_work import executions


def first_hit(base,history,target_mm):
    for index,row in enumerate(history):
        value=float(np.mean(row['selected_support_losses_mm']))
        if np.isfinite(value) and value<=target_mm+1e-6:
            prefix=history[:index+1];counts=totals(prefix)
            baseline_seconds=sum(r['isolated_execution_s'] for r in base['records'])+base.get('numerical_preflight_seconds',0.)
            return dict(candidates=index+1,objective_mm=value,
                episode_rollouts_with_initial=len(base['support'])+base.get('numerical_preflight_episode_rollouts',0)+counts['support_episode_requests'],
                numerical_preflight_seconds=base.get('numerical_preflight_seconds',0.),
                numerical_preflight_episode_rollouts=base.get('numerical_preflight_episode_rollouts',0),
                equivalent_serial_seconds_with_initial=baseline_seconds+counts['equivalent_full_solver_seconds']+counts['model_seconds']+counts['setup_seconds'],
                observed_method_seconds=sum(h['round_wall_seconds']+h.get('setup_seconds',0.) for h in prefix),
                **counts)
    return None


def compare_object(base,histories,finished,reductions):
    rows=[]
    for reduction in reductions:
        target=base['mean_mm']*(1-reduction)
        physical=first_hit(base,histories.get('physical_cma',[]),target)
        for method,history in histories.items():
            hit=first_hit(base,history,target)
            rows.append(dict(object=base['object'],method=method,reduction_fraction=reduction,
                baseline_mm=base['mean_mm'],target_mm=target,
                status='reached' if hit else ('not_reached' if finished[method] else 'pending'),
                compared_candidates=len(history),candidate_first_hit=hit['candidates'] if hit else None,
                achieved_mm=hit['objective_mm'] if hit else None,
                episode_rollouts_with_initial=hit['episode_rollouts_with_initial'] if hit else None,
                actual_new_support_episode_rollouts=hit['actual_new_episode_rollouts'] if hit else None,
                cached_support_episode_reuses=hit['cached_episode_reuses'] if hit else None,
                setup_seconds=hit['setup_seconds'] if hit else None,
                numerical_preflight_seconds=base.get('numerical_preflight_seconds',0.),
                numerical_preflight_episode_rollouts=base.get('numerical_preflight_episode_rollouts',0),
                model_seconds=hit['model_seconds'] if hit else None,
                equivalent_serial_seconds_with_initial=hit['equivalent_serial_seconds_with_initial'] if hit else None,
                observed_method_seconds=hit['observed_method_seconds'] if hit else None,
                cma_to_method_call_ratio=(physical['episode_rollouts_with_initial']/hit['episode_rollouts_with_initial']) if physical and hit else None,
                cma_to_method_equivalent_time_ratio=(physical['equivalent_serial_seconds_with_initial']/hit['equivalent_serial_seconds_with_initial']) if physical and hit else None))
    return rows


def summarize(run):
    run=Path(run);spec=read_json(run/'protocol.json');protocol=read_json(run/'comparison_protocol.json')
    rows=[];work=dict(initial=0,support=0,query=0,diagnostic=0,other=0)
    for obj in spec['test_objects']:
        path=run/'baselines'/f'{obj}.json'
        if not path.exists():continue
        histories={};finished={}
        for method in spec['methods']:
            folder=run/'inverse'/obj/method;path=folder/'history.json'
            histories[method]=read_json(path) if path.exists() else []
            finished[method]=(folder/'complete.json').exists()
        rows.extend(compare_object(read_json(run/'baselines'/f'{obj}.json'),histories,finished,protocol['relative_error_reductions']))
    solver_seconds=0.;attempts=executions(run)
    for attempt in attempts:
        requester=attempt['requester']
        kind='initial' if requester.startswith('initial/') else ('query' if requester.startswith('query/') else ('diagnostic' if requester.startswith('domain_diagnostic/') else ('support' if any(requester.endswith('/'+m) for m in spec['methods']) else 'other')))
        work[kind]+=1;solver_seconds+=attempt['seconds']
    replays=[read_json(p) for p in (run/'evaluation').glob('*/cases/*/*/*/replay.json')]
    diagnostic_replays=[read_json(p) for p in (run/'diagnostics/full_support_replay').glob('cases/*/*/*/replay.json')]
    result=dict(protocol=protocol,matched_accuracy=rows,actual_unique_rollouts_by_stage=work,
        actual_initial_support_query_solver_seconds=solver_seconds,
        solver_seconds_scope='All completed initial, support, query, and numerical diagnostic executions; adopted copies counted once',
        physical_failures=sum(not a['valid'] for a in attempts),
        retired_numerical_failures=sum(not a['valid'] and a['retired'] for a in attempts),
        failed_attempts=[a for a in attempts if not a['valid']],
        additional_completed_render_replays=len(replays),additional_render_replay_seconds=sum(r['elapsed_s'] for r in replays),
        diagnostic_completed_render_replays=len(diagnostic_replays),
        diagnostic_render_replay_seconds=sum(r['elapsed_s'] for r in diagnostic_replays),
        all_completed_full_solver_executions=sum(work.values())+len(replays)+len(diagnostic_replays))
    write_json(run/'comparison.json',result)
    if rows:
        with (run/'matched_accuracy.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result
