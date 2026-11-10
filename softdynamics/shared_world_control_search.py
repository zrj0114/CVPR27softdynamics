"""Prospective physical search with equal rollout budgets.

The learned direction is frozen before any candidate is executed. Its amplitude
is halved on a declared schedule. Random shooting and CMA-ES use the same action
coordinates, bounds, initial physical snapshot, goal, and candidate budget.
These evaluations do not provide response or gradient training targets.
"""
from pathlib import Path
import time

import numpy as np
import torch

from .io import write_json
from .shared_world_control import bounded_pose_sequence


def search_protocol(budget, seed):
    if budget < 4 or budget % 4:
        raise ValueError('Equal-budget CMA comparison requires a positive multiple of four candidates')
    return dict(candidate_budget=budget, baseline_rollouts_per_method=1,
        amplitude_scales=[.5**i for i in range(budget)],
        amplitude_selection='Lowest actual terminal mean surface distance, including the unchanged baseline',
        schedule='Fixed geometric halving; no early stopping or per-object tuning',
        random_standard_deviation=.25, cma_initial_standard_deviation=.25,
        cma_population_size=4, cma_seed=seed+1,
        action_coordinates='14 channels x 4 temporal knots, tanh bounded whole-window pose offsets',
        training_labels=False)


def update_search(trace, value, valid, baseline, elapsed, **metadata):
    """Charge every attempted rollout and retain the baseline on failed steps."""
    if baseline is None or not np.isfinite(baseline):
        raise ValueError('Search acceptance requires a valid physical baseline')
    valid=bool(valid and value is not None and np.isfinite(value))
    previous=trace[-1]['best_terminal_mm'] if trace else float(baseline)
    row=dict(candidate=len(trace), valid=valid, terminal_mm=float(value) if valid else None,
        best_terminal_mm=min(previous,float(value)) if valid else previous,
        candidate_rollouts=len(trace)+1, physical_rollouts_including_baseline=len(trace)+2,
        elapsed_seconds=(trace[-1]['elapsed_seconds'] if trace else 0.)+elapsed, **metadata)
    trace.append(row)
    return row


def compare_search(physical, pads0, geometry, raw, proposed, proposal_record, target,
                   hold_error, shooting_trace, budget, seed, output, planning_seconds, baseline_seconds):
    """Called by the control benchmark while its original simulator is live."""
    import cma
    output=Path(output);protocol=search_protocol(budget,seed)
    if len(shooting_trace)!=budget:
        raise ValueError('Random shooting and learned search must have equal candidate budgets')
    mask=physical.nodes['surface_mask'];traces={'world_amplitude':[], 'random':[], 'cma_es':[]}
    def error(states):
        return float(np.linalg.norm(states[-1,mask,:3]-target,axis=-1).mean()*1000)
    def report():
        write_json(output/'search_progress.json',dict(protocol=protocol,traces=traces,updated_at=time.time()))
    for row in shooting_trace:
        previous=traces['random'][-1]['elapsed_seconds'] if traces['random'] else 0.
        update_search(traces['random'],row['terminal_mm'],row['valid'],hold_error,
            row['elapsed_seconds']-previous)
    raw=torch.as_tensor(raw,device=pads0.device,dtype=pads0.dtype)
    with torch.no_grad():
        for index,alpha in enumerate(protocol['amplitude_scales']):
            if index==0:
                states,record=proposed,proposal_record
            else:
                pads=bounded_pose_sequence(pads0,raw,physical.horizon,geometry['scale_m'],
                    translation_m=.012*alpha,rotation_rad=.08*alpha,aperture_m=.006*alpha)[0]
                states,record=physical.rollout(pads[0].cpu().numpy(),output/'amplitude_search'/f'candidate_{index:03d}')
            update_search(traces['world_amplitude'],error(states) if record['valid'] else None,
                record['valid'],hold_error,record['elapsed_s'],scale=alpha,error=record['error'])
            report()
    optimizer=cma.CMAEvolutionStrategy(np.zeros(56),.25,
        {'popsize':4,'seed':seed+1,'verbose':-9,'verb_log':0,'verb_disp':0})
    with torch.no_grad():
        for generation in range(budget//4):
            candidates=optimizer.ask();scores=[]
            for candidate in candidates:
                value=torch.tensor(candidate.reshape(1,14,4),device=pads0.device,dtype=pads0.dtype)
                pads=bounded_pose_sequence(pads0,value,physical.horizon,geometry['scale_m'])[0]
                index=len(traces['cma_es'])
                states,record=physical.rollout(pads[0].cpu().numpy(),output/'physical_cma'/f'candidate_{index:03d}')
                metric=error(states) if record['valid'] else None
                row=update_search(traces['cma_es'],metric,record['valid'],hold_error,
                    record['elapsed_s'],generation=generation,error=record['error'])
                scores.append(row['terminal_mm'] if row['valid'] else 1e9)
                report()
            optimizer.tell(candidates,scores)
    if any(len(rows)!=budget for rows in traces.values()):
        raise RuntimeError('Incomplete equal-budget comparison')
    summary={}
    for method,rows in traces.items():
        summary[method]=dict(terminal_mm=rows[-1]['best_terminal_mm'],
            candidate_rollouts=budget,physical_rollouts_including_baseline=budget+1,
            physical_seconds=baseline_seconds+rows[-1]['elapsed_seconds'],
            planning_seconds=planning_seconds if method=='world_amplitude' else 0.,
            total_online_seconds=baseline_seconds+rows[-1]['elapsed_seconds']+(planning_seconds if method=='world_amplitude' else 0.),
            improvement_over_hold_mm=hold_error-rows[-1]['best_terminal_mm'])
    return dict(protocol=protocol,traces=traces,summary=summary,
        cma_version=cma.__version__,common_prefix_and_goal_generation_excluded_from_all_methods=True,
        reference_goal_trajectory_is_not_optimizer_input=True,
        physical_baseline_logical_cost_charged_to_each_method=True)
