"""Verified first-contribution output -> fixed-asset dynamics contract.

The source is the selection frozen before query evaluation. It includes the
actual confidence checkpoint, four-column weights and accepted CMA parameters.
No fallback to a manually configured material or a different contact runtime.
"""
from pathlib import Path
import numpy as np
from .io import read_json, sha256
from .parameters import PARAMETERS, ParameterSpace


def check_files(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen first-contribution input changed: {path}')


def audit_weighted_trace(selection, weights):
    """Verify saved candidate costs actually used the corresponding score column."""
    weights = np.asarray(weights, float)
    if weights.ndim != 2 or weights.shape[1] != 4 or not np.isfinite(weights).all() or (weights < 0).any() or (weights.sum(0) <= 0).any():
        raise ValueError('Expected finite, nonnegative, independent confidence columns')
    checked = 0; max_error = 0.
    for generation in selection['trace']:
        axis = PARAMETERS.index(generation['parameter'])
        losses = np.asarray(generation['support_losses_mm'], float)
        costs = np.asarray(generation['costs_mm'], float)
        if losses.shape != (len(costs), len(weights)):
            raise ValueError('CMA trace and support weights disagree')
        expected = losses @ weights[:, axis] / weights[:, axis].sum()
        if not np.array_equal(np.isfinite(costs), np.isfinite(expected)):
            raise ValueError('CMA trace validity does not match candidate losses')
        good = np.isfinite(costs)
        if good.any():
            error = float(np.max(np.abs(costs[good]-expected[good])))
            max_error = max(max_error, error); checked += int(good.sum())
    if checked == 0 or max_error > 1e-8:
        raise ValueError(f'First-contribution weighted trace mismatch: {max_error}')
    return dict(checked_candidates=checked, max_abs_cost_difference_mm=max_error,
                four_parameter_columns_preserved=True)


def load_handoff(fit_run, obj, method='learned', context='real_start0', budget=96):
    root = Path(fit_run).resolve(); plan_path = root/'plan.json'; plan = read_json(plan_path)
    if plan['schema'] != 'softdynamics.block-cma-recovery.v1' or method != 'learned' or context != 'real_start0':
        raise ValueError('Require the completed learned real-data CMA selection')
    selection_path = root/'selections'/obj/context/str(budget)/(method+'.json')
    selection = read_json(selection_path)
    if any(selection[k] != v for k,v in dict(object=obj, method=method, context=context, budget=budget).items()):
        raise ValueError('Selection identity mismatch')
    if selection['query_observations_used'] or selection['condition']['truth'] != 'real':
        raise ValueError('Selection must have been frozen without query observations')
    item = next(o for o in plan['objects'] if o['object'] == obj)
    if set(item['support']) & set(item['query']) or len(item['support']) < 3 or not item['query']:
        raise ValueError('Need disjoint support/query and a student validation episode')
    weights_path = root/'weights'/f'{obj}.json'; weights = read_json(weights_path)
    if weights['model_sha256'] != plan['model_sha256'] or sha256(plan['checkpoint']) != plan['model_sha256']:
        raise ValueError('First-contribution confidence checkpoint mismatch')
    if weights['teacher_scores_used'] or weights['support'] != item['support'] or weights['query'] != item['query']:
        raise ValueError('Weights must come from the frozen model on fitting episodes only')
    ctx = next(c for c in weights['contexts'] if c['id'] == context)
    if ctx['start'] != selection['start']:
        raise ValueError('Confidence and CMA initial guesses disagree')
    audit = audit_weighted_trace(selection, ctx['weights'][method])
    q = np.asarray(selection['selected_q'], float)
    expected = selection['candidate_q'] if selection['accepted'] else selection['start']
    if not np.array_equal(q, expected):
        raise ValueError('Selected parameters disagree with the frozen acceptance decision')
    parameters = ParameterSpace.from_config(plan['config']).parameters(q)
    entries = {str(e['episode']):e for e in item['episodes']}
    workspaces = {e['workspace'] for e in entries.values()}
    assets = {e['asset_sha256'] for e in entries.values()}
    if len(workspaces) != 1 or len(assets) != 1:
        raise ValueError('All interactions must use the same first-contribution asset')
    workspace = Path(next(iter(workspaces)))
    asset = read_json(workspace/'asset/asset.json')
    files = {str(plan_path):sha256(plan_path), str(selection_path):sha256(selection_path),
             str(weights_path):sha256(weights_path), plan['checkpoint']:plan['model_sha256'],
             str(workspace/'asset/asset.json'):next(iter(assets))}
    files.update({str(workspace/'asset'/name):digest for name,digest in asset['payload_sha256'].items()})
    for e in entries.values():
        files[str(workspace/f'initial_states/episode_{e["episode"]}.npz')] = e['initial_sha256']
        files[str(workspace/f'queries/episode_{e["episode"]}/actions.npz')] = e['actions_sha256']
        for path in (Path(e['row']), workspace/f'queries/episode_{e["episode"]}/episode.json',
                     workspace/f'initial_states/episode_{e["episode"]}.json'):
            files[str(path)] = sha256(path)
    check_files(files)
    return dict(schema=1, fit_run=str(root), object=obj, method=method, context=context, budget=budget,
                selection_path=str(selection_path), selected_q=q.tolist(), parameters=parameters,
                accepted=selection['accepted'], confidence_model_sha256=plan['model_sha256'],
                confidence_weights=ctx['weights'][method], weighted_trace_audit=audit,
                source_files=files, source_runtime=plan['runtime'], workspace=str(workspace),
                asset_sha256=next(iter(assets)), support=item['support'], query=item['query'],
                entries=entries, selection_frozen_at=selection['selection_frozen_at'],
                query_observations_used_for_fit=False, first_contribution_actually_used=True)
