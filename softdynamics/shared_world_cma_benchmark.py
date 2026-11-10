"""Prepare the frozen real-material model-assisted CMA comparison.

This replaces the one-gradient/three-step diagnostic, not the physical solver.
All three methods have 96 official continuous candidates, with 32/64/96 reports.
"""
import argparse
import copy
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .cma_benchmark import snapshot_code, supervise
from .io import read_json, sha256, write_json
from .parameters import ParameterSpace
from .recovery_data import trajectory_folder


def import_incumbent(root, plan, previous):
    """Only the exact incumbent, verified against the copied physics and inputs."""
    old = read_json(previous/'plan.json')
    if plan['incumbent_q'] != old['incumbent_q'] or plan['data_plan_sha256'] != old['data_plan_sha256']:
        raise ValueError('Changed incumbent material or data protocol')
    runtime = Path(plan['runtime'])
    data_hashes = read_json(Path(plan['data_root'])/'plan.json')['source_sha256']
    records = []
    for item in plan['objects']:
        source_item = next(o for o in old['objects'] if o['object'] == item['object'])
        for entry in item['episodes']:
            prior = next(e for e in source_item['episodes'] if e['episode'] == entry['episode'])
            for key in ('dtmax', 'domain_margin', 'initial_sha256', 'actions_sha256', 'asset_sha256',
                        'observation_sha256', 'surface_ids_sha256', 'real_surface_sha256'):
                if entry[key] != prior[key]:
                    raise ValueError('Changed incumbent episode protocol: '+key)
            # The recovery entry has no row_sha256 field. Its row identity is
            # frozen in the world-data source manifest instead.
            if entry['row'] != prior['row'] or sha256(entry['row']) != data_hashes[entry['row']]:
                raise ValueError('Changed incumbent prepared-data row')
            source = trajectory_folder(previous, item['object'], entry['episode'], plan['incumbent_q'])
            complete = read_json(source/'complete.json')
            if not complete['valid'] or complete.get('smoke') or complete['q'] != plan['incumbent_q']:
                raise ValueError('Invalid or wrong incumbent trajectory')
            if complete['parameters'] != ParameterSpace.from_config(plan['config']).parameters(plan['incumbent_q']):
                raise ValueError('Changed physical incumbent parameters')
            if complete['dtmax'] != entry['dtmax'] or complete['domain']['margin_m'] != entry['domain_margin']:
                raise ValueError('Changed incumbent solver settings')
            paths = dict(initial=Path(entry['workspace'])/f'initial_states/episode_{entry["episode"]}.npz',
                actions=Path(entry['workspace'])/f'queries/episode_{entry["episode"]}/actions.npz',
                asset=Path(entry['workspace'])/'asset/asset.json',
                pad_kernel=runtime/'gs-layer/pad_contact_kernels.py',
                adapter=runtime.parent/'softdynamics/adapters/pad_empm.py',
                initializer=runtime/'run_episode.py', transfer=runtime/'gs-layer/deterministic_transfer.py')
            # Keep the full original hash dictionary: each named active source
            # must have an explicit path, rather than checking a convenient subset.
            current = {key: sha256(path) for key, path in paths.items()}
            if current != complete['source_hashes']:
                raise ValueError('Incumbent physics sources differ: '+str(set(current)^set(complete['source_hashes'])))
            for name, digest in complete['payload_sha256'].items():
                if sha256(source/(name+'.npy')) != digest:
                    raise ValueError('Changed physical incumbent payload')
            destination = trajectory_folder(root, item['object'], entry['episode'], plan['incumbent_q'])
            destination.mkdir(parents=True)
            for name in ('complete.json', 'validation.json', 'tracks.npy', 'surface.npy'):
                os.link(source/name, destination/name)
            records.append(dict(object=item['object'], episode=entry['episode'],
                complete_sha256=sha256(source/'complete.json'), physical_seconds=complete['elapsed_s']))
    return records


def prepare(root, data_root, reference, state_reference, checkpoint):
    root, data_root, reference, state_reference, checkpoint = map(lambda p: Path(p).resolve(),
        (root, data_root, reference, state_reference, checkpoint))
    if root.exists():
        raise FileExistsError('Use a fresh run directory; completed protocols are immutable')
    data = read_json(data_root/'plan.json')
    plan = copy.deepcopy(read_json(reference/'plan.json'))
    if plan['data_plan_sha256'] != sha256(data_root/'plan.json') or len(plan['objects']) != 1:
        raise ValueError('Expected one object and its matching complete physical reference')
    item = plan['objects'][0]; obj = item['object']
    export = data_root/'first_module'/f'{obj}.json'; confidence = read_json(export)
    if (confidence['support'] != item['support'] or confidence['query'] != item['query']
            or set(item['support']) & set(item['query'])):
        raise ValueError('Changed first-module support/query split')
    initial = confidence['selected_q'] if confidence['selected_q'] is not None else confidence['initial_q']
    if initial != plan['incumbent_q'] or confidence['model_sha256'] != data['confidence_sha256']:
        raise ValueError('Changed first-module checkpoint or material handoff')
    if sha256(data['confidence_checkpoint']) != data['confidence_sha256']:
        raise ValueError('Changed first-module model file')
    if confidence['teacher_labels_used'] or confidence['query_observations_used']:
        raise ValueError('Invalid confidence provenance')
    from .shared_world_training import load_model
    model, bundle = load_model(checkpoint, 'cpu')
    if obj in bundle['training_objects'] or not model.direct_displacement:
        raise ValueError('Expected a held-out object and an observed-velocity world model')
    del model, bundle
    weights = np.asarray(confidence['confidence_weights'], float)
    if weights.shape != (len(item['support']), 4) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid four independent confidence columns')
    anchors = []
    for episode in item['support']:
        folder = state_reference/obj/f'episode_{episode}'
        record = read_json(folder/'complete.json')
        if (not record['valid'] or record['frames'] != record['planned_frames'] or record['query_states_used']
                or record['physical_plan_sha256'] != sha256(reference/'plan.json')
                or record['data_plan_sha256'] != sha256(data_root/'plan.json') or record['q'] != initial):
            raise ValueError('Incomplete or mismatched support state reference')
        if sha256(folder/'trajectory.npz') != record['trajectory_sha256']:
            raise ValueError('Changed physical state reference')
        anchors.append(dict(episode=episode, record_sha256=sha256(folder/'complete.json'),
            trajectory_sha256=record['trajectory_sha256'], capture_seconds=record['elapsed_s']))
    root.mkdir(parents=True)
    snapshot_code(root, plan)
    plan['runtime'] = str(root/'code/runtime')
    # Inputs are immutable copies; caches are separately verified below.
    for entry in item['episodes']:
        source = Path(entry['input']); destination = root/'inputs'/obj/source.name
        shutil.copytree(source, destination)
        entry['input'] = str(destination)
    imported = import_incumbent(root, plan, reference)
    plan.update(schema='softdynamics.shared-world-cma-real.v1', created_at=time.time(),
        previous_run=str(reference), methods=['uniform', 'learned', 'physical_uniform'],
        conditions=[dict(key='real', truth='real', noise_mm=0., seed=19)],
        unused_gpu_required=True,
        optimizer=dict(name='parameter-block CMA-ES with optional frozen-world preselection',
            library='cma==4.5.0', dimensions_per_block=1, population=4, block_generations=2,
            cycles=3, sigma=.2, seed=9101, bounds_q=[-.025, .025], CMA_active=False,
            maxstd='disabled (pycma N=1 bug)', budgets=[32, 64, 96], early_stop=False),
        world_preselection=dict(methods=['uniform', 'learned'], population=16,
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            data_root=str(data_root), data_plan_sha256=sha256(data_root/'plan.json'),
            window_horizon=32, batch_windows=8,
            objects={obj:dict(physical_reference=str(reference), physical_reference_sha256=sha256(reference/'plan.json'),
                state_reference=str(state_reference))},
            reference_cost=dict(dense_rollouts=imported, state_captures=anchors),
            ranking='Current parameter confidence column across support interactions; no four-column averaging',
            fitness='CMA.tell receives full physical scores only; surrogate ranking never becomes physical fitness'),
        method_definitions=dict(uniform='Frozen-world preselection and equal physical support weights',
            learned='Frozen-world preselection and learned independent per-parameter physical support weights',
            physical_uniform='Original full-physics block CMA; no world-model preselection'),
        fairness='Same initial material, recorded actions, local bounds, seeds, block order, physical population and 96 official candidate vectors. World arms additionally score 384 surrogate vectors; all extra costs reported.',
        selection_policy='No early stop; finish each 32-candidate sweep. Keep the weighted best within each block including its entering incumbent. Apply the same strict unweighted physical-support handoff gate.',
        query_policy='Freeze support choices at 32/64/96 before any candidate query scoring; report raw, accepted and baseline query scores.',
        cache_policy='Only exact verified incumbent imported; fresh continuous candidates. Identical candidates across arms may share physical rollouts; cache sharing is not a wallclock improvement.',
        interpretation='Model-assisted one-dimensional parameter-block CMA, not joint four-dimensional CMA or true-physics autodiff. Windowed surrogate covers every observation frame. These are development objects.',
        numerical_scope='Primary matched full-physics comparison' if obj=='048-butter-sponge' else 'Diagnostic; existing numerical convergence limitations remain unresolved',
        supersedes_diagnostic='One full-history gradient followed by three step lengths; those results do not constitute a CMA comparison',
        confidence_export_sha256=sha256(export), world_training=False, response_supervision=False,
        query_observations_used=False, recorded_actions_fixed=True)
    for key in ('proposals', 'grid_levels', 'passes', 'logical_candidates_per_fit'):
        plan.pop(key, None)
    write_json(root/'config.json', plan['config'])
    write_json(root/'cache_import.json', dict(incumbents=imported, other_candidates_imported=0, scores_recomputed=True))
    write_json(root/'weights'/f'{obj}.json', dict(model_sha256=data['confidence_sha256'],
        export_sha256=sha256(export), query_observations_used=False, teacher_labels_used=False,
        contexts=[dict(id='real_start_handoff', condition=plan['conditions'][0], start=initial,
            weights=dict(uniform=np.ones_like(weights), learned=weights, physical_uniform=np.ones_like(weights)))]))
    write_json(root/'plan.json', plan)
    write_json(root/'launch.json', dict(prepared_at=time.time(), plan_sha256=sha256(root/'plan.json'),
        source_manifest_sha256=sha256(root/'code_manifest.json')))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'supervise'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path); p.add_argument('--reference', type=Path)
    p.add_argument('--state-reference', type=Path); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--gpus', type=int, nargs='+', default=[4])
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.root, a.data_root, a.reference, a.state_reference, a.checkpoint)
    else:
        supervise(a.root, a.gpus)


if __name__ == '__main__':
    main()
