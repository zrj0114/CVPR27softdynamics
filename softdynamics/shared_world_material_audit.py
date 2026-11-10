"""Evaluate material responses against frozen full-physics trajectories.

This module never constructs a simulator or adds data to training replay. The
same cached initial state and recorded commands are used for every model and
material probe. Finite physical differences remain diagnostics, not labels.
"""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from .io import load_npz, read_json, sha256, write_json
from .parameters import PARAMETERS
from .shared_world_control import geometry_tensors, pose_rollout, reference_step_corrections, reference_step_anchors
from .shared_world_training import Replay, load_model
from .world_model import normalize_state, DT_REF


def response_agreement(predicted, actual):
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError('Response arrays must be finite and have identical shapes')
    pn, an = float(np.linalg.norm(predicted)), float(np.linalg.norm(actual))
    return dict(cosine=float(np.dot(predicted, actual) / (pn * an)) if min(pn, an) > 1e-12 else None,
                predicted_norm=pn, physical_norm=an, norm_ratio=pn / an if an > 1e-12 else None,
                relative_error=float(np.linalg.norm(predicted - actual) / an) if an > 1e-12 else None)


def validate_probe(baseline, probe, record, expected_theta, snapshot_sha256, horizon):
    """Reject stale or differently initialized/control-conditioned physics."""
    if record['snapshot_sha256'] != snapshot_sha256:
        raise ValueError('Physical snapshot changed')
    for key in ('origin_m', 'scale_m', 'initial_observed_velocity_m_s'):
        if not np.array_equal(baseline[key], probe[key]):
            raise ValueError('Physical probe changed ' + key)
    if not np.array_equal(baseline['state'][0], probe['state'][0]):
        raise ValueError('Physical probe changed initial state')
    frames = len(probe['state'])
    if not np.array_equal(probe['pads'], baseline['pads'][:frames]):
        raise ValueError('Physical probe changed recorded controls')
    if not np.array_equal(probe['times'], baseline['times'][:frames]):
        raise ValueError('Physical probe changed time intervals')
    if not np.array_equal(probe['theta'], np.asarray(expected_theta, dtype=probe['theta'].dtype)):
        raise ValueError('Physical probe material differs from its declared coordinate')
    if record['frames'] != frames or record['planned_frames'] != horizon + 1:
        raise ValueError('Incorrect physical frame count')
    if record['valid'] and (frames != horizon + 1 or not np.isfinite(probe['state']).all()):
        raise ValueError('Incomplete or nonfinite trajectory marked valid')


def load_case(root, folder):
    root, folder = Path(root), Path(folder)
    source = read_json(folder / 'result.json')
    if source.get('training_labels') is not False:
        raise ValueError('Expected a diagnostic-only physical cache')
    plan = read_json(root / 'plan.json')
    item = next(o for o in plan['objects'] if o['object'] == source['object'])
    if item['split'] == 'train':
        raise ValueError('Material response audit requires a held-out object')
    if sha256(item['nodes']) != item['nodes_sha256']:
        raise ValueError('Node mapping changed')
    fit_path = Path(plan['fit_run']) / 'plan.json'
    fit = read_json(fit_path)['config']
    base = np.array([fit['baseline'][k] for k in PARAMETERS], dtype=np.float64)
    scales = np.array(fit['coordinate_scales'], dtype=np.float64)
    q0 = np.array(source['q'], dtype=np.float64)
    theta0 = np.array([source['parameters'][k] for k in PARAMETERS], dtype=np.float64)
    np.testing.assert_allclose(base + scales * q0, theta0, rtol=1e-14, atol=1e-14)
    radii = sorted({float(row['radius']) for row in source['rows']}, reverse=True)
    if not radii or min(radii) <= 0:
        raise ValueError('Missing positive physical probe radii')
    definitions = {'baseline': dict(theta=theta0)}
    for radius in radii:
        for column, key in enumerate(PARAMETERS):
            for sign in (-1, 1):
                q = q0.copy(); q[column] += sign * radius
                definitions[f'h_{radius}_{key}_{sign}'] = dict(
                    theta=base + scales * q, radius=radius, parameter=key, sign=sign)
    arrays, records, hashes = {}, {}, {}
    for name, definition in definitions.items():
        path = folder / name / 'trajectory.npz'
        record = read_json(path.parent / 'complete.json')
        digest = sha256(path)
        if digest != record['trajectory_sha256']:
            raise ValueError('Physical payload changed: ' + str(path))
        arrays[name], records[name] = load_npz(path), record
        validate_probe(arrays['baseline'], arrays[name], record, definition['theta'],
                       source['snapshot_sha256'], source['horizon'])
        hashes[name] = dict(trajectory_sha256=digest, record_sha256=sha256(path.parent / 'complete.json'))
    if not records['baseline']['valid']:
        raise ValueError('Physical baseline is invalid')
    return dict(source=source, nodes=load_npz(item['nodes']), arrays=arrays, records=records,
                definitions=definitions, radii=radii,
                provenance=dict(source_result=str(folder / 'result.json'), source_sha256=sha256(folder / 'result.json'),
                    data_plan_sha256=sha256(root / 'plan.json'), fit_plan_sha256=sha256(fit_path),
                    nodes_sha256=item['nodes_sha256'], payloads=hashes))


def case_tensors(case, device):
    baseline = case['arrays']['baseline']
    origin, scale = baseline['origin_m'], float(baseline['scale_m'])
    state = baseline['state'][0].copy()
    state[:, 3:6] = baseline['initial_observed_velocity_m_s']
    # Match PhysicalControl.tensors(): float32 observation normalization followed
    # by FP64 model evaluation; no future-frame velocity enters the initial state.
    initial = torch.as_tensor(normalize_state(state, origin, scale)[None], device=device).double()
    geometry = geometry_tensors(case['nodes'], origin, device)
    geometry = {k: v.double() if v.is_floating_point() else v for k, v in geometry.items()}
    pads = torch.as_tensor(baseline['pads'][None], device=device, dtype=torch.float64)
    times = torch.as_tensor(baseline['times'][None], device=device, dtype=torch.float64)
    return initial, pads, geometry, times


@torch.no_grad()
def evaluate(root, physical_cache, checkpoints, output, device='cuda:0', step_anchor=False, stable_step_anchor=False):
    if stable_step_anchor:step_anchor=True
    root, physical_cache, output = Path(root), Path(physical_cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('Material audit results are immutable')
    with (output / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        complete = read_json(physical_cache / 'complete.json')
        cases = [load_case(root, physical_cache / f'episode_{e}') for e in complete['episodes']]
        checkpoint_hashes = {name: sha256(path) for name, path in checkpoints.items()}
        protocol = dict(physical_cache=str(physical_cache), cases=[c['provenance'] for c in cases],
            checkpoints={name: dict(path=str(path), sha256=checkpoint_hashes[name]) for name, path in checkpoints.items()},
            precision='float64', training_labels=False, new_physical_rollouts=0,
            reference_step_correction=step_anchor,
            stable_step_anchor=stable_step_anchor,
            baseline_error_interpretation='Reference reproduction by construction; not learned prediction accuracy' if step_anchor else 'Uncorrected model prediction accuracy',
            initial_velocity='Past observed displacement, matching the original audit',
            interpretation='Fixed-start local dynamics responses; not the gradient of a complete real-observation fitting objective')
        write_json(output / 'protocol.json', protocol)
        rows, predictions, started = [], {}, time.monotonic()
        for name, path in checkpoints.items():
            model, bundle = load_model(path, device)
            if sha256(path) != checkpoint_hashes[name]:
                raise ValueError('Checkpoint changed while loading')
            if bundle['data_plan_sha256'] != sha256(root / 'plan.json'):
                raise ValueError('Model uses a different data plan')
            if not model.direct_displacement:
                raise ValueError('This audit requires the observed-velocity model interface')
            model.double().requires_grad_(False)
            for case in cases:
                source = case['source']; obj, episode = source['object'], source['episode']
                if obj in bundle['training_objects']:
                    raise ValueError('Audit object was used to train the model')
                initial, pads, geometry, times = case_tensors(case, device)
                baseline = case['arrays']['baseline']
                values = {}; corrections=None; anchors=None; uncorrected_error=None
                if step_anchor:
                    states=normalize_state(baseline['state'],baseline['origin_m'],float(baseline['scale_m']))
                    states[1:,:,3:6]=np.diff(states[...,:3],axis=0)/(np.diff(baseline['times'])[:,None,None]/DT_REF)
                    reference=torch.as_tensor(states[None],device=device,dtype=torch.float64)
                    reference[:,0]=initial
                    theta0=torch.as_tensor(case['definitions']['baseline']['theta'][None],device=device,dtype=torch.float64)
                    if stable_step_anchor:anchors=reference_step_anchors(model,reference,pads,theta0,geometry,times)
                    else:corrections=reference_step_corrections(model,reference,pads,theta0,geometry,times)
                    uncorrected=pose_rollout(model,initial,pads,theta0,geometry,times)[0,:,:,:3].cpu().numpy()
                    uncorrected=uncorrected*float(baseline['scale_m'])+baseline['origin_m']
                    mask=case['nodes']['surface_mask']
                    uncorrected_error=float(np.linalg.norm(uncorrected[1:,mask]-baseline['state'][1:,mask,:3],axis=-1).mean()*1000)
                for probe, definition in case['definitions'].items():
                    theta = torch.as_tensor(definition['theta'][None], device=device, dtype=torch.float64)
                    states = pose_rollout(model, initial, pads, theta, geometry, times, step_corrections=corrections,step_anchors=anchors)
                    values[probe] = states[0, :, :, :3].cpu().numpy() * float(baseline['scale_m']) + baseline['origin_m']
                    if not np.isfinite(values[probe]).all():
                        raise FloatingPointError('Nonfinite model prediction')
                surface = case['nodes']['surface_mask']
                error = np.linalg.norm(values['baseline'][1:, surface] - baseline['state'][1:, surface, :3], axis=-1)
                responses = []
                for radius in case['radii']:
                    for key in PARAMETERS:
                        minus, plus = f'h_{radius}_{key}_-1', f'h_{radius}_{key}_1'
                        valid = case['records'][minus]['valid'] and case['records'][plus]['valid']
                        row = dict(parameter=key, radius=radius, valid=valid)
                        if valid:
                            actual = (case['arrays'][plus]['state'][1:, surface, :3].astype(np.float64) -
                                      case['arrays'][minus]['state'][1:, surface, :3]) / (2 * radius)
                            predicted = (values[plus][1:, surface] - values[minus][1:, surface]) / (2 * radius)
                            row.update(whole_window=response_agreement(predicted, actual),
                                terminal=response_agreement(predicted[-1], actual[-1]),
                                physical_plus_minus_mean_mm=float(np.linalg.norm(actual, axis=-1).mean() * 2 * radius * 1000))
                        responses.append(row)
                rows.append(dict(model=name, checkpoint_step=bundle['step'], object=obj, episode=episode,
                    baseline_position_error_mm=float(error.mean() * 1000), terminal_position_error_mm=float(error[-1].mean() * 1000),
                    uncorrected_baseline_position_error_mm=uncorrected_error,reference_step_correction=step_anchor,
                    responses=responses, physical_difference_stability=source['physical_difference_stability']))
                for probe, positions in values.items():
                    predictions[f'{name}__episode_{episode}__{probe}'] = positions
                write_json(output / 'progress.json', dict(completed_cases=len(rows), rows=rows, elapsed_s=time.monotonic() - started))
            del model, bundle
        np.savez_compressed(output / 'predictions.npz', **predictions)
        result = dict(**protocol, rows=rows, predictions_sha256=sha256(output / 'predictions.npz'),
                      elapsed_s=time.monotonic() - started)
        write_json(output / 'result.json', result)
        return result


@torch.no_grad()
def evaluate_training_materials(root, recent_manifest, checkpoints, output, device='cuda:0', max_horizon=None):
    """Can a model use materials even on its ordinary training experience?

    This is deliberately a training-fit diagnostic, separate from held-out
    physical verification. Common starts/commands let us swap material inputs
    exactly by permuting predictions, without additional model rollouts.
    """
    if max_horizon is not None and max_horizon<1:raise ValueError('Diagnostic horizon must be positive')
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('Material audit results are immutable')
    with (output / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        manifest = read_json(recent_manifest)
        if manifest['data_plan_sha256'] != sha256(root / 'plan.json'):
            raise ValueError('Material replay uses a different data plan')
        replay = Replay(root, ['train'], True); replay.refresh()
        groups = {}
        for identity, digest in manifest['trajectories'].items():
            t = replay.trajectories[identity]
            if t['record']['trajectory_sha256'] != digest:
                raise ValueError('Material experience payload changed')
            if t['record'].get('schema') != 'softdynamics.material-experience.v1':
                raise ValueError('Expected independent material experience')
            groups.setdefault(identity.rsplit('/', 1)[0], []).append(identity)
        for group, identities in groups.items():
            identities.sort(key=lambda i: replay.trajectories[i]['record']['material_sample_index'])
            base = replay.trajectories[identities[0]]
            if len(identities) < 2 or base['record']['material_sample_index'] != 0:
                raise ValueError('Each material group needs its physical nominal baseline')
            for identity in identities:
                t = replay.trajectories[identity]
                if t['record']['snapshot_sha256'] != base['record']['snapshot_sha256']:
                    raise ValueError('Training material group has different physical snapshots')
                for key in ('pads', 'times', 'origin_m', 'scale_m'):
                    if not np.array_equal(t[key], base[key]):
                        raise ValueError('Training material group changed ' + key)
                if not np.array_equal(t['state'][0], base['state'][0]) or t['minimum_start'] != 0:
                    raise ValueError('Training material group lacks a common observed initial state')
        protocol = dict(recent_manifest=str(recent_manifest), recent_manifest_sha256=sha256(recent_manifest),
            groups=groups, training_fit_diagnostic=True, held_out_generalization_evaluation=False,
            maximum_diagnostic_horizon=max_horizon,
            response_labels_used_for_training=False, new_physical_rollouts=0,
            checkpoints={name: dict(path=str(path), sha256=sha256(path)) for name, path in checkpoints.items()})
        write_json(output / 'protocol.json', protocol)
        rows, started = [], time.monotonic()
        for name, path in checkpoints.items():
            model, bundle = load_model(path, device); model.requires_grad_(False)
            if sha256(path) != protocol['checkpoints'][name]['sha256']:
                raise ValueError('Checkpoint changed while loading')
            if not model.direct_displacement or bundle['data_plan_sha256'] != manifest['data_plan_sha256']:
                raise ValueError('Incompatible model for the material replay')
            for group, identities in sorted(groups.items()):
                base = replay.trajectories[identities[0]]
                predictions, targets = [], []
                horizon = len(base['state']) - 1
                if max_horizon is not None:horizon=min(horizon,max_horizon)
                for identity in identities:
                    batch, geometry = replay.batch([(identity, 0)], horizon, device)
                    predicted = pose_rollout(model, batch['states'][:, 0], batch['pads'], batch['theta'], geometry, batch['times'])
                    predictions.append(predicted[0, 1:, :, :3].cpu().numpy())
                    targets.append(batch['states'][0, 1:, :, :3].cpu().numpy())
                mask = replay.geometry[base['task']['object']]['surface_mask']
                scale = float(base['scale_m'])
                predicted = np.stack(predictions).astype(np.float64)[:, :, mask] * scale
                actual = np.stack(targets).astype(np.float64)[:, :, mask] * scale
                def mae(value):
                    return float(np.linalg.norm(value - actual, axis=-1).mean() * 1000)
                correct, nominal, shuffled = mae(predicted), mae(predicted[:1]), mae(np.roll(predicted, 1, axis=0))
                actual_response, predicted_response = actual[1:] - actual[:1], predicted[1:] - predicted[:1]
                rows.append(dict(model=name, checkpoint_step=bundle['step'], group=group, object=base['task']['object'],
                    samples=len(identities), horizon=horizon, correct_material_mae_mm=correct,
                    nominal_material_mae_mm=nominal, shuffled_material_mae_mm=shuffled,
                    benefit_over_nominal_mm=nominal-correct, benefit_over_shuffled_mm=shuffled-correct,
                    physical_material_spread_mm=float(np.linalg.norm(actual_response, axis=-1).mean() * 1000),
                    predicted_material_spread_mm=float(np.linalg.norm(predicted_response, axis=-1).mean() * 1000),
                    material_response=response_agreement(predicted_response, actual_response)))
                write_json(output / 'progress.json', dict(rows=rows, elapsed_s=time.monotonic() - started))
            del model, bundle
        result = dict(**protocol, rows=rows, elapsed_s=time.monotonic() - started)
        write_json(output / 'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--physical-cache', type=Path)
    data.add_argument('--recent-manifest', type=Path)
    parser.add_argument('--checkpoint', nargs=2, action='append', metavar=('NAME', 'PATH'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--step-anchor',action='store_true')
    parser.add_argument('--stable-step-anchor',action='store_true',help='Subtract the reference prediction before adding the measured next state')
    parser.add_argument('--max-horizon',type=int,help='Limit the training-input diagnostic to the first transitions')
    args = parser.parse_args()
    checkpoints = {name: Path(path) for name, path in args.checkpoint}
    if len(checkpoints) != len(args.checkpoint):
        parser.error('Checkpoint names must be unique')
    if args.recent_manifest:
        if args.step_anchor or args.stable_step_anchor:parser.error('Step correction is a physical-cache diagnostic, not a training-fit metric')
        evaluate_training_materials(args.root, args.recent_manifest, checkpoints, args.output, args.device,args.max_horizon)
    else:
        if args.max_horizon is not None:parser.error('A shorter horizon is only supported for training-input diagnostics')
        evaluate(args.root, args.physical_cache, checkpoints, args.output, args.device, args.step_anchor,args.stable_step_anchor)


if __name__ == '__main__':
    main()
