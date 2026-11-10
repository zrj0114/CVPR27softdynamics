"""Fixed object/phase-balanced validation, including a no-motion baseline."""
from collections import defaultdict

import numpy as np
import torch

from .elastic_world_objective import autoregressive, phase_weights
from .shared_world_training import validate


def summarize_windows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['object'], row['phase'])].append(row)
    per_object, per_phase = {}, {}
    for obj in sorted({o for o, _ in groups}):
        values = {p: groups[obj, p] for p in (0, 1, 2) if (obj, p) in groups}
        mass = sum((.1, .45, .45)[p] for p in values)
        per_object[obj] = {field: sum((.1, .45, .45)[p] * np.mean([r[field] for r in rs])
            for p, rs in values.items()) / mass for field in ('position_mm', 'static_mm')}
    for phase in (0, 1, 2):
        values = [rs for (o, p), rs in groups.items() if p == phase]
        if values:
            per_phase[phase] = {field: float(np.mean([np.mean([r[field] for r in rs]) for rs in values]))
                for field in ('position_mm', 'static_mm')}
    return dict(mean_mm=float(np.mean([v['position_mm'] for v in per_object.values()])),
        static_mean_mm=float(np.mean([v['static_mm'] for v in per_object.values()])),
        per_object=per_object, per_phase=per_phase, window_count=len(rows))


def macro_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row['object']].append(row[field])
    return float(np.mean([np.mean(values) for values in groups.values()]))


@torch.no_grad()
def evaluate(model, val, confidence, tau, pairs, selection, legacy_selection, device):
    model.eval()
    local_rows, full_rows, cache = [], [], {}
    for offset in range(0, len(selection), 4):
        choices = selection[offset:offset+4]
        batch, geometry = val.batch([(k, s) for k, s, _ in choices], 32, device)
        prediction = autoregressive(model.world, batch, geometry, gradients=False)[..., :3]
        target = batch['states'][:, 1:, :, :3]
        scale = geometry['scale_m'][:, None, None]*1000
        error = (prediction-target).norm(dim=-1)*scale
        static = (batch['states'][:, :1, :, :3]-target).norm(dim=-1)*scale
        for i, (key, start, phase) in enumerate(choices):
            local_rows.append(dict(object=val.trajectories[key]['task']['object'], id=key, start=start,
                phase=phase, position_mm=float(error[i].mean()), static_mm=float(static[i].mean())))
    for key, trajectory in sorted(val.trajectories.items()):
        batch, geometry = val.batch([(key, 0)], len(trajectory['state'])-1, device)
        prediction = autoregressive(model.world, batch, geometry, gradients=False)[0, ..., :3]
        cache[key] = prediction.cpu()
        target = batch['states'][0, 1:, :, :3]
        phase = torch.as_tensor(trajectory['phase'][1:], device=device)
        weights = phase_weights(phase, prediction.dtype)
        scale = float(trajectory['scale_m'])*1000
        error = (prediction-target).norm(dim=-1).mean(1)*scale
        static = (batch['states'][0, :1, :, :3]-target).norm(dim=-1).mean(1)*scale
        full_rows.append(dict(object=trajectory['task']['object'], id=key,
            position_mm=float((error*weights).sum()), static_mm=float((static*weights).sum()),
            phases={p: dict(position_mm=float(error[phase==p].mean()), static_mm=float(static[phase==p].mean()))
                    for p in (0, 1, 2) if (phase==p).any()}))
    response_rows = []
    cfg = val.plan['config']
    for obj, choices in sorted(pairs.by_object.items()):
        for a, b in choices:
            x, y = val.trajectories[a], val.trajectories[b]
            weights = phase_weights(torch.as_tensor(x['phase'][1:]), torch.float32)
            scale = float(x['scale_m'])
            actual = torch.as_tensor(y['state'][1:, :, :3]-x['state'][1:, :, :3])*scale
            forecast = (cache[b]-cache[a])*scale
            qa, qb = [np.array([np.log(t['theta'][0]/cfg['reference_E_pa']),
                (t['theta'][1]-cfg['reference_nu'])/cfg['nu_scale']]) for t in (x, y)]
            floor = (.003*max(.1, np.linalg.norm(qb-qa)))**2
            energy = (actual.square().mean((1, 2))*weights).sum()
            response = ((forecast-actual).square().mean((1, 2))*weights).sum()/energy.clamp_min(floor)
            static_response = energy/energy.clamp_min(floor)
            aw = (actual*weights.sqrt()[:, None, None]).flatten()
            fw = (forecast*weights.sqrt()[:, None, None]).flatten()
            response_rows.append(dict(object=obj, a=a, b=b, response_relative_mse=float(response),
                static_response_relative_mse=float(static_response),
                response_cosine=float(torch.dot(aw, fw)/(aw.norm()*fw.norm()).clamp_min(1e-12))))
    confidence_errors = []
    for obj in sorted({r['object'] for r in confidence.rows}):
        ids = [i for i, r in enumerate(confidence.rows) if r['object']==obj][:12]
        values = []
        for offset in range(0, len(ids), 2):
            c, theta, target, mask = confidence.batch(ids[offset:offset+2], device, tau)
            values.extend((model.confidence(c, theta)-target).abs()[mask].cpu().tolist())
        confidence_errors.append(float(np.mean(values)))
    local = summarize_windows(local_rows)
    full_mm = macro_mean(full_rows, 'position_mm')
    response = macro_mean(response_rows, 'response_relative_mse')
    confidence_mae = float(np.mean(confidence_errors))
    # Keep the old selection available as a separately named comparison.
    legacy_local = validate(model.world, val, legacy_selection, 32, 2, device)
    full_by_id = {r['id']: r for r in full_rows}
    legacy_full = np.mean([(full_by_id[a]['position_mm']+full_by_id[b]['position_mm'])/2
                          for a, b, _ in pairs.validation_choices()])
    return dict(score=local['mean_mm']/3+full_mm/3+response+confidence_mae,
        local32=local, full_prefix_position_mm=full_mm, full_prefix_response_relative_mse=response,
        static_full_prefix_position_mm=macro_mean(full_rows, 'static_mm'),
        static_response_relative_mse=macro_mean(response_rows, 'static_response_relative_mse'),
        confidence_mae=confidence_mae, full_trajectory_count=len(full_rows), response_pair_count=len(response_rows),
        legacy_local32=legacy_local, legacy_four_pair_position_mm=float(legacy_full),
        local_records=local_rows, full_records=full_rows, response_records=response_rows)
