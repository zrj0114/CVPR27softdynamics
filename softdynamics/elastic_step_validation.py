"""Every held-out transition, independently conditioned on its known state."""
from collections import defaultdict

import numpy as np
import torch

from .elastic_step_objective import independent_steps
from .shared_world_model import DT_REF


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['object'], row['phase'])].append(row)
    fields = ('position_mm', 'static_mm', 'constant_velocity_mm')
    per_object = {}
    for obj in sorted({o for o, _ in groups}):
        phases = [p for p in (0, 1, 2) if (obj, p) in groups]
        mass = sum((.1, .45, .45)[p] for p in phases)
        per_object[obj] = {f: sum((.1, .45, .45)[p]*np.mean([v[f] for v in groups[obj, p]])
                                  for p in phases)/mass for f in fields}
    per_phase = {p: {f: float(np.mean([np.mean([v[f] for v in rs])
                    for (o, phase), rs in groups.items() if phase==p])) for f in fields}
                    for p in (0, 1, 2) if any(phase==p for _, phase in groups)}
    return dict(**{f: float(np.mean([v[f] for v in per_object.values()])) for f in fields},
        per_object=per_object, per_phase=per_phase)


@torch.no_grad()
def evaluate_confidence(model, confidence, tau, device, batch_size=4):
    """Evaluate every frozen label, with equal weight per object and parameter."""
    records = []
    for obj in sorted({r['object'] for r in confidence.rows}):
        ids = [i for i, r in enumerate(confidence.rows) if r['object']==obj]
        errors = [[], []]
        for offset in range(0, len(ids), batch_size):
            selected = ids[offset:offset+batch_size]
            c, theta, target, mask = confidence.batch(selected, device, tau)
            prediction = model.confidence(c, theta)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite confidence validation prediction')
            for j in range(2):
                errors[j].extend((prediction[:,j]-target[:,j]).abs()[mask[:,j]].cpu().tolist())
        records.append(dict(object=obj, rows=len(ids), valid_counts=[len(e) for e in errors],
            mae=[float(np.mean(e)) if e else None for e in errors]))
    by_parameter = [[r['mae'][j] for r in records if r['mae'][j] is not None] for j in range(2)]
    if not all(by_parameter):
        raise ValueError('Validation requires reliable labels for both material parameters')
    mae = [float(np.mean(values)) for values in by_parameter]
    return dict(mean_mae=float(np.mean(mae)), mae=mae, parameters=['E_pa','nu'],
        evaluated_rows=sum(r['rows'] for r in records), available_rows=len(confidence.rows),
        per_object=records, aggregation='Equal objects within each parameter, then equal parameters')


@torch.no_grad()
def evaluate(model, val, confidence, tau, device):
    model.eval()
    rows = []
    transitions = 0
    for key, trajectory in sorted(val.trajectories.items()):
        start = trajectory['minimum_start']
        batch, geo = val.batch([(key, start)], len(trajectory['state'])-start-1, device)
        prediction = independent_steps(model.world, batch, geo, gradients=False)[0, ..., :3]
        current, target = batch['states'][0, :-1], batch['states'][0, 1:, :, :3]
        dt = (batch['times'][0, 1:]-batch['times'][0, :-1])[:, None, None]/DT_REF
        cv = current[..., :3]+current[..., 3:6]*dt
        scale = float(trajectory['scale_m'])*1000
        curves = dict(position_mm=(prediction-target).norm(dim=-1).mean(-1)*scale,
            static_mm=(current[..., :3]-target).norm(dim=-1).mean(-1)*scale,
            constant_velocity_mm=(cv-target).norm(dim=-1).mean(-1)*scale)
        phase = torch.as_tensor(trajectory['phase'][start+1:], device=device)
        transitions += len(phase)
        for p in (0, 1, 2):
            mask = phase==p
            if mask.any():
                rows.append(dict(object=trajectory['task']['object'], id=key, phase=p,
                    transitions=int(mask.sum()), **{k: float(v[mask].mean()) for k, v in curves.items()}))
    confidence_result = evaluate_confidence(model, confidence, tau, device)
    summary = summarize(rows)
    return dict(score=summary['position_mm'], one_step=summary, confidence_mae=confidence_result['mean_mae'],
        confidence=confidence_result,
        trajectories=len(val.trajectories), independent_transitions=transitions,
        state_source='Ground-truth current state; velocity from current and previous positions only',
        prediction_feedback=False, recurrent_horizon=1, records=rows)
