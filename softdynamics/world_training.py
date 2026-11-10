"""Train and evaluate a fixed-asset dynamics surrogate on disjoint episodes."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .io import load_npz, read_json, sha256, write_json
from .fit_handoff import check_files
from .world_data import validate_splits
from .world_model import (DT_REF, MaterialDynamics, fit_statistics, normalize_state,
                          physical_state, position_error_mm, rollout, rollout_loss)


def load_trajectories(root, splits=('train', 'validation', 'test')):
    root = Path(root); plan = read_json(root/'plan.json')
    validate_splits(plan['config']['splits'])
    if sha256(root/'nodes.npz') != plan['nodes_sha256']:
        raise ValueError('Node identity changed')
    trajectories, excluded = [], []
    for task in plan['tasks']:
        if task['split'] not in splits:
            continue
        if task['episode'] not in plan['config']['splits'][task['split']]:
            raise ValueError('Task disagrees with frozen episode split')
        folder = root/'trajectories'/task['id']
        record = read_json(folder/'complete.json')
        if not record['valid']:
            if task['variant'] == 'nominal':
                raise ValueError(f'Nominal teacher unavailable: {task["id"]}')
            excluded.append(record)
            continue
        if record['diagnostic_truncation'] or record['task'] != task:
            raise ValueError('Truncated or mislabeled teacher trajectory')
        if 'handoff' in plan and record.get('first_contribution_sha256') != sha256(root/'first_contribution.json'):
            raise ValueError('Teacher is not bound to the selected first-contribution output')
        if sha256(folder/'trajectory.npz') != record['payload_sha256']:
            raise ValueError('Trajectory changed')
        t = load_npz(folder/'trajectory.npz')
        if not all(np.isfinite(t[k]).all() for k in ('state', 'pads', 'times')):
            raise ValueError(f'Non-finite trajectory: {task["id"]}')
        if (np.diff(t['times']) <= 0).any():
            raise ValueError('Non-monotonic time')
        t['scale_m'] = float(t['scale_m'])
        t['ground_y'] = float((-.01-t['origin_m'][1])/t['scale_m'])
        t['state'] = normalize_state(t['state'], t['origin_m'], t['scale_m'])
        t.update(task); t['record'] = record
        trajectories.append(t)
    return trajectories, excluded


def tensor_trajectories(trajectories, device):
    result = []
    for t in trajectories:
        result.append({**t, **{k: torch.as_tensor(t[k], dtype=torch.float32, device=device)
                              for k in ('state', 'pads', 'times', 'eligibility', 'ground_y')}})
    return result


def windows(trajectories, count, horizon, rng):
    """Sample episodes equally, then windows; augmentation never changes split."""
    result = []
    for _ in range(count):
        i = int(rng.integers(len(trajectories)))
        T = len(trajectories[i]['state'])
        if T <= horizon:
            raise ValueError('Trajectory shorter than rollout horizon')
        result.append((i, int(rng.integers(T-horizon))))
    return result


def batch_windows(trajectories, selection, horizon):
    states, pads, times, eligibility, ground = [], [], [], [], []
    for i, start in selection:
        t = trajectories[i]; end = start+horizon+1
        states.append(t['state'][start:end]); pads.append(t['pads'][start:end])
        times.append(t['times'][start:end]); eligibility.append(t['eligibility']); ground.append(t['ground_y'])
    return tuple(torch.stack(x) for x in (states, pads, times, eligibility, ground))


@torch.no_grad()
def validation_error(model, trajectories, selection, horizon, batch_size):
    errors = []
    model.eval()
    for start in range(0, len(selection), batch_size):
        choice = selection[start:start+batch_size]
        state, pads, times, eligibility, ground = batch_windows(trajectories, choice, horizon)
        predicted = rollout(model, state[:, 0], pads, eligibility, times, ground)
        scales = torch.tensor([trajectories[i]['scale_m'] for i, _ in choice], device=state.device)
        error = torch.linalg.vector_norm(predicted[..., :3]-state[:, 1:, :, :3], dim=-1)
        errors.extend((error.mean((1, 2))*scales*1000).cpu().tolist())
    model.train()
    return float(np.mean(errors))


def freeze_training_source(output):
    source = Path(__file__).resolve().parent
    target = output/'code'/'softdynamics'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    return {str(p.relative_to(output/'code')): sha256(p) for p in sorted(target.rglob('*.py'))}


def train(root, output, device='cuda:0', steps=None, variant=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if (output/'best.pt').exists() or (output/'manifest.json').exists():
        raise ValueError('Training output already exists; choose a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root/'plan.json'); cfg = copy.deepcopy(plan['config']); setting = copy.deepcopy(cfg['training'])
    if 'handoff' in plan:
        check_files(plan['handoff']['source_files'])
        if variant not in cfg['ablations']:
            raise ValueError('Choose a predeclared connected-model variant')
        experiment = cfg['ablations'][variant]
        cfg['model']['position_update'] = experiment['position_update']
        setting.update({k:experiment[k] for k in ('horizon','steps')})
    if steps is not None:
        setting['steps'] = steps
    seed = setting['seed']; np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    trajectories, excluded = load_trajectories(root, ('train', 'validation'))
    training = [t for t in trajectories if t['split'] == 'train']
    validation = [t for t in trajectories if t['split'] == 'validation']
    statistics = fit_statistics(training)
    nodes = load_npz(root/'nodes.npz')
    manifest = dict(schema=1, root=str(root), plan_sha256=sha256(root/'plan.json'),
                    nodes_sha256=plan['nodes_sha256'], config=cfg, training=setting,
                    first_contribution_sha256=sha256(root/'first_contribution.json') if 'handoff' in plan else None,
                    asset_sha256=plan['source_sha256'][str(Path(plan['workspace'])/'asset/asset.json')],
                    first_contribution=plan.get('handoff'), variant=variant,
                    train_ids=[t['id'] for t in training], validation_ids=[t['id'] for t in validation],
                    test_read_during_training=False, excluded=excluded,
                    trajectory_sha256={t['id']:t['record']['payload_sha256'] for t in trajectories},
                    source_sha256=freeze_training_source(output), started_at=time.time(),
                    device=device, torch_version=torch.__version__,
                    selection_metric=f'Validation mean material-node L2 mm over {max(cfg["evaluation_horizons"])}-step autoregressive windows')
    write_json(output/'manifest.json', manifest)
    write_json(output/'normalization.json', statistics)
    model = MaterialDynamics(nodes['canonical'], nodes['neighbors'], statistics, **cfg['model']).to(device)
    training = tensor_trajectories(training, device); validation = tensor_trajectories(validation, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting['learning_rate'], weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    val_horizon = max(cfg['evaluation_horizons'])
    val_windows = windows(validation, setting['validation_windows'], val_horizon, np.random.default_rng(seed+1))
    write_json(output/'validation_windows.json', dict(horizon=val_horizon, windows=val_windows))
    best = float('inf'); started = time.monotonic()
    for step in range(1, setting['steps']+1):
        # Same maximum horizon as the frozen configuration, with a short warmup.
        horizon = min(setting['horizon'], 1 if step <= 100 else 4 if step <= 250 else setting['horizon'])
        selection = windows(training, setting['batch_size'], horizon, rng)
        states, pads, times, eligibility, ground = batch_windows(training, selection, horizon)
        optimizer.zero_grad(set_to_none=True)
        predicted = rollout(model, states[:, 0], pads, eligibility, times, ground)
        loss = rollout_loss(predicted, states[:, 1:], model.loss_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            row = dict(step=step, horizon=horizon, loss=float(loss.detach()), grad_norm=float(grad_norm),
                       elapsed_s=time.monotonic()-started)
            with (output/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            write_json(output/'status.json', dict(status='training', **row))
            print('TRAIN', step, horizon, round(row['loss'], 6), round(row['elapsed_s'], 1), flush=True)
        if step == 1 or step % setting['evaluate_every'] == 0 or step == setting['steps']:
            error = validation_error(model, validation, val_windows, val_horizon, setting['batch_size'])
            with (output/'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step, mean_mm=error))+'\n')
            print('VALIDATION', step, error, flush=True)
            if np.isfinite(error) and error < best:
                best = error
                checkpoint = dict(model=model.state_dict(), model_config=cfg['model'],
                                  statistics=statistics, canonical=nodes['canonical'], neighbors=nodes['neighbors'],
                                  step=step, validation_mean_mm=error, manifest_sha256=sha256(output/'manifest.json'))
                torch.save(checkpoint, output/'best.tmp.pt'); (output/'best.tmp.pt').replace(output/'best.pt')
    if not (output/'best.pt').exists():
        raise FloatingPointError('No finite validation checkpoint')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=setting['steps']), output/'last.pt')
    write_json(output/'status.json', dict(status='trained', steps=setting['steps'], best_validation_mm=best,
               elapsed_s=time.monotonic()-started, completed_at=time.time(), best_sha256=sha256(output/'best.pt')))
    return output/'best.pt'


def load_checkpoint(checkpoint, device):
    # Checkpoints are generated locally by this module, never accepted from untrusted downloads.
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = MaterialDynamics(blob['canonical'], blob['neighbors'], blob['statistics'], **blob['model_config'])
    model.load_state_dict(blob['model']); model.to(device).eval()
    return model, blob


def summarize_error(prediction, target, scale, surface_mask):
    if not np.isfinite(prediction).all():
        return dict(valid=False, reason='Non-finite learned rollout')
    error = position_error_mm(prediction, target, scale)
    F = prediction[..., 6:15].reshape(*prediction.shape[:-1], 3, 3)+np.eye(3)
    det = np.linalg.det(F)
    return dict(valid=True, mean_mm=float(error.mean()), surface_mean_mm=float(error[..., surface_mask].mean()),
                endpoint_mm=float(error[-1].mean()), p95_mm=float(np.quantile(error, .95)),
                nonpositive_F_fraction=float((det<=0).mean()), min_F_determinant=float(det.min()))


def baselines(initial, times):
    H = len(times)-1
    persistence = np.repeat(initial[None], H, axis=0)
    velocity = persistence.copy()
    velocity[..., :3] += initial[None, :, 3:6] * ((times[1:]-times[0])/DT_REF)[:, None, None]
    return dict(persistence=persistence, constant_velocity=velocity)


@torch.no_grad()
def evaluate_windows(model, t, horizons, surface_mask):
    rows = []
    for horizon in horizons:
        starts = np.unique(np.linspace(0, len(t['state'])-horizon-1, min(24, len(t['state'])-horizon), dtype=int))
        for start in starts:
            end = start+horizon+1
            state = t['state'][start:end]; times = t['times'][start:end]
            pred = rollout(model, state[:1], t['pads'][None, start:end], t['eligibility'][None],
                           times[None], t['ground_y'][None])[0].cpu().numpy()
            target = state[1:].cpu().numpy()
            methods = {**baselines(state[0].cpu().numpy(), times.cpu().numpy()), 'learned':pred}
            for name, prediction in methods.items():
                rows.append(dict(id=t['id'], horizon=horizon, start=int(start), method=name,
                                 start_phase=int(t['phase'][start]),
                                 **summarize_error(prediction, target, t['scale_m'], surface_mask)))
    return rows


def action_gradient_check(model, t, horizon=8):
    """Check only the network's derivative, not its accuracy against EMPM."""
    valid = (t['pads'][:-1, :, 12]>.5) & (t['pads'][1:, :, 12]>.5)
    candidates = torch.nonzero(valid.any(1)).flatten().cpu().numpy()
    candidates = [int(i) for i in candidates if i+horizon<len(t['state'])]
    if not candidates:
        return dict(valid=False, reason='No valid controller interval')
    start = candidates[len(candidates)//2]; end = start+horizon+1
    initial = t['state'][None, start].double()
    times = t['times'][None, start:end].double(); pads = t['pads'][None, start:end].double()
    local_model = copy.deepcopy(model).double()
    eligibility = t['eligibility'][None].double(); ground = t['ground_y'][None].double()
    mask = (pads[..., 12]>.5).double()
    envelope = torch.linspace(0, 1, horizon+1, device=initial.device, dtype=torch.float64)[None, :, None]
    def objective(translation):
        displaced = torch.cat([pads[..., :3]+translation[None, None, None]*envelope[..., None]*mask[..., None],
                               pads[..., 3:]], -1)
        final = rollout(local_model, initial, displaced, eligibility, times, ground)[:, -1]
        return (final[..., :3].mean(1)*final.new_tensor([.7, .2, -.3])).sum()
    action = torch.zeros(3, device=initial.device, dtype=torch.float64, requires_grad=True)
    value = objective(action); gradient = torch.autograd.grad(value, action)[0]
    numerical = []
    epsilon = 1e-5
    with torch.no_grad():
        for k in range(3):
            delta = torch.zeros_like(action); delta[k] = epsilon
            numerical.append(float((objective(delta)-objective(-delta))/(2*epsilon)))
    analytic = gradient.detach().cpu().numpy(); numerical = np.array(numerical)
    relative = float(np.linalg.norm(analytic-numerical)/max(np.linalg.norm(numerical), 1e-12))
    return dict(valid=bool(np.isfinite(analytic).all() and relative<1e-3 and np.linalg.norm(analytic)>1e-10),
                id=t['id'], start=start, horizon=horizon, autograd=analytic, finite_difference=numerical,
                relative_error=relative, epsilon_normalized=epsilon,
                teacher_gradient_tested=False, interpretation='Autograd plumbing only; no control-success claim')


@torch.no_grad()
def benchmark(model, t, repeats=100):
    state=t['state'][None,0]; pads=t['pads'][None,:2]; eligible=t['eligibility'][None]
    dt=(t['times'][1]-t['times'][0])[None]; ground=t['ground_y'][None]
    def sync():
        if state.device.type == 'cuda': torch.cuda.synchronize(state.device)
    for _ in range(20): model(state,pads[:,0],pads[:,1],eligible,dt,ground)
    sync(); started=time.perf_counter()
    for _ in range(repeats): model(state,pads[:,0],pads[:,1],eligible,dt,ground)
    sync(); resident_s=(time.perf_counter()-started)/repeats
    host=[x.cpu() for x in (state,pads[:,0],pads[:,1],eligible,dt,ground)]
    sync(); started=time.perf_counter()
    for _ in range(repeats):
        model(*[x.to(state.device) for x in host]).cpu()
    sync(); transfer_s=(time.perf_counter()-started)/repeats
    teacher_s=t['record']['physics_s']/(t['record']['frames']-1)
    return dict(batch=1,nodes=state.shape[1],teacher_particles=t['record']['particles'],
                learned_device_resident_ms=resident_s*1000,learned_roundtrip_ms=transfer_s*1000,
                teacher_control_interval_ms=teacher_s*1000,
                teacher_over_learned_device_resident=teacher_s/resident_s,
                teacher_over_learned_roundtrip=teacher_s/transfer_s,
                caveat=f'Learned timing covers {state.shape[1]} material nodes; teacher evolves {t["record"]["particles"]} particles. Dense interpolation/rendering excluded. Teacher includes validity checks.')


def action_response_report(output, split):
    """Paired finite-amplitude action responses from the same initial state.

    This compares the learned action response with actual simulated changes,
    unlike an autograd-vs-network finite-difference plumbing test. A +/-4 mm
    intervention is not an infinitesimal derivative at a contact switch.
    """
    rows=[]
    for nominal_path in sorted(Path(output).glob(f'episode_*_nominal_{split}_prediction.npz')):
        prefix=nominal_path.name.split('_nominal_')[0]
        plus=Path(output)/f'{prefix}_translate_positive_{split}_prediction.npz'
        minus=Path(output)/f'{prefix}_translate_negative_{split}_prediction.npz'
        if not plus.exists() or not minus.exists():
            rows.append(dict(episode=prefix,valid=False,reason='Paired perturbed teacher unavailable'))
            continue
        positive,negative=load_npz(plus),load_npz(minus)
        true=(positive['teacher'][...,:3].astype('d')-negative['teacher'][...,:3].astype('d'))/2
        pred=(positive['predicted'][...,:3].astype('d')-negative['predicted'][...,:3].astype('d'))/2
        active=np.linalg.norm(true,axis=-1).mean(1)>1e-4
        finite=np.isfinite(pred).all()
        if not active.any() or not finite:
            rows.append(dict(episode=prefix,valid=False,reason='No measurable teacher response or non-finite student'))
            continue
        t,p=true[active],pred[active];delta=p-t
        denom=np.linalg.norm(t)*np.linalg.norm(p)
        rows.append(dict(episode=prefix,valid=True,frames=int(active.sum()),
            teacher_response_mm=float(np.linalg.norm(t,axis=-1).mean()*1000),
            learned_response_mm=float(np.linalg.norm(p,axis=-1).mean()*1000),
            response_error_mm=float(np.linalg.norm(delta,axis=-1).mean()*1000),
            zero_response_baseline_mm=float(np.linalg.norm(t,axis=-1).mean()*1000),
            cosine=float(np.sum(t*p)/denom) if denom>1e-12 else None,
            relative_l2_error=float(np.linalg.norm(delta)/np.linalg.norm(t))))
    return dict(rows=rows,definition='Half of positive-minus-negative full-rollout displacement at +/-4 mm smooth common gripper translations',
        teacher_reference='Actual first-contribution physics',
        infinitesimal_gradient_claim=False,teacher_response_threshold_mm=.1)


def evaluate(root, output, device='cuda:0', split='test'):
    root, output=Path(root),Path(output)
    destination=output/f'evaluation_{split}.json'
    if destination.exists():
        raise ValueError('Frozen evaluation already exists; do not tune against repeated test evaluations')
    manifest=read_json(output/'manifest.json')
    if sha256(root/'plan.json') != manifest['plan_sha256']:
        raise ValueError('Teacher plan changed since training')
    model, blob=load_checkpoint(output/'best.pt',device)
    if sha256(output/'manifest.json') != blob['manifest_sha256']:
        raise ValueError('Training manifest changed')
    torch.set_num_threads(1)
    trajectories,excluded=load_trajectories(root,(split,)); nodes=load_npz(root/'nodes.npz')
    trajectories=tensor_trajectories(trajectories,device)
    rows=[]; full=[]; phase_rows=[]
    for t in trajectories:
        rows.extend(evaluate_windows(model,t,manifest['config']['evaluation_horizons'],nodes['surface_mask']))
        with torch.no_grad():
            predicted=rollout(model,t['state'][:1],t['pads'][None],t['eligibility'][None],
                              t['times'][None],t['ground_y'][None])[0].cpu().numpy()
        target=t['state'][1:].cpu().numpy()
        for name,prediction in {**baselines(t['state'][0].cpu().numpy(),t['times'].cpu().numpy()),'learned':predicted}.items():
            full.append(dict(id=t['id'],method=name,steps=len(prediction),
                             **summarize_error(prediction,target,t['scale_m'],nodes['surface_mask'])))
            for phase in (0,1,2):
                mask=t['phase'][1:]==phase
                if mask.any():
                    phase_rows.append(dict(id=t['id'],method=name,phase=phase,
                            **summarize_error(prediction[mask],target[mask],t['scale_m'],nodes['surface_mask'])))
        np.savez_compressed(output/(t['id'].replace('/','_')+f'_{split}_prediction.npz'),
                            predicted=physical_state(predicted,t['origin_m'],t['scale_m']),
                            teacher=physical_state(target,t['origin_m'],t['scale_m']),
                            times=t['times'][1:].cpu().numpy(),phase=t['phase'][1:],surface_mask=nodes['surface_mask'])
        print('EVALUATE',t['id'],full[-1],flush=True)
    aggregates=[]
    for horizon in manifest['config']['evaluation_horizons']:
        for method in ('persistence','constant_velocity','learned'):
            group=[r for r in rows if r['horizon']==horizon and r['method']==method and r['id'].endswith('/nominal')]
            episode_scores=[]
            for identity in sorted({r['id'] for r in group}):
                episode=[r for r in group if r['id']==identity]
                episode_scores.append(np.mean([r['mean_mm'] for r in episode]) if all(r['valid'] for r in episode) else None)
            aggregates.append(dict(horizon=horizon,method=method,episodes=len(episode_scores),
                                   invalid_windows=sum(not r['valid'] for r in group),
                                   mean_mm=float(np.mean(episode_scores)) if all(e is not None for e in episode_scores) else None))
    report=dict(split=split,checkpoint_sha256=sha256(output/'best.pt'),checkpoint_step=blob['step'],
                selection_validation_mm=blob['validation_mean_mm'],aggregates=aggregates,
                windows=rows,full_rollouts=full,full_rollout_phases=phase_rows,excluded=excluded,
                speed=benchmark(model,trajectories[0]),action_gradient=action_gradient_check(model,trajectories[0]),
                completed_at=time.time(),error_reference='Same-material-node EMPM teacher positions, not real camera observations',
                test_protocol='Source-episode holdout on an historically inspected object; development pilot',
                sensitivity_network_modified=False,physical_parameters_updated_by_student=False,
                first_contribution=manifest.get('first_contribution'),variant=manifest.get('variant'),
                primary_aggregates='Nominal trajectories only; perturbations are paired within each source episode',
                action_response=action_response_report(output,split))
    write_json(destination,report)
    return report


def pipeline(root, output, device='cuda:0', steps=None, variant=None):
    """Wait for this collection only, then train and evaluate once; failures stop."""
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    while True:
        status=read_json(root/'collection_status.json')
        if status['status']=='complete': break
        if status['status']=='failed': raise RuntimeError(status)
        pid=status.get('pid')
        if pid:
            try: os.kill(pid,0)
            except ProcessLookupError: raise RuntimeError('Collection supervisor exited without completion')
        write_json(output/'pipeline_status.json',dict(status='waiting_for_teacher',updated_at=time.time()))
        time.sleep(15)
    train(root,output,device,steps,variant)
    write_json(output/'pipeline_status.json',dict(status='evaluating',updated_at=time.time()))
    evaluate(root,output,device,'validation')
    evaluate(root,output,device,'test')
    from .world_report import write_report
    write_report(output)
    write_json(output/'pipeline_status.json',dict(status='complete',completed_at=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate','pipeline'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--steps',type=int)
    p.add_argument('--variant',choices=['connected_short','connected_long'])
    p.add_argument('--split',choices=['validation','test'],default='test')
    a=p.parse_args()
    try:
        if a.mode=='train': train(a.root,a.output,a.device,a.steps,a.variant)
        elif a.mode=='evaluate': evaluate(a.root,a.output,a.device,a.split)
        else: pipeline(a.root,a.output,a.device,a.steps,a.variant)
    except Exception as exc:
        write_json(a.output/'failure.json',dict(error=repr(exc),updated_at=time.time()))
        raise


if __name__=='__main__': main()
