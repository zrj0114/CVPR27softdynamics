"""Train the frozen-label recurrent surrogate with coupled E/nu supervision."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .io import read_json, write_json, sha256, json_value
from .elastic_world_data import freeze_data, frozen_replay
from .elastic_train import atomic_checkpoint
from .teacher_contact import require_selected_architecture, contract
from .teacher_contact_replay import require_aligned_replay
from .coupled_world_data import JointLabels, TemporalPairs, temporal_batch, fit_statistics_streaming
from .coupled_world_model import CoupledWorld
from .coupled_world_validation import evaluate

LOSS_NAMES = ['rollout', 'first_step', 'sensitivity', 'correlation', 'local_response',
              'head_consistency', 'paired_absolute', 'paired_response']


def source_hashes():
    root = Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(root.glob('*.py'))}


def curriculum(step, total=40000):
    fraction = step/total
    if fraction < .025:
        return 1, 32
    if fraction < .10:
        return 4, 128
    if fraction < .25:
        return 8, None
    return 16, None


def prepare(data_root, output, steps=40000):
    output = Path(output).resolve()
    if (output/'config.json').exists():
        raise ValueError('Use a new output directory for a new protocol')
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze_data(data_root, output)
    print('AUDIT contact/actions for', len(manifest['records']), 'frozen labels', flush=True)
    require_aligned_replay(manifest, output/'teacher_contact_alignment.json')
    plan = read_json(Path(data_root)/'plan.json')
    architecture = plan['config']['model']
    require_selected_architecture(architecture)
    cfg = dict(schema='softdynamics.coupled-temporal-training.v1', steps=steps, seed=20261110,
        architecture=architecture, model_class='CoupledWorld', data_root=str(Path(data_root).resolve()),
        data_manifest_sha256=sha256(output/'data_manifest.json'), source_sha256=source_hashes(),
        batch_size=2, sensitivity_batch_size=1, history_frames=3,
        curriculum=[[.025, 1, 32], [.1, 4, 128], [.25, 8, None], [1., 16, None]],
        local_response_interval=8, paired_response_interval=16, paired_gradient_tail=16,
        loss_names=LOSS_NAMES, loss_weights=[1., .2, .1, .025, .1, .05, .1, .2],
        learning_rate=1e-4, minimum_learning_rate=1e-5, warmup_steps=500,
        validation_interval=500, validation_inverse_steps=12, save_interval=100,
        initialization='random', checkpoint_files=['best.pt', 'latest.pt'],
        max_raw_gradient_norm=100000., prediction_feedback=True,
        material_trajectory_response_loss=True, temporal_dynamics='Per-node GRU over shared graph encodings',
        temporal_scoring='Ordered GRU with timestamps, intervals and phase indicators',
        joint_output='Positive strengths and bounded correlation, PSD 2x2 Gram matrix',
        local_response='Teacher-aligned fixed-history finite secants, in normalized teacher q coordinates',
        paired_response='Matched trajectories from identical frame zero; no observed-state resets',
        pair_weight_gradient='Backpropagate through final 16 intervals; preceding states remain predicted',
        best_selection='16-step mm/3 + .1 paired response MSE + .1 local response MSE + .1 inverse normalized error',
        inverse_validation='Fixed validation objects only; 12-step single-start short-window diagnostic; not MPM validation',
        test_objects_excluded=manifest['excluded_objects'], contact_contract=contract(),
        contact_alignment_sha256=sha256(output/'teacher_contact_alignment.json'))
    write_json(output/'config.json', cfg)
    return cfg


def train(output, resume=False, device='cuda:0'):
    output = Path(output).resolve()
    cfg = read_json(output/'config.json')
    if cfg['source_sha256'] != source_hashes():
        raise ValueError('Training source changed since prepare')
    if cfg['data_manifest_sha256'] != sha256(output/'data_manifest.json'):
        raise ValueError('Frozen manifest changed')
    if cfg['contact_contract'] != contract():
        raise ValueError('Teacher contact/actions changed')
    if cfg['contact_alignment_sha256'] != sha256(output/'teacher_contact_alignment.json'):
        raise ValueError('Alignment audit changed')
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = output/'training'
    out.mkdir(exist_ok=True)
    lock = (out/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'latest.pt').exists() and not resume:
        raise ValueError('Existing checkpoint requires --resume')
    old = torch.load(out/'latest.pt', map_location='cpu', weights_only=False) if resume else None
    if old and old['config_sha256'] != sha256(output/'config.json'):
        raise ValueError('Resume protocol changed')
    def status(stage, **kwargs):
        row = dict(status=stage, pid=os.getpid(), updated_at=time.time(), **kwargs)
        write_json(out/'status.json', row)
        print('COUPLED_STATUS', json.dumps(json_value(row)), flush=True)
        return row
    status('loading_trajectories', step=old['step'] if old else 0, total_steps=cfg['steps'])
    manifest = read_json(output/'data_manifest.json')
    replay = frozen_replay(manifest, 'train')
    val = frozen_replay(manifest, 'validation')
    status('loading_joint_labels', training_trajectories=len(replay.trajectories), validation_trajectories=len(val.trajectories))
    auxiliary, va = JointLabels(cfg['data_root'], replay), JointLabels(cfg['data_root'], val)
    status('checking_material_pairs', training_labels=len(auxiliary.rows), validation_labels=len(va.rows))
    pairs, vp = TemporalPairs(replay), TemporalPairs(val)
    statistics = old['statistics'] if old else fit_statistics_streaming(replay)
    tau = np.asarray(old['tau'] if old else auxiliary.calibrate())
    material = replay.plan['config']
    model = CoupledWorld(cfg['architecture'], tau, material['reference_E_pa'], material['reference_nu'], material['nu_scale']).to(device)
    model.world.set_statistics(statistics)
    if old:
        model.load_state_dict(old['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    rng = np.random.default_rng(cfg['seed'])
    if old:
        optimizer.load_state_dict(old['optimizer'])
        rng.bit_generator.state = old['rng']['numpy']
        torch.set_rng_state(old['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(old['rng']['cuda'], device)
    protocol = dict(cfg, training_trajectories=len(replay.trajectories), validation_trajectories=len(val.trajectories),
        training_objects=replay.object_names(), validation_objects=val.object_names(),
        training_ids=sorted(replay.trajectories), validation_ids=sorted(val.trajectories),
        training_sensitivity_labels=len(auxiliary.rows), validation_sensitivity_labels=len(va.rows),
        training_joint_valid=sum(r['correlation_valid'] for r in auxiliary.rows),
        material_pairs=pairs.report(), validation_material_pairs=vp.report(), tau=tau.tolist(),
        parameter_count=sum(p.numel() for p in model.parameters()), device=str(device),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
        scalar_response_scales='Training-only median; no validation or test calibration')
    write_json(out/'protocol.json', protocol)
    write_json(out/'statistics.json', statistics)
    start = old['step'] if old else 0
    best = old['best_score'] if old else float('inf')
    metrics = old['metrics'] if old else None
    weights = torch.tensor(cfg['loss_weights'], device=device)
    started = time.monotonic()
    last_status = status('ready', step=start, total_steps=cfg['steps'], parameters=protocol['parameter_count'])
    def save(step, improved=False):
        payload = dict(schema='softdynamics.coupled-temporal-checkpoint.v1', model=model.state_dict(),
            model_class='CoupledWorld', architecture=cfg['architecture'], optimizer=optimizer.state_dict(),
            step=step, best_score=best, metrics=metrics, protocol=protocol, statistics=statistics, tau=tau.tolist(),
            material_coordinates={k:material[k] for k in ('reference_E_pa', 'reference_nu', 'nu_scale')},
            config_sha256=sha256(output/'config.json'), data_manifest_sha256=cfg['data_manifest_sha256'],
            rng=dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
        atomic_checkpoint(out/'latest.pt', payload)
        if improved:
            atomic_checkpoint(out/'best.pt', payload)
    for step in range(start, cfg['steps']):
        horizon, prefix = curriculum(step, cfg['steps'])
        choices = replay.sample(cfg['batch_size'], horizon, rng)
        batch, geo = temporal_batch(replay, choices, horizon, device)
        label_ids = auxiliary.sample(cfg['sensitivity_batch_size'], rng)
        aux = auxiliary.batch(label_ids, tau, device)
        use_local = step % cfg['local_response_interval'] == 0
        pair_choice = pairs.sample(rng, prefix) if step % cfg['paired_response_interval'] == 0 else None
        pair = pairs.batch(pair_choice, device) if pair_choice else None
        lr = (cfg['minimum_learning_rate']+.5*(cfg['learning_rate']-cfg['minimum_learning_rate'])*
              (1+math.cos(math.pi*step/cfg['steps'])))*min(1., (step+1)/cfg['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            losses = model(batch, geo, aux, pair=pair, local_response=use_local, pair_tail=cfg['paired_gradient_tail'])
            objective = (losses*weights).sum()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite forward loss')
            objective.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            if norm > cfg['max_raw_gradient_norm']:
                raise FloatingPointError(f'Excessive raw gradient norm {norm}')
            optimizer.step()
        except Exception as error:
            write_json(out/'failure.json', dict(step=step+1, choices=choices, label_ids=label_ids,
                pair_choice=pair_choice, error=repr(error), time=time.time()))
            status('failed', step=step+1, error=repr(error))
            raise
        if step == start or (step+1) % 20 == 0:
            last_status = status('running', step=step+1, total_steps=cfg['steps'], horizon=horizon,
                losses=dict(zip(LOSS_NAMES, losses.detach().cpu().tolist())), gradient_norm=norm,
                learning_rate=lr, elapsed_s=time.monotonic()-started,
                peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type=='cuda' else 0.)
            with (out/'train.jsonl').open('a') as stream:
                stream.write(json.dumps(json_value(last_status))+'\n')
        # Preserve progress before potentially lengthy validation.
        if step == start or (step+1) % cfg['save_interval'] == 0:
            save(step+1)
        due = step == 0 or (step+1) % cfg['validation_interval'] == 0 or step+1 == cfg['steps']
        if due:
            status('validating', step=step+1, total_steps=cfg['steps'])
            metrics = evaluate(model, val, va, vp, device, cfg['validation_inverse_steps'])
            improved = math.isfinite(metrics['score']) and metrics['score'] < best
            if improved:
                best = metrics['score']
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(json_value(dict(step=step+1, **metrics)))+'\n')
            save(step+1, improved)
            status('running', step=step+1, total_steps=cfg['steps'], best_score=best,
                   validation_mm=metrics['dynamics']['mean_mm'], latest_checkpoint_step=step+1)
        if (output/'pause.request.json').exists():
            save(step+1)
            status('paused', step=step+1, resumable_checkpoint_step=step+1)
            os.replace(output/'pause.request.json', output/'pause.fulfilled.json')
            return
    status('complete', step=cfg['steps'], best_score=best)
    write_json(out/'complete.json', dict(step=cfg['steps'], best_score=best, time=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'train'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.data_root, args.output, args.steps)
    else:
        train(args.output, args.resume, args.device)


if __name__ == '__main__':
    main()
