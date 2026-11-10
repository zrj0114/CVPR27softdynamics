"""Train a differentiable MPM surrogate for predominantly neural E/nu search.

Only best.pt/latest.pt are model artifacts. MPM replay is immutable; historical
network weights are never loaded unless explicitly resuming THIS training run.
"""
import argparse
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .elastic_data import PARAMETERS
from .elastic_train import ConfidenceReplay, atomic_checkpoint, clip_gradients
from .elastic_world_data import freeze_data, frozen_replay
from .elastic_step_objective import StepWorld
from .elastic_step_validation import evaluate
from .io import read_json, write_json, sha256, json_value
from .shared_world_training import fit_statistics
from .teacher_contact import selected_architecture,require_selected_architecture,contract
from .teacher_contact_replay import require_aligned_replay


def source_hashes():
    root = Path(__file__).parent
    return {name: sha256(root/name) for name in ('elastic_world_train.py', 'elastic_world_data.py',
        'elastic_step_objective.py', 'shared_world_model.py', 'joint_world.py',
        'shared_world_training.py', 'elastic_train.py', 'elastic_step_validation.py',
        'elastic_step_observation.py', 'elastic_world_search.py','teacher_contact.py',
        'teacher_contact_support.py','teacher_contact_replay.py')}


def prepare(data_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = freeze_data(data_root, output)
    require_aligned_replay(manifest,output/'teacher_contact_alignment.json')
    architecture = selected_architecture(read_json(Path(data_root)/'plan.json')['config']['model'])
    cfg = dict(schema='softdynamics.world-step-training.v3', steps=40000, seed=20261102,
        architecture=architecture, data_root=str(Path(data_root).resolve()),
        data_manifest_sha256=sha256(output/'data_manifest.json'),
        batch_per_rank=16, confidence_batch_per_rank=2,
        loss_weights=[1., .1], learning_rate=1e-4, minimum_learning_rate=1e-5,
        warmup_steps=500, validation_interval=500, save_interval=100,
        max_raw_gradient_norm=10000., recurrent_horizon=1, prediction_feedback=False,
        checkpoint_files=['best.pt', 'latest.pt'], initialization='random',
        sampling='Object and phase balanced independent one-step transitions',
        dynamics_supervision='MPM current-state/action/material to next state; no predicted-state feedback',
        material_trajectory_response_loss=False, response_is_exact_jacobian=False,
        train_velocity='Backward position difference, never future velocity',
        test_objects_excluded=manifest['excluded_objects'],
        best_selection='All held-out independent one-step position errors, balanced by object and phase',
        validation_baselines=['No motion', 'Constant current observed velocity'],
        source_sha256=source_hashes(),contact_contract=contract(),
        contact_alignment_sha256=sha256(output/'teacher_contact_alignment.json'))
    path = output/'config.json'
    if path.exists() and read_json(path) != cfg:
        raise ValueError('Refusing to replace an existing training configuration')
    write_json(path, cfg)
    return cfg


def gradients(wrapper, model, optimizer, arguments, weights, world_size, max_raw_norm=10000.):
    """One-step derivatives; no time truncation or recurrent gradient graph."""
    attempts = []
    for scale in (1., 2.**-16, 2.**-32, 2.**-64):
        optimizer.zero_grad(set_to_none=True)
        losses = wrapper(*arguments)
        loss = (losses*weights).sum()
        valid = torch.isfinite(loss).to(torch.int32)
        if world_size > 1:
            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid:
            raise FloatingPointError('Nonfinite one-step forward loss; failed step is not counted')
        (loss*scale).backward()
        norm = loss.new_tensor(float('inf'), dtype=torch.float64)
        try:
            norm = clip_gradients(model.parameters(), 1., scale).double()
        except RuntimeError as error:
            if 'non-finite' not in str(error):
                raise
        if world_size > 1:
            dist.all_reduce(norm, op=dist.ReduceOp.MAX)
        raw = float(norm)
        attempts.append(dict(backward_scale=scale, raw_norm=raw))
        if math.isfinite(raw):
            if raw > max_raw_norm:
                raise FloatingPointError(f'Excessive one-step gradient: {raw}')
            return losses, raw, dict(backward_scale=scale, gradient_attempts=attempts,
                clip_coefficient=min(1., 1./(raw+1e-6)))
    raise FloatingPointError(f'Nonfinite one-step backward persisted: {attempts}')


def train(output, resume=False):
    output = Path(output).resolve()
    cfg = read_json(output/'config.json')
    require_selected_architecture(cfg['architecture'])
    if cfg.get('contact_contract')!=contract():
        raise ValueError('Training configuration differs from the selected teacher contact/actions')
    if cfg['source_sha256'] != source_hashes():
        raise ValueError('Training code differs from frozen configuration')
    if cfg['data_manifest_sha256'] != sha256(output/'data_manifest.json'):
        raise ValueError('Frozen data manifest changed')
    rank, size = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if size > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45), device_id=device)
    out = output/'training'
    out.mkdir(exist_ok=True)
    if rank == 0:
        lock = (out/'training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'latest.pt').exists() and not resume:
            raise ValueError('Existing run requires --resume')
    old = torch.load(out/'latest.pt', map_location='cpu', weights_only=False) if resume else None
    if old and old['config_sha256'] != sha256(output/'config.json'):
        raise ValueError('Resume configuration changed')
    manifest = read_json(output/'data_manifest.json')
    require_aligned_replay(manifest)
    if rank == 0:
        write_json(out/'status.json', dict(status='loading_data', step=0, total_steps=cfg['steps'],
            world_size=size, pid=os.getpid(), updated_at=time.time()))
        print('WORLD_LOADING frozen trajectories', flush=True)
    replay, val = frozen_replay(manifest, 'train'), frozen_replay(manifest, 'validation')
    if rank == 0:
        print('WORLD_LOADING auxiliary sensitivity labels and normalization', flush=True)
    confidence, vc = ConfidenceReplay(cfg['data_root'], 'train'), ConfidenceReplay(cfg['data_root'], 'validation')
    for dataset, source in ((confidence, replay), (vc, val)):
        dataset.refresh()
        dataset.rows = sorted([r for r in dataset.rows if r['context'] in source.trajectories], key=lambda r:r['source'])
    statistics = old['statistics'] if old else fit_statistics(replay)
    tau = np.asarray(old['tau'] if old else confidence.calibrate())
    model = StepWorld(cfg['architecture'], PARAMETERS).to(device)
    model.world.set_statistics(statistics)
    if old:
        model.load_state_dict(old['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    if old:
        optimizer.load_state_dict(old['optimizer'])
    wrapper = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if size>1 else model
    rng = np.random.default_rng(cfg['seed']+rank)
    if old and len(old['rng_states']) == size:
        state = old['rng_states'][rank]
        rng.bit_generator.state = state['numpy']
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'], device)
    protocol = dict(cfg, world_size=size, training_objects=replay.object_names(), validation_objects=val.object_names(),
        training_trajectories=len(replay.trajectories), validation_trajectories=len(val.trajectories),
        validation_ids=sorted(val.trajectories), torch_version=torch.__version__,
        training_ids=sorted(replay.trajectories), gradient_clip=1.,
        backward_retry_scales=[1.,2.**-16,2.**-32,2.**-64],
        initial_state_source='Each independent example uses its actual current state',
        predicted_state_as_input=False, maximum_prediction_depth=1)
    if rank == 0:
        write_json(out/'protocol.json', protocol)
        print('STEP_DATA_READY', dict(train=len(replay.trajectories), validation=len(val.trajectories),
            world_size=size, independent_batch_per_rank=cfg['batch_per_rank']), flush=True)
    weights = torch.tensor(cfg['loss_weights'], device=device)
    start = old['step'] if old else 0
    best = old['best_score'] if old else float('inf')
    last_metrics = old['metrics'] if old else None
    started = time.monotonic()
    status = dict(status='running', pid=os.getpid(), step=start, total_steps=cfg['steps'], world_size=size)
    for step in range(start, cfg['steps']):
        horizon = 1
        choices = replay.sample(cfg['batch_per_rank'], horizon, rng)
        batch, geometry = replay.batch(choices, horizon, device)
        cb = confidence.batch(confidence.sample(cfg['confidence_batch_per_rank'], rng), device, tau)
        model.train()
        lr = (cfg['minimum_learning_rate']+.5*(cfg['learning_rate']-cfg['minimum_learning_rate'])*
              (1+math.cos(math.pi*step/cfg['steps'])))*min(1.,(step+1)/cfg['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        try:
            losses, norm, gradient_info = gradients(wrapper, model, optimizer, (batch, geometry, cb), weights, size, cfg['max_raw_gradient_norm'])
        except (FloatingPointError, RuntimeError) as error:
            write_json(out/f'failure_rank_{rank}.json', dict(step=step+1, choices=choices,
                error=repr(error), created_at=time.time()))
            raise
        optimizer.step()
        if (step+1)%20 == 0 or step==start or step+1==cfg['steps'] or len(gradient_info['gradient_attempts'])>1:
            values = losses.detach().clone()
            if size > 1:
                dist.all_reduce(values)
                values /= size
            if rank == 0:
                status = dict(status='running', pid=os.getpid(), step=step+1, total_steps=cfg['steps'],
                    local_horizon=1, prediction_feedback=False, independent_batch_per_rank=cfg['batch_per_rank'],
                    loss_names=['one_step_dynamics','sensitivity'],
                    losses=values.cpu().tolist(), gradient_norm=norm, **gradient_info,
                    learning_rate=lr, world_size=size, elapsed_s=time.monotonic()-started,
                    peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device)/1e9, updated_at=time.time())
                write_json(out/'status.json', status)
                with (out/'train.jsonl').open('a') as stream:
                    stream.write(json.dumps(json_value(status))+'\n')
                print('WORLD_TRAIN', json.dumps(json_value(status)), flush=True)
        pause = torch.tensor(int(rank==0 and (output/'pause.request.json').exists()), device=device)
        if size > 1:
            dist.broadcast(pause, src=0)
        due = step==0 or (step+1)%cfg['validation_interval']==0 or step+1==cfg['steps']
        improved = False
        if due:
            if size > 1:
                dist.barrier()
            if rank == 0:
                write_json(out/'status.json', dict(status, status='validating', step=step+1, updated_at=time.time()))
                last_metrics = evaluate(model,val,vc,tau,device)
                score = last_metrics['score']
                improved = math.isfinite(score) and score < best
                if improved:
                    best = score
                row = dict(step=step+1, best_score=best, **last_metrics)
                with (out/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(json_value(row))+'\n')
                print('WORLD_VALIDATE', json.dumps(json_value({k:v for k,v in row.items() if k!='records'})), flush=True)
            if size > 1:
                dist.barrier()
        if due or (step+1)%cfg['save_interval']==0 or pause:
            local = dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
            states = [None]*size
            if size > 1:
                dist.all_gather_object(states, local)
            else:
                states = [local]
            if rank == 0:
                payload = dict(schema='softdynamics.elastic-world-step-checkpoint.v3',
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    architecture=model.world.architecture, confidence_parameters=list(PARAMETERS),
                    optimizer=optimizer.state_dict(), step=step+1, total_steps=cfg['steps'], best_score=best,
                    statistics=statistics,tau=tau.tolist(),rng_states=states,protocol=protocol,metrics=last_metrics,
                    config_sha256=sha256(output/'config.json'),data_manifest_sha256=cfg['data_manifest_sha256'])
                atomic_checkpoint(out/'latest.pt', payload)
                if improved:
                    atomic_checkpoint(out/'best.pt', payload)
                write_json(out/'status.json', dict(status, status='running', step=step+1,
                    latest_checkpoint_step=step+1, best_score=best, updated_at=time.time()))
            if size > 1:
                dist.barrier()
        if pause:
            if rank == 0:
                write_json(out/'status.json', dict(status, status='paused', step=step+1,
                    resumable_checkpoint_step=step+1, updated_at=time.time()))
                os.replace(output/'pause.request.json', output/'pause.fulfilled.json')
            if size > 1:
                dist.destroy_process_group()
            return
    if rank == 0:
        write_json(out/'complete.json', dict(status='complete', step=cfg['steps'], best_score=best, finished_at=time.time()))
        status.update(status='complete', step=cfg['steps'], updated_at=time.time())
        write_json(out/'status.json', status)
    if size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','train'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.data_root, args.output)
    else:
        train(args.output, args.resume)


if __name__ == '__main__':
    main()
