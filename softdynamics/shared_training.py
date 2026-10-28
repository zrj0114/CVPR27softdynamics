"""Object-disjoint motion pretraining and four independent sensitivity heads."""
import argparse
import copy
import itertools
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from .inference import extract, validate_input
from .io import load_npz, read_json, sha256, write_json
from .motion import CausalMotion, make_batch, baselines
from .parameters import PARAMETERS, ParameterSpace
from .training import fit, predict


def tensors(path, device):
    a = load_npz(path)
    validate_input(a)
    if len(a['times']) < 32 or a['actions'].shape[1] != 52:
        raise ValueError('Expected >=32 frames with independent pad actions')
    return {k:torch.as_tensor(a[k],device=device) for k in
            ['positions','valid','geometry','actions','times','scale_m']}


@torch.inference_mode()
def motion_evaluation(model, rows, device, windows=8):
    model.eval()
    totals, count = np.zeros(3), 0
    for row in rows:
        d = tensors(row['input'],device)
        for start in np.unique(np.linspace(0,len(d['times'])-32,windows).astype(int)):
            b,target,mask,scale = make_batch({'e':d},['e'],np.random.default_rng(17100+int(start)),starts=[int(start)])
            for j,p in enumerate([model(b),*baselines(b)]):
                errors = torch.linalg.vector_norm(p-target,dim=-1)*scale[:,None,None]*1000
                totals[j] += float(errors[mask].sum())
            count += int(mask.sum())
    if count == 0:
        raise ValueError('No visible held-out motion targets')
    return dict(zip(['model_mm','hold_mm','constant_velocity_mm'],(totals/count).tolist()))


def motion_train(rows, config, output, device, pilot):
    training = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    data, by_object = {}, defaultdict(list)
    for row in training:
        for path in [row['input'],row['real']]:
            if path not in data:
                data[path] = tensors(path,device)
                by_object[row['object']].append(path)
    objects = sorted(by_object)
    model = CausalMotion(action_dim=52).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    rng = np.random.default_rng(1300)
    spec = config['training']
    steps = spec['pilot_motion_steps'] if pilot else spec['motion_steps']
    history, best, best_step, best_state = [], float('inf'), 0, None
    for step in range(steps):
        keys = [str(rng.choice(by_object[str(obj)])) for obj in rng.choice(objects,spec['batch_size'])]
        batch,target,mask,_ = make_batch(data,keys,rng)
        if not mask.any():
            continue
        model.train()
        prediction = model(batch)
        loss = F.smooth_l1_loss(prediction[mask],target[mask],beta=.01)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step % 200 == 0 or step == steps-1:
            evaluation = motion_evaluation(model,validation,device,windows=8)
            item = dict(step=step,loss=float(loss.detach()),validation=evaluation)
            history.append(item)
            if evaluation['model_mm'] < best:
                best,best_step = evaluation['model_mm'],step
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            write_json(output/'motion_progress.json',dict(history=history,best_step=best_step,total_steps=steps))
            print('MOTION_TRAIN',item,flush=True)
    if best_state is None:
        raise ValueError('Motion training produced no valid checkpoint')
    model.load_state_dict(best_state)
    torch.save(dict(model=best_state,action_dim=52,use_actions=True,
                    training_objects=objects,best_validation_step=best_step),output/'encoder.pt')
    del data
    torch.cuda.empty_cache() if str(device).startswith('cuda') else None
    return model


def score_metrics(scores, target, valid, entries, test):
    metrics = []
    for j,key in enumerate(PARAMETERS):
        ids = test[valid[test,j] & np.isfinite(scores[test,j])]
        groups = defaultdict(list)
        for i in ids:
            row = entries[i]
            groups[(row['object'],row['truth'],row['workpoint'])].append(i)
        correlations, correct, pairs = [], 0., 0
        for group in groups.values():
            if len(group)>=3 and np.ptp(target[group,j])>1e-8 and np.ptp(scores[group,j])>1e-8:
                correlations.append(float(spearmanr(scores[group,j],target[group,j]).statistic))
            for a,b in itertools.combinations(group,2):
                gap = target[a,j]-target[b,j]
                if abs(gap)<.02:
                    continue
                pred = scores[a,j]-scores[b,j]
                correct += .5 if abs(pred)<1e-10 else float(pred*gap>0)
                pairs += 1
        metrics.append(dict(parameter=key,count=len(ids),
                    confidence_mae=float(np.abs(scores[ids,j]-target[ids,j]).mean()) if len(ids) else None,
                    within_object_spearman=float(np.mean(correlations)) if correlations else None,
                    comparable_pairs=pairs,pairwise_accuracy=correct/pairs if pairs else None))
    return metrics


def train(manifest, config_path, output, device, pilot=False):
    started = time.monotonic()
    config = read_json(config_path)
    rows = [read_json(p) for p in read_json(manifest)['teachers']]
    objects = {split:{r['object'] for r in rows if r['split']==split} for split in ['train','validation','test']}
    if len(objects['train'])<3 or not objects['validation'] or not objects['test']:
        raise ValueError(f'Insufficient object-disjoint labels: {objects}')
    if any(objects[a]&objects[b] for a,b in itertools.combinations(objects,2)):
        raise ValueError('Object leakage across splits')
    for r in rows:
        if r['config_sha256'] != sha256(config_path):
            raise ValueError('Teacher configuration does not match training')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = motion_train(rows,config,output,device,pilot)
    teacher_entries, X, S, masks = [], [], [], []
    for row in rows:
        d = tensors(row['input'],device)
        feature = extract(encoder,d)
        for record in row['records']:
            path = Path(row['input']).parent/record['path']
            if sha256(path) != record['sha256']:
                raise ValueError('Teacher payload changed')
            label = load_npz(path)
            X.append(np.r_[feature,label['q']/config['query_step']])
            S.append(label['S']);masks.append(label['valid'])
            teacher_entries.append(dict(id=row['id'],object=row['object'],split=row['split'],
                         truth=row['truth'],workpoint=record['workpoint'],teacher=str(path)))
    X,S,valid = np.array(X),np.array(S),np.array(masks)
    groups = np.array([r['id'] for r in teacher_entries])
    indices = {s:np.array([i for i,r in enumerate(teacher_entries) if r['split']==s],dtype=int)
               for s in objects}
    np.savez_compressed(output/'features.npz',X=X,S=S,valid=valid,episode_ids=groups)
    write_json(output/'rows.json',teacher_entries)
    heads,records = [],[]
    for seed in config['training']['seeds']:
        state,history = fit(X,S,valid,groups,indices['train'],indices['validation'],seed,
                            config['training']['head_epochs'],config['training']['patience'],device)
        heads.append(state);records.append(history)
        print('HEADS_TRAINED',seed,history['best_epoch'],history['validation_loss'],flush=True)
    tau = heads[0]['tau'].numpy()
    prediction = np.mean([predict(h,X,device) for h in heads],axis=0)
    scores,target = prediction/(prediction+tau),S/(S+tau)
    calibration = np.array([np.nanmean(target[indices['train'],j][valid[indices['train'],j]])
                            if valid[indices['train'],j].any() else np.nan for j in range(4)])
    constants = np.broadcast_to(calibration,S.shape)
    space = ParameterSpace.from_config(config)
    torch.save(dict(encoder={k:v.detach().cpu() for k,v in encoder.state_dict().items()},heads=heads,
                    parameters=list(PARAMETERS),base=torch.tensor([space.base[k] for k in PARAMETERS],dtype=torch.float64),
                    scales=torch.tensor(space.scales),tau=heads[0]['tau'],query_step=config['query_step'],
                    action_dim=52,training_objects=sorted(objects['train'])),output/'model.pt')
    np.savez_compressed(output/'predictions.npz',confidence=scores,target=target,valid=valid,
                        test_indices=indices['test'],validation_indices=indices['validation'])
    motion_test = motion_evaluation(encoder,[r for r in rows if r['split']=='test'],device)
    report = dict(status='complete',pilot=pilot,objects={s:sorted(v) for s,v in objects.items()},
                  teachers=len(rows),label_vectors=len(S),reliable_labels=valid.sum(0),tau=tau,
                  heads=records,test=score_metrics(scores,target,valid,teacher_entries,indices['test']),
                  constant=score_metrics(constants,target,valid,teacher_entries,indices['test']),
                  motion_test=motion_test,elapsed_s=time.monotonic()-started,
                  model_sha256=sha256(output/'model.pt'),manifest_sha256=sha256(manifest),
                  interpretation='Four independent relative sensitivity scores, not correctness probabilities; physical parameter recovery is a separate evaluation')
    write_json(output/'report.json',report)
    print('TRAIN_COMPLETE',report['test'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    train(a.manifest,a.config,a.output,a.device,a.pilot)


if __name__=='__main__':
    main()
