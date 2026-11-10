"""GPU integration audit and held-out evaluation of frozen E/nu selections."""
import argparse
import csv
import fcntl
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import render_evaluation as evaluation
from . import joint_acceleration as ja
from .elastic_downstream import (config_for, folder_for, install_adapter,
    simulation_coordinates, tensor_parameters, wide_key)
from .io import read_json, write_json, sha256, load_npz
from .joint_world import load_joint, context_batch
from .physics_protocol import cache_compatible
from .elastic_gradient_audit import material_precision_audit,relative_error

METRICS=['dist_mm','track_mm','iou_percent','psnr_db','ssim','lpips']


def gradient_audit(run,obj,device):
    install_adapter();torch.set_num_threads(2)
    run=Path(run);spec,physics=config_for(run)
    item=next(o for o in physics['objects'] if o['object']==obj)
    episode=item['support'][0];records={}
    dest=run/'audits'/f'{obj}.json'
    if dest.exists():
        previous=read_json(dest)
        if previous['status']!='passed' or previous['episode']!=episode:
            raise ValueError('Existing audit belongs to another support input')
        audited_plan=previous.get('physics_plan_sha256',read_json(run/'baselines'/f'{obj}.json')['physics_plan_sha256'])
        if not cache_compatible(run,audited_plan,obj,episode):
            raise ValueError('Physical audit protocol changed')
        if 'state_sha256' in previous and sha256(folder_for(run,obj,episode,[0.,0.])/'state.npz')!=previous['state_sha256']:
            raise ValueError('Audited support state changed')
        for name,source in spec['checkpoints'].items():
            if previous['records'][name]['checkpoint_sha256']!=source['sha256'] or sha256(source['path'])!=source['sha256']:
                raise ValueError('Audited model changed')
        return previous
    for name,source in spec['checkpoints'].items():
        if sha256(source['path'])!=source['sha256']:raise ValueError('Checkpoint changed')
        model,_=load_joint(source['path'],device);model.eval();model.requires_grad_(False)
        window=ja.EndpointWindows(run,obj,episode,device)
        q=torch.zeros(2,device=device,requires_grad=True)
        window.set_anchor(model.world,[0.]*4,tensor_parameters(q,spec['elastic_config']))
        loss=window.loss(model.world,tensor_parameters(q,spec['elastic_config']))
        gradient=torch.autograd.grad(loss,q)[0]
        finite=[];epsilon=.001
        with torch.no_grad():
            for j in range(2):
                delta=torch.zeros_like(q);delta[j]=epsilon
                plus=window.loss(model.world,tensor_parameters(q+delta,spec['elastic_config']))
                minus=window.loss(model.world,tensor_parameters(q-delta,spec['elastic_config']))
                finite.append(float((plus-minus)/(2*epsilon)))
            confidence=model.confidence(context_batch([window.confidence_context],device),
                tensor_parameters(q,spec['elastic_config'])[None])
        exact=gradient.cpu().numpy();finite=np.asarray(finite)
        if not np.isfinite(exact).all() or np.linalg.norm(exact)<1e-8:
            raise ValueError('The frozen model does not provide usable material gradients')
        error=np.linalg.norm(exact-finite)/max(np.linalg.norm(exact),np.linalg.norm(finite),1e-8)
        if confidence.shape!=(1,2) or not torch.isfinite(confidence).all():raise ValueError('Invalid confidence output')
        # A finite float32 step can cross rapidly changing local geometry.
        # Check convergence at smaller steps in float64, and independently
        # verify the actual float32 optimizer gradient against that derivative.
        precise=material_precision_audit(window,model.world,spec['elastic_config'],exact)
        fine=precise['finite_differences'][-1]
        records[name]=dict(checkpoint_sha256=source['sha256'],step=source['step'],
            support_endpoint_objective_mm=float(loss.detach()),gradient=exact,finite_difference=fine['gradient'],
            finite_difference_epsilon=fine['epsilon'],relative_error=relative_error(exact,fine['gradient']),
            float32_difference=dict(epsilon=epsilon,gradient=finite,relative_error=error),
            precision_check=precise,confidence=confidence.cpu().tolist(),
            network_weights_frozen=True,query_observations_used=False)
        if not precise['passed']:
            write_json(run/'audits'/f'{obj}.failed.json',dict(status='failed',object=obj,episode=episode,records=records))
            raise ValueError(f'Material derivative convergence check failed: {name}, {precise["failure_reasons"]}')
        del model,window,loss,q,gradient,confidence;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    write_json(dest,dict(status='passed',object=obj,episode=episode,records=records,
        physics_plan_sha256=spec['physics_plan_sha256'],
        state_sha256=sha256(folder_for(run,obj,episode,[0.,0.])/'state.npz')))


def frozen_choices(run,obj):
    run=Path(run);spec,_=config_for(run)
    choices=[dict(method='initial',budget=0,q=[0.,0.],selection=str(run/'baselines'/f'{obj}.json'))]
    for method in spec['methods']:
        if not (run/'inverse'/obj/method/'complete.json').exists():
            raise ValueError('Every method must finish support selection before query evaluation')
        budgets=spec['inverse']['cma_report_budgets'] if method=='physical_cma' else spec['inverse']['report_budgets']
        for budget in budgets:
            path=run/'inverse'/obj/method/f'selection_{budget}.json';selection=read_json(path)
            if selection['query_observations_used']:raise ValueError('Query leaked into selection')
            choices.append(dict(method=method,budget=budget,q=selection['q'],selection=str(path),selection_sha256=sha256(path)))
    for choice in choices:
        choice['selection_sha256']=sha256(choice['selection'])
    return choices


def prepare_queries(run,obj):
    install_adapter();run=Path(run);spec,physics=config_for(run)
    dest=run/'query_plans'/f'{obj}.json'
    if dest.exists():return read_json(dest)
    item=next(o for o in physics['objects'] if o['object']==obj)
    choices=frozen_choices(run,obj);queries={}
    for choice in choices:
        legacy=simulation_coordinates(choice['q'],spec['elastic_config'],physics['config'])
        for episode in item['query']:
            key=f'{obj}/episode_{episode}/{wide_key(legacy)}'
            queries[key]=dict(object=obj,episode=episode,q=choice['q'])
    result=dict(object=obj,frozen_at=time.time(),choices=choices,queries=queries,
        query_observations_used_for_selection=False)
    write_json(dest,result);return read_json(dest)


def prepare_metrics(run,obj,empm):
    install_adapter();run=Path(run);empm=Path(empm);spec,physics=config_for(run)
    root=run/'evaluation'/obj;dest=root/'plan.json'
    if dest.exists():return read_json(dest)
    query_plan=read_json(run/'query_plans'/f'{obj}.json')
    item=next(o for o in physics['objects'] if o['object']==obj)
    rows=[];cases={};query_results=[]
    for choice in query_plan['choices']:
        if sha256(choice['selection'])!=choice['selection_sha256']:raise ValueError('Frozen support selection changed')
        legacy=simulation_coordinates(choice['q'],spec['elastic_config'],physics['config'])
        query_losses=[]
        for episode in item['query']:
            folder=folder_for(run,obj,episode,choice['q']);result=read_json(folder/'complete.json')
            score=read_json(folder/'scores.json')
            query_losses.append(score['conditions']['real']['objective_mm'] if result['valid'] else np.inf)
        query_results.append(dict(**choice,query_losses_mm=query_losses,query_mean_mm=float(np.mean(query_losses))))
        # All budget checkpoints get held-out geometry objectives; the six-
        # metric table uses the matched budget 12 plus CMA's budget 48 reference.
        if choice['budget'] not in (0,12,48):continue
        for split,episodes in [('fit_interaction',item['support']),('held_out_interaction',item['query'])]:
            for episode in episodes:
                entry=next(e for e in item['episodes'] if e['episode']==episode)
                folder=folder_for(run,obj,episode,choice['q']);complete=read_json(folder/'complete.json')
                key=f'{obj}/episode_{episode}/{wide_key(legacy)}'
                camera_path=Path(entry['workspace'])/'queries'/f'episode_{episode}'/'cameras.npz'
                cases[key]=dict(object=obj,episode=episode,q=legacy,elastic_q=choice['q'],entry=entry,
                    trajectory=str(folder),complete_sha256=sha256(folder/'complete.json'),
                    frames=list(range(1,entry['frames'])),cameras=[0],camera_path=str(camera_path),
                    camera_sha256=sha256(camera_path),valid=complete['valid'])
                rows.append(dict(object=obj,episode=episode,split=split,method=choice['method'],budget=choice['budget'],
                    case=key,selection=choice['selection'],selection_sha256=choice['selection_sha256'],query_valid=complete['valid']))
    protocol=dict(image_metric_profile='empm-script',
        selection='All methods and budget selections frozen before any query simulation; no query-driven acceptance, updates, or fallback',
        split='fit_interaction and held_out_interaction are separate interaction episodes, not the EMPM temporal split',
        frames='All recorded observation frames after initialization, equally weighted; no phase weighting or Huber in reporting metrics',
        dist='Observed-to-predicted surface one-direction L1 nearest-neighbor mean in mm',
        track='Visible corresponding 3D-point L2 mean in mm',
        image_size=[1280,720],cameras='Fixed camera index 0, same as the upstream EMPM evaluation caller',
        IoU='Nonzero rounded uint8 alpha, intersection/union multiplied by 100',
        PSNR='Full-image RGB [0,1], rendered RGB rounded to uint8 precision, 100 dB cap',
        SSIM='Original EMPM 11x11 Gaussian window implementation',
        LPIPS='Original EMPM AlexNet v0.1 caller with inputs [0,1]; standard [-1,1] also saved per camera',
        RGB='Original undistorted video masked by the refined object mask on black; prediction rendered on black without GT masking',
        appearance='Frozen existing first-frame Gaussians and material binding; no appearance optimization',
        occlusion='Separate hand/robot occlusion masks are unavailable',
        aggregation='Equal frames per episode, equal episodes within each object, equal objects for macro means; splits separate',
        limitations='Historical development objects, dataset/point sampling/binding/split differ from EMPM paper tables')
    source_data=read_json(Path(spec['fit_run'])/'plan.json')['source_data']
    result=dict(schema='softdynamics.elastic-six-metrics.v1',run=str(run/'physics'),source_data=source_data,
        run_plan_sha256=sha256(run/'physics/plan.json'),empm=str(empm),rows=rows,cases=cases,
        unavailable=[],protocol=protocol,query_plan_sha256=sha256(run/'query_plans'/f'{obj}.json'),
        upstream_sha256={name:sha256(empm/'scripts/eval'/name) for name in ['evaluate_chamfer.py','evaluate_track.py','evaluate_render.py']})
    write_json(root/'query_results.json',query_results)
    write_json(dest,result);return read_json(dest)


def evaluate_case(run,obj,key,device):
    run=Path(run);root=run/'evaluation'/obj;plan=read_json(root/'plan.json');case=plan['cases'][key]
    if not cache_compatible(run,plan['run_plan_sha256'],obj,case['episode']):raise ValueError('Relevant episode physics changed')
    out=root/'cases'/key
    if (out/'complete.json').exists():return
    if not case['valid']:
        evaluation.geometry(root,plan,key)
        write_json(out/'complete.json',dict(status='physical_failure',valid=False,reason='Held-out solver failed; no parameter fallback using query data'))
        return
    # Per-case workers write only their own progress and metric payloads.
    # The CPU controller alone aggregates the shared report.
    evaluation.report=lambda *_:None
    evaluation.progress=lambda _root,stage,**fields:write_json(out/'progress.json',dict(stage=stage,pid=os.getpid(),updated_at=time.time(),**fields))
    camera_source=evaluation.camera_source
    def locked_camera_source(root,plan,case,index):
        path=root/'reference'/case['object']/f'episode_{case["episode"]}'/f'camera_{index}.lock'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return camera_source(root,plan,case,index)
    evaluation.camera_source=locked_camera_source
    started=time.monotonic();geometry=evaluation.geometry(root,plan,key)
    replay=evaluation.replay(root,plan,key,device)
    evaluation.render(root,plan,key,device)
    for index in case['cameras']:
        names=load_npz(case['camera_path'])['camera_names']
        record=read_json(out/'cameras'/f'{names[index]}.json')
        if record['frames']!=len(case['frames']):raise ValueError('Missing render frames')
        if not all(np.isfinite(record['mean'][m]) for m in METRICS[2:]):raise ValueError('Nonfinite image metric')
    write_json(out/'complete.json',dict(status='complete',valid=True,elapsed_s=time.monotonic()-started,
        replay_coordinate_error_m=replay['max_replay_coordinate_error_m'],all_frames=True))


def summarize(run):
    run=Path(run);spec,physics=config_for(run);all_rows=[];groups=[];query=[];fit=[];complete=True
    for obj in spec['test_objects']:
        root=run/'evaluation'/obj
        for method in spec['methods']:
            for path in sorted((run/'inverse'/obj/method).glob('selection_*.json')):
                fit.append(read_json(path))
        if not (root/'plan.json').exists():complete=False;continue
        plan=read_json(root/'plan.json');report=evaluation.report(root,plan)
        all_rows.extend(report['rows']);query.extend(read_json(root/'query_results.json'))
        for case in plan['cases']:
            if not (root/'cases'/case/'complete.json').exists():complete=False
    for key in sorted(set((r['object'],r['split'],r['method'],r['budget']) for r in all_rows)):
        group=[r for r in all_rows if (r['object'],r['split'],r['method'],r['budget'])==key]
        valid=all(r['complete'] for r in group)
        groups.append(dict(object=key[0],split=key[1],method=key[2],budget=key[3],complete=valid,
            episodes=len(group),physical_failures=sum(not r['query_valid'] for r in group),
            **{m:float(np.mean([r[m] for r in group])) if valid else None for m in METRICS}))
    macro=[]
    for key in sorted(set((g['split'],g['method'],g['budget']) for g in groups)):
        items=[g for g in groups if (g['split'],g['method'],g['budget'])==key]
        valid=len(items)==len(spec['test_objects']) and all(g['complete'] for g in items)
        macro.append(dict(split=key[0],method=key[1],budget=key[2],complete=valid,
            **{m:float(np.mean([g[m] for g in items])) if valid else None for m in METRICS}))
    from .elastic_comparison import summarize as summarize_comparison
    comparison=summarize_comparison(run)
    result=dict(status='complete' if complete and groups else 'partial',updated_at=time.time(),
        checkpoints=spec['checkpoints'],groups=groups,object_equal_macro=macro,rows=all_rows,
        held_out_objectives=query,support_identification=fit,comparison=comparison,
        actual_unique_episode_rollouts=sum(comparison['actual_unique_rollouts_by_stage'].values()),
        physical_failures=comparison['physical_failures'],
        retired_numerical_failures=comparison['retired_numerical_failures'],
        limitations='Single optimizer seed; historical development objects; no direct numerical comparison to EMPM paper table; no MPC/RL implemented')
    write_json(run/'report.json',result)
    if groups:
        with (run/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['audit','queries','prepare','case','report'],required=True)
    p.add_argument('--object');p.add_argument('--case');p.add_argument('--device',default='cuda:0')
    p.add_argument('--empm',type=Path,default=Path('/home/rongjin/empm-study/empm'))
    a=p.parse_args()
    if a.mode=='audit':gradient_audit(a.run,a.object,a.device)
    elif a.mode=='queries':prepare_queries(a.run,a.object)
    elif a.mode=='prepare':prepare_metrics(a.run,a.object,a.empm)
    elif a.mode=='case':evaluate_case(a.run,a.object,a.case,a.device)
    else:summarize(a.run)


if __name__=='__main__':main()
