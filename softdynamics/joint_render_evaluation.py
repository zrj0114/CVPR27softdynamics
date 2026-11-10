"""Six reporting metrics for frozen joint-world selections, with explicit splits.

Uses every observed frame after initialization. A fitted episode is labelled
``fit_interaction`` and a query episode ``held_out_interaction``; neither is
relabeled as the upstream EMPM within-sequence temporal test split.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from . import render_evaluation as evaluation
from .io import read_json, sha256, write_json
from .recovery_data import trajectory_folder, qkey


def prepare(root, run, empm, obj, budgets):
    root,run,empm=map(lambda p:Path(p).resolve(),[root,run,empm])
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        return read_json(root/'plan.json')
    # The per-object repair keeps the exact solver/domain used by optimization.
    repair=run/'repairs/whitebear_grid32'
    effective=repair if obj=='187-white-bear' and (repair/'physics/plan.json').exists() else run
    physics=effective/'physics';original=read_json(physics/'plan.json')
    item=next(i for i in original['objects'] if i['object']==obj)
    rows=[];cases={};unavailable=[]
    choices=[dict(method='initial',budget=0,q=[0.]*4,selection=str(effective/'baselines'/f'{obj}.json'))]
    for method in read_json(run/'protocol.json')['methods']:
        for budget in budgets:
            if method!='physical_uniform_cma' and budget>12:continue
            path=effective/'inverse'/obj/method/f'selection_{budget}.json'
            if not path.exists():
                unavailable.append(dict(object=obj,method=method,budget=budget,reason='Selection not yet frozen'))
                continue
            selected=read_json(path)
            choices.append(dict(method=method,budget=budget,q=selected['q'],selection=str(path)))
    for choice in choices:
        selection=Path(choice['selection'])
        if not selection.exists():
            unavailable.append(dict(**choice,reason='Initial simulation not completed'));continue
        for split,episodes in [('fit_interaction',item['support']),('held_out_interaction',item['query'])]:
            for episode in episodes:
                folder=trajectory_folder(physics,obj,episode,choice['q'])
                if not (folder/'complete.json').exists():
                    unavailable.append(dict(**choice,object=obj,episode=episode,split=split,reason='Selected trajectory not yet cached'))
                    continue
                entry=next(e for e in item['episodes'] if e['episode']==episode)
                complete=read_json(folder/'complete.json')
                key=f'{obj}/episode_{episode}/{qkey(choice["q"])}'
                camera_path=Path(entry['workspace'])/'queries'/f'episode_{episode}'/'cameras.npz'
                cases[key]=dict(object=obj,episode=episode,q=choice['q'],entry=entry,
                    trajectory=str(folder),complete_sha256=sha256(folder/'complete.json'),
                    frames=list(range(1,entry['frames'])),cameras=[0],camera_path=str(camera_path),
                    camera_sha256=sha256(camera_path))
                rows.append(dict(object=obj,episode=episode,split=split,method=choice['method'],budget=choice['budget'],
                    case=key,selection=str(selection),selection_sha256=sha256(selection),query_valid=complete['valid']))
    protocol=dict(
        image_metric_profile='empm-script',
        selection='Frozen parameters only; no fitting, adaptation or selection using reporting metrics',
        split='fit_interaction and held_out_interaction are separate groups, NOT the upstream within-sequence train/test split',
        frames='All recorded frames after initialization, equally weighted; no phase weights or Huber transform',
        dist='Observed-to-predicted-surface single-direction L1 nearest-neighbor mean, mm',
        track='Visible corresponding 3D-point L2 mean, mm',
        image_size=[1280,720],cameras='First camera (index 0), as in the released EMPM evaluate_render.py; fixed before scoring',
        IoU='Nonzero rounded uint8 alpha, intersection/union multiplied by 100',
        PSNR='Full-image RGB [0,1], prediction rounded to uint8 PNG precision; 100 dB cap for exact equality',
        SSIM='Original EMPM 11x11 Gaussian-window implementation',
        LPIPS='Original EMPM AlexNet v0.1 caller inputs [0,1]; standard [-1,1] result also saved per camera',
        RGB='Original undistorted RGB times refined object mask on black; prediction rendered on black without GT masking',
        appearance='Existing frozen first-frame Gaussians and material binding; no appearance optimization',
        occlusion='Separate hand/robot masks used by EMPM are unavailable in this prepared data; this difference remains explicit',
        aggregation='Equal frames per episode, equal episodes within object, separate method/budget/split groups',
        limitations='Deform360 data, point samples, track binding and split differ from EMPM; these values cannot be compared directly to its paper table')
    upstream=empm/'scripts/eval'
    plan=dict(schema='softdynamics.joint-six-metrics.v1',created_at=time.time(),run=str(physics),
        run_plan_sha256=sha256(physics/'plan.json'),experiment_run=str(run),effective_run=str(effective),
        empm=str(empm),rows=rows,cases=cases,unavailable=unavailable,protocol=protocol,
        upstream_sha256={name:sha256(upstream/name) for name in ['evaluate_chamfer.py','evaluate_track.py','evaluate_render.py']},
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(evaluation.__file__),Path(evaluation.__file__).with_name('evaluation_metrics.py')]})
    write_json(root/'plan.json',plan)
    return read_json(root/'plan.json')


def summarize(root,plan):
    report=evaluation.report(root,plan)
    keys=['dist_mm','track_mm','iou_percent','psnr_db','ssim','lpips']
    groups={}
    for row in report['rows']:
        key=(row['object'],row['method'],row['budget'],row['split'])
        groups.setdefault(key,[]).append(row)
    aggregate=[]
    for (obj,method,budget,split),rows in groups.items():
        missing=[r for r in plan['unavailable'] if r.get('object')==obj and r.get('method')==method
                 and r.get('budget')==budget and r.get('split',split)==split]
        record=dict(object=obj,method=method,budget=budget,split=split,episodes=[r['episode'] for r in rows],
                    unavailable_episodes=len(missing),complete=not missing and all(r['complete'] for r in rows))
        for key in keys:
            record[key]=float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
        aggregate.append(record)
    report['object_split_means']=aggregate
    if plan['unavailable']:report['status']='partial'
    write_json(root/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--empm',type=Path,required=True);p.add_argument('--object',required=True)
    p.add_argument('--budgets',nargs='+',type=int,default=[12]);p.add_argument('--device',default='cuda:0')
    p.add_argument('--mode',choices=['prepare','geometry','all','report'],default='all')
    a=p.parse_args();root=a.root.resolve();plan=prepare(root,a.run,a.empm,a.object,a.budgets)
    if a.mode=='prepare':return
    try:
        if a.mode!='report':
            for key in sorted(plan['cases']):
                geometry=evaluation.geometry(root,plan,key)
                if geometry['valid'] and a.mode=='all':
                    evaluation.replay(root,plan,key,a.device)
                    evaluation.render(root,plan,key,a.device)
                summarize(root,plan)
        report=summarize(root,plan)
        write_json(root/'status.json',dict(status='complete',mode=a.mode,report_status=report['status'],updated_at=time.time()))
    except Exception as error:
        write_json(root/'status.json',dict(status='failed',mode=a.mode,error=str(error),updated_at=time.time()))
        raise


if __name__=='__main__':main()
