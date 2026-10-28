"""Read-only paired reporting and audit for the frozen recovery experiment."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .io import load_npz, read_json, sha256, write_json
from .parameters import PARAMETERS
from .recovery_benchmark import bank_for, shuffled_scores
from .recovery_data import condition_key, qkey, trajectory_folder
from .search import cycle


def average(values):
    return float(np.mean(values)) if len(values) else None


def paired_summary(rows, method):
    """Only identical completed object/condition/start pairs enter a comparison."""
    lookup={(r['object'],r['context'],r['method']):r for r in rows}
    pairs=[(u,lookup[(o,c,method)]) for (o,c,m),u in lookup.items()
           if m=='uniform' and (o,c,method) in lookup]
    valid=[(u,r) for u,r in pairs if u['query_valid'] and r['query_valid']]
    delta=np.array([r['parameter_error_steps']-u['parameter_error_steps'] for u,r in pairs])
    return dict(pairs=len(pairs),same_parameters=sum(u['selected_q']==r['selected_q'] for u,r in pairs),
                parameter_wins=int(np.sum(delta < -1e-9)),parameter_ties=int(np.sum(abs(delta)<=1e-9)),
                parameter_losses=int(np.sum(delta > 1e-9)),parameter_error_delta=average(delta),
                paired_query_valid=len(valid),
                paired_query_delta_mm=average([r['query_objective_mm']-u['query_objective_mm'] for u,r in valid]),
                uniform_query_failures=sum(not u['query_valid'] for u,r in pairs),
                method_query_failures=sum(not r['query_valid'] for u,r in pairs),
                method_only_query_failures=sum(u['query_valid'] and not r['query_valid'] for u,r in pairs))


def synthetic_groups(rows, methods):
    lookup={(r['object'],r['context'],r['method']):r for r in rows}
    groups=[]
    for obj in sorted({r['object'] for r in rows}):
        for noise in [0.,.5,2.]:
            contexts=sorted({r['context'] for r in rows if r['object']==obj and r['condition']['noise_mm']==noise
                             and all((obj,r['context'],m) in lookup for m in methods)})
            if not contexts:continue
            common_valid=[c for c in contexts if all(lookup[(obj,c,m)]['query_valid'] for m in methods)]
            metrics={}
            for method in methods:
                selected=[lookup[(obj,c,method)] for c in contexts]
                metrics[method]=dict(trials=len(selected),
                    parameter_error_steps=average([r['parameter_error_steps'] for r in selected]),
                    per_parameter_error_steps=np.mean([r['per_parameter_error_steps'] for r in selected],0),
                    exact_fraction=average([r['exact'] for r in selected]),
                    query_valid=sum(r['query_valid'] for r in selected),
                    # Same set for every method: a failed rollout is not silently removed from just one bar.
                    query_objective_mm_common_valid=average([lookup[(obj,c,method)]['query_objective_mm'] for c in common_valid]))
            matched=[lookup[(obj,c,m)] for c in contexts for m in methods]
            groups.append(dict(object=obj,noise_mm=noise,contexts=len(contexts),common_query_valid=len(common_valid),
                               methods=metrics,paired={m:paired_summary(matched,m) for m in methods if m!='uniform'}))
    return groups


def baseline_query(root, plan, item, condition):
    values=[]
    for episode in item['query']:
        path=trajectory_folder(root,item['object'],episode,plan['real_initial_guess'])/'scores.json'
        if not path.exists():return None
        value=read_json(path)['conditions'].get(condition)
        if not value or not value['valid']:return None
        values.append(value)
    return {k:average([v[k] for v in values]) for k in ['objective_mm','surface_mm','track_mm']}


def audit(root, plan, rows, verify_payload=False):
    """Replay each finished search using original search.cycle, without new physics."""
    errors=[];cache={};checked=0;trajectories=set()
    items={o['object']:o for o in plan['objects']};contexts={}
    def check(condition, message):
        if not condition:errors.append(message)
    check(sha256(plan['checkpoint'])==plan['model_sha256'],'Checkpoint hash changed')
    manifest=read_json(root/'code_manifest.json')
    for name,digest in manifest['files'].items():
        check(sha256(root/'code'/name)==digest,'Frozen code changed: '+name)
    for obj,item in items.items():
        check(not (set(item['support']) & set(item['query'])),obj+': support/query overlap')
        check(len({e['asset_sha256'] for e in item['episodes']})==1,obj+': episodes do not share canonical asset')
        for entry in item['episodes']:
            for truth in plan['config']['truths']:
                folder=trajectory_folder(root,obj,entry['episode'],truth['q']);path=folder/'scores.json'
                if not path.exists():continue
                value=read_json(path)['conditions'][condition_key(truth['name'],0.,19)]
                check(value['valid'] and value['objective_mm']==0.,str(folder)+': nonzero noiseless truth self-loss')
                trajectories.add(folder)
        path=root/'weights'/f'{obj}.json'
        if not path.exists():continue
        weights=read_json(path)
        check(weights['model_sha256']==plan['model_sha256'],obj+': wrong weights checkpoint')
        check(weights['teacher_scores_used'] is False,obj+': teacher used for weights')
        check(weights['support']==item['support'] and weights['query']==item['query'],obj+': wrong weights episodes')
        contexts[obj]={c['id']:c for c in weights['contexts']}
        for context in weights['contexts']:
            w=np.asarray(context['weights']['learned'])
            check(w.shape==(len(item['support']),4) and np.isfinite(w).all() and (w>=0).all() and (w<=1).all(),
                  obj+'/'+context['id']+': invalid confidence')
            check(np.array_equal(np.array(context['weights']['shuffled']),shuffled_scores(w)),
                  obj+'/'+context['id']+': wrong shuffled control')
    for row in rows:
        obj=row['object'];item=items[obj];context=contexts[obj][row['context']]
        label=obj+'/'+row['context']+'/'+row['method']
        key=(obj,row['condition']['key'])
        if key not in cache:cache[key]=bank_for(root,item,row['condition'])
        bank=cache[key]
        def losses(q):return bank[qkey(q)]
        expected,trace=cycle(losses,np.array(context['weights'][row['method']]),context['start'],
                             plan['grid_levels'],plan['passes'])
        check(np.allclose(expected,row['candidate_q'],atol=1e-10,rtol=0),label+': candidate differs from original search')
        check(row['logical_episode_evaluations']==36*len(item['support']),label+': unequal logical budget')
        check(len(row['trace'])==len(trace)==12,label+': wrong trace length')
        for actual,original in zip(row['trace'],trace):
            check(actual['parameter']==original['parameter'] and actual['new_level']==original['new_level'],
                  label+': different coordinate update')
            costs=np.array([np.inf if c is None else c for c in actual['costs']])
            check(np.allclose(costs,original['costs'],atol=1e-10,rtol=1e-10),label+': wrong full-trajectory objective')
            for q in actual['tested_q']:
                trajectories.update(trajectory_folder(root,obj,e,q) for e in item['support'])
        final=losses(expected);initial=losses(context['start'])
        accepted=True if row['condition']['truth']!='real' else bool(np.isfinite(final).all() and np.isfinite(initial).all()
                                                                       and final.mean()<initial.mean()-1e-12)
        selected=expected if accepted else np.array(context['start'])
        check(accepted==row['accepted'] and np.allclose(selected,row['selected_q'],atol=1e-10,rtol=0),label+': wrong acceptance')
        check(row['query_observations_used'] is False,label+': query used for selection')
        selection_path=root/'selections'/obj/row['context']/f'{row["method"]}.json'
        frozen=read_json(selection_path)
        check('query' not in frozen and frozen['selected_q']==row['selected_q'] and
              frozen['selection_frozen_at']==row['selection_frozen_at'],label+': unfrozen query selection')
        target=condition_key(row['condition']['truth'],0.,19)
        for query in row['query']:
            path=trajectory_folder(root,obj,query['episode'],selected);trajectories.add(path)
            check(query==dict(episode=query['episode'],**read_json(path/'scores.json')['conditions'][target]),label+': wrong query target')
        if row['condition']['truth']!='real':
            truth=next(t for t in plan['config']['truths'] if t['name']==row['condition']['truth'])
            delta=abs(selected-np.array(truth['q']))/.025
            check(np.allclose(delta,row['per_parameter_error_steps']) and np.isclose(delta.mean(),row['parameter_error_steps']),
                  label+': incorrect parameter error')
            check(np.allclose(delta*.025*np.array(plan['config']['coordinate_scales']),row['parameter_absolute_errors']),
                  label+': incorrect physical units')
        checked+=1
    invalid=[];payload_files=0;entry_cache={}
    for folder in sorted(trajectories):
        result=read_json(folder/'complete.json');entry_key=(result['object'],result['episode'])
        if entry_key not in entry_cache:
            entry=next(e for e in items[result['object']]['episodes'] if e['episode']==result['episode'])
            actions=load_npz(Path(entry['workspace'])/'queries'/f'episode_{entry["episode"]}'/'actions.npz')
            entry_cache[entry_key]=(entry,len(actions['poses'])-1)
        entry,last=entry_cache[entry_key]
        label=str(folder.relative_to(root));validation=read_json(folder/'validation.json')
        for name,k in [('asset','asset_sha256'),('initial','initial_sha256'),('actions','actions_sha256')]:
            check(result['source_hashes'][name]==entry[k],label+': wrong '+name)
        check(result['domain']['margin_m']==entry['domain_margin'],label+': changed simulation domain')
        check(len(validation)>0 and [v['frame'] for v in validation]==list(range(1,len(validation)+1)),label+': incomplete timeline')
        if not result['valid']:
            invalid.append(dict(object=result['object'],episode=result['episode'],q=result['q'],
                                frame=result['invalid_frame'],check=result['check']))
            check(not validation[-1]['valid'] and validation[-1]['frame']==result['invalid_frame'],label+': invalid result not recorded')
            continue
        check(result['smoke'] is False and len(validation)==last and all(v['valid'] for v in validation),label+': incomplete full rollout')
        check(result['dtmax']==entry['dtmax'],label+': changed time step')
        if verify_payload:
            for name,digest in result['payload_sha256'].items():
                check(sha256(folder/(name+'.npy'))==digest,label+': changed '+name)
                payload_files+=1
    return dict(passed=not errors,errors=errors,replayed_trials=checked,checked_trajectories=len(trajectories),
                invalid_trajectories=invalid,payload_files_hashed=payload_files,
                physics_rerun=False,checkpoint_sha256=plan['model_sha256'])


def collect(root, run_audit=True, verify_payload=False):
    root=Path(root);plan=read_json(root/'plan.json');status=read_json(root/'status.json')
    rows=[read_json(p) for p in sorted((root/'results').glob('*/*/*.json'))]
    synthetic=[r for r in rows if r['condition']['truth']!='real'];real=[]
    for item in plan['objects']:
        values=[r for r in rows if r['object']==item['object'] and r['condition']['truth']=='real']
        if not values:continue
        real.append(dict(object=item['object'],baseline_query=baseline_query(root,plan,item,'real'),
                         methods={r['method']:{k:r[k] for k in ['accepted','selected_q','support_initial_mm','support_candidate_mm',
                                'query_valid','query_objective_mm','query_surface_mm','query_track_mm']} for r in values}))
    groups=synthetic_groups(synthetic,plan['methods']);aggregate=[]
    for noise in [0.,.5,2.]:
        complete=[g for g in groups if g['noise_mm']==noise and status['objects'][g['object']]['stage']=='complete']
        if not complete:continue
        aggregate.append(dict(noise_mm=noise,objects=[g['object'] for g in complete],
           methods={m:{k:average([g['methods'][m][k] for g in complete if g['methods'][m][k] is not None])
                       for k in ['parameter_error_steps','exact_fraction','query_objective_mm_common_valid']} for m in plan['methods']}))
    return dict(updated_at=time.time(),status=status['status'],object_status=status['objects'],
                completed_trials=len(rows),planned_trials=len(plan['objects'])*87,model_sha256=plan['model_sha256'],
                parameter_names=PARAMETERS,physical_units_per_grid_step=np.array(plan['config']['coordinate_scales'])*.025,
                groups=groups,complete_object_aggregate=aggregate,real=real,
                audit=audit(root,plan,rows,verify_payload) if run_audit else None,
                interpretation='Paired contexts; object-equal aggregate uses completed objects only. Query means use the same valid contexts for all three methods. Report failures separately. Noise seeds and starts are repeated trials, not independent objects.')


def update_model_index(root, report):
    """Update experiment metadata only; checkpoint and model-selection files are untouched."""
    plan=read_json(Path(root)/'plan.json');path=Path(plan['pretrain'])/'latest_model.json'
    if not path.exists():return
    index=read_json(path)
    if index.get('model_sha256')!=report['model_sha256']:return
    evaluated=[obj for obj,value in report['object_status'].items() if value['stage']=='complete']
    complete=(report['status']=='complete' and report['completed_trials']==report['planned_trials']
              and len(evaluated)==len(report['object_status']) and report['audit']['passed'])
    index.update(recovery_experiment=str(Path(root).resolve()),recovery_status=report['status'],
                 recovery_analysis=str(Path(root).resolve()/'analysis/recovery_report.md'),
                 recovery_evaluated_objects=evaluated,recovery_completed_trials=report['completed_trials'],
                 physical_recovery_evaluated=complete,
                 physical_recovery_evaluation_status='complete' if complete else 'partial',
                 recovery_checked_at=report['updated_at'])
    write_json(path,index)


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(report):
    stamp=datetime.fromtimestamp(report['updated_at'],timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    text=['# 冻结全量模型的参数优化对照', '',f'更新：{stamp}；状态：{report["status"]}；完成 {report["completed_trials"]}/{report["planned_trials"]} 个方法试验。', '',
          '模型、初始几何、真实控制动作、完整前向模拟、搜索预算均固定，只改变每个参数对应的交互权重。对照为当前等权坐标搜索流程。', '',
          '参数误差为四个归一化参数误差除以 0.025 后的平均值，即平均相差几个网格步。一个网格步依次是 '+
          '、'.join(number(v,5) for v in report['physical_units_per_grid_step'])+'（Pa、无量纲、无量纲、s⁻¹）。', '',
          '## 同条件合成恢复', '',
          '仅比较三种方法都完成的相同物体／真值／噪声／种子／初值。查询误差为无噪声留出交互的综合表面与跟踪目标；各方法均使用三者共同有效的上下文，失败另列。', '',
          '| 物体 | 噪声 mm | 上下文 | 参数误差：等权 / 模型 / 打乱 | 查询误差 mm：等权 / 模型 / 打乱 | 查询有效：等权 / 模型 / 打乱 |',
          '|---|---:|---:|---|---|---|']
    for group in report['groups']:
        metrics=group['methods'];order=['uniform','learned','shuffled']
        text.append(f'| {group["object"]} | {group["noise_mm"]:g} | {group["contexts"]} | '+
             ' / '.join(number(metrics[m]['parameter_error_steps'],4) for m in order)+' | '+
             ' / '.join(number(metrics[m]['query_objective_mm_common_valid']) for m in order)+' | '+
             ' / '.join(f'{metrics[m]["query_valid"]}/{group["contexts"]}' for m in order)+' |')
    text+=['','## 已完成物体等权汇总','',
           '先在每个物体内平均，再对已完成物体等权平均。未完成物体不进入该表，查询失败仍以逐物体表为准。', '',
           '| 噪声 mm | 完成物体数 | 方法 | 参数误差（网格步） | 完全恢复比例 | 查询综合误差 mm（共同有效样本） |',
           '|---|---:|---|---:|---:|---:|']
    for group in report['complete_object_aggregate']:
        for method,row in group['methods'].items():
            text.append(f'| {group["noise_mm"]:g} | {len(group["objects"])} | {method} | '+
                        f'{number(row["parameter_error_steps"],4)} | {number(row["exact_fraction"],4)} | '+
                        f'{number(row["query_objective_mm_common_valid"])} |')
    text+=['','## 模型与等权的配对差异','','| 物体 | 噪声 mm | 参数相同 | 参数误差胜 / 平 / 负 | 双方查询有效 | 查询差值 mm（模型 − 等权） | 模型独有查询失败 |',
           '|---|---:|---:|---|---:|---:|---:|']
    for g in report['groups']:
        p=g['paired']['learned']
        text.append(f'| {g["object"]} | {g["noise_mm"]:g} | {p["same_parameters"]}/{p["pairs"]} | '+
                    f'{p["parameter_wins"]} / {p["parameter_ties"]} / {p["parameter_losses"]} | {p["paired_query_valid"]}/{p["pairs"]} | '+
                    f'{number(p["paired_query_delta_mm"],6)} | {p["method_only_query_failures"]} |')
    text+=['','负差值表示模型更好；模拟失败不作为低误差处理。重复初值、噪声种子不等同于更多独立物体。','',
           '## 真实数据与留出交互', '',
           '真实物理参数没有标注，因此这里评价运动拟合与泛化。候选只按无权重拟合集改善决定接受，查询数据不参与决策。', '',
           '| 物体 | 方法 | 接受 | 拟合前 → 后 mm | 查询初始 → 最终 mm | 查询表面 mm | 查询跟踪 mm |',
           '|---|---|---|---|---|---:|---:|']
    for item in report['real']:
        baseline=item['baseline_query'];before=baseline['objective_mm'] if baseline else None
        for method,row in item['methods'].items():
            text.append(f'| {item["object"]} | {method} | {row["accepted"]} | '+
                        f'{number(row["support_initial_mm"])} → {number(row["support_candidate_mm"])} | '+
                        f'{number(before)} → {number(row["query_objective_mm"])} | '+
                        f'{number(row["query_surface_mm"])} | {number(row["query_track_mm"])} |')
    text+=['','综合目标、表面距离、跟踪 RMS 是不同指标，不能直接与旧实验的单独表面指标比较。','']
    if report['audit'] is not None:
        a=report['audit'];text+=['## 执行核查','',f'核查通过：{a["passed"]}。用原始 `search.cycle` 重放 {a["replayed_trials"]} 次已完成搜索，'+
              f'检查 {a["checked_trajectories"]} 条候选时间轴、冻结参数、来源哈希与固定模拟设置；哈希核对 {a["payload_files_hashed"]} 个轨迹文件。', '']
        for invalid in a['invalid_trajectories']:
            text.append(f'- 数值失败：{invalid["object"]} episode {invalid["episode"]}，q={invalid["q"]}，frame {invalid["frame"]}，'+
                        f'原因 {invalid["check"].get("failure_kind", "物理状态检查")}，min J={invalid["check"].get("min_J", "—")}。')
        if a['errors']:text+=['']+['- '+error for error in a['errors']]
    text+=['',f'checkpoint SHA-256：`{report["model_sha256"]}`。', '',
           '这是局部三档、三轮坐标搜索验证，没有物理反向传播或连续收敛保证。尚未完成的物体不进入完整物体汇总。', '']
    return '\n'.join(text)


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=report['groups']
    if not groups:return
    colors=['#777777','#1976d2','#d48731'];methods=['uniform','learned','shuffled']
    fig,axes=plt.subplots(1,2,figsize=(max(11,len(groups)*1.5),4.6))
    x=np.arange(len(groups));width=.24
    for j,m in enumerate(methods):
        for ax,key in zip(axes,['parameter_error_steps','query_objective_mm_common_valid']):
            values=[g['methods'][m][key] for g in groups]
            ax.bar(x+(j-1)*width,[np.nan if v is None else v for v in values],width,color=colors[j],label=m)
    labels=[g['object'].split('-')[0]+f'\nnoise {g["noise_mm"]:g} mm\nn={g["contexts"]}; valid={g["common_query_valid"]}' for g in groups]
    for ax,title in zip(axes,['Parameter error (mean grid steps)','Held-out objective (mm; common valid contexts)']):
        ax.set_title(title);ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.suptitle(f'Frozen model recovery: {report["completed_trials"]}/{report["planned_trials"]} trials, {report["status"]}')
    fig.tight_layout();fig.savefig(output/'recovery_comparison.png',dpi=180);fig.savefig(output/'recovery_comparison.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--verify-payload',action='store_true')
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=30)
    a=p.parse_args();output=a.output or a.root/'analysis';output.mkdir(parents=True,exist_ok=True)
    previous=None
    while True:
        status=read_json(a.root/'status.json')
        key=(status['completed_trials'],status['status'],tuple(sorted((o,v['stage']) for o,v in status['objects'].items())))
        if key!=previous:
            report=collect(a.root,verify_payload=a.verify_payload)
            write_json(output/'recovery_report.json',report)
            temporary=output/'recovery_report.md.tmp';temporary.write_text(markdown(report));temporary.replace(output/'recovery_report.md')
            plot(report,output)
            print('REPORT',report['completed_trials'],report['status'],'audit',report['audit']['passed'],flush=True)
            if not report['audit']['passed']:raise RuntimeError('Recovery audit failed; see recovery_report.json')
            update_model_index(a.root,report)
            previous=key
        if not a.watch or status['status'] in ['complete','failed']:return
        time.sleep(max(1,min(60,a.interval)))


if __name__=='__main__':main()
