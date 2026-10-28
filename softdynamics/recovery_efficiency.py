"""Paired budget curves from completed frozen searches; no new physical rollouts."""
import argparse
from pathlib import Path
import time

import numpy as np

from .io import read_json, write_json
from .parameters import PARAMETERS
from .recovery_data import qkey


def search_path(row):
    q=np.array(row['start'],dtype=float);path=[q.copy()];seen=set();unique=[0]
    for step,t in enumerate(row['trace']):
        j=step%4
        if t['parameter']!=PARAMETERS[j]:raise ValueError('Unexpected search coordinate order')
        seen.update(qkey(candidate) for candidate in t['tested_q'])
        q=np.array(t['tested_q'][t['new_level']],dtype=float)
        path.append(q.copy());unique.append(len(seen))
    if not np.allclose(path[-1],row['candidate_q'],atol=1e-10,rtol=0):
        raise ValueError('Trace does not reconstruct final candidate')
    return np.array(path),np.array(unique)


def first_and_stable(mask):
    ids=np.flatnonzero(mask)
    first=int(ids[0]) if len(ids) else None
    stable=next((i for i in range(len(mask)) if np.all(mask[i:])),None)
    return first,stable


def unchanged_cycle_stop(row):
    """A deployable rule: stop after a full E/nu/friction/decay cycle with no changes."""
    for start in range(0,len(row['trace']),4):
        cycle=row['trace'][start:start+4]
        if len(cycle)==4 and all(t['old_level']==t['new_level'] for t in cycle):return start+4
    return len(row['trace'])


def average(values):
    return float(np.mean(values)) if len(values) else None


def collect(root):
    root=Path(root);plan=read_json(root/'plan.json');status=read_json(root/'status.json')
    rows=[read_json(p) for p in sorted((root/'results').glob('*/*/*.json'))]
    lookup={(r['object'],r['context'],r['method']):r for r in rows}
    items={o['object']:o for o in plan['objects']}
    curves=[];reaching=[];paired=[];stops=[]
    # Only fully completed three-method contexts; partial contexts cannot bias a comparison.
    for (obj,context,method),uniform in lookup.items():
        if method!='uniform' or not all((obj,context,m) in lookup for m in plan['methods']):continue
        paths={};targets={};n=len(items[obj]['support'])
        for method in plan['methods']:
            row=lookup[(obj,context,method)];path,unique=search_path(row);paths[method]=(path,unique)
            stop=unchanged_cycle_stop(row)
            stops.append(dict(object=obj,context=context,method=method,truth=row['condition']['truth'],
                              coordinate_updates=stop,logical_candidate_vectors=3*stop,
                              logical_episode_evaluations=3*stop*n,unique_candidate_vectors=int(unique[stop]),
                              unique_episode_rollouts=int(unique[stop])*n,full_budget_unique_vectors=int(unique[-1]),
                              same_candidate_as_full=bool(np.allclose(path[stop],row['candidate_q'])),
                              role='Hypothetical no-change-cycle stopping rule replayed on the frozen path; actual jobs used three cycles.'))
            if row['condition']['truth']=='real':continue
            errors=abs(path-np.array(row['truth_q']))/.025
            mean_errors=errors.mean(1)
            for step,value in enumerate(mean_errors):
                curves.append(dict(object=obj,context=context,truth=row['condition']['truth'],noise_mm=row['condition']['noise_mm'],
                                   method=method,coordinate_updates=step,logical_candidate_vectors=3*step,
                                   logical_episode_evaluations=3*step*n,unique_candidate_vectors=int(unique[step]),
                                   parameter_error_steps=float(value),exact=bool(errors[step].max()<1e-7)))
            for threshold in [0.,.25,.5,1.]:
                first,stable=first_and_stable(mean_errors<=threshold+1e-8)
                reaching.append(dict(object=obj,context=context,method=method,threshold_steps=threshold,
                   first_hit_update=first,stable_hit_update=stable,
                   first_hit_logical_vectors=3*first if first is not None else None,
                   stable_hit_logical_vectors=3*stable if stable is not None else None,
                   first_hit_unique_vectors=int(unique[first]) if first is not None else None,
                   stable_hit_unique_vectors=int(unique[stable]) if stable is not None else None))
            targets[method]=mean_errors
        for method in plan['methods']:
            if method=='uniform':continue
            row=lookup[(obj,context,method)];upath,uunique=paths['uniform'];path,unique=paths[method]
            target=np.array(uniform['candidate_q'])
            uf,us=first_and_stable(np.all(np.isclose(upath,target,atol=1e-10,rtol=0),axis=1))
            mf,ms=first_and_stable(np.all(np.isclose(path,target,atol=1e-10,rtol=0),axis=1))
            same=bool(np.allclose(path[-1],upath[-1],atol=1e-10,rtol=0))
            paired.append(dict(object=obj,context=context,truth=row['condition']['truth'],noise_mm=row['condition']['noise_mm'],method=method,
                 same_final_candidate=same,uniform_first_hit_update=uf,method_first_hit_update=mf,
                 uniform_stable_hit_update=us,method_stable_hit_update=ms,
                 stable_updates_delta=ms-us if same else None,
                 stable_unique_vectors_delta=int(unique[ms])-int(uunique[us]) if same else None))
    summaries=[]
    for obj in items:
        for method in plan['methods']:
            if method=='uniform':continue
            selected=[r for r in paired if r['object']==obj and r['method']==method]
            if not selected:continue
            comparable=[r for r in selected if r['same_final_candidate']]
            delta=[r['stable_updates_delta'] for r in comparable]
            summaries.append(dict(object=obj,method=method,paired_contexts=len(selected),same_final_contexts=len(comparable),
                earlier=sum(d<0 for d in delta),same=sum(d==0 for d in delta),later=sum(d>0 for d in delta),
                mean_stable_updates_delta=average(delta),mean_stable_unique_vectors_delta=average([r['stable_unique_vectors_delta'] for r in comparable])))
    truth_recovery=[]
    for obj in items:
        for threshold in [0.,.25,.5,1.]:
            index={(r['context'],r['method']):r for r in reaching if r['object']==obj and r['threshold_steps']==threshold}
            pairs=[(u,index[(context,'learned')]) for (context,method),u in index.items() if method=='uniform']
            if not pairs:continue
            for criterion in ['first_hit_update','stable_hit_update']:
                common=[(u,l) for u,l in pairs if u[criterion] is not None and l[criterion] is not None]
                truth_recovery.append(dict(object=obj,threshold_steps=threshold,criterion=criterion,contexts=len(pairs),
                    uniform_reached=sum(u[criterion] is not None for u,l in pairs),learned_reached=sum(l[criterion] is not None for u,l in pairs),
                    both_reached=len(common),learned_earlier=sum(l[criterion]<u[criterion] for u,l in common),
                    same=sum(l[criterion]==u[criterion] for u,l in common),learned_later=sum(l[criterion]>u[criterion] for u,l in common)))
    budget_summary=[]
    for obj in items:
        for noise in [0.,.5,2.]:
            for step in [0,4,8,12]:
                methods={}
                for method in plan['methods']:
                    group=[r for r in curves if r['object']==obj and r['noise_mm']==noise and r['method']==method and r['coordinate_updates']==step]
                    if group:methods[method]=dict(contexts=len(group),parameter_error_steps=average([r['parameter_error_steps'] for r in group]),
                                              exact_fraction=average([r['exact'] for r in group]))
                if methods:budget_summary.append(dict(object=obj,noise_mm=noise,cycles=step//4,logical_candidate_vectors=3*step,methods=methods))
    return dict(updated_at=time.time(),completed_trials=len(rows),status=status['status'],object_status=status['objects'],
                model_sha256=plan['model_sha256'],paired_summary=summaries,budget_summary=budget_summary,
                curves=curves,reaching=reaching,truth_recovery=truth_recovery,paired=paired,no_change_cycle_stop=stops,
                protocol='Retrospective first/stable hits use known synthetic truths or the uniform final candidate, never a proposed online stopping signal. Actual methods all run 12 coordinate updates and 36 logical candidate vectors. Unique rollouts count a separate empty cache for each method, excluding inference and query evaluation. Shared benchmark wall time is not a fair method runtime measurement.')


def markdown(report):
    text=['# 加权与等权：达到相同结果的预算对照','',f'基于 {report["completed_trials"]} 个已完成方法试验，状态 {report["status"]}。主实验不变，仅读取搜索记录。','',
          '实际代码全部运行三轮，每轮 E、ν、摩擦、衰减各更新一次：12 次参数更新、36 个逻辑候选向量。每个向量模拟全部拟合交互；因此当前程序没有因加权自动少跑步骤。', '',
          '## 最早稳定到达等权最终候选', '',
          '“稳定到达”表示从该更新起，所有后续记录都保持同一组参数。只比较最终候选相同的配对；结果不同的上下文单独排除，不强行当作同质量加速。该指标是事后回看，不能直接用来在线早停。', '',
          '| 物体 | 方法 | 已配对条件 | 最终参数相同 | 更早 / 相同 / 更晚 | 平均更新数差（方法 − 等权） | 独立缓存唯一候选数差 |',
          '|---|---|---:|---:|---|---:|---:|']
    for r in report['paired_summary']:
        text.append(f'| {r["object"]} | {r["method"]} | {r["paired_contexts"]} | {r["same_final_contexts"]} | '+
                    f'{r["earlier"]} / {r["same"]} / {r["later"]} | {r["mean_stable_updates_delta"]:.3f} | {r["mean_stable_unique_vectors_delta"]:.3f} |')
    text+=['','## 达到已知真值的更新预算','',
           '只对合成实验核查四参数完全恢复。未恢复的条件保留为未达到，不把它们当作快速结束。', '',
           '| 物体 | 条件数 | 完全恢复：等权 / 模型 | 双方均恢复 | 模型更早 / 相同 / 更晚（首次到达） |',
           '|---|---:|---|---:|---|']
    for r in report['truth_recovery']:
        if r['threshold_steps']!=0 or r['criterion']!='first_hit_update':continue
        text.append(f'| {r["object"]} | {r["contexts"]} | {r["uniform_reached"]} / {r["learned_reached"]} | {r["both_reached"]} | '+
                    f'{r["learned_earlier"]} / {r["same"]} / {r["learned_later"]} |')
    text+=['','## 固定一／二／三轮后的合成恢复','',
           '每轮为 4 次参数更新、12 个逻辑候选向量。下表使用已完成的同条件三方法配对。没有按真值挑“过程中最好的候选”；列出当时真正选中的参数。', '',
           '| 物体 | 噪声 mm | 轮数 | 条件数 | 参数误差：等权 / 模型 / 打乱 | 完全恢复比例：等权 / 模型 / 打乱 |',
           '|---|---:|---:|---:|---|---|']
    for g in report['budget_summary']:
        if g['cycles']==0:continue
        methods=g['methods'];order=['uniform','learned','shuffled']
        text.append(f'| {g["object"]} | {g["noise_mm"]:g} | {g["cycles"]} | {methods["uniform"]["contexts"]} | '+
                    ' / '.join(f'{methods[m]["parameter_error_steps"]:.4f}' for m in order)+' | '+
                    ' / '.join(f'{methods[m]["exact_fraction"]:.3f}' for m in order)+' |')
    text+=['','## 可执行的早停规则诊断','','规则：完整一轮中四个参数都没有改变，就停止。规则不读取真值、教师或查询集。固定权重与确定性搜索下，下一轮会重复相同决策。这里仅重放规则，运行中的主实验仍保持三轮。','',
           '| 物体 | 方法 | 条件数 | 平均早停更新数 | 平均唯一候选数 | 最终候选一致 |','|---|---|---:|---:|---:|---:|']
    for obj in report['object_status']:
        for method in ['uniform','learned','shuffled']:
            rows=[r for r in report['no_change_cycle_stop'] if r['object']==obj and r['method']==method]
            if not rows:continue
            text.append(f'| {obj} | {method} | {len(rows)} | {average([r["coordinate_updates"] for r in rows]):.3f} | '+
                        f'{average([r["unique_candidate_vectors"] for r in rows]):.3f} | {sum(r["same_candidate_as_full"] for r in rows)}/{len(rows)} |')
    text+=['','真正的时间收益需将模型推理计入，并在相同硬件、独立缓存、相同早停规则下比较；当前共享多卡缓存的完成先后不能当作方法加速。', '']
    return '\n'.join(text)


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    objects=[o for o in report['object_status'] if any(g['object']==o for g in report['curves'])]
    if not objects:return
    fig,axes=plt.subplots(len(objects),2,figsize=(11,3.5*len(objects)),squeeze=False)
    for i,obj in enumerate(objects):
        for method,color in zip(['uniform','learned','shuffled'],['#777777','#1976d2','#d48731']):
            groups=[[r for r in report['curves'] if r['object']==obj and r['method']==method and r['coordinate_updates']==s] for s in range(13)]
            axes[i,0].plot(np.arange(13)*3,[average([r['parameter_error_steps'] for r in g]) for g in groups],label=method,color=color,marker='o',markersize=3)
            axes[i,1].plot(np.arange(13)*3,[average([r['exact'] for r in g]) for g in groups],label=method,color=color,marker='o',markersize=3)
        for j,title in enumerate(['Parameter error (mean grid steps)','Exact recovery fraction']):
            axes[i,j].set_title(obj+': '+title);axes[i,j].set_xlabel('Logical candidate vectors');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    fig.suptitle('Matched completed contexts; all noise/truth/start conditions pooled within each object')
    fig.tight_layout();fig.savefig(output/'efficiency_budget_curves.png',dpi=180);fig.savefig(output/'efficiency_budget_curves.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
    output=a.output or a.root/'analysis';output.mkdir(parents=True,exist_ok=True);previous=None
    while True:
        status=read_json(a.root/'status.json');key=(status['completed_trials'],status['status'])
        if key!=previous:
            report=collect(a.root);write_json(output/'efficiency_report.json',report)
            temporary=output/'efficiency_report.md.tmp';temporary.write_text(markdown(report));temporary.replace(output/'efficiency_report.md')
            plot(report,output);print('EFFICIENCY',report['completed_trials'],report['status'],flush=True);previous=key
        if not a.watch or status['status'] in ['complete','failed']:return
        time.sleep(30)


if __name__=='__main__':main()
