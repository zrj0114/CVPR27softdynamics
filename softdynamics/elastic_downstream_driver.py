"""Schedule support-only identification before any held-out observation scoring."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

from .io import read_json,write_json,sha256
from .elastic_downstream_evaluation import prepare_queries,prepare_metrics,summarize
from .elastic_prefetch import update_jobs as prefetch_jobs


def run(root):
    root=Path(root);spec=read_json(root/'protocol.json');python=sys.executable
    material=[python,'-m','softdynamics.elastic_downstream','--run',str(root)]
    metric=[python,'-m','softdynamics.elastic_downstream_evaluation','--run',str(root)]
    with (root/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            plan=read_json(root/'jobs.json');jobs={j['id']:j for j in plan['jobs']};added=[]
            def add(key,command,after):
                row=dict(id=key,command=command,after=after)
                if key in jobs:
                    if jobs[key]!=row:raise ValueError('Existing scheduled job changed: '+key)
                    return
                jobs[key]=row;added.append(key)
            def done(key):
                path=root/'job_results'/f'{key}.json'
                return path.exists() and read_json(path)['returncode']==0
            for obj in spec['test_objects']:
                baseline=f'baseline_{obj}';audit=plan.get('audit_jobs',{}).get(obj,f'audit_{obj}')
                add(audit,metric+['--mode','audit','--object',obj,'--device','{device}'],[baseline])
                add(f'inverse_{obj}_physical_cma',material+['--mode','cma','--object',obj,'--device','{device}'],[baseline])
                for method in spec['methods']:
                    if method=='physical_cma':continue
                    add(f'inverse_{obj}_{method}',material+['--mode','inverse','--object',obj,'--method',method,'--device','{device}'],[audit])
                inverse_jobs=[f'inverse_{obj}_{m}' for m in spec['methods']]
                if not all(done(key) for key in inverse_jobs):continue
                query=prepare_queries(root,obj);query_jobs=[]
                for case,row in query['queries'].items():
                    key=f'query_{obj}_{row["episode"]}_{case.rsplit("_",1)[-1][:12]}'
                    add(key,material+['--mode','physics','--object',obj,'--episode',str(row['episode']),
                        '--q',*[repr(float(q)) for q in row['q']],'--device','{device}','--no-capture',
                        '--requester',f'query/{obj}'],inverse_jobs)
                    query_jobs.append(key)
                if not all(done(key) for key in query_jobs):continue
                evaluation=prepare_metrics(root,obj,'/home/rongjin/empm-study/empm')
                for case,row in evaluation['cases'].items():
                    key=f'metrics_{obj}_{row["episode"]}_{case.rsplit("_",1)[-1][:12]}'
                    add(key,metric+['--mode','case','--object',obj,'--case',case,'--device','{device}'],query_jobs)
            plan['jobs']=list(jobs.values())
            plan['jobs'],prefetched,retired=prefetch_jobs(root,plan,python)
            jobs={j['id']:j for j in plan['jobs']}
            if added or prefetched or retired:
                write_json(root/'jobs.json',plan)
                print('JOBS','added',len(added),'prefetched',len(prefetched),'retired',len(retired),'total',len(jobs),flush=True)
            report=summarize(root)
            failures={p.stem:read_json(p) for p in (root/'job_results').glob('*.json') if read_json(p)['returncode']!=0}
            write_json(root/'driver_status.json',dict(pid=os.getpid(),updated_at=time.time(),jobs=len(jobs),
                completed=sum(done(key) for key in jobs),failures=failures,report_status=report['status']))
            if report['status']=='complete' and all(done(key) for key in jobs):
                for source in spec['checkpoints'].values():
                    if sha256(source['path'])!=source['sha256']:raise ValueError('Pretrained model changed')
                plan['sealed']=True;write_json(root/'jobs.json',plan)
                write_json(root/'complete.json',dict(status='complete',completed_at=time.time(),jobs=len(jobs),
                    report_sha256=sha256(root/'report.json'),checkpoints_unchanged=True,
                    all_methods_completed=True,query_excluded_from_fitting=True,
                    physical_failures=report['physical_failures']))
                return
            time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run)
