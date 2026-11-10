import pytest
from softdynamics.io import write_json,sha256
from softdynamics.teacher50_schedule import (choose_task,workload,result_path,archive_incomplete,required_free_mib)

def task(obj,variant=0):return dict(id=f'{obj}/episode_0/sample_{variant}',object=obj,episode=0,variant=variant,particles=9999999)
def work():return dict(object_particles={'a':20000,'b':90000,'large':2000000},episodes={
    'a/episode_0':dict(particles=20000,frames_estimate=300),
    'b/episode_0':dict(particles=90000,frames_estimate=300),
    'large/episode_0':dict(particles=2000000,frames_estimate=300)})

def test_large_object_is_last_and_running_small_object_does_not_take_all_cards():
    pending=[task('large'),task('a'),task('b')]
    assert choose_task(pending,[],work(),20000)['object']=='a'
    assert choose_task(pending,['a'],work(),20000)['object']=='b'
    assert choose_task([task('large')],[],work(),20000)['object']=='large'
    assert choose_task(pending,[],work(),1000) is None
    assert required_free_mib(task('large'),work())>required_free_mib(task('a'),work())

def test_workload_uses_repaired_episode_particle_counts_and_largest_episode_per_object(tmp_path):
    episodes=[]
    for ep,n in [(0,4000),(1,15000)]:
        row=tmp_path/f'row_{ep}.json';write_json(row,dict(frames=200))
        receipt=tmp_path/f'prepared_{ep}.json';write_json(receipt,dict(particles=n,row=str(row)))
        episodes.append(dict(id=f'a/episode_{ep}',object='a',preparation=str(receipt),particles_estimate=10000000))
    w=workload(dict(episodes=episodes))
    assert w['object_particles']=={'a':15000}
    assert w['episodes']['a/episode_0']['particles']==4000

def test_successful_retry_overrides_historical_failure_and_checks_integrity(tmp_path):
    t=task('a');original=tmp_path/'trajectories'/t['id']/'complete.json';write_json(original,dict(valid=False))
    success=tmp_path/'retries/v1'/t['id']/'complete.json';write_json(success,dict(valid=True,full_rollout=True))
    idx={t['id']:dict(folder=str(success.parent.relative_to(tmp_path)),complete_sha256=sha256(success))}
    assert result_path(tmp_path,t,idx)==success
    write_json(success,dict(valid=False))
    with pytest.raises(ValueError,match='retry receipt changed'):result_path(tmp_path,t,idx)

def test_interrupted_teacher_directory_is_preserved_and_fresh_attempt_can_start(tmp_path):
    old=tmp_path/'attempt_0/teacher/workpoint_0.npz';old.parent.mkdir(parents=True);old.write_bytes(b'partial-label')
    (tmp_path/'task.lock').touch();write_json(tmp_path/'progress.json',dict(frame=50))
    archived=archive_incomplete(tmp_path)
    assert (archived/'attempt_0/teacher/workpoint_0.npz').read_bytes()==b'partial-label'
    assert (tmp_path/'task.lock').exists() and not (tmp_path/'attempt_0').exists()
    assert archive_incomplete(tmp_path) is None
    (tmp_path/'attempt_0/teacher').mkdir(parents=True,exist_ok=False)
    write_json(tmp_path/'complete.json',dict(valid=True))
    assert archive_incomplete(tmp_path) is None
