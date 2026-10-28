import shutil

import pytest

from softdynamics import cma_benchmark as scheduler
from softdynamics.io import read_json, write_json
from softdynamics.recovery_data import trajectory_folder


def setup_run(root, objects):
    items=[dict(object=obj,support=[0],query=[1],
                episodes=[dict(particles=1,frames=2)]) for obj in objects]
    write_json(root/'plan.json',dict(objects=items,methods=['uniform','learned'],
        conditions=[dict(truth='real'),dict(truth='center')],
        config=dict(initial_guesses=[[.025]*4,[-.025]*4],
                    truths=[dict(name='center',q=[0.]*4)])))


def invalid_reference(root, obj):
    write_json(trajectory_folder(root,obj,0,[0.]*4)/'complete.json',
               dict(valid=False,invalid_frame=25,check=dict(failure_kind='domain_boundary')))


def no_resources(monkeypatch):
    monkeypatch.setattr(scheduler,'gpu_state',lambda: {})
    monkeypatch.setattr(scheduler,'foreign_compute_gpus',lambda: set())
    monkeypatch.setattr(scheduler.shutil,'disk_usage',lambda _: shutil._ntuple_diskusage(100,100,0))


def test_bad_reference_does_not_abort_or_launch_work_for_bad_object(tmp_path,monkeypatch):
    setup_run(tmp_path,['bad','healthy']);invalid_reference(tmp_path,'bad')
    no_resources(monkeypatch)
    class EndIteration(BaseException):
        pass
    def stop(_):
        raise EndIteration()
    monkeypatch.setattr(scheduler.time,'sleep',stop)
    with pytest.raises(EndIteration):scheduler.supervise(tmp_path,[])
    status=read_json(tmp_path/'status.json')
    assert status['status']=='waiting_resources'
    assert status['objects']['bad']['stage']=='numerically_unavailable'
    assert status['objects']['healthy']['stage']=='weights_or_truth'
    # Only the healthy object's two references remain requested.
    assert status['requested_simulations']==2
    assert status['completed_searches']==0
    assert status['total_searches']==12
    assert status['unavailable_searches']==status['eligible_searches']==6
    assert 'disk_free_below_10_GiB' in status['waiting_reasons']


def test_unavailable_searches_are_not_counted_as_successful_completion(tmp_path,monkeypatch):
    setup_run(tmp_path,['bad']);invalid_reference(tmp_path,'bad')
    no_resources(monkeypatch)
    scheduler.supervise(tmp_path,[])
    status=read_json(tmp_path/'status.json')
    assert status['status']=='complete_with_unavailable'
    assert status['completed_searches']==status['completed_budget_evaluations']==0
    assert status['unavailable_searches']==status['total_searches']==6
    report=read_json(tmp_path/'report.json')
    assert report['numerically_unavailable']['bad']['references'][0]['invalid_frame']==25


def test_supervisor_exception_replaces_stale_running_status(tmp_path,monkeypatch):
    setup_run(tmp_path,[])
    write_json(tmp_path/'status.json',dict(status='running',updated_at=0))
    def broken_hardware():
        raise RuntimeError('hardware query failed')
    monkeypatch.setattr(scheduler,'gpu_state',broken_hardware)
    with pytest.raises(RuntimeError,match='hardware query failed'):
        scheduler.supervise(tmp_path,[])
    status=read_json(tmp_path/'status.json')
    assert status['status']=='failed'
    assert status['supervisor_error']['message']=='hardware query failed'
    assert status['updated_at']>0
