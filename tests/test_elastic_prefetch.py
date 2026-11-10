import os
import pickle
from pathlib import Path

import numpy as np

from softdynamics.elastic_prefetch import current_candidate,update_jobs
from softdynamics.elastic_downstream import bounds
from softdynamics.io import write_json
from test_elastic_downstream import CONFIG,PHYSICS


def fixture(tmp_path,method='best_uniform'):
    write_json(tmp_path/'protocol.json',dict(elastic_config=CONFIG,methods=[method]))
    write_json(tmp_path/'physics/plan.json',dict(config=PHYSICS,objects=[dict(object='test',support=[0,2],query=[8])]))
    write_json(tmp_path/'baselines/test.json',dict(valid=True,records=[dict(episode=0,isolated_execution_s=10.),dict(episode=2,isolated_execution_s=20.)]))
    write_json(tmp_path/'farm_status.json',dict(active={f'inverse_test_{method}':dict(pid=os.getpid(),gpu=0)}))
    folder=tmp_path/'inverse/test'/method;folder.mkdir(parents=True)
    return folder,dict(jobs=[],gpus=[0,1],max_active=2)


def test_prefetch_only_remaining_support_of_current_neural_candidate(tmp_path):
    folder,plan=fixture(tmp_path)
    write_json(folder/'proposal.json',dict(round=1,candidate_q=[.1,-.1]))
    jobs,added,_=update_jobs(tmp_path,plan,'python',[(0,'gpu0'),(1,'gpu1')])
    assert len(added)==1
    assert jobs[0]['prefetch']['episode']==2
    assert jobs[0]['prefetch']['query_observations_used'] is False
    assert jobs[0]['command'][-2:]==['--requester','test/best_uniform']
    # A stale proposal must not launch more work after the round is committed.
    write_json(folder/'history.json',[dict(round=1)])
    assert update_jobs(tmp_path,plan,'python',[(0,'gpu0'),(1,'gpu1')])[1]==[]


def test_cma_prefetch_never_advances_to_future_population_member(tmp_path):
    folder,_=fixture(tmp_path,'physical_cma')
    pending=dict(candidates=[[.2,.3],[.7,.8],[.4,.5],[.1,.2]],costs=[4.])
    (folder/'optimizer.pkl').write_bytes(pickle.dumps(dict(pending=pending,history=[{}])))
    round_index,q=current_candidate(folder,'physical_cma',CONFIG)
    lo,hi=bounds(CONFIG)
    assert round_index==2
    np.testing.assert_array_equal(q,lo+np.array([.7,.8])*(hi-lo))


def test_busy_farm_and_completed_cache_do_not_add_extra_work(tmp_path):
    folder,plan=fixture(tmp_path)
    write_json(folder/'proposal.json',dict(round=1,candidate_q=[.1,-.1]))
    assert update_jobs(tmp_path,plan,'python',[(0,'gpu0')])[1]==[]
    jobs,_,_=update_jobs(tmp_path,plan,'python',[(0,'gpu0'),(1,'gpu1')])
    write_json(Path(jobs[0]['prefetch']['folder'])/'complete.json',dict(valid=True))
    plan['jobs']=jobs
    remaining,added,removed=update_jobs(tmp_path,plan,'python',[(0,'gpu0'),(1,'gpu1')])
    assert not remaining and not added
    assert removed==[jobs[0]['id']]
