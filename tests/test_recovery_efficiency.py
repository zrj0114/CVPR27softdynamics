import numpy as np

from softdynamics.recovery_efficiency import first_and_stable, search_path, unchanged_cycle_stop
from softdynamics.search import cycle


def trace_row():
    start=np.full(4,.025);levels=np.array([-.025,0.,.025])
    q,original=cycle(lambda q:np.array([np.sum(q*q)]),np.ones((1,4)),start)
    trace=[{**t,'tested_q':[levels[list(indices)].tolist() for indices in t['tested_indices']]} for t in original]
    return dict(start=start.tolist(),candidate_q=q.tolist(),trace=trace)


def test_budget_curve_tracks_selected_parameters_and_repeated_candidates():
    row=trace_row();path,unique=search_path(row)
    assert path.shape==(13,4)
    np.testing.assert_array_equal(path[0],row['start'])
    np.testing.assert_array_equal(path[4:],np.zeros((9,4)))
    assert unique[0]==0 and np.all(np.diff(unique)>=0)
    assert unique[-1]<36
    assert unique[8]==unique[12]


def test_no_change_cycle_stop_preserves_result_without_truth_or_query():
    row=trace_row();path,unique=search_path(row)
    stop=unchanged_cycle_stop(row)
    assert stop==8
    np.testing.assert_array_equal(path[stop],row['candidate_q'])
    assert first_and_stable(np.all(path==0,axis=1))==(4,4)


def test_first_hit_is_distinct_from_stable_hit_and_non_recovery():
    assert first_and_stable(np.array([True,False,True,True]))==(0,2)
    assert first_and_stable(np.array([False,False]))==(None,None)
    assert first_and_stable(np.array([False,True,False]))==(1,None)
