import itertools

import numpy as np
from scipy.spatial import cKDTree

from softdynamics.objective import score
from softdynamics.recovery_benchmark import frontier,shuffled_scores
from softdynamics.recovery_data import noise_arrays,qkey
from softdynamics.search import cycle


def test_resumable_frontier_matches_original_full_loss_optimizer():
    rng=np.random.default_rng(17)
    A=rng.normal(size=(3,6,4));b=rng.normal(size=(3,6))*.015
    weights=rng.uniform(.1,1,(3,4));start=np.full(4,.025)
    def loss(q):return ((A@q-b)**2).sum(1)
    expected,trace=cycle(loss,weights,start)
    bank={}
    for _ in range(40):
        result,partial,missing=frontier(bank,weights,start)
        if result is not None:break
        assert missing
        for q in missing:bank[qkey(q)]=loss(q)
    assert result is not None
    np.testing.assert_array_equal(result['q'],expected)
    assert result['logical_candidate_vectors']==36
    assert len(result['trace'])==len(trace)==12
    for actual,original in zip(result['trace'],trace):
        assert actual['new_level']==original['new_level']
        np.testing.assert_allclose(actual['costs'],original['costs'])


def test_score_shuffle_preserves_each_parameter_distribution():
    scores=np.arange(20).reshape(5,4)/20
    shuffled=shuffled_scores(scores)
    np.testing.assert_array_equal(np.sort(shuffled,axis=0),np.sort(scores,axis=0))
    assert not np.array_equal(scores,shuffled)
    # Parameter columns are not normalized against one another.
    assert not np.allclose(shuffled.sum(1),1)


def test_noise_is_paired_reproducible_and_zero_at_original_initial_frame():
    x=np.zeros((24,20,3),np.float32);surface=np.ones((24,30,3),np.float32)
    a,b=noise_arrays(x,surface,2.,17)
    c,d=noise_arrays(x,surface,2.,17)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    np.testing.assert_array_equal(a[0],x[0]);np.testing.assert_array_equal(b[0],surface[0])
    assert np.std(a[1:])>0
    p,q=noise_arrays(x,surface,0.,17)
    np.testing.assert_array_equal(p,x);np.testing.assert_array_equal(q,surface)


def test_original_surface_track_objective_zero_truth_and_phase_weights():
    points=np.array([[0.,0,0],[.01,0,0],[0,.01,0]],np.float32)
    tracks=np.broadcast_to(points,(6,3,3)).copy()
    obs=dict(surface=list(tracks),tracks=tracks,visible=np.ones((6,3),bool),
             meta=dict(frames=np.arange(6),clip_start=2,clip_end=3),trees=[cKDTree(p) for p in tracks])
    config=dict(huber_delta_mm=5.,surface_weight=.5,track_weight=.5,
                phase_weights=dict(before=.1,interaction=.45,release=.45))
    assert score(dict(surface=tracks,tracks=tracks),obs,config)['objective_mm']==0
    offset=tracks.copy();offset[4:,:,2]=.002
    r=score(dict(surface=offset,tracks=offset),obs,config)
    assert abs(r['objective_mm']-.45*(.5*2+.5*.5*2**2/5))<1e-6
