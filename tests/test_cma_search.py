import numpy as np
import pytest

from softdynamics.cma_search import BlockCMA,weighted_loss
from softdynamics.recovery_data import qkey


def test_continuous_cache_keys_do_not_alias_or_mutate():
    center=np.zeros(4)
    near=center.copy();near[0]=1e-12
    other=near.copy();other[0]=np.nextafter(near[0],np.inf)
    assert len({qkey(center),qkey(near),qkey(other)})==3
    assert qkey([.025,-.025,.025,-.025])=='2020'
    signed=np.array([-0.,0.,0.,0.]);qkey(signed)
    assert np.signbit(signed[0])
    with pytest.raises(ValueError):qkey([0,0,0,.026])


def run(state, objective):
    while not state.done:
        p=state.ask()
        q=np.asarray(p['q'])
        j=state.block%4
        assert np.all(abs(q)<=.025)
        np.testing.assert_array_equal(q[:,[k for k in range(4) if k!=j]],
                                      np.tile(state.anchor[[k for k in range(4) if k!=j]],(state.population,1)))
        state.tell([objective(row) for row in q],objective(p['incumbent']))
    return state


def test_cma_converges_and_constant_columns_match_equal_weights():
    pytest.importorskip('cma')
    target=np.array([.013,-.017,.009,-.006])
    def objective(q):return np.repeat(np.sum(((q-target)/.025)**2),3)
    a=BlockCMA(np.zeros(4),np.ones((3,4)),cycles=4,block_generations=8)
    b=BlockCMA(np.zeros(4),np.tile([.1,.8,.4,.2],(3,1)),cycles=4,block_generations=8)
    run(a,objective);run(b,objective)
    assert np.linalg.norm(a.current-target)<.0008
    np.testing.assert_allclose(a.current,b.current,atol=1e-10)
    assert a.logical_candidates==4*4*8*4


def test_saved_pending_population_and_rng_resume_exactly(tmp_path):
    pytest.importorskip('cma')
    state=BlockCMA(np.zeros(4),np.ones((2,4)),cycles=1,block_generations=2)
    p=state.ask();checkpoint=tmp_path/'state.pkl';state.save(checkpoint)
    resumed=BlockCMA.load(checkpoint)
    np.testing.assert_array_equal(resumed.ask()['q'],p['q'])
    objective=lambda q:np.repeat(np.sum(q*q),2)
    run(state,objective);run(resumed,objective)
    np.testing.assert_array_equal(state.current,resumed.current)
    for a,b in zip(state.trace,resumed.trace):
        np.testing.assert_array_equal(a['q'],b['q'])
    assert state.logical_candidates==32


def test_independent_parameter_weights_and_invalid_rollout_penalty():
    pytest.importorskip('cma')
    confidence=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    losses=[1.,10.]
    assert weighted_loss(losses,confidence[:,0])==1
    assert weighted_loss(losses,confidence[:,1])==10
    assert np.isinf(weighted_loss([1.,np.inf],confidence[:,0]))
    state=BlockCMA(np.zeros(4),confidence,cycles=1,block_generations=1)
    state.ask();state.tell(np.full((4,2),np.inf),losses)
    np.testing.assert_array_equal(state.current,np.zeros(4))
