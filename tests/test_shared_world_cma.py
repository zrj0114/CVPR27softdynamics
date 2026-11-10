import numpy as np
import pytest
import torch

from softdynamics.shared_world_cma import PreselectedBlockCMA, batched_window_reference_rollout


def finish(state, surrogate, physical):
    while not state.done:
        pool = state.ask()
        state.preselect([surrogate(q) for q in pool['pool_q']])
        pending = state.ask()
        state.tell([physical(q) for q in pending['q']], physical(pending['incumbent']))
    return state


def test_continuous_cma_runs_all_96_candidates_and_preserves_four_objectives():
    pytest.importorskip('cma')
    weights = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], float)
    targets = np.array([[.014]*4, [-.012]*4])
    objective = lambda q: np.sum(((q-targets)/.025)**2, axis=1)
    state = PreselectedBlockCMA(np.zeros(4), weights, cycles=3, block_generations=2)
    finish(state, objective, objective)
    assert state.logical_candidates == 96
    assert state.surrogate_candidates == 384
    assert len(state.trace) == 24
    assert len(state.seen) == 97
    assert np.sign(state.current).tolist() == [1, -1, 1, -1]
    for row in state.trace:
        j = row['block'] % 4
        other = [k for k in range(4) if k != j]
        assert np.all(abs(row['pool_q']) <= .025)
        assert np.unique(row['pool_q'][:, j]).size > 3
        np.testing.assert_array_equal(row['pool_q'][:, other], np.tile(row['pool_q'][0, other], (16, 1)))
        assert row['candidate_vectors'] == row['generation']*4+row['block']*8
    assert [r['candidate_vectors'] for r in state.trace if r['candidate_vectors'] % 32 == 0] == [32, 64, 96]


def test_constant_confidence_columns_exactly_match_uniform_preselection():
    pytest.importorskip('cma')
    objective = lambda q: np.repeat(np.sum((q-.011)**2), 2)
    a = PreselectedBlockCMA(np.zeros(4), np.ones((2, 4)), cycles=1, block_generations=2)
    b = PreselectedBlockCMA(np.zeros(4), np.tile([.1, .8, .4, .2], (2, 1)), cycles=1, block_generations=2)
    finish(a, objective, objective); finish(b, objective, objective)
    for x, y in zip(a.trace, b.trace):
        np.testing.assert_array_equal(x['pool_q'], y['pool_q'])
        np.testing.assert_array_equal(x['selected_indices'], y['selected_indices'])
    np.testing.assert_array_equal(a.current, b.current)


@pytest.mark.parametrize('selected', [False, True])
def test_pool_and_selected_population_resume_without_resampling(tmp_path, selected):
    pytest.importorskip('cma')
    state = PreselectedBlockCMA(np.zeros(4), np.ones((2, 4)), cycles=1, block_generations=2)
    objective = lambda q: np.repeat(np.sum(q*q), 2)
    pool = state.ask()['pool_q'].copy()
    if selected:
        state.preselect([objective(q) for q in pool])
    state.save(tmp_path/'state.pkl')
    resumed = PreselectedBlockCMA.load(tmp_path/'state.pkl')
    np.testing.assert_array_equal(resumed.ask()['pool_q'], pool)
    for s in (state, resumed):
        if not selected:
            s.preselect([objective(q) for q in pool])
        p = s.ask(); s.tell([objective(q) for q in p['q']], objective(p['incumbent']))
        finish(s, objective, objective)
    for x, y in zip(state.trace, resumed.trace):
        np.testing.assert_array_equal(x['pool_q'], y['pool_q'])
        np.testing.assert_array_equal(x['q'], y['q'])


def test_invalid_surrogates_fall_back_and_physical_fitness_controls_incumbent():
    pytest.importorskip('cma')
    state = PreselectedBlockCMA(np.zeros(4), np.ones((2, 4)), cycles=1, block_generations=1)
    p = state.ask(); original = p['q'].copy()
    with pytest.raises(ValueError, match='preselection'):
        state.tell(np.ones((4, 2)), np.ones(2))
    state.preselect(np.full((16, 2), np.nan))
    np.testing.assert_array_equal(state.ask()['q'], original)
    # Every proposal is physically worse than the entering incumbent.
    state.tell(np.full((4, 2), 10.), [1., 1.])
    np.testing.assert_array_equal(state.current, np.zeros(4))
    assert state.logical_candidates == 4
    assert state.surrogate_candidates == 16


class AffineWorld(torch.nn.Module):
    kinematic_only = True
    def forward(self, state, pads0, pads1, theta, geometry, dt):
        signal = theta[:, :1, None] + pads1[:, :1, :1]
        return torch.cat((1.2*state[..., :6]+.3+signal, state[..., 6:]), -1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('length,horizon,batch', [(70, 32, 2), (7, 1, 8), (5, 32, 8)])
def test_window_baseline_is_exact_and_partial_window_material_gradient_is_local(dtype, length, horizon, batch):
    reference = torch.linspace(.1, .7, length, dtype=dtype)[None, :, None, None].expand(1, -1, 2, 24).clone().requires_grad_()
    pads = torch.zeros((1, length, 4, 13), dtype=dtype)
    theta = torch.zeros((1, 4), dtype=dtype, requires_grad=True)
    times = torch.arange(length, dtype=dtype)[None]/30
    model = AffineWorld()
    nominal = batched_window_reference_rollout(model, reference, pads, theta.detach(), {}, times, horizon, batch)
    assert all(not value.requires_grad for value in nominal)
    result = batched_window_reference_rollout(model, reference, pads, theta, {}, times, horizon, batch, nominal)
    torch.testing.assert_close(result[..., :6], reference[0, ..., :6], rtol=0, atol=0)
    assert result.shape == (length, 2, 24)
    # Gradients at every frame depend on steps since its own window start,
    # including the final partial window, rather than the complete history.
    for frame in [1, length-1, min(horizon+1, length-1)]:
        steps = (frame-1) % horizon+1
        expected = sum(1.2**k for k in range(steps))
        gradient = torch.autograd.grad(result[frame, 0, 0], theta, retain_graph=True)[0]
        torch.testing.assert_close(gradient[0, 0], theta.new_tensor(expected), rtol=2e-6, atol=2e-6)
    assert reference.grad is None


def test_window_batching_rejects_missing_nominal_groups():
    ref = torch.zeros(1, 8, 2, 24)
    with pytest.raises(ValueError, match='batching'):
        batched_window_reference_rollout(AffineWorld(), ref, torch.zeros(1, 8, 4, 13),
            torch.zeros(1, 4), {}, torch.zeros(1, 8), horizon=2, batch_windows=2, predictions=[])
