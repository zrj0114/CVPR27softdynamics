import os
from pathlib import Path

import numpy as np

from softdynamics import cma_benchmark as scheduler
from softdynamics.io import read_json, sha256, write_json
from softdynamics.recovery_data import trajectory_folder


def test_complete_scheduler_uses_physical_fitness_and_freezes_query_at_all_budgets(tmp_path, monkeypatch):
    """Exercise the actual scheduler, replacing only expensive external workers."""
    methods = ['uniform', 'learned', 'physical_uniform']
    confidence = np.array([[.9, .1, .9, .1], [.1, .9, .1, .9]])
    plan = dict(objects=[dict(object='fixture', support=[0, 1], query=[2],
                    episodes=[dict(particles=1, frames=2)])], methods=methods,
        config=dict(initial_guesses=[], truths=[]),
        conditions=[dict(key='real', truth='real', noise_mm=0., seed=19)],
        optimizer=dict(seed=9101, cycles=3, block_generations=2, population=4, sigma=.2, budgets=[32, 64, 96]),
        unused_gpu_required=True,
        world_preselection=dict(methods=['uniform', 'learned'], population=16,
            checkpoint='fixture', checkpoint_sha256='fixture', data_root='fixture', data_plan_sha256='fixture',
            window_horizon=32, batch_windows=8, reference_cost={},
            objects=dict(fixture=dict(physical_reference='fixture', physical_reference_sha256='fixture', state_reference='fixture'))))
    write_json(tmp_path/'plan.json', plan)
    write_json(tmp_path/'weights/fixture.json', dict(contexts=[dict(id='real', start=[0.]*4,
        condition=plan['conditions'][0], weights=dict(uniform=np.ones((2, 4)), learned=confidence,
                                                      physical_uniform=np.ones((2, 4))))]))
    monkeypatch.setattr(scheduler, 'gpu_state', lambda: {4: (0, 0)})
    monkeypatch.setattr(scheduler, 'foreign_compute_gpus', lambda: set())
    monkeypatch.setattr(scheduler, 'occupied_compute_gpus', lambda: set())
    monkeypatch.setattr(scheduler.time, 'sleep', lambda _: None)
    query_calls, surrogate_calls = [], []

    def physical(q, episode):
        # Two support interactions favor opposite parameter values.
        target = .011 if episode == 0 else -.013
        return float(1.+np.sum(((np.array(q)-target)/.025)**2))

    class Worker:
        def __init__(self, command, **kwargs):
            self.pid = os.getpid()  # A real /proc identity for the scheduler.
            args = command[command.index('-m')+2:]
            def get(flag): return args[args.index(flag)+1]
            if command[command.index('-m')+1] == 'softdynamics.shared_world_cma':
                request = Path(get('--request')); r = read_json(request)
                surrogate_calls.append(r)
                # Deliberately misleading surrogate: fitness must still come
                # from physical workers and incumbents remain physically valid.
                values = [[50.-physical(q, e) for e in r['support']] for q in r['pool_q']]
                write_json(get('--output'), dict(request_sha256=sha256(request),
                    support_losses_mm=values, physical_fitness=False, query_observations_used=False, elapsed_s=.01))
                return
            episode = int(get('--episode'))
            q = [float(v) for v in args[args.index('--q')+1:args.index('--q')+5]]
            folder = trajectory_folder(tmp_path, 'fixture', episode, q)
            if get('--mode') == 'simulate':
                if episode == 2:
                    selections = [read_json(p) for p in (tmp_path/'selections').rglob('*.json')]
                    assert any(q == row['selected_q'] or q == row['candidate_q'] for row in selections)
                    assert all(row['query_observations_used'] is False for row in selections)
                    query_calls.append(q)
                write_json(folder/'complete.json', dict(valid=True, elapsed_s=.1))
            else:
                write_json(folder/'scores.json', dict(conditions=dict(real=dict(valid=True, objective_mm=physical(q, episode)))))

        def poll(self): return 0

    monkeypatch.setattr(scheduler.subprocess, 'Popen', Worker)
    for episode in [0, 1, 2]:
        folder = trajectory_folder(tmp_path, 'fixture', episode, [0.]*4)
        write_json(folder/'complete.json', dict(valid=True, elapsed_s=.1))
        write_json(folder/'scores.json', dict(conditions=dict(real=dict(valid=True, objective_mm=physical([0.]*4, episode)))))
    scheduler.supervise(tmp_path, [4])
    status = read_json(tmp_path/'status.json')
    assert status['status'] == 'complete'
    assert status['completed_searches'] == 3
    assert status['completed_budget_evaluations'] == 9
    rows = [read_json(p) for p in (tmp_path/'results').rglob('*.json')]
    assert {(r['method'], r['budget']) for r in rows} == {(m, b) for m in methods for b in [32, 64, 96]}
    for row in rows:
        assert row['logical_episode_evaluations'] == row['budget']*2
        assert row['episode_evaluations_including_start'] == (row['budget']+1)*2
        assert row['surrogate_candidate_vectors'] == (row['budget']*4 if row['method'] != 'physical_uniform' else 0)
        assert row['query_scoring_started_after_selection'] is True
        assert row['support_candidate_losses_mm'] == [physical(row['candidate_q'], e) for e in [0, 1]]
        assert row['query_objective_mm'] == physical(row['selected_q'], 2)
    assert query_calls and surrogate_calls
    assert all(r['support'] == [0, 1] for r in surrogate_calls)
