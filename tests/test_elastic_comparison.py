from softdynamics.elastic_comparison import compare_object,first_hit


BASE=dict(object='test',mean_mm=10.,support=[0,1],records=[dict(isolated_execution_s=4.),dict(isolated_execution_s=6.)])


def row(value,owned=True,setup=0.,model=0.):
    return dict(selected_support_losses_mm=[value-1,value+1],
        records=[dict(actual_new_rollout=owned,isolated_execution_s=5.)]*2,
        round_wall_seconds=12.,setup_seconds=setup,proposal_seconds=model)


def test_fixed_accuracy_uses_first_qualifying_support_round_and_includes_setup():
    history=[row(9.5,setup=2.,model=1.),row(8.5,owned=False,model=3.),row(7.)]
    hit=first_hit(BASE,history,9.)
    assert hit['candidates']==2
    assert hit['objective_mm']==8.5
    assert hit['episode_rollouts_with_initial']==6
    assert hit['actual_new_episode_rollouts']==2
    assert hit['cached_episode_reuses']==2
    assert hit['equivalent_serial_seconds_with_initial']==36.
    assert hit['observed_method_seconds']==26.


def test_unreached_and_pending_targets_do_not_receive_speedup_claims():
    histories=dict(physical_cma=[row(9.5),row(8.5)],best_uniform=[row(9.2)],latest_confidence=[])
    finished=dict(physical_cma=True,best_uniform=True,latest_confidence=False)
    rows=compare_object(BASE,histories,finished,[.1,.2])
    by_key={(r['method'],r['reduction_fraction']):r for r in rows}
    assert by_key['physical_cma',.1]['status']=='reached'
    assert by_key['best_uniform',.1]['status']=='not_reached'
    assert by_key['latest_confidence',.1]['status']=='pending'
    assert by_key['physical_cma',.2]['cma_to_method_call_ratio'] is None
    assert by_key['best_uniform',.1]['cma_to_method_equivalent_time_ratio'] is None


def test_shared_numerical_preflight_is_included_in_matched_accuracy_cost():
    base=dict(BASE,numerical_preflight_seconds=7.,numerical_preflight_episode_rollouts=2)
    hit=first_hit(base,[row(9.)],9.)
    assert hit['episode_rollouts_with_initial']==6
    assert hit['equivalent_serial_seconds_with_initial']==27.
    assert hit['numerical_preflight_episode_rollouts']==2
