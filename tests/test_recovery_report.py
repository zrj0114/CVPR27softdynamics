from softdynamics.io import read_json, write_json
from softdynamics.recovery_report import paired_summary, synthetic_groups, update_model_index


def result(context, method, error=0., query=1., valid=True):
    return dict(object='test-object',context=context,method=method,
                condition=dict(truth='center',noise_mm=2.),selected_q=[error,0.,0.,0.],
                parameter_error_steps=error,per_parameter_error_steps=[error]*4,exact=error==0,
                query_valid=valid,query_objective_mm=query if valid else None)


def test_paired_report_preserves_query_failures_and_uses_identical_contexts():
    rows=[result('a','uniform',query=1.),result('a','learned',query=2.),result('a','shuffled',query=3.),
          result('b','uniform',query=100.),result('b','learned',valid=False),result('b','shuffled',query=80.),
          result('not-complete','uniform',query=900.)]
    group,=synthetic_groups(rows,['uniform','learned','shuffled'])
    assert group['contexts']==2 and group['common_query_valid']==1
    assert group['methods']['uniform']['query_objective_mm_common_valid']==1.
    assert group['methods']['learned']['query_objective_mm_common_valid']==2.
    assert group['methods']['learned']['query_valid']==1
    paired=group['paired']['learned']
    assert paired['pairs']==2 and paired['method_only_query_failures']==1
    assert paired['paired_query_delta_mm']==1.


def test_parameter_error_and_query_outcomes_are_reported_separately():
    rows=[result('a','uniform',error=1.,query=1.),result('a','learned',error=0.,query=2.),
          result('b','uniform',error=0.,query=2.),result('b','learned',error=1.,query=1.),
          result('c','uniform',error=1.,valid=False),result('c','learned',error=1.,valid=False)]
    summary=paired_summary(rows,'learned')
    assert (summary['parameter_wins'],summary['parameter_ties'],summary['parameter_losses'])==(1,1,1)
    assert summary['parameter_error_delta']==0.
    assert summary['paired_query_valid']==2 and summary['paired_query_delta_mm']==0.
    assert summary['same_parameters']==1
    assert summary['uniform_query_failures']==summary['method_query_failures']==1


def test_partial_recovery_does_not_mark_full_model_evaluation_complete(tmp_path):
    pretrain=tmp_path/'pretrain';root=tmp_path/'recovery'
    write_json(root/'plan.json',dict(pretrain=str(pretrain)))
    write_json(pretrain/'latest_model.json',dict(model_sha256='frozen',checkpoint='model.pt',status='complete'))
    report=dict(model_sha256='frozen',status='complete',completed_trials=87,planned_trials=348,
                object_status={'rabbit':dict(stage='complete'),'bear':dict(stage='numerically_unavailable')},
                audit=dict(passed=True),updated_at=1.)
    update_model_index(root,report)
    index=read_json(pretrain/'latest_model.json')
    assert index['physical_recovery_evaluated'] is False
    assert index['recovery_evaluated_objects']==['rabbit']
    assert index['checkpoint']=='model.pt' and index['status']=='complete'
    report['completed_trials']=348
    update_model_index(root,report)
    assert read_json(pretrain/'latest_model.json')['physical_recovery_evaluated'] is False
    report['object_status']['bear']['stage']='complete'
    update_model_index(root,report)
    assert read_json(pretrain/'latest_model.json')['physical_recovery_evaluated'] is True
