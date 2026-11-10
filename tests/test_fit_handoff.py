import copy
import numpy as np
import pytest
from softdynamics.fit_handoff import audit_weighted_trace, check_files, load_handoff
from softdynamics.parameters import PARAMETERS
from softdynamics.io import sha256, write_json


def test_handoff_audits_each_parameter_column_and_rejects_averaging():
    weights=np.array([[.1,.3,.8,.4],[.7,.2,.1,.2],[.2,.5,.1,.4]])
    losses=np.array([[1.,5.,9.],[2.,7.,4.]])
    selection=dict(trace=[dict(parameter=key,support_losses_mm=losses.tolist(),
                       costs_mm=(losses@weights[:,j]/weights[:,j].sum()).tolist())
                          for j,key in enumerate(PARAMETERS)])
    assert audit_weighted_trace(selection,weights)['checked_candidates']==8
    with pytest.raises(ValueError,match='trace mismatch'):
        audit_weighted_trace(selection,np.broadcast_to(weights.mean(1)[:,None],weights.shape))
    changed=copy.deepcopy(selection);changed['trace'][2]['costs_mm'][0]+=.01
    with pytest.raises(ValueError,match='trace mismatch'):
        audit_weighted_trace(changed,weights)


def test_handoff_rejects_changed_frozen_source(tmp_path):
    source=tmp_path/'selection.json';source.write_text('{"selected_q":[0,0,0,0]}')
    files={str(source):sha256(source)};check_files(files)
    source.write_text('{"selected_q":[1,0,0,0]}')
    with pytest.raises(ValueError,match='changed'):
        check_files(files)


def test_handoff_rejects_missing_parameter_confidence_column():
    with pytest.raises(ValueError,match='independent confidence'):
        audit_weighted_trace(dict(trace=[]),np.zeros((3,4)))


@pytest.mark.parametrize('accepted',[True,False])
def test_handoff_uses_frozen_acceptance_decision_and_authenticated_confidence(tmp_path,accepted):
    fit=tmp_path/'fit'; workspace=tmp_path/'asset_workspace'
    checkpoint=tmp_path/'confidence.pt';checkpoint.write_bytes(b'local fixture checkpoint')
    asset=workspace/'asset/asset.json';write_json(asset,dict(payload_sha256={}))
    entries=[]
    for e in range(4):
        initial=workspace/f'initial_states/episode_{e}.npz';initial.parent.mkdir(exist_ok=True)
        initial.write_bytes(f'initial {e}'.encode())
        actions=workspace/f'queries/episode_{e}/actions.npz';actions.parent.mkdir(parents=True)
        actions.write_bytes(f'actions {e}'.encode())
        row=workspace/f'row_{e}.json';write_json(row,{})
        write_json(workspace/f'queries/episode_{e}/episode.json',{})
        write_json(workspace/f'initial_states/episode_{e}.json',{})
        entries.append(dict(episode=e,workspace=str(workspace),asset_sha256=sha256(asset),
                            initial_sha256=sha256(initial),actions_sha256=sha256(actions),row=str(row)))
    cfg=dict(baseline=dict(E_pa=4000.,nu=.3,ground_friction=.8,velocity_decay_rate_s_inv=.2,
                          gripper_static_friction=2.,gripper_sliding_friction=.7),
             coordinate_scales=[1000.,.1,.5,.2])
    write_json(fit/'plan.json',dict(schema='softdynamics.block-cma-recovery.v1',config=cfg,
        checkpoint=str(checkpoint),model_sha256=sha256(checkpoint),runtime=str(fit/'runtime'),
        objects=[dict(object='fixture',support=[0,1,2],query=[3],episodes=entries)]))
    weights=np.array([[.1,.3,.8,.4],[.7,.2,.1,.2],[.2,.5,.1,.4]])
    losses=np.array([[1.,5.,9.],[2.,7.,4.]])
    write_json(fit/'weights/fixture.json',dict(model_sha256=sha256(checkpoint),teacher_scores_used=False,
        support=[0,1,2],query=[3],contexts=[dict(id='real_start0',start=[0.]*4,
                                             weights=dict(learned=weights))]))
    candidate=[.1]*4;selected=candidate if accepted else [0.]*4
    write_json(fit/'selections/fixture/real_start0/96/learned.json',dict(
        object='fixture',method='learned',context='real_start0',budget=96,condition=dict(truth='real'),
        query_observations_used=False,start=[0.]*4,candidate_q=candidate,selected_q=selected,
        accepted=accepted,selection_frozen_at=123.,trace=[dict(parameter='E_pa',
            support_losses_mm=losses,costs_mm=losses@weights[:,0])]))
    handoff=load_handoff(fit,'fixture')
    assert handoff['selected_q']==selected
    assert handoff['parameters']['E_pa']==(4100. if accepted else 4000.)
    assert handoff['parameters']['gripper_sliding_friction']==.7
    assert handoff['query_observations_used_for_fit'] is False
    checkpoint.write_bytes(b'different confidence model')
    with pytest.raises(ValueError,match='checkpoint mismatch'):
        load_handoff(fit,'fixture')
