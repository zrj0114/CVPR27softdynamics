import numpy as np
import pytest
import torch

from softdynamics.elastic_downstream import bounds, project, simulation_coordinates, tensor_parameters, totals, wide_key
from softdynamics.joint_acceleration import normalized_weights, weighted_parameter_gradient
from softdynamics.parameters import ParameterSpace


CONFIG=dict(reference_E_pa=3500.,reference_nu=.27,nu_scale=.2,E_range_pa=[1000.,12000.],nu_range=[.08,.42],
    baseline=dict(E_pa=3500.,nu=.27,ground_friction=.853391,velocity_decay_rate_s_inv=.359543,
                  gripper_static_friction=2.15827,gripper_sliding_friction=.770921))
PHYSICS=dict(baseline=CONFIG['baseline'],coordinate_scales=[3500.,.4,.853391,1.])


def test_full_elastic_domain_reaches_original_solver_without_changing_fixed_parameters():
    for q in [*bounds(CONFIG),np.zeros(2),np.array([.3,-.4])]:
        adapter=simulation_coordinates(q,CONFIG,PHYSICS)
        material=ParameterSpace.from_config(PHYSICS).parameters(adapter)
        predicted=tensor_parameters(torch.tensor(q),CONFIG).numpy()
        np.testing.assert_allclose([material['E_pa'],material['nu']],predicted,atol=1e-8)
        assert all(material[k]==v for k,v in CONFIG['baseline'].items() if k not in ('E_pa','nu'))
        assert np.all(adapter[2:]==0)
        assert wide_key(adapter)==wide_key(adapter.copy())
    with pytest.raises(ValueError):wide_key([0.,0.,.01,0.])
    with pytest.raises(ValueError):simulation_coordinates([4.,0.],CONFIG,PHYSICS)


def test_material_derivative_matches_analytic_and_finite_difference():
    q=torch.tensor([.2,-.1],dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(lambda v:tensor_parameters(v,CONFIG),q)
    expected=torch.diag(torch.tensor([3500.*np.exp(.2),.2],dtype=q.dtype))
    torch.testing.assert_close(jac,expected)
    for j in range(2):
        delta=torch.zeros_like(q);delta[j]=1e-6
        finite=(tensor_parameters(q+delta,CONFIG)-tensor_parameters(q-delta,CONFIG))/(2e-6)
        torch.testing.assert_close(jac[:,j],finite,rtol=1e-8,atol=1e-8)


def test_confidence_changes_gradient_direction_without_updating_weights():
    q=torch.tensor([0.,0.],requires_grad=True)
    losses=[3*q[0]+q[1],-q[0]+4*q[1]]
    raw=torch.tensor([[.9,.1],[.1,.9]],requires_grad=True)
    weights=normalized_weights(raw)
    assert not weights.requires_grad
    gradient=weighted_parameter_gradient(losses,q,weights)
    torch.testing.assert_close(gradient,torch.tensor([2.6,3.7]))
    anchor=q.detach();lo=torch.full((2,),-2.);hi=-lo
    uniform=project(anchor-torch.tensor([1.,2.5]),anchor,.25,lo,hi)
    weighted=project(anchor-gradient,anchor,.25,lo,hi)
    assert not torch.allclose(uniform,weighted)
    assert weighted.norm()<=.250001


def test_counters_separate_real_solver_work_from_reuse():
    a=dict(actual_new_rollout=True,isolated_execution_s=7.)
    b=dict(actual_new_rollout=False,isolated_execution_s=11.)
    result=totals([dict(records=[a,b],proposal_seconds=2.)])
    assert result['support_candidate_validations']==1
    assert result['actual_new_episode_rollouts']==1
    assert result['cached_episode_reuses']==1
    assert result['actual_new_solver_seconds']==7.
    assert result['equivalent_full_solver_seconds']==18.


def test_cma_resumes_mid_generation_without_repeating_completed_candidates(tmp_path,monkeypatch):
    from softdynamics import elastic_downstream as downstream
    from softdynamics.io import read_json,write_json
    options=dict(cma_candidates=8,cma_report_budgets=[4,8],cma_population=4,cma_sigma=.2,seed=9101)
    write_json(tmp_path/'protocol.json',dict(elastic_config=CONFIG,inverse=options,checkpoints={}))
    write_json(tmp_path/'physics/plan.json',dict(config=PHYSICS))
    write_json(tmp_path/'baselines'/'test.json',dict(support=[0],query=[1],mean_mm=2.,losses_mm=[2.],
        records=[dict(isolated_execution_s=1.)]))
    completed=[];attempts=[];crashed=False
    def fake_physics(run,obj,episodes,q,device,**kwargs):
        nonlocal crashed
        attempts.append(np.asarray(q).copy())
        if len(completed)==2 and not crashed:
            crashed=True
            raise RuntimeError('simulated process failure before third candidate completed')
        completed.append(np.asarray(q).copy())
        loss=float(np.linalg.norm(q-np.array([.1,.1]))**2)
        return np.array([loss]),[dict(actual_new_rollout=True,isolated_execution_s=1.)]
    monkeypatch.setattr(downstream,'physical_result',fake_physics)
    with pytest.raises(RuntimeError,match='simulated process failure'):
        downstream.cma(tmp_path,'test','cpu')
    downstream.cma(tmp_path,'test','cpu')
    assert len(completed)==8
    assert len({tuple(q) for q in completed})==8
    np.testing.assert_array_equal(attempts[2],attempts[3])
    result=read_json(tmp_path/'inverse/test/physical_cma/complete.json')
    assert result['dimensions']==2
    assert result['actual_new_episode_rollouts']==8
    assert (tmp_path/'inverse/test/physical_cma/selection_4.json').exists()
    assert (tmp_path/'inverse/test/physical_cma/selection_8.json').exists()
