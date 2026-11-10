import copy

import numpy as np
import pytest
import torch

from softdynamics.io import sha256, write_json
from softdynamics.shared_world_material_audit import response_agreement, validate_probe, evaluate_training_materials
from softdynamics.shared_world_control import reference_step_corrections, pose_rollout


def test_physical_audit_rejects_changes_that_would_confuse_material_responses():
    baseline = dict(state=np.zeros((3, 4, 24), dtype=np.float32), pads=np.zeros((3, 4, 13)),
                    times=np.arange(3) * .04, origin_m=np.zeros(3), scale_m=.2,
                    initial_observed_velocity_m_s=np.zeros((4, 3)), theta=np.array([3500, .27, .8, .4]))
    record = dict(snapshot_sha256='same-full-state', frames=3, planned_frames=3, valid=True)
    expected = np.array([4000, .27, .8, .4])
    probe = copy.deepcopy(baseline); probe['theta'] = expected.copy()
    probe['state'][1:, :, :3] += .001
    validate_probe(baseline, probe, record, expected, 'same-full-state', 2)
    changes = [('state', (0, 0, 0), 'initial state'), ('pads', (1, 0, 0), 'controls'),
               ('times', (1,), 'time intervals'), ('theta', (2,), 'material'),
               ('initial_observed_velocity_m_s', (0, 0), 'velocity')]
    for key, index, message in changes:
        corrupted = copy.deepcopy(probe); corrupted[key][index] += .001
        with pytest.raises(ValueError, match=message):
            validate_probe(baseline, corrupted, record, expected, 'same-full-state', 2)
    with pytest.raises(ValueError, match='snapshot'):
        validate_probe(baseline, probe, record, expected, 'different-full-state', 2)


def test_response_audit_distinguishes_direction_magnitude_and_absent_signal():
    actual = np.array([.002, -.003, .004])
    same_direction = response_agreement(2 * actual, actual)
    assert same_direction['cosine'] == pytest.approx(1)
    assert same_direction['norm_ratio'] == pytest.approx(2)
    assert same_direction['relative_error'] == pytest.approx(1)
    assert response_agreement(-actual, actual)['cosine'] == pytest.approx(-1)
    assert response_agreement(actual, np.zeros(3))['cosine'] is None
    with pytest.raises(ValueError, match='finite'):
        response_agreement([np.nan], [1])


def test_step_correction_preserves_candidate_gradients_without_reference_gradients():
    class BiasedDynamics(torch.nn.Module):
        kinematic_only = True
        def forward(self, state, p0, p1, theta, geometry, dt):
            out=state.clone()
            out[...,0]=2*state[...,0]+theta[:,:1]+p1[:,:1,0]
            return out

    world=BiasedDynamics()
    reference=torch.zeros(1,4,2,24,dtype=torch.float64)
    reference[0,:,:,0]=torch.tensor([1.,3.,9.,27.])[:,None]
    reference.requires_grad_(True)
    reference_pads=torch.zeros(1,4,4,13,dtype=torch.float64,requires_grad=True)
    reference_theta=torch.ones(1,4,dtype=torch.float64,requires_grad=True)
    times=torch.arange(4.,dtype=torch.float64)[None]
    correction=reference_step_corrections(world,reference,reference_pads,reference_theta,{},times)
    assert not correction.requires_grad
    theta=reference_theta.detach().clone().requires_grad_(True)
    pads=reference_pads.detach().clone().requires_grad_(True)
    prediction=pose_rollout(world,reference[:,0].detach(),pads,theta,{},times,step_corrections=correction)
    torch.testing.assert_close(prediction,reference,rtol=0,atol=0)
    gradients=torch.autograd.grad(prediction[:,-1,:,0].sum(),(theta,pads,reference,reference_theta,reference_pads),allow_unused=True)
    torch.testing.assert_close(gradients[0],torch.tensor([[14.,0.,0.,0.]],dtype=torch.float64))
    torch.testing.assert_close(gradients[1][0,:,0,0],torch.tensor([0.,8.,4.,2.],dtype=torch.float64))
    assert gradients[2:]==(None,None,None)


@pytest.mark.parametrize('max_horizon',[None,1])
def test_material_input_ablation_recognizes_known_correct_conditional_dynamics(tmp_path, monkeypatch,max_horizon):
    nodes = tmp_path / 'nodes.npz'
    np.savez(nodes, canonical=np.zeros((3, 3)), neighbors=np.array([[1, 2], [0, 2], [0, 1]]),
             surface_mask=np.ones(3, bool), scale_m=1.)
    write_json(tmp_path / 'plan.json', dict(tasks=[], objects=[dict(object='o', split='train',
               nodes=str(nodes), nodes_sha256=sha256(nodes))]))
    data_hash = sha256(tmp_path / 'plan.json'); trajectories = {}
    for index, stiffness in enumerate([3500., 4500., 2500.]):
        identity = f'experience/material_0/o/episode_0/sample_{index:03d}'
        folder = tmp_path / identity; folder.mkdir(parents=True)
        state = np.zeros((4, 3, 24), np.float32)
        state[:, :, 0] = np.arange(4)[:, None] * stiffness / 3500 * .001
        pads = np.zeros((4, 4, 13), np.float32); pads[:, :2, 12] = 1
        np.savez(folder / 'trajectory.npz', state=state, pads=pads, times=np.arange(4) / 30,
                 phase=np.ones(4), theta=np.array([stiffness, .27, .8, .4], np.float32),
                 origin_m=np.zeros(3), scale_m=1., initial_observed_velocity_m_s=np.zeros((3, 3)))
        digest = sha256(folder / 'trajectory.npz'); trajectories[identity] = digest
        write_json(folder / 'complete.json', dict(schema='softdynamics.material-experience.v1',
            valid=True, full_rollout=True, source_data_plan_sha256=data_hash, nodes_sha256=sha256(nodes),
            trajectory_sha256=digest, task=dict(id=identity, object='o', split='train', variant=index),
            snapshot_sha256='same', material_sample_index=index, policy_training_eligible=False))
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, dict(data_plan_sha256=data_hash, trajectories=trajectories))
    model_path = tmp_path / 'known_dynamics.pt'; model_path.write_text('Test uses analytical dynamics')

    class KnownDynamics(torch.nn.Module):
        direct_displacement = True

        def forward(self, state, pads0, pads1, theta, geometry, dt):
            nxt = state.clone(); nxt[..., 0] += theta[:, :1] / 3500 * .001
            return nxt

    monkeypatch.setattr('softdynamics.shared_world_material_audit.load_model',
                        lambda *args: (KnownDynamics(), dict(step=1, data_plan_sha256=data_hash)))
    result = evaluate_training_materials(tmp_path, manifest, {'correct': model_path}, tmp_path / 'audit', 'cpu',max_horizon)
    row = result['rows'][0]
    assert row['horizon']==(3 if max_horizon is None else max_horizon)
    assert row['correct_material_mae_mm'] < 1e-6
    assert row['benefit_over_nominal_mm'] > (.3 if max_horizon is None else .1)
    assert row['benefit_over_shuffled_mm'] > (.3 if max_horizon is None else .1)
    assert row['material_response']['cosine'] == pytest.approx(1)
    assert row['material_response']['relative_error'] < 1e-6
    assert result['training_fit_diagnostic'] and not result['held_out_generalization_evaluation']
