import numpy as np
import pytest
import torch

from softdynamics.world_data import augmented_actions, validate_splits
from softdynamics.world_model import (MaterialDynamics, fit_statistics, normalize_state,
                                     physical_state, position_error_mm, rollout)
from softdynamics.world_training import action_gradient_check, baselines, pipeline
from softdynamics.world_inference import predict
from softdynamics.io import read_json, sha256, write_json


def fixture(position_update='independent'):
    torch.manual_seed(19)
    N, T = 8, 12
    rng = np.random.default_rng(19)
    canonical = rng.normal(size=(N, 3)).astype('f')*.1
    neighbors = np.stack([np.roll(np.arange(N), -i-1)[:3] for i in range(N)])
    statistics = dict(mean=np.zeros(24), std=np.ones(24), output_scale=np.ones(24)*.01,
                      loss_scale=np.ones(24))
    model = MaterialDynamics(canonical, neighbors, statistics, hidden=16, layers=1, position_update=position_update)
    state = torch.zeros(T, N, 24); state[..., :3] = torch.tensor(canonical)
    pads = torch.zeros(T, 4, 13); pads[:, :2, 12] = 1
    pads[:, :2, 3:12] = torch.eye(3).reshape(9)
    pads[:, 0, 0] = torch.linspace(0, .01, T)
    t = dict(state=state, pads=pads, times=torch.arange(T)/30, eligibility=torch.ones(N, 4),
             ground_y=torch.tensor(-.5), id='synthetic', split='train')
    return model, t


def test_episode_split_cannot_leak_across_variants():
    validate_splits(dict(train=[0, 2], validation=[1], test=[6, 9]))
    with pytest.raises(ValueError):
        validate_splits(dict(train=[0, 2], validation=[1], test=[2]))
    with pytest.raises(ValueError):
        validate_splits(dict(train=[0, 0], validation=[1], test=[2]))


def test_action_augmentation_preserves_reference_aperture_and_missingness():
    poses = np.broadcast_to(np.eye(4), (9, 2, 2, 4, 4)).copy()
    poses[:, :, 1, 0, 3] = .06
    valid = np.ones((9, 2, 2), bool); valid[4, 1, 0] = False
    poses[4, 1, 0] = np.nan
    result = augmented_actions(poses, valid, np.arange(9)/30, 2, 6, 'translate_positive', .004)
    np.testing.assert_allclose(result[:3], poses[:3])
    np.testing.assert_allclose(result[6:], poses[6:], atol=1e-16)
    np.testing.assert_allclose(result[:, 0, 1, :3, 3]-result[:, 0, 0, :3, 3],
                               poses[:, 0, 1, :3, 3]-poses[:, 0, 0, :3, 3])
    assert np.isnan(result[4, 1, 0]).all()
    assert np.linalg.norm(result[4, 0, 0, :3, 3]) == pytest.approx(.004)


def test_physical_units_roundtrip_and_material_node_mm():
    x = np.random.default_rng(3).normal(size=(7, 8, 24)).astype('f')
    origin = np.array([.2, -.5, .1]); scale = .2
    normalized = normalize_state(x, origin, scale)
    np.testing.assert_allclose(physical_state(normalized, origin, scale), x, atol=2e-6)
    pred = normalized.copy(); pred[..., 0] += .015
    np.testing.assert_allclose(position_error_mm(pred, normalized, scale), 3., atol=1e-4)


def test_normalizers_refuse_validation_or_test_data():
    t = dict(state=np.zeros((5, 8, 24), 'f'), times=np.arange(5)/30,
             id='e0', split='train', scale_m=.2)
    fitted = fit_statistics([t])
    assert fitted['training_ids'] == ['e0']
    for split in ('validation', 'test'):
        with pytest.raises(ValueError):
            fit_statistics([t, {**t, 'split':split, 'state':np.full((5, 8, 24), 1e6)}])


def test_missing_action_endpoint_disables_pad_even_with_nan_storage():
    model, t = fixture()
    p0, p1 = t['pads'][None, 0].clone(), t['pads'][None, 1].clone()
    p1[:, 0, 12] = 0
    kwargs = (t['eligibility'][None], torch.tensor([1/30]), t['ground_y'][None])
    base = model(t['state'][:1], p0, p1, *kwargs)
    p0[:, 0, :12] = float('nan'); p1[:, 0, :12] = float('nan')
    masked = model(t['state'][:1], p0, p1, *kwargs)
    assert torch.isfinite(masked).all()
    torch.testing.assert_close(masked, base)


def test_autoregressive_rollout_does_not_read_future_state():
    model, t = fixture()
    initial = t['state'][:1].clone()
    predict = lambda: rollout(model, initial, t['pads'][None], t['eligibility'][None],
                              t['times'][None], t['ground_y'][None])
    expected = predict()
    t['state'][1:] = float('nan')
    torch.testing.assert_close(predict(), expected)
    assert expected.shape == (1, 11, 8, 24)


@pytest.mark.parametrize('position_update', ['independent', 'velocity_consistent'])
def test_action_jacobian_matches_finite_difference(position_update):
    model, t = fixture(position_update)
    report = action_gradient_check(model, t, horizon=4)
    assert report['valid'], report
    assert report['teacher_gradient_tested'] is False


def test_constant_velocity_baseline_uses_elapsed_time():
    initial = np.zeros((2, 24), 'f'); initial[:, 3] = .01
    times = np.array([2., 2.1, 2.3])
    predictions = baselines(initial, times)
    np.testing.assert_allclose(predictions['constant_velocity'][:, :, 0], [[.03, .03], [.09, .09]])
    assert (predictions['persistence'][..., :3] == 0).all()


@pytest.mark.parametrize('fit_identity', [None, 'learned-material-selection'])
def test_exported_checkpoint_predicts_from_initial_state_and_actions_only(tmp_path, fit_identity):
    model, t = fixture()
    manifest = dict(nodes_sha256='nodes-identity', asset_sha256='asset-identity')
    if fit_identity:
        manifest['first_contribution_sha256'] = fit_identity
    write_json(tmp_path/'manifest.json', manifest)
    statistics = {key:getattr(model,key).numpy() for key in ('mean','std','output_scale','loss_scale')}
    torch.save(dict(model=model.state_dict(), model_config=dict(hidden=16,layers=1),
                    statistics=statistics, canonical=model.canonical.numpy(), neighbors=model.neighbors.numpy(),
                    manifest_sha256=sha256(tmp_path/'manifest.json')), tmp_path/'best.pt')
    origin=np.array([.1,.2,.3],dtype='f'); scale=.2
    query=dict(initial_state=physical_state(t['state'][0].numpy(),origin,scale),
               pads=t['pads'].numpy(),eligibility=t['eligibility'].numpy(),times=t['times'].numpy(),
               origin_m=origin,scale_m=scale,**manifest)
    np.savez(tmp_path/'query.npz',**query)
    predicted=predict(tmp_path/'best.pt',tmp_path/'query.npz',tmp_path/'prediction.npz','cpu')
    assert predicted.shape==(11,8,24) and np.isfinite(predicted).all()
    ground=torch.tensor([(-.01-origin[1])/scale])
    with torch.no_grad():
        expected=rollout(model,t['state'][:1],t['pads'][None],t['eligibility'][None],t['times'][None],ground)[0].numpy()
    np.testing.assert_allclose(predicted,physical_state(expected,origin,scale),atol=1e-6)
    if fit_identity:
        query['first_contribution_sha256']='same-geometry-wrong-material'
        np.savez(tmp_path/'wrong_material.npz',**query)
        with pytest.raises(ValueError,match='material selection'):
            predict(tmp_path/'best.pt',tmp_path/'wrong_material.npz',tmp_path/'bad_material.npz','cpu')
        query['first_contribution_sha256']=fit_identity
    query['asset_sha256']='wrong-asset'
    np.savez(tmp_path/'bad.npz',**query)
    with pytest.raises(ValueError,match='Wrong fixed asset'):
        predict(tmp_path/'best.pt',tmp_path/'bad.npz',tmp_path/'bad_prediction.npz','cpu')


def test_small_pipeline_writes_checkpoint_frozen_test_and_report(tmp_path):
    """Synthetic software integration fixture, never research-result evidence."""
    model,t=fixture();root=tmp_path/'teacher';root.mkdir()
    np.savez(root/'nodes.npz',canonical=model.canonical.numpy(),neighbors=model.neighbors.numpy(),
             surface_mask=np.arange(8)<6)
    config=dict(splits=dict(train=[0],validation=[1],test=[2]),model=dict(hidden=16,layers=1),
                 training=dict(seed=19,steps=2,batch_size=2,horizon=8,learning_rate=3e-4,
                               evaluate_every=2,validation_windows=2),evaluation_horizons=[1,4,8,16,32])
    tasks=[]
    for split,episode in [('train',0),('validation',1),('test',2)]:
        task=dict(id=f'episode_{episode}/nominal',episode=episode,variant='nominal',split=split,padding=0)
        tasks.append(task);folder=root/'trajectories'/task['id'];folder.mkdir(parents=True)
        origin=np.array([0.,0.,0.],dtype='f');scale=.2
        state=np.repeat(physical_state(t['state'][:1].numpy(),origin,scale),40,axis=0)
        pads=np.repeat(t['pads'][:1].numpy(),40,axis=0)
        np.savez(folder/'trajectory.npz',state=state,pads=pads,eligibility=t['eligibility'].numpy(),
                 times=np.arange(40)/30,origin_m=origin,scale_m=scale,phase=np.ones(40,dtype='int8'))
        write_json(folder/'complete.json',dict(valid=True,task=task,diagnostic_truncation=False,
                    payload_sha256=sha256(folder/'trajectory.npz'),frames=40,particles=8,physics_s=1.,
                    synthetic_test_fixture=True))
    write_json(root/'plan.json',dict(config=config,tasks=tasks,nodes_sha256=sha256(root/'nodes.npz'),
                workspace=str(root/'workspace'),source_sha256={str(root/'workspace/asset/asset.json'):'fixture'}))
    write_json(root/'collection_status.json',dict(status='complete'))
    output=tmp_path/'model';pipeline(root,output,'cpu')
    assert read_json(output/'pipeline_status.json')['status']=='complete'
    assert (output/'best.pt').exists() and (output/'report.md').exists()
    assert (output/'prediction_errors.png').stat().st_size>1000
    report=read_json(output/'evaluation_test.json')
    assert all(row['id']=='episode_2/nominal' for row in report['windows'])
    assert all(row['valid'] for row in report['full_rollouts'])
    assert read_json(output/'manifest.json')['test_read_during_training'] is False


def test_velocity_consistent_integrator_uses_predicted_velocity_and_elapsed_time():
    model,t=fixture('velocity_consistent')
    with torch.no_grad():
        model.decoder[-1].weight.zero_(); model.decoder[-1].bias.zero_()
        model.decoder[-1].bias[0]=.2
    initial=t['state'][:1].clone(); initial[...,3:6]=.03
    ratio=2.
    result=model(initial,t['pads'][None,0],t['pads'][None,1],t['eligibility'][None],
                 torch.tensor([ratio/30]),t['ground_y'][None])
    expected_v=initial[...,3:6].clone();expected_v[...,0]+=.2*.01*ratio
    torch.testing.assert_close(result[...,3:6],expected_v)
    torch.testing.assert_close(result[...,:3],initial[...,:3]+.5*(initial[...,3:6]+expected_v)*ratio)


def test_action_response_compares_with_teacher_not_network_finite_differences(tmp_path):
    from softdynamics.world_training import action_response_report
    state=np.zeros((3,2,24),dtype='f')
    np.savez(tmp_path/'episode_9_nominal_test_prediction.npz',teacher=state,predicted=state)
    for variant,sign in [('translate_positive',1),('translate_negative',-1)]:
        teacher=state.copy();teacher[...,0]=sign*.002
        np.savez(tmp_path/f'episode_9_{variant}_test_prediction.npz',teacher=teacher,predicted=teacher)
    report=action_response_report(tmp_path,'test');row=report['rows'][0]
    assert row['valid'] and row['response_error_mm']==0
    assert row['teacher_response_mm']==pytest.approx(2.)
    assert row['cosine']==pytest.approx(1.)
    assert report['infinitesimal_gradient_claim'] is False
