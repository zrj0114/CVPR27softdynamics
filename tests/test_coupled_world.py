import copy
import numpy as np
import pytest
import torch

from softdynamics.coupled_world_model import CoupledWorld, forecast, response_summary
from softdynamics.coupled_world_data import teacher_context, to_device
from softdynamics.coupled_world_train import curriculum


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    rng = np.random.default_rng(4)
    T, N = 24, 12
    x = rng.normal(size=(N, 3)).astype('f')*.1
    positions = x[None] + np.arange(T, dtype='f')[:, None, None]*.001
    visible = np.ones((T, N), bool)
    visible[7, 2] = False
    actions = np.zeros((T, 4, 13), 'f')
    actions[:, :2, :12] = np.c_[np.eye(3), np.zeros(3)].reshape(12)
    actions[:, :2, 12] = 1
    starts = np.array([3, 6, 9, 12, 15, 18])
    label = dict(starts=starts, ends=starts+1, point_ids=np.arange(N)[::-1],
                 J_fine=np.zeros((int(visible[starts+1].sum())*3, 2)))
    windows = [dict(start=int(s), end=int(s+1), phase=('before', 'interaction', 'release')[i//2],
                    weight=(.05, .05, .225, .225, .225, .225)[i]) for i, s in enumerate(starts)]
    visual = dict(positions=positions, valid=visible, times=np.arange(T, dtype='f')/30,
                  actions=actions.reshape(T, 52), point_ids=np.arange(N), scale_m=np.float32(.2),
                  origin_m=np.zeros(3, 'f'))
    context = to_device(teacher_context(visual, label, windows, [3500., .27]), 'cpu')
    model = CoupledWorld(dict(hidden=16, layers=1, heads=2, kinematic_only=True,
        direct_displacement=True, contact_geometry=True, material_dim=2), [.1, .2])
    return model, context, visual, label, windows


def test_joint_matrix_is_psd_and_both_parameters_share_the_head():
    model, c, *_ = fixture()
    score, rho, H = model.confidence([c], torch.zeros(1, 2))
    assert (torch.linalg.eigvalsh(H) >= -1e-7).all()
    assert ((score > 0) & (score < 1)).all() and rho.abs().max() < 1
    (score.sum()+rho.sum()).backward()
    assert model.sequence.weight_ih.grad.abs().sum() > 0
    assert model.world.state_encoder[0].weight.grad.abs().sum() > 0
    assert (model.joint_head[-1][-1].weight.grad.abs().sum(1) > 0).all()


def test_teacher_response_preserves_signed_correlation():
    J = torch.tensor([[1., -2.], [2., -4.]])
    score, rho, H = response_summary(J, torch.ones(2))
    torch.testing.assert_close(rho, torch.tensor(-1.))
    torch.testing.assert_close(H, J.T@J)
    assert score[1] > score[0]


def test_forecast_never_reads_future_teacher_states_and_parameter_gradients_exist():
    model, c, *_ = fixture()
    state = c['states'][:1]
    pads = torch.stack([c['pads0'][0]]*5)[None]
    history = {k: v[:1] for k, v in c['history'].items()}
    geo = {k: v[:1] for k, v in c['geometry'].items()}
    q = torch.zeros(1, 2, requires_grad=True)
    batch = dict(states=state[:, None].expand(-1, 5, -1, -1).clone(), pads=pads,
        times=torch.arange(5)[None]/30, theta=model.theta(q), history=history)
    predicted = forecast(model.world, batch, geo)
    corrupted = dict(batch, states=batch['states'].clone())
    corrupted['states'][:, 1:] = 99999
    torch.testing.assert_close(predicted, forecast(model.world, corrupted, geo), rtol=0, atol=0)
    grad = torch.autograd.grad(predicted[:, -1, :, :3].square().sum(), q)[0]
    assert torch.isfinite(grad).all() and (grad.abs() > 0).all()


def test_teacher_context_has_exact_point_order_masks_and_no_future_state_input():
    _, c, visual, label, windows = fixture()
    changed = copy.deepcopy(visual)
    changed['positions'][-1] += 300
    other = to_device(teacher_context(changed, label, windows, [3500., .27]), 'cpu')
    torch.testing.assert_close(c['states'], other['states'])
    torch.testing.assert_close(c['history']['states'], other['history']['states'])
    np.testing.assert_allclose(c['states'][0, :, :3].numpy(), visual['positions'][3, ::-1])
    assert int(c['target_visible'].sum())*3 == label['J_fine'].shape[0]


def test_local_secants_train_dynamics_without_second_order_autograd():
    model, c, *_ = fixture()
    J = model.local_response(c, torch.zeros(2), torch.tensor([.025, .025]))
    assert J.shape == (int(c['target_visible'].sum())*3, 2)
    assert torch.isfinite(J).all()
    J.square().mean().backward()
    assert model.world.material_encoder[0].weight.grad.abs().sum() > 0
    assert model.world.memory.weight_ih.grad.abs().sum() > 0


@pytest.mark.parametrize('step,horizon,prefix', [(0, 1, 32), (999, 1, 32), (1000, 4, 128),
    (3999, 4, 128), (4000, 8, None), (10000, 16, None), (39999, 16, None)])
def test_curriculum_boundaries(step, horizon, prefix):
    assert curriculum(step) == (horizon, prefix)
