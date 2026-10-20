import numpy as np
import pytest
import torch

from softdynamics.data import normalize_episode
from softdynamics.inference import validate_input
from softdynamics.training import fit


def test_geometry_normalization_and_missing_gripper():
    rng = np.random.default_rng(1)
    initial = rng.normal(size=(128, 3))*.01
    tracks = np.broadcast_to(initial, (32, 128, 3)).copy()
    poses = np.broadcast_to(np.eye(4), (32, 1, 4, 4)).copy()
    poses[3] = np.nan
    data = normalize_episode(tracks, np.ones((32, 128), dtype=bool), initial,
        initial-initial.mean(0), np.broadcast_to(np.eye(3), (128, 3, 3)), poses,
        np.ones((32, 1))*.02, np.arange(32)/30, point_ids=np.arange(128))
    validate_input(data)
    assert not data['actions'][:, 14:].any()
    assert not data['actions'][3].any()
    np.testing.assert_allclose(data['positions'][0].mean(0), 0, atol=1e-7)


def test_calibration_excludes_validation_and_missing_labels():
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 580))
    S = np.abs(rng.normal(size=(10, 4)))
    valid = np.ones_like(S, dtype=bool)
    valid[:2, 0] = False
    S[:2, 0] = np.nan
    groups = np.repeat(np.arange(5).astype(str), 2)
    state, _ = fit(X, S, valid, groups, np.arange(6), [6, 7], epochs=2, patience=1)
    changed = S.copy()
    changed[6:] = 10000
    other, _ = fit(X, changed, valid, groups, np.arange(6), [6, 7], epochs=2, patience=1)
    np.testing.assert_array_equal(state['tau'], other['tau'])
    assert not state['ready'][0]  # Only two entire training episodes have labels on axis 0.
    with pytest.raises(ValueError, match='disjoint'):
        fit(X, S, valid, groups, [0, 2, 4], [1, 6], epochs=1)
