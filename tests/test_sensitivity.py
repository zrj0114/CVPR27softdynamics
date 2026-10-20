import numpy as np

from softdynamics.labels import audit_columns, central_column, TrainingCalibration
from softdynamics.search import cycle


def test_unequal_rounded_steps_and_unreliable_labels():
    # Quadratic motion has a known derivative even with asymmetric rounding.
    base = np.array([2., 3.])
    slope = np.array([4., -5.])
    curvature = np.array([7., 11.])
    hp, hm = .10000001, .09999998
    plus = base+hp*slope+hp**2*curvature
    minus = base-hm*slope+hm**2*curvature
    derivative, _ = central_column(base, plus, minus, hp, hm)
    np.testing.assert_allclose(derivative, slope, atol=1e-12)
    fine = np.column_stack([slope, slope, np.zeros(2)])
    coarse = fine.copy()
    coarse[:, 1] *= 2
    result = audit_columns(coarse, fine, np.ones(3, dtype=bool))
    assert result['valid'].tolist() == [True, False, True]
    assert np.isnan(result['S'][1])
    assert result['S'][2] == 0


def test_constant_weight_invariance_and_known_parameter_recovery():
    truth = np.array([.025, -.025, .025, 0])
    directions = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
    def losses(q):
        return (directions*(q-truth)**2).sum(1)
    a, _ = cycle(losses, np.ones((2, 4)), np.full(4, -.025))
    b, _ = cycle(losses, np.broadcast_to([.1, .3, .7, .2], (2, 4)), np.full(4, -.025))
    np.testing.assert_array_equal(a, truth)
    np.testing.assert_array_equal(a, b)


def test_same_interaction_can_be_informative_for_multiple_parameters():
    S = np.array([[10., 10.], [1., 1.], [2., 2.]])
    calibration = TrainingCalibration.fit(S, np.ones_like(S, dtype=bool), [0, 1, 2], [0, 1, 2])
    score = calibration.transform(S, np.ones_like(S, dtype=bool))
    # The implementation returns scores and masks; both large scores can coexist.
    if isinstance(score, tuple):
        score = score[0]
    assert np.all(score[0] > .8)
