import hashlib

import numpy as np

from softdynamics.parameters import ParameterSpace
from softdynamics.teacher import generate


class LinearRuntime:
    """Analytically differentiable motion; tests branching independently of MPM."""
    def __init__(self, space, response):
        self.space, self.response = space, response
        self.frame, self.times = 0, np.arange(10)/30
        self.x = np.arange(12, dtype=float).reshape(4, 3)*.01
        self.parameters = space.parameters(np.zeros(4))

    def snapshot(self):
        return dict(x=self.x.copy(), frame=self.frame)

    @staticmethod
    def digest(snapshot):
        return hashlib.sha256(snapshot['x'].tobytes()+str(snapshot['frame']).encode()).hexdigest()

    def restore(self, snapshot, parameters):
        self.x, self.frame = snapshot['x'].copy(), snapshot['frame']
        self.parameters = parameters

    def advance(self, end):
        self.x += (end-self.frame)*np.einsum('ncj,j->nc', self.response,
                                           self.space.coordinate(self.parameters))
        self.frame = end
        return dict(valid=True)

    def positions(self, ids):
        return self.x[ids]

    def state_hashes(self):
        return self.digest(self.snapshot())


def test_teacher_matches_analytic_derivatives_and_restores_state(tmp_path):
    space = ParameterSpace(dict(E_pa=1000., nu=.3, ground_friction=.5,
                               velocity_decay_rate_s_inv=.3), np.array([1000., .4, .5, 1.]))
    response = np.random.default_rng(5).normal(size=(4, 3, 4))*.01
    runtime = LinearRuntime(space, response)
    initial = runtime.x.copy()
    data = dict(tracks=np.broadcast_to(initial, (10, 4, 3)), track_visible=np.ones((10, 4), bool),
                initial=initial, saved_ids=np.arange(4), binding=np.arange(4)[:, None],
                weights=np.ones((4, 1)), frames=np.arange(10), point_ids=np.arange(4))
    spec = dict(windows_per_phase=2, points=4, workpoints=[[0., 0., 0., 0.]],
                radii=[.1, .05], refinement_radii=[.025, .0125], relative_column_tolerance=.25)
    rows = generate(runtime, data, dict(clip_start=3, clip_end=6), space, spec,
                    dict(before=.1, interaction=.45, release=.45), tmp_path/'teacher')
    expected = np.linalg.norm(response.reshape(-1, 4), axis=0)*1000/np.sqrt(4)
    np.testing.assert_allclose(rows[0]['S'], expected, rtol=1e-10)
    assert rows[0]['valid'].all()
    assert all(w['zero_repeat_identical'] for w in rows[0]['windows'])
