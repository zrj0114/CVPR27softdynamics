"""Fixed physical scales; query coordinates are not all percentages."""
from dataclasses import dataclass

import numpy as np

PARAMETERS = ('E_pa', 'nu', 'ground_friction', 'velocity_decay_rate_s_inv')


@dataclass
class ParameterSpace:
    base: dict
    scales: np.ndarray

    @classmethod
    def from_config(cls, config):
        scales = np.asarray(config['coordinate_scales'], dtype=float)
        if scales.shape != (4,) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Four positive, finite coordinate scales are required')
        result = cls(dict(config['baseline']), scales)
        result.parameters(np.zeros(4))
        return result

    def parameters(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError('Expected four finite normalized parameter coordinates')
        p = dict(self.base)
        for j, key in enumerate(PARAMETERS):
            p[key] = round(float(self.base[key] + self.scales[j] * q[j]), 8)
        if not (p['E_pa'] > 0 and 0 < p['nu'] < .49 and
                p['ground_friction'] >= 0 and p['velocity_decay_rate_s_inv'] >= 0):
            raise ValueError('Query outside the supported physical domain')
        return p

    def coordinate(self, parameters):
        return np.array([(parameters[k] - self.base[k]) / self.scales[j]
                         for j, k in enumerate(PARAMETERS)])
