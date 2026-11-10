"""Read only particles referenced by fixed observation bindings.

The original eight-neighbor interpolation order and host float precision stay
unchanged. Deduplication changes only the positions requested from the runtime.
"""
import numpy as np


class BoundObservation:
    def __init__(self, data, points=None, compact=False):
        points = slice(None) if points is None else points
        binding = np.asarray(data['binding'][points])
        self.initial = data['initial'][binding]
        self.weights = data['weights'][points]
        self.origin = data['tracks'][0, points]
        if compact:
            self.particle_ids, inverse = np.unique(data['saved_ids'][binding], return_inverse=True)
            self.binding = inverse.reshape(binding.shape)
        else:
            self.particle_ids = data['saved_ids']
            self.binding = binding

    def capture(self, runtime):
        p = runtime.positions(self.particle_ids)
        return self.origin + np.sum((p[self.binding] - self.initial) * self.weights[..., None], axis=1)
