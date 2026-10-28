"""Finite corrected-pad colliders in pad-center coordinates.

The impulse law is the existing unilateral Coulomb law. Geometry comes from
the delivered 64 x 25 mm rectangles; it is not shifted to match object motion.
All grid material can meet either side, without selecting future contact IDs.
"""
import warp as wp
from material_contact_kernels import MaterialContact, MPMStateStruct, MPMModelStruct, contact_delta


@wp.kernel
def accumulate_pad_contact(t: float, dt: float, state: MPMStateStruct,
                           model: MPMModelStruct, c: MaterialContact):
    i, j, k = wp.tid()
    if state.grid_m[i, j, k] > 1.e-12:
        node = wp.vec3(float(i), float(j), float(k))/c.inv_dx
        free = state.grid_v_out[i, j, k]
        delta = wp.vec3(0.)
        weight = float(0.)
        for side in range(2):
            if c.enabled[side] != 0:
                tau = wp.max(0., t-c.frame_time)
                tr = c.transforms[side]
                q = wp.transform_get_rotation(tr)
                omega = c.angular[side]
                speed = wp.length(omega)
                if speed > 1.e-10:
                    q = wp.quat_from_axis_angle(omega/speed, speed*tau)*q
                origin = wp.transform_get_translation(tr)+tau*c.linear[side]
                local = wp.quat_rotate_inv(q, node-origin)
                sign = float(1.)
                if side == 1:
                    sign = -1.
                distance = sign*local[0]
                if wp.abs(local[1]) <= c.footprint_y and wp.abs(local[2]) <= c.footprint_z and distance <= 0. and distance >= -c.radius:
                    normal = wp.quat_rotate(q, wp.vec3(sign, 0., 0.))
                    controller = c.linear[side]+wp.cross(omega, node-origin)
                    relative = free-controller
                    if wp.dot(relative, normal) < 0.:
                        delta = delta+contact_delta(relative, normal, c.contact_friction, c.sliding_friction)
                        weight = weight+1.
        c.momentum[i, j, k] = c.momentum[i, j, k]+delta
        c.weight[i, j, k] = c.weight[i, j, k]+weight
