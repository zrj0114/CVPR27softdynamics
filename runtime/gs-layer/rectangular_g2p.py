"""Original G2P with signed storage indices and per-axis allocation bounds."""
import warp as wp
from material_contact_kernels import MPMStateStruct, MPMModelStruct
from mpm_utils import compute_dweight, update_cov_differentiable

@wp.kernel
def g2p_rectangular(
    state: MPMStateStruct, next_state: MPMStateStruct, model: MPMModelStruct, dt: float, offset: wp.vec3i
):
    """
    Compute:
        next_state.particle_v, next_state.particle_x, next_state.particle_C, next_state.particle_F_trial
    """
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(wp.floor(grid_pos[0] - 0.5))
        base_pos_y = wp.int(wp.floor(grid_pos[1] - 0.5))
        base_pos_z = wp.int(wp.floor(grid_pos[2] - 0.5))
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        # new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_C = wp.mat33(new_v, new_v, new_v)
        
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i + offset[0]
                    iy = base_pos_y + j + offset[1]
                    iz = base_pos_z + k + offset[2]
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = (
                        new_v + grid_v * weight
                    )  # TODO, check gradient from static loop
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        next_state.particle_v[p] = new_v

        # add clip here:
        new_x = state.particle_x[p] + dt * new_v
        dx = 1.0 / model.inv_dx
        a_min = wp.vec3(float(2-offset[0]), float(2-offset[1]), float(2-offset[2])) * dx
        a_max = wp.vec3(float(model.grid_dim_x-offset[0]-2), float(model.grid_dim_y-offset[1]-2), float(model.grid_dim_z-offset[2]-2)) * dx

        new_x_clamped = wp.vec3(
            wp.clamp(new_x[0], a_min[0], a_max[0]),
            wp.clamp(new_x[1], a_min[1], a_max[1]),
            wp.clamp(new_x[2], a_min[2], a_max[2]),
        )
        next_state.particle_x[p] = new_x_clamped

        # next_state.particle_x[p] = new_x

        next_state.particle_C[p] = new_C

        I33_1 = wp.vec3(1.0, 0.0, 0.0)
        I33_2 = wp.vec3(0.0, 1.0, 0.0)
        I33_3 = wp.vec3(0.0, 0.0, 1.0)
        I33 = wp.mat33(I33_1, I33_2, I33_3)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        next_state.particle_F_trial[p] = F_tmp

        if 0:
            update_cov_differentiable(state, next_state, p, new_F, dt)

