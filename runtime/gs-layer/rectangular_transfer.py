"""Rectangular storage and signed grid indices; physical coordinates are unchanged.

Force expressions are derived from deterministic_transfer.py. Storage offsets
are integers and never added to particle positions or pad transforms.
"""
import warp as wp
from warp.utils import radix_sort_pairs
from material_contact_kernels import MPMStateStruct, MPMModelStruct
from mpm_utils import compute_dweight, compute_stress_from_F_trial
from deterministic_transfer import DeterministicTransfer as BaseTransfer
from rectangular_g2p import g2p_rectangular
import rectangular_active_grid as active
wp.set_module_options({'enable_backward': False, 'max_unroll': 1})
ACTIVE_INDEX_THREADS = 32768

@wp.struct
class CellIndex:
    keys: wp.array(dtype=wp.int64)
    ids: wp.array(dtype=int)
    begin: wp.array(dtype=int)
    end: wp.array(dtype=int)
    marked: wp.array(dtype=int)
    nodes: wp.array(dtype=int)
    count: wp.array(dtype=int)
    invalid: wp.array(dtype=int)
    n: int
    gx: int
    gy: int
    gz: int
    ox: int
    oy: int
    oz: int

@wp.kernel
def particle_keys(state: MPMStateStruct, model: MPMModelStruct, index: CellIndex):
    p = wp.tid()
    g = state.particle_x[p] * model.inv_dx
    x = int(wp.floor(g[0] - .5)) + index.ox; y = int(wp.floor(g[1] - .5)) + index.oy; z = int(wp.floor(g[2] - .5)) + index.oz
    if x < 0 or y < 0 or z < 0 or x + 2 >= index.gx or y + 2 >= index.gy or z + 2 >= index.gz:
        wp.atomic_max(index.invalid, 0, 1)
        index.keys[p] = wp.int64(index.gx)*wp.int64(index.gy)*wp.int64(index.gz)*wp.int64(index.n)
    else:
        cell = (x * index.gy + y) * index.gz + z
        index.keys[p] = wp.int64(cell) * wp.int64(index.n) + wp.int64(p)
    index.ids[p] = p

@wp.kernel
def cell_ranges_and_nodes(index: CellIndex):
    s = wp.tid()
    cell = int(index.keys[s] / wp.int64(index.n))
    if cell < index.gx * index.gy * index.gz:
        first = bool(s == 0)
        if s > 0:
            first = int(index.keys[s-1] / wp.int64(index.n)) != cell
        last = bool(s == index.n-1)
        if s < index.n-1:
            last = int(index.keys[s+1] / wp.int64(index.n)) != cell
        if last:
            index.end[cell] = s + 1
        if first:
            index.begin[cell] = s
            x = cell / (index.gy * index.gz)
            y = (cell / index.gz) % index.gy
            z = cell % index.gz
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        node = ((x+i)*index.gy+y+j)*index.gz+z+k
                        old = wp.atomic_max(index.marked, node, 1)
                        if old == 0:
                            slot = wp.atomic_add(index.count, 0, 1)
                            index.nodes[slot] = node

@wp.func
def p2g_contribution(state: MPMStateStruct, model: MPMModelStruct, dt: float,
                     p: int, i: int, j: int, k: int):
    # The expression and matrix layout below are copied from p2g_apic_with_stress.
    stress = state.particle_stress[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    base = wp.vec3(float(int(wp.floor(grid_pos[0]-.5))), float(int(wp.floor(grid_pos[1]-.5))), float(int(wp.floor(grid_pos[2]-.5))))
    fx = grid_pos - base
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(wp.cw_mul(wa, wa)*.5,
        wp.vec3(0.,0.,0.)-wp.cw_mul(wb,wb)+wp.vec3(.75), wp.cw_mul(wc,wc)*.5)
    dw = wp.mat33(fx-wp.vec3(1.5), -2.0*(fx-wp.vec3(1.0)), fx-wp.vec3(.5))
    dpos = (wp.vec3(float(i),float(j),float(k))-fx)*model.dx
    weight = w[0,i]*w[1,j]*w[2,k]
    dweight = compute_dweight(model,w,dw,i,j,k)
    C = state.particle_C[p]
    C = (1.0-model.rpic_damping)*C + model.rpic_damping/2.0*(C-wp.transpose(C))
    if model.rpic_damping < -.001:
        C = wp.mat33(0.,0.,0.,0.,0.,0.,0.,0.,0.)
    elastic_force = -state.particle_vol[p]*stress*dweight
    momentum = weight*state.particle_mass[p]*(state.particle_v[p]+C*dpos)+dt*elastic_force
    return wp.vec4(momentum[0],momentum[1],momentum[2],weight*state.particle_mass[p])

@wp.kernel
def gather_p2g(state: MPMStateStruct, model: MPMModelStruct, dt: float, index: CellIndex):
    slot = wp.tid()
    if slot < index.count[0]:
        node = index.nodes[slot]
        x = node / (index.gy*index.gz); y = (node/index.gz)%index.gy; z = node%index.gz
        total = wp.vec4(0.)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bx=x-i; by=y-j; bz=z-k
                    if bx>=0 and by>=0 and bz>=0:
                        cell=(bx*index.gy+by)*index.gz+bz
                        start=index.begin[cell]
                        if start>=0:
                            for s in range(start,index.end[cell]):
                                p=index.ids[s]
                                if state.particle_selection[p]==0:
                                    total=total+p2g_contribution(state,model,dt,p,i,j,k)
        state.grid_v_in[x,y,z]=wp.vec3(total[0],total[1],total[2])
        state.grid_m[x,y,z]=total[3]

@wp.kernel
def reset_used_index(index: CellIndex):
    # Every occupied base cell is also its own (0,0,0) support node, so
    # clearing the previous support union resets ALL previously written
    # begin entries and marks. end is read only when begin is nonnegative.
    for slot in range(wp.tid(),index.count[0],ACTIVE_INDEX_THREADS):
        node=index.nodes[slot]
        index.begin[node]=-1
        index.marked[node]=0

class DeterministicTransfer(BaseTransfer):
    def __init__(self, model, device, offset):
        n = model.n_particles
        shape = (model.grid_dim_x, model.grid_dim_y, model.grid_dim_z)
        volume = shape[0] * shape[1] * shape[2]
        self.device = device
        index = CellIndex(); index.n = n
        index.gx, index.gy, index.gz = shape
        index.ox, index.oy, index.oz = map(int, offset)
        index.keys = wp.empty(2*n, dtype=wp.int64, device=device)
        index.ids = wp.empty(2*n, dtype=int, device=device)
        for name in ('begin', 'end', 'marked'):
            setattr(index, name, wp.empty(volume, dtype=int, device=device))
        index.nodes = wp.empty(min(27*n, volume), dtype=int, device=device)
        index.count = wp.zeros(1, dtype=int, device=device)
        index.invalid = wp.zeros(1, dtype=int, device=device)
        self.index = index; self.offset = wp.vec3i(*map(int, offset))
        self.active_grid = self.sparse_index_reset = self.active_p2g = True
        self._index_initialized = False
        self.p2g_threads = max(512*256, ((n+255)//256)*256)

    def prepare(self, state, model):
        index = self.index
        if self._index_initialized:
            wp.launch(reset_used_index, dim=ACTIVE_INDEX_THREADS, inputs=[index], device=self.device)
        else:
            index.begin.fill_(-1); index.marked.zero_()
        index.count.zero_()
        wp.launch(particle_keys, dim=index.n, inputs=[state, model, index], device=self.device)
        radix_sort_pairs(index.keys, index.ids, index.n)
        wp.launch(cell_ranges_and_nodes, dim=index.n, inputs=[index], device=self.device)
        self._index_initialized = True

    def p2g(self, state, model, dt):
        wp.launch(gather_p2g, dim=self.index.nodes.size, inputs=[state, model, dt, self.index],
                  device=self.device, max_blocks=self.p2g_threads//256)

    def step(self, solver, model, state, next_state, dt):
        from pad_contact_kernels import accumulate_pad_contact
        from material_contact_kernels import material_grid_boundary
        if (solver.particle_velocity_modifiers or solver.pre_p2g_operations or solver.impulse_params or
                any(bc is not None for bc in solver.modify_bc) or not solver.grid_postprocess or
                solver.grid_postprocess[-1] is not material_grid_boundary or
                any(k not in (accumulate_pad_contact, material_grid_boundary) for k in solver.grid_postprocess)):
            raise ValueError('Rectangular transfer requires the selected finite-pad contact pipeline')
        self.prepare(state, model)
        args = [self.index.nodes, self.index.count, self.index.gy, self.index.gz, self.offset]
        def launch(kernel, inputs):
            wp.launch(kernel, dim=active.ACTIVE_THREADS, inputs=inputs+args, device=self.device)
        launch(active.active_zero_grid, [state, model])
        wp.launch(compute_stress_from_F_trial, dim=model.n_particles, inputs=[state, model, dt], device=self.device)
        self.p2g(state, model, dt)
        launch(active.active_grid_normalization_and_gravity, [state, model, dt])
        if model.grid_v_damping_scale < 1.:
            launch(active.active_add_damping_via_grid, [state, model.grid_v_damping_scale])
        mapping = {accumulate_pad_contact: active.active_accumulate_pad_contact,
                   material_grid_boundary: active.active_material_grid_boundary}
        for kernel, contact in zip(solver.grid_postprocess, solver.collider_params):
            launch(mapping[kernel], [solver.time, dt, state, model, contact])
        wp.launch(g2p_rectangular, dim=model.n_particles, inputs=[state, next_state, model, dt, self.offset], device=self.device)
        solver.time += dt
