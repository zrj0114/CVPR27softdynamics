"""Forward-only, fixed-order replacements for the two floating-point scatters.

Sort by (base cell, immutable particle index). Each grid node has one writer,
which visits its 27 supporting cells in stencil order, then particles in ID
order. Integer atomics only enumerate distinct active nodes; enumeration order
cannot change a node's sum. Particle contributions retain the original FP32
expressions. Constitutive, contact impulse, ground and G2P kernels are reused.

This is a forward evaluator for the existing derivative-free identification;
sorting/gather topology is not an implementation of the solver's adjoint.
"""
import warp as wp
from warp.utils import radix_sort_pairs
from material_contact_kernels import MPMStateStruct, MPMModelStruct, MaterialContact, scatter_material_velocity
from mpm_utils import (zero_grid, compute_stress_from_F_trial, compute_dweight,
                       grid_normalization_and_gravity, add_damping_via_grid,
                       g2p_differentiable)

wp.set_module_options({'enable_backward': False, 'max_unroll': 1})


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
    grid: int


@wp.kernel
def particle_keys(state: MPMStateStruct, model: MPMModelStruct, index: CellIndex):
    p = wp.tid()
    g = state.particle_x[p] * model.inv_dx
    x = int(g[0] - .5); y = int(g[1] - .5); z = int(g[2] - .5)
    if x < 0 or y < 0 or z < 0 or x + 2 >= index.grid or y + 2 >= index.grid or z + 2 >= index.grid:
        wp.atomic_max(index.invalid, 0, 1)
        index.keys[p] = wp.int64(index.grid)*wp.int64(index.grid)*wp.int64(index.grid)*wp.int64(index.n)
    else:
        cell = (x * index.grid + y) * index.grid + z
        index.keys[p] = wp.int64(cell) * wp.int64(index.n) + wp.int64(p)
    index.ids[p] = p


@wp.kernel
def cell_ranges_and_nodes(index: CellIndex):
    s = wp.tid()
    cell = int(index.keys[s] / wp.int64(index.n))
    if cell < index.grid * index.grid * index.grid:
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
            x = cell / (index.grid * index.grid)
            y = (cell / index.grid) % index.grid
            z = cell % index.grid
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        node = ((x+i)*index.grid+y+j)*index.grid+z+k
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
    base = wp.vec3(float(int(grid_pos[0]-.5)), float(int(grid_pos[1]-.5)), float(int(grid_pos[2]-.5)))
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
        x = node / (index.grid*index.grid); y = (node/index.grid)%index.grid; z = node%index.grid
        total = wp.vec4(0.)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bx=x-i; by=y-j; bz=z-k
                    if bx>=0 and by>=0 and bz>=0:
                        cell=(bx*index.grid+by)*index.grid+bz
                        start=index.begin[cell]
                        if start>=0:
                            for s in range(start,index.end[cell]):
                                p=index.ids[s]
                                if state.particle_selection[p]==0:
                                    total=total+p2g_contribution(state,model,dt,p,i,j,k)
        state.grid_v_in[x,y,z]=wp.vec3(total[0],total[1],total[2])
        state.grid_m[x,y,z]=total[3]


@wp.func
def contact_mass_contribution(state: MPMStateStruct, c: MaterialContact,
                              p: int, i: int, j: int, k: int):
    # Preserve the original contact scatter's expression (distinct from P2G).
    grid=state.particle_x[p]*c.inv_dx
    fx=grid-wp.vec3(float(int(grid[0]-.5)),float(int(grid[1]-.5)),float(int(grid[2]-.5)))
    w=wp.mat33(.5*(1.5-fx[0])*(1.5-fx[0]),.75-(fx[0]-1.)*(fx[0]-1.),.5*(fx[0]-.5)*(fx[0]-.5),
               .5*(1.5-fx[1])*(1.5-fx[1]),.75-(fx[1]-1.)*(fx[1]-1.),.5*(fx[1]-.5)*(fx[1]-.5),
               .5*(1.5-fx[2])*(1.5-fx[2]),.75-(fx[2]-1.)*(fx[2]-1.),.5*(fx[2]-.5)*(fx[2]-.5))
    return state.particle_mass[p]*w[0,i]*w[1,j]*w[2,k]


@wp.kernel
def gather_contact(t: float, dt: float, state: MPMStateStruct, c: MaterialContact, index: CellIndex):
    slot=wp.tid()
    if slot<index.count[0] and t>=c.start_time and t<c.end_time:
        node_id=index.nodes[slot]
        x=node_id/(index.grid*index.grid); y=(node_id/index.grid)%index.grid; z=node_id%index.grid
        node=wp.vec3(float(x),float(y),float(z))/c.inv_dx
        for side in range(2):
            if c.enabled[side]!=0:
                tau=wp.max(0.,t-c.frame_time)
                tr=c.transforms[side]; q=wp.transform_get_rotation(tr)
                omega=c.angular[side]; speed=wp.length(omega)
                if speed>1.e-10:q=wp.quat_from_axis_angle(omega/speed,speed*tau)*q
                origin=wp.transform_get_translation(tr)+tau*c.linear[side]
                point=wp.quat_rotate_inv(q,node-origin)
                center=c.plane_centers[side]
                sign=float(1.)
                if side==1:sign=-1.
                distance=sign*(point[0]-center[0])
                if wp.abs(point[1]-center[1])<=c.footprint_y and wp.abs(point[2]-center[2])<=c.footprint_z and distance<=0. and distance>=-.03:
                    total=float(0.)
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                bx=x-i; by=y-j; bz=z-k
                                if bx>=0 and by>=0 and bz>=0:
                                    cell=(bx*index.grid+by)*index.grid+bz
                                    start=index.begin[cell]
                                    if start>=0:
                                        for s in range(start,index.end[cell]):
                                            p=index.ids[s]
                                            if c.labels[p]==side:
                                                total=total+contact_mass_contribution(state,c,p,i,j,k)
                    if side==0:c.mass[x,y,z]=total
                    else:c.right_mass[x,y,z]=total


class DeterministicTransfer:
    def __init__(self, model, device):
        n=model.n_particles; grid=model.grid_dim_x
        if (grid,grid)!=(model.grid_dim_y,model.grid_dim_z):
            raise ValueError('Deterministic transfer requires the existing cubic grid')
        self.device=device
        index=CellIndex(); index.n=n; index.grid=grid
        index.keys=wp.empty(2*n,dtype=wp.int64,device=device)
        index.ids=wp.empty(2*n,dtype=int,device=device)
        for name in ['begin','end','marked']:
            setattr(index,name,wp.empty(grid**3,dtype=int,device=device))
        index.nodes=wp.empty(min(27*n,grid**3),dtype=int,device=device)
        index.count=wp.zeros(1,dtype=int,device=device)
        index.invalid=wp.zeros(1,dtype=int,device=device)
        self.index=index

    def prepare(self,state,model):
        index=self.index
        index.begin.fill_(-1); index.marked.zero_(); index.count.zero_()
        wp.launch(particle_keys,dim=index.n,inputs=[state,model,index],device=self.device)
        radix_sort_pairs(index.keys,index.ids,index.n)
        wp.launch(cell_ranges_and_nodes,dim=index.n,inputs=[index],device=self.device)

    def p2g(self,state,model,dt):
        wp.launch(gather_p2g,dim=self.index.nodes.size,inputs=[state,model,dt,self.index],device=self.device)

    def contact(self,t,dt,state,c):
        wp.launch(gather_contact,dim=self.index.nodes.size,inputs=[t,dt,state,c,self.index],device=self.device)

    def check(self):
        if self.index.invalid.numpy()[0]:
            raise RuntimeError('Particle stencil outside allocated grid in deterministic transfer')

    def step(self,solver,model,state,next_state,dt):
        if solver.particle_velocity_modifiers or any(k is not scatter_material_velocity for k in solver.pre_p2g_operations):
            raise ValueError('This forward transfer adapter only supports the existing material-contact pipeline')
        grid=(model.grid_dim_x,model.grid_dim_y,model.grid_dim_z)
        device=self.device
        wp.launch(zero_grid,dim=grid,inputs=[state,model],device=device)
        self.prepare(state,model)
        for c in solver.impulse_params:self.contact(solver.time,dt,state,c)
        wp.launch(compute_stress_from_F_trial,dim=model.n_particles,inputs=[state,model,dt],device=device)
        self.p2g(state,model,dt)
        wp.launch(grid_normalization_and_gravity,dim=grid,inputs=[state,model,dt],device=device)
        if model.grid_v_damping_scale<1.:
            wp.launch(add_damping_via_grid,dim=grid,inputs=[state,model.grid_v_damping_scale],device=device)
        for k,kernel in enumerate(solver.grid_postprocess):
            wp.launch(kernel,dim=grid,inputs=[solver.time,dt,state,model,solver.collider_params[k]],device=device)
            if solver.modify_bc[k] is not None:solver.modify_bc[k](solver.time,dt,solver.collider_params[k])
        wp.launch(g2p_differentiable,dim=model.n_particles,inputs=[state,next_state,model,dt],device=device)
        solver.time+=dt
