"""Finite material-associated unilateral contact with Coulomb friction.

Only compressive relative normal motion generates an impulse. Opposing
grippers accumulate impulses from the same unconstrained grid velocity, then
apply their weighted average once, avoiding gripper-order bias.
"""
import warp as wp
from benchmark_mpm_cpu import MPMStateStruct,MPMModelStruct,MPMWARPDiff


@wp.struct
class MaterialContact:
    enabled: wp.array(dtype=int)
    labels: wp.array(dtype=int)
    local: wp.array(dtype=wp.vec3)
    transforms: wp.array(dtype=wp.transform)
    linear: wp.array(dtype=wp.vec3)
    angular: wp.array(dtype=wp.vec3)
    correction: wp.array(dtype=wp.vec3)
    mass: wp.array3d(dtype=float)
    right_mass: wp.array3d(dtype=float)
    momentum: wp.array3d(dtype=wp.vec3)
    weight: wp.array3d(dtype=float)
    inv_dx: float
    frame_time: float
    start_time: float
    end_time: float
    ground: float
    friction: float
    contact_friction: float
    sliding_friction: float
    gain: float
    mode: int
    plane_centers: wp.array(dtype=wp.vec3)
    radius: float
    footprint_y: float
    footprint_z: float


@wp.func
def contact_delta(relative:wp.vec3,normal:wp.vec3,friction:float,sliding_friction:float):
    vn=wp.dot(relative,normal)
    delta=wp.vec3(0.)
    if vn<0.:
        normal_impulse=-vn
        tangent=relative-vn*normal
        scale=float(1.)
        if wp.length(tangent)>friction*normal_impulse:
            scale=sliding_friction*normal_impulse/wp.max(wp.length(tangent),1.e-12)
        delta=normal_impulse*normal-scale*tangent
    return delta


@wp.kernel
def scatter_material_velocity(t:float,dt:float,state:MPMStateStruct,c:MaterialContact):
    p=wp.tid()
    side=c.labels[p]
    enabled=int(0)
    if side>=0:enabled=c.enabled[side]
    if enabled!=0 and t>=c.start_time and t<c.end_time:
        tau=wp.max(0.,t-c.frame_time)
        tr=c.transforms[side];q=wp.transform_get_rotation(tr)
        omega=c.angular[side];speed=wp.length(omega)
        if speed>1.e-10:q=wp.quat_from_axis_angle(omega/speed,speed*tau)*q
        origin=wp.transform_get_translation(tr)+tau*c.linear[side]
        grid=state.particle_x[p]*c.inv_dx
        bx=int(grid[0]-.5);by=int(grid[1]-.5);bz=int(grid[2]-.5)
        fx=grid-wp.vec3(float(bx),float(by),float(bz))
        w=wp.mat33(.5*(1.5-fx[0])*(1.5-fx[0]),.75-(fx[0]-1.)*(fx[0]-1.),.5*(fx[0]-.5)*(fx[0]-.5),
                   .5*(1.5-fx[1])*(1.5-fx[1]),.75-(fx[1]-1.)*(fx[1]-1.),.5*(fx[1]-.5)*(fx[1]-.5),
                   .5*(1.5-fx[2])*(1.5-fx[2]),.75-(fx[2]-1.)*(fx[2]-1.),.5*(fx[2]-.5)*(fx[2]-.5))
        center=c.plane_centers[side]
        sign=float(1.)
        if side==1:sign=-1.
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    node=wp.vec3(float(bx+i),float(by+j),float(bz+k))/c.inv_dx
                    point=wp.quat_rotate_inv(q,node-origin)
                    distance=sign*(point[0]-center[0])
                    if wp.abs(point[1]-center[1])<=c.footprint_y and wp.abs(point[2]-center[2])<=c.footprint_z and distance<=0. and distance>=-.03:
                        mass=state.particle_mass[p]*w[0,i]*w[1,j]*w[2,k]
                        if side==0:wp.atomic_add(c.mass,bx+i,by+j,bz+k,mass)
                        else:wp.atomic_add(c.right_mass,bx+i,by+j,bz+k,mass)


@wp.kernel
def accumulate_material_contact(t:float,dt:float,state:MPMStateStruct,model:MPMModelStruct,c:MaterialContact):
    i,j,k=wp.tid()
    grid_mass=state.grid_m[i,j,k]
    if grid_mass>1.e-12:
        free_velocity=state.grid_v_out[i,j,k]
        node=wp.vec3(float(i),float(j),float(k))/c.inv_dx
        delta=wp.vec3(0.);weight=float(0.)
        for side in range(2):
            mass=c.mass[i,j,k]
            if side==1:mass=c.right_mass[i,j,k]
            if mass>1.e-12:
                fraction=mass/grid_mass
                blend=wp.clamp((fraction-.02)/.13,0.,1.)
                tau=wp.max(0.,t-c.frame_time)
                tr=c.transforms[side];q=wp.transform_get_rotation(tr)
                omega=c.angular[side];speed=wp.length(omega)
                if speed>1.e-10:q=wp.quat_from_axis_angle(omega/speed,speed*tau)*q
                origin=wp.transform_get_translation(tr)+tau*c.linear[side]
                controller=c.linear[side]+wp.cross(omega,node-origin)
                sign=float(1.)
                if side==1:sign=-1.
                normal=wp.quat_rotate(q,wp.vec3(sign,0.,0.))
                relative=free_velocity-controller
                if wp.dot(relative,normal)<0.:
                    delta=delta+blend*contact_delta(relative,normal,c.contact_friction,c.sliding_friction)
                    weight=weight+blend
        c.momentum[i,j,k]=c.momentum[i,j,k]+delta
        c.weight[i,j,k]=c.weight[i,j,k]+weight
    c.mass[i,j,k]=0.;c.right_mass[i,j,k]=0.


@wp.kernel
def material_grid_boundary(t:float,dt:float,state:MPMStateStruct,model:MPMModelStruct,c:MaterialContact):
    i,j,k=wp.tid()
    if state.grid_m[i,j,k]>1.e-12:
        v=state.grid_v_out[i,j,k]+c.momentum[i,j,k]/wp.max(1.,c.weight[i,j,k])
        # The original unilateral ground and ground-friction update.
        if float(j)*model.dx<c.ground and v[1]<0.:
            speed=wp.sqrt(v[0]*v[0]+v[2]*v[2])
            factor=wp.max(0.,1.+c.friction*v[1]/wp.max(speed,1.e-12))
            v=wp.vec3(v[0]*factor,0.,v[2]*factor)
        state.grid_v_out[i,j,k]=v
    c.momentum[i,j,k]=wp.vec3(0.);c.weight[i,j,k]=0.
