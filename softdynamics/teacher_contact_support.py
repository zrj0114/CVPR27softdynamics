"""Conservative, finite, swept input support matching the teacher pad geometry.

The teacher collides grid nodes; a particle receives their velocities through
quadratic B-spline interpolation. Expand by that WORLD-axis support, not a large
arbitrary sphere. Translation is swept continuously. Eight geodesic rotation
segments use a provable corner-motion bound, so crossings between samples are
not dropped. This boolean spatial support is piecewise constant; physical
features still provide ordinary differentiable action/material conditioning.
"""
import math
import torch
from .teacher_contact import FACE_SIZE_M,GRID_DX_M,SLAB_CELLS,GRID_SUPPORT_CELLS

def matrix_to_quaternion(matrix):
    """w,x,y,z; stable largest-diagonal branch, used only for boolean support."""
    m=matrix;d=torch.stack([1+m[...,0,0]+m[...,1,1]+m[...,2,2],
        1+m[...,0,0]-m[...,1,1]-m[...,2,2],1-m[...,0,0]+m[...,1,1]-m[...,2,2],
        1-m[...,0,0]-m[...,1,1]+m[...,2,2]],-1).clamp_min(0).sqrt()
    candidates=torch.stack([
        torch.stack([d[...,0].square(),m[...,2,1]-m[...,1,2],m[...,0,2]-m[...,2,0],m[...,1,0]-m[...,0,1]],-1),
        torch.stack([m[...,2,1]-m[...,1,2],d[...,1].square(),m[...,1,0]+m[...,0,1],m[...,0,2]+m[...,2,0]],-1),
        torch.stack([m[...,0,2]-m[...,2,0],m[...,1,0]+m[...,0,1],d[...,2].square(),m[...,2,1]+m[...,1,2]],-1),
        torch.stack([m[...,1,0]-m[...,0,1],m[...,2,0]+m[...,0,2],m[...,2,1]+m[...,1,2],d[...,3].square()],-1)],-2)
    candidates=candidates/(2*d.clamp_min(.1)[...,None])
    q=candidates.gather(-2,d.argmax(-1)[...,None,None].expand(*d.shape[:-1],1,4)).squeeze(-2)
    return q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)

def quaternion_matrix(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)

def segment_box(a,b,low,high):
    delta=b-a;parallel=delta.abs()<1e-12
    safe=torch.where(parallel,torch.ones_like(delta),delta)
    t0=(low-a)/safe;t1=(high-a)/safe
    enter=torch.where(parallel,torch.full_like(t0,-torch.inf),torch.minimum(t0,t1)).amax(-1).clamp_min(0)
    leave=torch.where(parallel,torch.full_like(t0,torch.inf),torch.maximum(t0,t1)).amin(-1).clamp_max(1)
    outside=(parallel&((a<low)|(a>high))).any(-1)
    return (enter<=leave)&~outside

@torch.no_grad()
def finite_pad_grid_swept_support(state,pads0,pads1,scale_m,dt,segments=8):
    B,N,_=state.shape;P=pads0.shape[1]
    if P%2 or P>4 or segments<1:raise ValueError('Paired pad sides and positive sweep subdivision required')
    if pads0.shape!=pads1.shape or pads0.shape!=(B,P,13):raise ValueError('Expected t,R,existence pad representation')
    active=(pads0[...,12]>.5)&(pads1[...,12]>.5)
    eye=torch.eye(3,dtype=state.dtype,device=state.device)
    R0=torch.where(active[...,None,None],pads0[...,3:12].reshape(B,P,3,3),eye)
    R1=torch.where(active[...,None,None],pads1[...,3:12].reshape(B,P,3,3),eye)
    q0=matrix_to_quaternion(R0);q1=matrix_to_quaternion(R1)
    dot=(q0*q1).sum(-1);q1=torch.where((dot<0)[...,None],-q1,q1);dot=dot.abs().clamp(0,1)
    half_angle=dot.acos();angle=2*half_angle
    scale=scale_m.reshape(B,1,1,1)
    x=state[:,:,None,:3]*scale
    movement=state[:,:,None,3:6]*scale*(dt.reshape(B,1,1,1)/(1/30))
    p0=torch.where(active[...,None],pads0[...,:3],0.)[:,None]*scale
    p1=torch.where(active[...,None],pads1[...,:3],0.)[:,None]*scale
    sign=state.new_tensor([1.,-1.]*(P//2)).reshape(1,1,P,1)
    depth=SLAB_CELLS*GRID_DX_M
    radius=math.sqrt(depth**2+(FACE_SIZE_M[0]/2)**2+(FACE_SIZE_M[1]/2)**2)
    # Maximum corner motion away from each interval's midpoint orientation.
    rotation_bound=2*radius*torch.sin(angle/(4*segments))[:,None,:,None]
    lower=state.new_tensor([-depth,-FACE_SIZE_M[0]/2,-FACE_SIZE_M[1]/2])
    upper=state.new_tensor([0.,FACE_SIZE_M[0]/2,FACE_SIZE_M[1]/2])
    result=torch.zeros((B,N,P),dtype=torch.bool,device=state.device)
    for k in range(segments):
        u0=k/segments;u1=(k+1)/segments;u=(u0+u1)/2
        denom=half_angle.sin().clamp_min(1e-12)
        s0=torch.sin((1-u)*half_angle)/denom;s1=torch.sin(u*half_angle)/denom
        q=torch.where((dot>.9995)[...,None],(1-u)*q0+u*q1,s0[...,None]*q0+s1[...,None]*q1)
        q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12);R=quaternion_matrix(q)
        def local(t):
            world=x+t*movement-(p0+t*(p1-p0))
            v=torch.einsum('bnpi,bpij->bnpj',world,R)
            return torch.cat([v[...,:1]*sign,v[...,1:]],-1)
        support=GRID_SUPPORT_CELLS*GRID_DX_M*R.abs().sum(-2)[:,None]+rotation_bound
        result|=segment_box(local(u0),local(u1),lower-support,upper+support)&active[:,None]
    return result.to(state.dtype)
