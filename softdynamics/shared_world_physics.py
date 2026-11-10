"""Differentiable material/state features for the learned transition.

These descriptors do not replace the physical solver or provide derivative
labels. Local deformation is estimated from the observed geometry, not read
from the simulator's hidden F state.
"""
import torch


def geometric_deformation(positions,canonical,neighbors,ridge=1e-3):
    batch=torch.arange(len(positions),device=positions.device)[:,None,None]
    rest=canonical[batch,neighbors]-canonical[:,:,None]
    current=positions[batch,neighbors]-positions[:,:,None]
    weights=rest.square().sum(-1).clamp_min(1e-12).reciprocal()
    weights=weights/weights.sum(-1,keepdim=True)
    covariance=torch.einsum('bnk,bnki,bnkj->bnij',weights,rest,rest)
    cross=torch.einsum('bnk,bnki,bnkj->bnij',weights,current-rest,rest)
    scale=covariance.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-12)/3
    eye=torch.eye(3,device=positions.device,dtype=positions.dtype)
    regularized=covariance+ridge*scale[...,None,None]*eye
    return eye+torch.linalg.solve(regularized,cross.transpose(-2,-1)).transpose(-2,-1)


def corotated_stress(deformation,theta):
    """FCR Kirchhoff stress with a smooth iterative polar approximation."""
    eye=torch.eye(3,device=deformation.device,dtype=deformation.dtype)
    rotation=deformation/torch.linalg.vector_norm(deformation,dim=(-2,-1),keepdim=True).clamp_min(1e-6)
    # Avoid the undefined SVD backward at repeated singular values, including I.
    for _ in range(10):
        rotation=.5*rotation@(3*eye-rotation.transpose(-2,-1)@rotation)
    E,nu=theta[:,0],theta[:,1]
    shear=E/(2*(1+nu))
    lame=E*nu/((1+nu)*(1-2*nu))
    determinant=torch.linalg.det(deformation)
    return (2*shear[:,None,None,None]*(deformation-rotation)@deformation.transpose(-2,-1)+
        lame[:,None,None,None]*determinant[...,None,None]*(determinant[...,None,None]-1)*eye)


def physical_state_features(state,theta,geometry,dt,dt_reference=1/30):
    deformation=geometric_deformation(state[...,:3],geometry['canonical'],geometry['neighbors'])
    stress=corotated_stress(deformation,theta).flatten(-2)
    scale=geometry['scale_m'][:,None,None]
    velocity=state[...,3:6]*scale/dt_reference
    distance=(state[...,1:2]-geometry['ground_y'][:,None,None])*scale
    proximity=torch.sigmoid(-distance/.002)
    tangent=velocity*state.new_tensor([1.,0.,1.])
    # A friction/contact descriptor, not a claim to compute the normal force.
    friction=-theta[:,2,None,None]*proximity*tangent
    decay=torch.expm1(-theta[:,3,None,None]*dt[:,None,None])*velocity
    return torch.cat((torch.asinh(stress/3500.),torch.asinh(friction/.1),torch.asinh(decay/.1)),-1)
