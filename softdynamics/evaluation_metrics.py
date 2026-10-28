"""Reporting metrics, independent of the weighted fitting objective."""
import numpy as np
from scipy.spatial import cKDTree


def geometry_metrics(surface, reference_surface, tracks, reference_tracks, visible):
    """EMPM-style GT->surface L1 distance and fixed-correspondence L2 track error."""
    surface=np.asarray(surface,dtype=np.float64)
    reference_surface=np.asarray(reference_surface,dtype=np.float64)
    tracks=np.asarray(tracks,dtype=np.float64)
    reference_tracks=np.asarray(reference_tracks,dtype=np.float64)
    if not np.isfinite(surface).all() or not np.isfinite(tracks).all():
        raise ValueError('Nonfinite prediction must be reported as failure')
    reference_surface=reference_surface[np.isfinite(reference_surface).all(1)]
    keep=np.asarray(visible,dtype=bool)&np.isfinite(reference_tracks).all(1)
    dist=float(cKDTree(surface).query(reference_surface,p=1)[0].mean()*1000) if len(reference_surface) and len(surface) else None
    track=float(np.linalg.norm(tracks[keep]-reference_tracks[keep],axis=1).mean()*1000) if keep.any() else None
    return dict(dist_mm=dist,track_mm=track,visible_tracks=int(keep.sum()),reference_points=len(reference_surface))


def mask_iou(prediction,reference):
    prediction,reference=np.asarray(prediction,bool),np.asarray(reference,bool)
    if prediction.shape!=reference.shape:raise ValueError('Mask shape mismatch')
    union=np.logical_or(prediction,reference).sum()
    return float(np.logical_and(prediction,reference).sum()/union) if union else 1.


def psnr_from_mse(mse,floor=1e-10):
    """Finite serialization convention: exact RGB matches are capped at 100 dB."""
    if not np.isfinite(mse) or mse<0:raise ValueError('Invalid RGB MSE')
    return float(-10*np.log10(max(mse,floor)))


def transport_gaussians(positions,gradients,indices,weights,inverse_initial_gradients,offset,initial_covariance):
    """Fixed eight-particle material binding, with relative affine covariance transport."""
    relative=np.einsum('gnij,gnjk->gnik',gradients[indices],inverse_initial_gradients)
    affine=np.einsum('gn,gnij->gij',weights,relative)
    means=np.einsum('gn,gni->gi',weights,positions[indices])+np.einsum('gij,gj->gi',affine,offset)
    covariance=affine@initial_covariance@affine.transpose(0,2,1)
    return means,covariance


def pack_covariance(covariance):
    return covariance[:,[0,0,0,1,1,2],[0,1,2,1,2,2]]
