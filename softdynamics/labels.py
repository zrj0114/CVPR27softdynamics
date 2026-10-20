"""Numerically audited sensitivity labels and split-safe confidence calibration.

Interaction sensitivity and teacher reliability are distinct quantities. An
unreliable finite difference is missing supervision, not an insensitive action.
"""
from dataclasses import dataclass
import numpy as np

def central_column(r0,rplus,rminus,hplus,hminus):
    """Derivative at zero of the quadratic interpolant; allow rounded offsets."""
    if not (hplus>0 and hminus>0):raise ValueError('Both realized perturbation magnitudes must be positive')
    r0,rplus,rminus=[np.asarray(r,dtype=np.float64) for r in [r0,rplus,rminus]]
    if not (r0.shape==rplus.shape==rminus.shape):raise ValueError('Residual correspondence changed across probes')
    forward=(rplus-r0)/hplus;backward=(r0-rminus)/hminus
    j=(hminus*forward+hplus*backward)/(hplus+hminus)
    curvature=float(np.linalg.norm(forward-backward)/max(np.linalg.norm(forward)+np.linalg.norm(backward),1e-12))
    return j,curvature

def audit_columns(coarse,fine,rollout_valid,relative_tolerance=.25,absolute_floor=1e-10):
    a,b=np.asarray(coarse,float),np.asarray(fine,float)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Expected two residual-dimension by parameter Jacobians')
    if np.asarray(rollout_valid).shape!=(a.shape[1],):raise ValueError('One rollout-valid flag is required per parameter')
    if not (np.isfinite(relative_tolerance) and relative_tolerance>0 and np.isfinite(absolute_floor) and absolute_floor>0):
        raise ValueError('Stability tolerances must be finite and positive')
    finite=np.isfinite(a).all(0)&np.isfinite(b).all(0)
    raw_coarse=np.linalg.norm(a,axis=0);raw_fine=np.linalg.norm(b,axis=0)
    difference=np.linalg.norm(b-a,axis=0)
    stable=difference<=relative_tolerance*raw_fine+absolute_floor
    valid=np.asarray(rollout_valid,bool)&finite&stable
    reasons=[]
    for j in range(a.shape[1]):
        if not rollout_valid[j]:reasons.append('invalid_rollout')
        elif not finite[j]:reasons.append('nonfinite_response')
        elif not stable[j]:reasons.append('unstable_step_response')
        else:reasons.append('accepted')
    denominator=np.maximum(raw_coarse*raw_fine,absolute_floor**2)
    cosine=np.sum(a*b,axis=0)/denominator
    both_zero=(raw_coarse==0)&(raw_fine==0);cosine[both_zero]=1.
    return dict(S=np.where(valid,raw_fine,np.nan),valid=valid,reason=reasons,
        raw_S_coarse=raw_coarse,raw_S_fine=raw_fine,
        relative_J_change=difference/np.maximum(raw_fine,absolute_floor),cosine=cosine,
        stable=stable,finite=finite,H_fine=b.T@b)

@dataclass
class TrainingCalibration:
    tau: np.ndarray
    training_ids: tuple
    valid_counts: np.ndarray

    @classmethod
    def fit(cls,S,valid,training_indices,episode_ids,min_valid=3,tau_floor=1e-8):
        S=np.asarray(S,float);valid=np.asarray(valid,bool);idx=np.asarray(training_indices,int)
        if S.shape!=valid.shape or S.ndim!=2:raise ValueError('S and validity must have identical sample by parameter shapes')
        if len(episode_ids)!=len(S) or not ((idx>=0)&(idx<len(S))).all():raise ValueError('Training identifiers are inconsistent with the label array')
        if len(idx)!=len(set(idx.tolist())):raise ValueError('Repeated training indices do not create independent labels')
        if len({episode_ids[i] for i in idx})!=len(idx):raise ValueError('Repeated episode identifiers cannot meet independent-label requirements')
        effective=valid[idx]&np.isfinite(S[idx])&(S[idx]>=0);counts=effective.sum(0)
        if (counts<min_valid).any():
            raise ValueError(f'Insufficient reliable training labels per parameter: {counts.tolist()}; require {min_valid}. Do not fill missing teachers with zero.')
        tau=np.array([max(float(np.median(S[idx,j][effective[:,j]])),tau_floor) for j in range(S.shape[1])])
        return cls(tau,tuple(episode_ids[i] for i in idx),counts)

    def transform(self,S,valid):
        S=np.asarray(S,float);mask=np.asarray(valid,bool)&np.isfinite(S)&(S>=0)
        safe=np.where(mask,S,0.)
        return np.where(mask,safe/(safe+self.tau),np.nan),mask

def masked_confidence_loss(prediction,target,valid):
    """No NaNs enter a loss graph; empty supervision is explicit to the trainer."""
    import torch
    from torch.nn import functional as F
    mask=valid.bool()&torch.isfinite(target)
    if prediction.shape!=target.shape or target.shape!=mask.shape:raise ValueError('Confidence/mask shapes differ')
    if not mask.any():raise ValueError('This batch has no reliable teacher labels; skip it explicitly')
    return F.smooth_l1_loss(prediction[mask],target[mask],beta=.1)
