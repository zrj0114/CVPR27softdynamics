"""Verify local material derivatives independently of float32 differences."""
import numpy as np
import torch

from .elastic_downstream import tensor_parameters
from .shared_world_inverse import project_observations


def relative_error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-8))


def precision_audit(function,point,optimizer_gradient):
    q=point.detach().double().requires_grad_();loss=function(q)
    exact=torch.autograd.grad(loss,q)[0].detach().cpu().numpy()
    rows=[]
    with torch.no_grad():
        for epsilon in (1e-4,3e-5,1e-5):
            finite=[]
            for j in range(q.numel()):
                delta=torch.zeros_like(q);delta[j]=epsilon
                plus=function(q+delta);minus=function(q-delta)
                finite.append(float((plus-minus)/(2*epsilon)))
            rows.append(dict(epsilon=epsilon,gradient=finite,relative_error=relative_error(exact,finite)))
    agreement=relative_error(optimizer_gradient,exact)
    convergence=relative_error(rows[-1]['gradient'],rows[-2]['gradient'])
    reasons=[]
    if not np.isfinite(exact).all() or np.linalg.norm(exact)<1e-8:reasons.append('Unusable double-precision derivative')
    if not np.isfinite(agreement) or agreement>.02:reasons.append('Optimizer gradient differs from the double-precision derivative')
    if any(not np.isfinite(row['relative_error']) or row['relative_error']>.01 for row in rows[-2:]):
        reasons.append('Small-step finite differences disagree with autograd')
    if not np.isfinite(convergence) or convergence>.01:reasons.append('Small-step finite differences did not converge')
    return dict(passed=not reasons,failure_reasons=reasons,loss_mm=float(loss.detach()),gradient=exact,
        finite_differences=rows,optimizer_gradient_relative_error=agreement,
        finite_difference_convergence=convergence,optimizer_agreement_tolerance=.02,
        finite_difference_tolerance=.01,dtype='float64',network_weights_updated=False)


def _double(value):
    if isinstance(value,torch.Tensor):return value.double() if value.is_floating_point() else value
    if isinstance(value,dict):return {k:_double(v) for k,v in value.items()}
    if isinstance(value,list):return [_double(v) for v in value]
    if isinstance(value,tuple):return tuple(_double(v) for v in value)
    return value


def material_precision_audit(window,model,config,optimizer_gradient):
    """Use a temporary audit instance; fitting keeps its original float32 model."""
    model.double()
    for obj in (window.real,window):
        for key,value in vars(obj).copy().items():setattr(obj,key,_double(value))
    q=torch.zeros(2,device=window.device,dtype=torch.float64)
    with torch.no_grad():
        material=tensor_parameters(q,config)[None].expand(len(window.indices),2)
        window.nominal=[]
        for t in range(window.horizon):
            f=window.indices[:,t]
            window.nominal.append(model(window.reference[:,t],window.real.pads[0,f],
                window.real.pads[0,f+1],material,window.geo,
                window.real.times[0,f+1]-window.real.times[0,f])[...,:6].detach())
        xyz=window.reference[:,-1,:,:3]*window.real.scale+torch.as_tensor(window.real.origin,device=window.device)
        window.projected_anchor=project_observations(xyz,window.real.projection)
    return precision_audit(lambda point:window.loss(model,tensor_parameters(point,config)),q,optimizer_gradient)
