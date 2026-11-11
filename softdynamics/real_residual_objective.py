"""Initial-only real rollout and visibility-safe, differentiable GT scoring."""
import numpy as np
import torch

from .coupled_world_model import burn_history, call_step


def causal_batch(data, theta, device, end=None):
    end = len(data['times'])-1 if end is None else int(end)
    if not 1 <= end < len(data['times']):
        raise ValueError('Rollout endpoint outside recorded actions')
    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)
    initial = tensor(data['state0'])[None]
    geometry = {k:tensor(data[k])[None] for k in ('canonical','scale')}
    geometry['scale_m'] = geometry.pop('scale')
    geometry['neighbors'] = tensor(data['neighbors'],torch.long)[None]
    geometry['surface_mask'] = tensor(data['surface_mask'],torch.bool)[None]
    geometry['ground_y'] = tensor([(-.01-float(data['origin'][1]))/float(data['scale'])])
    # No future GT/state tensor is supplied to the world model.
    history = dict(states=initial.new_empty((1,0,len(initial[0]),24)),
        pads0=tensor(np.empty((1,0,4,13))), pads1=tensor(np.empty((1,0,4,13))),
        dt=tensor(np.empty((1,0))), present=tensor(np.empty((1,0),bool),torch.bool))
    batch = dict(states=initial[:,None], pads=tensor(data['pads'][:end+1])[None],
        times=tensor(data['times'][:end+1])[None], theta=theta.reshape(1,2), history=history)
    return batch, geometry


def real_rollout(model, data, theta, device, end=None, gradients=False, gradient_tail=None):
    if model.residual_config is None:
        raise ValueError('A residual-enabled coupled model is required')
    batch, geometry = causal_batch(data,theta,device,end)
    state = batch['states'][:,0]
    horizon = batch['pads'].shape[1]-1
    if gradient_tail is not None and gradient_tail < 1:
        raise ValueError('Positive gradient tail required')
    boundary = max(0,horizon-gradient_tail) if gradient_tail else 0
    memory = burn_history(model.world,batch['history'],batch['theta'],geometry,gradients)
    predictions, corrections = [state], []
    for t in range(horizon):
        active = gradients and t >= boundary
        if boundary and t==boundary:
            state, memory = state.detach(), memory.detach()
        with torch.set_grad_enabled(active):
            state, memory, correction = call_step(model.world.step_with_correction,
                (state,batch['pads'][:,t],batch['pads'][:,t+1],batch['theta'],geometry,
                 batch['times'][:,t+1]-batch['times'][:,t],memory),active)
        predictions.append(state); corrections.append(correction)
    predicted = torch.stack(predictions,1)[0]
    correction = torch.stack(corrections,1)[0]
    positions = predicted[...,:3]*geometry['scale_m'][0]+torch.as_tensor(
        data['origin'],dtype=predicted.dtype,device=device)
    return positions, correction, boundary+1


def observation_objective(positions, data, first=1, surface_points=128):
    """mm: 0.5 GT-to-material-surface distance + 0.5 visible track Huber.

    GT-to-surface is one-sided because observations are partial. Hidden NaNs
    are removed BEFORE subtraction/cdist, never multiplied by zero afterwards.
    No initial frame or unobserved withdrawal frame is treated as a target.
    """
    if surface_points < 1 or first < 1 or len(positions) > len(data['times']):
        raise ValueError('Invalid scoring range or point budget')
    device, dtype = positions.device, positions.dtype
    def tensor(value, kind=None):
        return torch.as_tensor(value,dtype=kind or dtype,device=device)
    delta = positions-tensor(data['x0'])
    tracks = tensor(data['track0'])[None]+torch.einsum('pn,tnc->tpc',tensor(data['P']),delta)
    ids = np.linspace(0,len(data['surface0'])-1,min(surface_points,len(data['surface0']))).astype(int)
    binding = tensor(data['surface_binding'][ids],torch.long)
    weights = tensor(data['surface_weights'][ids])
    surface = tensor(data['surface0'][ids])[None]+(delta[:,binding]*weights[None,:,:,None]).sum(2)
    rows, phases, distances, errors = [], [], [], []
    for frame in range(first,len(positions)):
        pieces = []
        visible = np.flatnonzero(data['visible'][frame])
        if len(visible):
            error = (tracks[frame,visible]-tensor(data['tracks'][frame,visible])).norm(dim=-1)*1000
            pieces.append(torch.where(error<5.,.5*error.square()/5.,error-2.5).mean())
            errors.append(error.mean())
        reference = data['reference'][frame,data['reference_valid'][frame]]
        if len(reference):
            reference = reference[np.linspace(0,len(reference)-1,min(surface_points,len(reference))).astype(int)]
            distance = torch.cdist(tensor(reference),surface[frame],
                compute_mode='donot_use_mm_for_euclid_dist').amin(-1).mean()*1000
            pieces.append(distance); distances.append(distance)
        if pieces:
            rows.append(torch.stack(pieces).mean()); phases.append(int(data['phase'][frame]))
    if not rows:
        raise ValueError('No valid real GT in scored rollout segment')
    masses = [(.1,.45,.45)[p] for p in sorted(set(phases))]
    means = [torch.stack([v for v,p in zip(rows,phases) if p==phase]).mean()
             for phase in sorted(set(phases))]
    objective = sum(w*v for w,v in zip(masses,means))/sum(masses)
    metrics = dict(objective_mm=float(objective.detach()), scored_frames=len(rows),
        gt_surface_mm=float(torch.stack(distances).mean().detach()) if distances else None,
        track_mm=float(torch.stack(errors).mean().detach()) if errors else None)
    return objective, metrics


def residual_regularizers(corrections, neighbors, maximum, first=0):
    c = corrections[first:]/maximum
    if not len(c):
        raise ValueError('No active residual intervals')
    edges = torch.as_tensor(neighbors,dtype=torch.long,device=c.device)
    return c.square().mean(), (c[:,:,None]-c[:,edges]).square().mean()
