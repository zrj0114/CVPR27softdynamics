"""Original phase-weighted surface/Huber objective in millimeters."""
import numpy as np
from scipy.spatial import cKDTree

def score(pred,obs,config,return_curves=False):
    cfg=config;frames=np.array(obs['meta']['frames']);meta=obs['meta']
    surface=[]
    for x,y,tree in zip(pred['surface'],obs['surface'],obs['trees']):
        surface.append((tree.query(x)[0].mean()+cKDTree(x).query(y)[0].mean())*500)
    surface=np.array(surface)
    distances=np.linalg.norm(np.asarray(pred['tracks'],dtype=np.float64)-obs['tracks'],axis=-1)*1000.
    visible=obs['visible'];count=visible.sum(1);assert (count>0).all()
    track=(distances*visible).sum(1)/count
    delta=cfg['huber_delta_mm'];robust=np.where(distances<delta,.5*distances**2/delta,distances-.5*delta)
    robust=(robust*visible).sum(1)/count
    loss=cfg['surface_weight']*surface+cfg['track_weight']*robust
    masks=dict(before=frames<meta['clip_start'],interaction=(frames>=meta['clip_start'])&(frames<=meta['clip_end']),release=frames>meta['clip_end'])
    phases={k:dict(surface_mm=float(surface[v].mean()),track_mm=float(track[v].mean()),loss_mm=float(loss[v].mean())) for k,v in masks.items()}
    value=sum(cfg['phase_weights'][k]*phases[k]['loss_mm'] for k in masks)
    result=dict(objective_mm=float(value),surface_mm=float(surface.mean()),track_mm=float(track.mean()),phases=phases)
    if return_curves:result['per_frame_loss_mm']=loss.tolist()
    return result
