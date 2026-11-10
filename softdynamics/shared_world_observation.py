"""Read-only GPU observation of the unchanged full-resolution MPM state.

Zero-copy Torch views avoid transferring every particle to CPU when recording
512 nodes. Validation still checks every particle and every original dynamic
field. No simulation state, forces, constitutive law or integration is changed.
"""
import numpy as np
import torch
from contextlib import contextmanager

from .adapters.empm import DYNAMICS


@contextmanager
def padded_simulation_grid(cells,lower_cells=None):
    """Extend by whole cells, optionally assigning more space to one direction."""
    from .adapters import pad_empm
    if cells==0:
        if lower_cells is not None:raise ValueError('Directed padding requires a positive grid extension')
        yield;return
    if cells<0 or cells%16:raise ValueError('Grid padding must be a positive multiple of 16 cells')
    lower=np.full(3,cells) if lower_cells is None else np.asarray(lower_cells)
    if lower.shape!=(3,) or not np.all(lower==np.floor(lower)) or np.any(lower<0) or np.any(lower>2*cells):
        raise ValueError('Lower-side padding must be three integers between zero and the total extension')
    original=pad_empm.simulation_domain
    def expanded(initial,poses,valid,dx,*args,**kwargs):
        size,shift=original(initial,poses,valid,dx,*args,**kwargs)
        if size+2*cells>384:raise ValueError('Padded grid exceeds the original 384-cell allocation limit')
        offset=cells*dx if lower_cells is None else (lower*dx).astype(shift.dtype)
        return size+2*cells,shift+offset
    pad_empm.simulation_domain=expanded
    try:yield
    finally:pad_empm.simulation_domain=original


class GPUObservation:
    def __init__(self,sim,ids):
        import warp as wp
        self.sim=sim
        self.views=[{name:wp.to_torch(getattr(state,name)) for name in DYNAMICS} for state in sim.states]
        self.ids=torch.as_tensor(ids,dtype=torch.long,device=self.views[0]['particle_x'].device)
        self.shadow_checks=[]

    @torch.inference_mode()
    def capture(self):
        fields=self.views[self.sim.current];ids=self.ids
        state=torch.cat((fields['particle_x'][ids],fields['particle_v'][ids],
            fields['particle_F_trial'][ids].reshape(-1,9),fields['particle_C'][ids].reshape(-1,9)),-1).cpu().numpy()
        # Match the original NumPy double-precision world-shift subtraction.
        state[:,:3]=state[:,:3].astype(np.float64)-self.sim.shift
        return state

    @torch.inference_mode()
    def check(self):
        sim=self.sim;fields=self.views[sim.current];x=fields['particle_x'];F=fields['particle_F_trial']
        finite=torch.stack([torch.isfinite(value).all() for value in fields.values()]).all()
        # NumPy computes float32 determinant through double-precision LAPACK,
        # then casts back. Match that precision policy before reducing all nodes.
        det=torch.linalg.det(F.double()).float()
        axis=torch.minimum(x,sim.model.grid_lim-x).amin(0);margin=axis.amin()
        values=torch.stack((finite.float(),det.amin(),det.amax(),margin)).cpu().numpy()
        finite=bool(values[0]);min_j,max_j,margin=map(float,values[1:])
        valid=finite and min_j>0 and margin>2.01*sim.model.dx
        result=dict(valid=valid,finite=finite,min_J=min_j,max_J=max_j,domain_margin_m=margin)
        if not valid:
            result['axis_domain_margins_m']=axis.cpu().tolist()
            result['lower_domain_margins_m']=x.amin(0).cpu().tolist()
            result['upper_domain_margins_m']=(sim.model.grid_lim-x.amax(0)).cpu().tolist()
            speed=fields['particle_v'].norm(dim=-1)
            result['max_particle_speed_m_s']=float(speed.amax().cpu())
            result['particles_at_boundary']=int((torch.minimum(x,sim.model.grid_lim-x).amin(-1)<=2.01*sim.model.dx).sum().cpu())
            if finite and min_j>0 and margin<=2.01*sim.model.dx:result['failure_kind']='domain_boundary'
        return result

    def install_validation(self,shadow_interval=100):
        reference=self.sim.check
        def checked():
            result=self.check()
            if self.sim.frame==1 or self.sim.frame%shadow_interval==0 or not result['valid']:
                cpu=reference()
                for key in ('finite','valid'):
                    if result[key]!=cpu[key]:raise ValueError('GPU observer validity mismatch')
                for key in ('min_J','max_J','domain_margin_m'):
                    np.testing.assert_allclose(result[key],cpu[key],rtol=2e-6,atol=1e-8)
                self.shadow_checks.append(dict(frame=self.sim.frame,valid=result['valid'],reference=cpu))
            return result
        self.sim.check=checked
