"""Past-only point encoder and future-action-conditioned residual forecast."""
import numpy as np
import torch
from torch import nn

PAST=16
FUTURE=16

def make_batch(data,keys,rng,points=128,starts=None):
    rows=[];truth=[];masks=[];scale=[]
    for row,key in enumerate(keys):
        d=data[key];T,N=d['positions'].shape[:2]
        start=int(rng.integers(T-PAST-FUTURE+1)) if starts is None else starts[row]
        ids=torch.as_tensor(rng.choice(N,min(points,N),replace=False),device=d['positions'].device)
        x=d['positions'][start:start+PAST+FUTURE,ids]
        v=d['valid'][start:start+PAST+FUTURE,ids]
        t=d['times'][start:start+PAST+FUTURE]
        past=x[:PAST];anchor=past[-1].clone()
        dt=(t[PAST-1]-t[PAST-3]).clamp_min(1e-6)
        velocity=(past[-1]-past[-3])/dt
        observed=v[PAST-1]&v[PAST-3]
        velocity=torch.where(observed[:,None],velocity,0.)
        action=d['actions'][start:start+PAST+FUTURE].clone()
        for g in range(2):
            mat=action[:,g*14:g*14+12].reshape(-1,3,4)
            r0=mat[PAST-1,:,:3].clone();t0=mat[PAST-1,:,3].clone()
            mat[:,:,:3]=mat[:,:,:3]@r0.T-torch.eye(3,device=mat.device)
            mat[:,:,3]-=t0
            mat*=action[:,g*14+13,None,None]
        rows.append(dict(past=past,valid=v[:PAST],geometry=d['geometry'][ids],
            anchor=anchor,velocity=velocity,actions=action,
            past_times=t[:PAST]-t[PAST-1],future_times=t[PAST:]-t[PAST-1]))
        truth.append(x[PAST:]);masks.append(v[PAST:]&v[PAST-1][None]);scale.append(d['scale_m'])
    return {k:torch.stack([r[k] for r in rows]) for k in rows[0]},torch.stack(truth),torch.stack(masks),torch.stack(scale)

class CausalMotion(nn.Module):
    def __init__(self,actions=True):
        super().__init__();self.use_actions=actions
        self.point=nn.Sequential(nn.Linear(23,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.history=nn.GRU(64,64,batch_first=True)
        self.action=nn.Sequential(nn.Linear(29,64),nn.GELU(),nn.Linear(64,64))
        self.action_history=nn.GRU(64,64,batch_first=True)
        self.global_point=nn.Linear(128,64)
        self.decoder=nn.Sequential(nn.Linear(64+64+64+15+3+2,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def encode(self,b):
        x=b['past'];B,L,N,_=x.shape;valid=b['valid']
        relative=torch.where(valid[...,None],x-b['anchor'][:,None],0.)
        dx=torch.cat([torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]],dim=1)
        dt=torch.cat([torch.ones_like(b['past_times'][:,:1]),b['past_times'][:,1:]-b['past_times'][:,:-1]],dim=1)
        pair=valid&torch.cat([torch.zeros_like(valid[:,:1]),valid[:,:-1]],dim=1)
        velocity=torch.where(pair[...,None],dx/dt[:,:,None,None].clamp_min(1e-6),0.)
        geom=b['geometry'][:,None].expand(-1,L,-1,-1)
        clock=b['past_times'][:,:,None,None].expand(-1,-1,N,1)
        z=self.point(torch.cat([geom,relative,velocity,valid[...,None].float(),clock],-1))
        z=z.permute(0,2,1,3).reshape(B*N,L,64)
        _,h=self.history(z);return h[0].reshape(B,N,64)

    def forward(self,b):
        h=self.encode(b);B,N,D=h.shape;H=b['future_times'].shape[1]
        global_feature=self.global_point(torch.cat([h.mean(1),h.amax(1)],-1))
        times=torch.cat([b['past_times'],b['future_times']],dim=1)
        actions=b['actions'] if self.use_actions else torch.zeros_like(b['actions'])
        ah,_=self.action_history(self.action(torch.cat([actions,times[...,None]],-1)))
        ah=ah[:,-H:]
        features=torch.cat([h[:,None].expand(-1,H,-1,-1),global_feature[:,None,None].expand(-1,H,N,-1),
            ah[:,:,None].expand(-1,-1,N,-1),b['geometry'][:,None].expand(-1,H,-1,-1),
            b['velocity'][:,None].expand(-1,H,-1,-1),
            torch.stack([b['future_times'],b['future_times']**2],-1)[:,:,None].expand(-1,-1,N,-1)],-1)
        correction=self.decoder(features)
        horizon=b['future_times'][:,:,None,None]
        return b['anchor'][:,None]+horizon*b['velocity'][:,None]+horizon*correction

def baselines(b):
    hold=b['anchor'][:,None].expand(-1,FUTURE,-1,-1)
    cv=hold+b['future_times'][:,:,None,None]*b['velocity'][:,None]
    return hold,cv
