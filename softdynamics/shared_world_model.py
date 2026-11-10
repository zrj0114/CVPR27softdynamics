"""One material-conditioned 3-D world model shared across object geometries.

Ordinary next-state/multi-step supervision only. Geometry is a batch input,
never a checkpoint-specific object buffer. Model weights may be frozen while
autograd still differentiates material values and continuous pad actions.
"""
import math

import torch
from torch import nn
from .shared_world_physics import physical_state_features
from .teacher_contact import CONTACT_ID, SUPPORT_ID, FACE_SIZE_M, GRID_DX_M, SLAB_CELLS
from .teacher_contact_support import finite_pad_grid_swept_support

DT_REF = 1/30


def mlp(inputs,hidden,outputs):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,outputs))


def physical_parameters(q, base, scales):
    """Differentiable counterpart of ParameterSpace; no NumPy/float/round casts."""
    return base+q*scales


def material_features(theta):
    if theta.shape[-1] == 2:
        return torch.stack((torch.log(theta[...,0]/3500.), (theta[...,1]-.27)/.12), -1)
    return torch.stack((torch.log(theta[...,0]/3500.),
                        (theta[...,1]-.27)/.12,
                        (theta[...,2]-.8)/.7,
                        (theta[...,3]-.4)/.8),-1)


def recurrent_channels(model):
    """State fields consumed by this dynamics representation; C is optional."""
    if getattr(model,'deformation_memory',False):return 15
    return 6 if getattr(model,'kinematic_only',True) else 24


def gather_neighbors(values,neighbors):
    batch=torch.arange(values.shape[0],device=values.device)[:,None,None]
    return values[batch,neighbors]


def swept_pad_influence(state, pads0, pads1, scale_m, inner_m=.01, outer_m=.05):
    """C2 spatial support around the pad's swept bounding sphere.

    The sphere encloses the finite pad for every interpolated orientation.
    A fast translation crossing a node therefore remains active even when
    both endpoints are distant. Only current state and commanded poses enter.
    """
    if not 0 <= inner_m < outer_m:
        raise ValueError('Pad influence requires 0 <= inner < outer')
    x=state[:,:,None,:3];a=pads0[:,None,:,:3];b=pads1[:,None,:,:3]
    delta=b-a
    fraction=((x-a)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)
    nearest=a+fraction.clamp(0,1)[...,None]*delta
    radius=math.sqrt(.00625**2+.032**2+.0125**2)
    distance=torch.linalg.vector_norm(x-nearest,dim=-1)*scale_m[:,None,None]-radius
    u=((distance-inner_m)/(outer_m-inner_m)).clamp(0,1)
    return (1-10*u**3+15*u**4-6*u**5).clamp(0,1)


def pad_contact_features(state,pads0,pads1,scale_m,dt):
    """Known signed pad geometry from the unchanged corrected-pad collider.

    Left/right normals have opposite signs, even for equal marker rotations.
    Ordinary position prediction remains the only supervision. The smooth
    support values below are features, not an approximate replacement solver.
    """
    B,N,_=state.shape
    scale=scale_m.reshape(B,1,1,1)
    active=(pads0[...,12]>.5)&(pads1[...,12]>.5)
    p0=torch.where(active[...,None],pads0[...,:12],0.)
    p1=torch.where(active[...,None],pads1[...,:12],0.)
    R0=p0[...,3:12].reshape(B,4,3,3);R1=p1[...,3:12].reshape(B,4,3,3)
    local0=torch.einsum('bnpi,bpij->bnpj',state[:,:,None,:3]-p0[:,None,:,:3],R0)*scale
    local1=torch.einsum('bnpi,bpij->bnpj',state[:,:,None,:3]-p1[:,None,:,:3],R1)*scale
    sign=state.new_tensor([1.,-1.,1.,-1.]).reshape(1,1,4,1)
    def geometry(local):
        signed=torch.cat((local[...,:1]*sign,local[...,1:]),-1)
        depth=SLAB_CELLS*GRID_DX_M;half_y,half_z=FACE_SIZE_M[0]/2,FACE_SIZE_M[1]/2
        unit=signed/state.new_tensor([depth,half_y,half_z])
        smooth=.0015;distance=signed[...,:1]
        support=(torch.sigmoid(-distance/smooth)*torch.sigmoid((distance+depth)/smooth)*
            torch.sigmoid((half_y-local[...,1:2].abs())/smooth)*torch.sigmoid((half_z-local[...,2:3].abs())/smooth))
        return torch.asinh(unit),support
    g0,support0=geometry(local0);g1,support1=geometry(local1)
    normal0=R0[:,:,:,0][:,None]*sign;normal1=R1[:,:,:,0][:,None]*sign
    # Exact displacement of the current point under the pad's rigid motion.
    carried=p1[:,None,:,:3]+torch.einsum('bpij,bnpj->bnpi',R1,local0/scale)
    pad_velocity=(carried-state[:,:,None,:3])*scale/dt.reshape(B,1,1,1)
    node_velocity=state[:,:,None,3:6]*scale/DT_REF
    relative=torch.asinh((node_velocity-pad_velocity)/.1)
    features=torch.cat((g0,g1,normal0.expand(B,N,4,3),normal1.expand(B,N,4,3),relative,
        support0,support1,sign.expand(B,N,4,1)),-1)
    return features*active[:,None,:,None]


class LocalPointAttention(nn.Module):
    def __init__(self,hidden,heads):
        super().__init__()
        if hidden%heads:raise ValueError('hidden must be divisible by heads')
        self.heads=heads;self.dim=hidden//heads
        self.norm=nn.LayerNorm(hidden)
        self.qkv=nn.Linear(hidden,3*hidden)
        self.edge=mlp(9,64,heads)
        self.edge_value=nn.Linear(9,hidden)
        self.project=nn.Linear(hidden,hidden)
        self.feed=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,2*hidden),nn.SiLU(),nn.Linear(2*hidden,hidden))

    def forward(self,h,neighbors,edges,condition,node_valid=None):
        B,N,D=h.shape
        scale,shift=condition.chunk(2,-1)
        z=self.norm(h)*(1+.1*torch.tanh(scale[:,None]))+shift[:,None]
        q,k,v=self.qkv(z).chunk(3,-1)
        q=q.reshape(B,N,self.heads,self.dim)
        k=gather_neighbors(k,neighbors).reshape(B,N,-1,self.heads,self.dim)
        v=gather_neighbors(v,neighbors)+self.edge_value(edges)
        v=v.reshape(B,N,-1,self.heads,self.dim)
        logits=(q[:,:,None]*k).sum(-1)/math.sqrt(self.dim)+self.edge(edges)
        if node_valid is None:
            attention=logits.softmax(2)
        else:
            available=gather_neighbors(node_valid[...,None],neighbors)[...,0]
            logits=logits.masked_fill(~available[...,None],torch.finfo(logits.dtype).min)
            attention=logits.softmax(2)*available[...,None]
            attention=attention/attention.sum(2,keepdim=True).clamp_min(1e-12)
        aggregate=(attention[...,None]*v).sum(2).reshape(B,N,D)
        h=h+self.project(aggregate)
        return h+self.feed(h)


class SharedWorldModel(nn.Module):
    def __init__(self,hidden=256,layers=4,heads=4,kinematic_only=False,direct_displacement=False,contact_geometry=False,
                 affine_material=False,material_reference=None,residual_material=False,deformation_memory=False,
                 physical_features=False,physical_feature_reference=None,material_dim=4,
                 pad_locality=False,pad_locality_inner_m=.01,pad_locality_outer_m=.05,
                 pad_locality_geometry='swept_sphere_legacy',contact_contract=None):
        super().__init__()
        if physical_features and (not direct_displacement or affine_material or residual_material or deformation_memory):
            raise ValueError('Physical features require the original observed-velocity dynamics')
        if physical_feature_reference is not None and not physical_features:
            raise ValueError('A physical feature reference requires physical features')
        if deformation_memory and (not direct_displacement or kinematic_only or affine_material or residual_material):
            raise ValueError('Deformation memory requires the original direct-displacement head and a non-kinematic state')
        if affine_material and residual_material:raise ValueError('Choose one material output representation')
        if (affine_material or residual_material) and not direct_displacement:
            raise ValueError('Material output adaptation requires observed-velocity dynamics')
        if material_reference is not None and not (affine_material or residual_material):
            raise ValueError('A material reference requires material output adaptation')
        if material_dim not in (2,4): raise ValueError('Material input must have two or four dimensions')
        self.material_dim=material_dim
        if pad_locality_geometry not in ('swept_sphere_legacy',SUPPORT_ID):raise ValueError('Unknown pad support geometry')
        if contact_contract not in (None,CONTACT_ID):raise ValueError('Unknown teacher contact contract')
        if pad_locality_geometry==SUPPORT_ID and not (pad_locality and contact_geometry and contact_contract==CONTACT_ID):
            raise ValueError('Selected finite-pad support requires the selected contact contract and geometry features')
        if contact_contract==CONTACT_ID and pad_locality_geometry!=SUPPORT_ID:
            raise ValueError('Selected teacher policy cannot use the legacy sphere action gate')
        self.pad_locality_geometry=pad_locality_geometry;self.contact_contract=contact_contract
        if not 0 <= pad_locality_inner_m < pad_locality_outer_m:
            raise ValueError('Invalid pad locality radii')
        self.pad_locality=pad_locality
        self.pad_locality_inner_m=pad_locality_inner_m
        self.pad_locality_outer_m=pad_locality_outer_m
        self.direct_displacement=direct_displacement
        self.kinematic_only=(kinematic_only or direct_displacement) and not deformation_memory
        self.deformation_memory=deformation_memory
        self.contact_geometry=contact_geometry
        self.affine_material=affine_material
        self.residual_material=residual_material
        self.physical_features=physical_features
        self.architecture=dict(hidden=hidden,layers=layers,heads=heads,kinematic_only=self.kinematic_only,
            direct_displacement=direct_displacement,contact_geometry=contact_geometry)
        if material_dim != 4:self.architecture['material_dim']=material_dim
        if pad_locality:
            self.architecture['pad_locality']=True
            if pad_locality_geometry=='swept_sphere_legacy':
                self.architecture.update(pad_locality_inner_m=pad_locality_inner_m,pad_locality_outer_m=pad_locality_outer_m)
            else:self.architecture.update(pad_locality_geometry=pad_locality_geometry,contact_contract=contact_contract)
        if deformation_memory:self.architecture['deformation_memory']=True
        if physical_features:self.architecture['physical_features']=True
        if physical_feature_reference is not None:
            reference=list(physical_feature_reference)
            if (len(reference)!=4 or not all(math.isfinite(v) for v in reference) or reference[0]<=0
                or not -1<reference[1]<.5 or min(reference[2:])<0):
                raise ValueError('Invalid reference material for physical features')
            self.architecture['physical_feature_reference']=reference
            self.register_buffer('physical_feature_reference',torch.tensor(reference,dtype=torch.float32))
        if affine_material or residual_material:
            reference=[3500.,.27,.8,.4] if material_reference is None else list(material_reference)
            if len(reference)!=4 or not all(math.isfinite(v) for v in reference) or reference[0]<=0:
                raise ValueError('Need four finite reference parameters and positive Young modulus')
            self.architecture.update(material_reference=reference)
            self.architecture['affine_material' if affine_material else 'residual_material']=True
            self.register_buffer('material_reference',torch.tensor(reference,dtype=torch.float32))
        # x/v/(F-I)/(C*dt), canonical, ground clearance, metric size, dt, surface.
        self.state_encoder=mlp(24+3+4,hidden,hidden)
        # Each pad: local start/end positions, displacement, R0, R1-R0, valid.
        self.pad_encoder=mlp(46 if contact_geometry else 28,hidden,hidden)
        self.material_encoder=mlp(material_dim,hidden,hidden)
        self.condition=nn.ModuleList([nn.Linear(hidden,2*hidden) for _ in range(layers)])
        self.blocks=nn.ModuleList([LocalPointAttention(hidden,heads) for _ in range(layers)])
        output_dimensions=15 if affine_material else 3 if direct_displacement else 21
        self.output=nn.Sequential(nn.LayerNorm(hidden),mlp(hidden,hidden,output_dimensions))
        nn.init.normal_(self.output[-1][-1].weight,std=1e-3)
        nn.init.zeros_(self.output[-1][-1].bias)
        self.register_buffer('state_mean',torch.zeros(24))
        self.register_buffer('state_std',torch.ones(24))
        self.register_buffer('output_scale',torch.ones(21)*.01)
        self.register_buffer('loss_scale',torch.ones(24))
        if direct_displacement:self.register_buffer('velocity_scale',torch.ones(3)*.01)
        if physical_features:
            self.physical_encoder=mlp(15,hidden,hidden)
            nn.init.zeros_(self.physical_encoder[-1].weight)
            nn.init.zeros_(self.physical_encoder[-1].bias)
        if residual_material:
            self.material_residual=mlp(hidden,hidden,12)
            nn.init.zeros_(self.material_residual[-1].weight)
            nn.init.zeros_(self.material_residual[-1].bias)
        if deformation_memory:
            self.deformation_encoder=mlp(9,hidden,hidden)
            self.deformation_transition=mlp(hidden,hidden,9)
            for module in (self.deformation_encoder,self.deformation_transition):
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)

    def initialize_physical_features(self,source_state):
        """Preserve the existing transition and all input derivatives initially."""
        if not self.physical_features:raise ValueError('Target requires physical features')
        current=self.state_dict()
        added={key for key in current if key.startswith('physical_encoder.') or key=='physical_feature_reference'}
        if set(source_state)!=set(current)-added:
            raise ValueError('Physical-feature initialization requires an unchanged parent architecture')
        nn.init.zeros_(self.physical_encoder[-1].weight)
        nn.init.zeros_(self.physical_encoder[-1].bias)
        values={**source_state,**{key:value for key,value in self.state_dict().items() if key in added}}
        self.load_state_dict(values)

    def initialize_deformation_memory(self,source_state):
        """Keep the complete original function, then learn recurrent F dynamics."""
        if not self.deformation_memory:raise ValueError('Target requires deformation memory')
        if source_state['output.1.2.weight'].shape[0]!=3 or 'material_reference' in source_state:
            raise ValueError('Source must have the original three-channel velocity head')
        values=dict(source_state)
        for module in (self.deformation_encoder,self.deformation_transition):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        for key,value in self.state_dict().items():
            if key.startswith(('deformation_encoder.','deformation_transition.')):values[key]=value
        self.load_state_dict(values)

    def initialize_residual_material(self,source_state):
        """Preserve the complete parent function at every material initially."""
        if not self.residual_material:raise ValueError('Target requires a residual material output')
        if 'material_reference' in source_state:raise ValueError('Source already changes material output')
        nn.init.zeros_(self.material_residual[-1].weight)
        nn.init.zeros_(self.material_residual[-1].bias)
        values=dict(source_state)
        values['material_reference']=self.material_reference.clone()
        for key,value in self.state_dict().items():
            if key.startswith('material_residual.'):values[key]=value
        self.load_state_dict(values)

    def train_material_residual_only(self):
        if not self.residual_material:raise ValueError('No material residual to train')
        self.requires_grad_(False)
        self.material_residual.requires_grad_(True)

    def initialize_affine_material(self,source_state):
        """Preserve the source function at the reference material only.

        Four initially zero output fields learn material dependence from
        ordinary state losses. This conversion does not preserve the source
        model's material derivatives or its predictions at other materials.
        """
        if not self.affine_material:raise ValueError('Target must have affine material output')
        weight,bias='output.1.2.weight','output.1.2.bias'
        if source_state[weight].shape[0]!=3 or 'material_reference' in source_state:
            raise ValueError('Source must have the original three-channel velocity output')
        values=dict(source_state)
        values['material_reference']=self.material_reference.clone()
        for key in (weight,bias):
            expanded=torch.zeros_like(self.state_dict()[key])
            expanded[:3].copy_(source_state[key])
            values[key]=expanded
        self.load_state_dict(values)

    def set_statistics(self,statistics):
        for key in ('state_mean','state_std','output_scale','loss_scale'):
            getattr(self,key).copy_(torch.as_tensor(statistics[key],dtype=torch.float32))
        if self.direct_displacement:self.velocity_scale.copy_(self.state_std[3:6])

    def encode(self,state,pads0,pads1,theta,geometry,dt):
        """Shared causal geometry/state/action/material representation."""
        B,N,_=state.shape;ratio=(dt/DT_REF).reshape(B,1,1)
        canonical=geometry['canonical'];neighbors=geometry['neighbors']
        node_valid=geometry.get('node_valid')
        if node_valid is not None:
            if node_valid.shape != state.shape[:2] or node_valid.dtype != torch.bool:
                raise ValueError('Node validity must be a boolean mask per input point')
            if not torch.isfinite(state[...,:6][node_valid]).all():
                raise ValueError('Nonfinite state on a valid observed node')
            # Invalid observations cannot enter either node or edge features.
            fallback=torch.cat((canonical,torch.zeros_like(state[...,3:])),dim=-1)
            state=torch.where(node_valid[...,None],state,fallback)
        scale=geometry['scale_m'].reshape(B,1,1)
        ground=state[...,1:2]-geometry['ground_y'].reshape(B,1,1)
        normalized=(state-self.state_mean)/self.state_std
        deformation=(normalized[...,6:15] if self.deformation_memory else None)
        if self.kinematic_only or self.deformation_memory:
            normalized=torch.cat((normalized[...,:6],torch.zeros_like(normalized[...,6:])),-1)
        inputs=torch.cat((normalized,canonical,ground,
                          scale.expand(B,N,1),ratio.expand(B,N,1),
                          geometry['surface_mask'][...,None].to(state.dtype)),-1)
        h=self.state_encoder(inputs)
        if deformation is not None:h=h+self.deformation_encoder(deformation)
        active=(pads0[...,12]>.5)&(pads1[...,12]>.5)
        p0=torch.where(active[...,None],pads0[...,:12],0.)
        p1=torch.where(active[...,None],pads1[...,:12],0.)
        R0=p0[...,3:12].reshape(B,4,3,3)
        rel0=state[:,:,None,:3]-p0[:,None,:,:3]
        rel1=state[:,:,None,:3]-p1[:,None,:,:3]
        local0=torch.einsum('bnpi,bpij->bnpj',rel0,R0)
        local1=torch.einsum('bnpi,bpij->bnpj',rel1,R0)
        action=torch.cat(((p1[...,:3]-p0[...,:3])/ratio,p0[...,3:12],
                          p1[...,3:12]-p0[...,3:12]),-1)
        features=torch.cat((local0,local1,action[:,None].expand(B,N,4,21),
                            active[:,None,:,None].expand(B,N,4,1).to(state.dtype)),-1)
        if self.contact_geometry:
            features=torch.cat((features,pad_contact_features(state,pads0,pads1,geometry['scale_m'],dt)),-1)
        encoded=self.pad_encoder(features)*active[:,None,:,None]
        if self.pad_locality:
            if self.pad_locality_geometry==SUPPORT_ID:
                influence=finite_pad_grid_swept_support(state,pads0,pads1,geometry['scale_m'],dt)
            else:
                influence=swept_pad_influence(state,p0,p1,geometry['scale_m'],
                    self.pad_locality_inner_m,self.pad_locality_outer_m)*active[:,None]
            # The denominator must also be local: toggling a remote valid pad
            # must not rescale the contribution of a nearby pad.
            h=h+(encoded*influence[...,None]).sum(2)/influence.sum(2).clamp_min(1)[...,None]
        else:
            h=h+encoded.sum(2)/active.sum(1).clamp_min(1)[:,None,None]
        conditioning_theta=self.material_reference.expand(B,-1) if self.affine_material else theta
        material=self.material_encoder(material_features(conditioning_theta));h=h+material[:,None]
        if self.physical_features:
            feature_theta=self.physical_feature_reference.expand(B,-1) if hasattr(self,'physical_feature_reference') else theta
            h=h+self.physical_encoder(physical_state_features(state,feature_theta,geometry,dt))
        edges=torch.cat((gather_neighbors(state[...,:3],neighbors)-state[:,:,None,:3],
                         gather_neighbors(state[...,3:6],neighbors)-state[:,:,None,3:6],
                         gather_neighbors(canonical,neighbors)-canonical[:,:,None]),-1)
        for block,condition in zip(self.blocks,self.condition):
            h=block(h,neighbors,edges,condition(material),node_valid)
            if node_valid is not None:
                h=torch.where(node_valid[...,None],h,0.)
        return h

    def forward(self,state,pads0,pads1,theta,geometry,dt):
        B,N,_=state.shape;ratio=(dt/DT_REF).reshape(B,1,1)
        h=self.encode(state,pads0,pads1,theta,geometry,dt)
        if self.direct_displacement:
            velocity_output=self.output(h)
            if self.affine_material:
                # Affine in the four normalized material features for a fixed
                # current state; recurrent trajectories can still be nonlinear.
                fields=velocity_output.reshape(B,N,5,3)
                delta=material_features(theta)-material_features(self.material_reference)
                velocity_output=fields[:,:,0]+(fields[:,:,1:]*delta[:,None,:,None]).sum(2)
            if self.residual_material:
                # The existing material-conditioned function is retained. This
                # correction is zero at the reference for every state/action,
                # even after training; all supervision remains absolute state.
                fields=self.material_residual(h).reshape(B,N,4,3)
                delta=material_features(theta)-material_features(self.material_reference)
                velocity_output=velocity_output+(fields*delta[:,None,:,None]).sum(2)
            interval_velocity=velocity_output*self.velocity_scale
            if self.deformation_memory:
                # Multiplicative deformation updates retain positive orientation
                # in exact arithmetic. F is a recurrent state, never a future
                # target read by forward(). The APIC C fields remain unused.
                identity=torch.eye(3,dtype=state.dtype,device=state.device)
                F=state[...,6:15].reshape(B,N,3,3)+identity
                generator=(self.deformation_transition(h)*self.output_scale[3:12]).reshape(B,N,3,3)
                increment=torch.matrix_exp(generator*ratio[...,None])-identity
                next_F=state[...,6:15]+(increment@F).reshape(B,N,9)
                return torch.cat((state[...,:3]+interval_velocity*ratio,interval_velocity,next_F,state[...,15:]),-1)
            return torch.cat((state[...,:3]+interval_velocity*ratio,interval_velocity,state[...,6:]),-1)
        next_dynamic=state[...,3:]+self.output(h)*self.output_scale*ratio
        next_x=state[...,:3]+.5*(state[...,3:6]+next_dynamic[...,:3])*ratio
        if self.kinematic_only:
            # Keep the 24-column data interface, but auxiliary F/C columns are
            # carried unchanged and are explicitly not physical predictions.
            return torch.cat((next_x,next_dynamic[...,:3],state[...,6:]),-1)
        return torch.cat((next_x,next_dynamic),-1)


def rollout(model,initial,pads,theta,geometry,times):
    state=initial;states=[]
    for t in range(pads.shape[1]-1):
        state=model(state,pads[:,t],pads[:,t+1],theta,geometry,times[:,t+1]-times[:,t])
        states.append(state)
    return torch.stack(states,1)


def dynamics_loss(prediction,target,model,scale_m):
    # Metric positional target is 3 mm regardless of object diameter.
    position_scale=(.003/scale_m).reshape(-1,1,1,1)
    x=((prediction[...,:3]-target[...,:3])/position_scale).square().mean()
    error=((prediction[...,3:]-target[...,3:])/model.loss_scale[3:]).square()
    if model.deformation_memory:return x+.2*error[...,:3].mean()+.05*error[...,3:12].mean()
    if model.kinematic_only:return x+.2*error[...,:3].mean()
    return x+.2*error[...,:3].mean()+.05*error[...,3:12].mean()+.05*error[...,12:].mean()


def observation_loss(predicted_surface,target_surface,predicted_tracks,target_tracks,visible,
                     surface_weight=.5,track_weight=.5,huber_delta_mm=5.):
    """Same symmetric-distance and track Huber definitions as objective.score.

    Tensors in metres; return one loss per time step in millimetres. Visibility
    and material-to-observation bindings are fixed measured data, not optimized.
    """
    # Subtract coordinates directly: the GEMM distance identity loses precision
    # for millimetre separations in metre-valued coordinates.
    distance=torch.cdist(predicted_surface,target_surface,compute_mode='donot_use_mm_for_euclid_dist')*1000
    surface=.5*(distance.amin(-1).mean(-1)+distance.amin(-2).mean(-1))
    track=torch.linalg.vector_norm(predicted_tracks-target_tracks,dim=-1)*1000
    robust=torch.where(track<huber_delta_mm,.5*track.square()/huber_delta_mm,track-.5*huber_delta_mm)
    count=visible.sum(-1)
    if torch.any(count==0):raise ValueError('No visible tracks')
    return surface_weight*surface+track_weight*(robust*visible).sum(-1)/count


def confidence_gradient(losses,parameters,weights):
    """Each material column uses its own frozen interaction weights.

    losses: one scalar per interaction; parameters: shared four-vector. This is
    a column-wise update, not differentiation of one averaged confidence score.
    """
    if weights.shape!=(len(losses),4):raise ValueError('Need four confidence columns')
    jacobian=torch.stack([torch.autograd.grad(loss,parameters,retain_graph=True)[0] for loss in losses])
    return (jacobian*weights).sum(0)/weights.sum(0).clamp_min(1e-8)


def initialize_goal_module(module,state_dict,source_architecture):
    """Optionally add a zero goal-offset projection without changing behavior."""
    source_pointwise=source_architecture.get('pointwise_goal',False)
    expected={k:v for k,v in module.architecture.items() if k!='pointwise_goal'}
    if {k:v for k,v in source_architecture.items() if k!='pointwise_goal'}!=expected:
        raise ValueError('Goal-module initialization cannot change the existing architecture')
    if source_pointwise and not module.pointwise_goal:
        raise ValueError('Cannot discard an initialized pointwise goal representation')
    values=dict(state_dict)
    keys=('goal_condition.weight','goal_condition.bias')
    if any(key in values for key in keys)!=source_pointwise:
        raise ValueError('Goal representation disagrees with checkpoint architecture')
    if module.pointwise_goal and not source_pointwise:
        for key in keys:values[key]=torch.zeros_like(module.state_dict()[key])
    module.load_state_dict(values)


def goal_point_features(module,state,geometry,goal):
    inputs=torch.cat((state[...,:6],geometry['canonical']),-1)
    if not module.pointwise_goal:return module.points(inputs)
    # Fuse the indexed target before the pointwise nonlinearity and pooling.
    # Pooling a separate linear offset branch would still lose correspondence.
    hidden=module.points[0](inputs)+module.goal_condition(goal-state[...,:3])
    return module.points[2](module.points[1](hidden))


class GoalPolicy(nn.Module):
    """Shared actor consumes only visible-style x/v, geometry, goal and estimated material."""
    def __init__(self,hidden=128,pointwise_goal=False):
        super().__init__()
        self.pointwise_goal=pointwise_goal;self.architecture=dict(hidden=hidden)
        self.points=mlp(9,hidden,hidden)
        self.goals=mlp(3,hidden,hidden)
        self.actor=mlp(2*hidden+4+52+1,hidden,14)
        nn.init.normal_(self.actor[-1].weight,std=1e-3);nn.init.zeros_(self.actor[-1].bias)
        if pointwise_goal:
            self.architecture['pointwise_goal']=True
            self.goal_condition=nn.Linear(3,hidden)
            nn.init.zeros_(self.goal_condition.weight);nn.init.zeros_(self.goal_condition.bias)

    def forward(self,state,geometry,goal,pads,theta):
        surface=geometry['surface_mask'].to(state.dtype)[...,None]
        features=goal_point_features(self,state,geometry,goal)
        pooled=(features*surface).sum(1)/surface.sum(1).clamp_min(1)
        target=(self.goals(goal)*surface).sum(1)/surface.sum(1).clamp_min(1)
        # Two grippers, each with translation 3, rotation 3, aperture 1.
        return torch.tanh(self.actor(torch.cat((pooled,target,material_features(theta),pads.flatten(1),geometry['scale_m'][:,None]),-1))).reshape(-1,2,7)


def rotation_matrix(rotvec):
    """Smooth exponential map, including finite derivatives at zero rotation."""
    x,y,z=rotvec.unbind(-1);zero=torch.zeros_like(x)
    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*rotvec.shape[:-1],3,3)
    angle=torch.linalg.vector_norm(rotvec,dim=-1)
    a=torch.sinc(angle/math.pi)
    b=.5*torch.sinc(angle/(2*math.pi)).square()
    eye=torch.eye(3,device=rotvec.device,dtype=rotvec.dtype)
    return eye+a[...,None,None]*K+b[...,None,None]*(K@K)


def apply_gripper_action(pads,action,scale_m,translation_m=.004,rotation_rad=.03,aperture_m=.002):
    """Differentiable rigid gripper transform plus bounded aperture increment."""
    B=pads.shape[0];old=pads.reshape(B,2,2,13);p=old[...,:3]
    center=p.mean(2);axis=p[:,:,1]-p[:,:,0]
    width=torch.linalg.vector_norm(axis,dim=-1,keepdim=True).clamp_min(1e-5)
    scale=scale_m.reshape(B,1,1)
    delta=action[...,:3]*translation_m/scale
    R=rotation_matrix(action[...,3:6]*rotation_rad)
    dw=action[...,6:7]*aperture_m/scale
    # Smoothly reduce aperture range for near-closed fingers.
    dw=dw*torch.tanh(width/(2*aperture_m/scale))
    sign=torch.tensor([-1.,1.],device=pads.device,dtype=pads.dtype).reshape(1,1,2,1)
    offset=p-center[:,:,None]+sign*.5*dw[:,:,None]*axis[:,:,None]/width[:,:,None]
    next_p=center[:,:,None]+delta[:,:,None]+torch.einsum('bgij,bgsj->bgsi',R,offset)
    old_R=old[...,3:12].reshape(B,2,2,3,3)
    next_R=R[:,:,None]@old_R
    values=torch.cat((next_p,next_R.flatten(-2),old[...,12:13]),-1)
    active=old[...,12:13]>.5
    return torch.where(active,values,old).reshape(B,4,13)


def policy_action_step(actor,state,geometry,goal,current_pads,theta,reference_pads,command):
    """Use the same whole-window action domain in imagined and physical control."""
    action=actor(state,geometry,goal,current_pads,theta)
    if getattr(actor,'bounded_actions',False):
        command=command+.25*action
        next_pads=apply_gripper_action(reference_pads,command.tanh(),geometry['scale_m'],
            translation_m=.012,rotation_rad=.08,aperture_m=.006)
    else:
        next_pads=apply_gripper_action(current_pads,action,geometry['scale_m'])
    return next_pads,action,command


class GoalValue(nn.Module):
    def __init__(self,hidden=128,pointwise_goal=False):
        super().__init__()
        self.pointwise_goal=pointwise_goal;self.architecture=dict(hidden=hidden)
        self.points=mlp(9,hidden,hidden);self.goals=mlp(3,hidden,hidden)
        self.value=mlp(2*hidden+4+52+1,hidden,1)
        if pointwise_goal:
            self.architecture['pointwise_goal']=True
            self.goal_condition=nn.Linear(3,hidden)
            nn.init.zeros_(self.goal_condition.weight);nn.init.zeros_(self.goal_condition.bias)

    def forward(self,state,geometry,goal,theta,pads):
        surface=geometry['surface_mask'].to(state.dtype)[...,None]
        features=goal_point_features(self,state,geometry,goal)
        h=(features*surface).sum(1)/surface.sum(1).clamp_min(1)
        target=(self.goals(goal)*surface).sum(1)/surface.sum(1).clamp_min(1)
        return self.value(torch.cat((h,target,material_features(theta),pads.flatten(1),geometry['scale_m'][:,None]),-1)).squeeze(-1)
