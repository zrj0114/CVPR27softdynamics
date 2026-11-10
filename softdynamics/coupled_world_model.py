"""Causal recurrent dynamics and a teacher-anchored joint E/nu response head.

The recurrent state is learned, not a claimed physical deformation gradient.
Local fixed-state secants and full-history trajectory differences stay distinct.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .shared_world_model import SharedWorldModel, DT_REF, mlp, dynamics_loss


def load_coupled(path, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle.get('schema') != 'softdynamics.coupled-temporal-checkpoint.v1':
        raise ValueError('Expected a coupled temporal checkpoint')
    coordinates = bundle['material_coordinates']
    model = CoupledWorld(bundle['architecture'], bundle['tau'], coordinates['reference_E_pa'],
                         coordinates['reference_nu'], coordinates['nu_scale']).to(device)
    model.load_state_dict(bundle['model'])
    return model, bundle


class TemporalWorld(SharedWorldModel):
    def __init__(self, **architecture):
        super().__init__(**architecture)
        if not self.kinematic_only or not self.direct_displacement or self.material_dim != 2:
            raise ValueError('Temporal world requires the two-parameter kinematic displacement model')
        if self.affine_material or self.residual_material or self.deformation_memory:
            raise ValueError('Incompatible legacy material/state head')
        self.memory = nn.GRUCell(architecture['hidden'], architecture['hidden'])

    def remember(self, h, memory):
        if memory is None:
            memory = torch.zeros_like(h)
        return self.memory(h.flatten(0, 1), memory.flatten(0, 1)).reshape_as(h)

    def forward(self, state, pads0, pads1, theta, geometry, dt, memory=None):
        h = self.encode(state, pads0, pads1, theta, geometry, dt)
        memory = self.remember(h, memory)
        velocity = self.output(h + .1 * memory) * self.velocity_scale
        position = state[..., :3] + velocity * (dt / DT_REF)[:, None, None]
        return torch.cat((position, velocity, state[..., 6:]), -1), memory


def call_step(world, args, gradients):
    return checkpoint(world, *args, use_reentrant=False, preserve_rng_state=False) if gradients else world(*args)


def burn_history(world, history, theta, geometry, gradients):
    memory = None
    for i in range(history['states'].shape[1]):
        args = (history['states'][:, i], history['pads0'][:, i], history['pads1'][:, i],
                theta, geometry, history['dt'][:, i], memory)
        _, updated = call_step(world, args, gradients)
        previous = torch.zeros_like(updated) if memory is None else memory
        memory = torch.where(history['present'][:, i, None, None], updated, previous)
    return memory


def forecast(world, batch, geometry, gradients=True, tail_gradient=None):
    """Only the initial state/history are observations. All later states recur.

For long paired prefixes, tail_gradient limits weight backpropagation, while
the entire prefix still runs from frame zero without teacher-state resets.
Inference defaults to a full parameter-gradient path.
"""
    memory = burn_history(world, batch['history'], batch['theta'], geometry, gradients)
    state = batch['states'][:, 0]
    predictions = []
    horizon = batch['pads'].shape[1] - 1
    boundary = max(0, horizon - tail_gradient) if tail_gradient else 0
    for t in range(horizon):
        active = gradients and t >= boundary
        if t == boundary and boundary:
            state = state.detach()
            memory = memory.detach()
        with torch.set_grad_enabled(active):
            state, memory = call_step(world, (state, batch['pads'][:, t], batch['pads'][:, t+1],
                batch['theta'], geometry, batch['times'][:, t+1]-batch['times'][:, t], memory), active)
        predictions.append(state)
    return torch.stack(predictions, 1)


def response_summary(jacobian, tau):
    """J is residual-dimension by parameter, in the teacher's q coordinates."""
    gram = jacobian.T @ jacobian
    strength = torch.sqrt(torch.diagonal(gram).clamp_min(1e-20))
    score = strength / (strength + tau)
    rho = (gram[0, 1] / (strength[0]*strength[1]).clamp_min(1e-20)).clamp(-1, 1)
    return score, rho, gram


class CoupledWorld(nn.Module):
    def __init__(self, architecture, tau, reference_E=3500., reference_nu=.27, nu_scale=.2):
        super().__init__()
        self.world = TemporalWorld(**architecture)
        hidden = architecture['hidden']
        self.time_encoder = mlp(5, 32, 16)
        self.sequence = nn.GRUCell(2*hidden + 16, 128)
        self.joint_head = nn.Sequential(nn.LayerNorm(384), mlp(384, 128, 3))
        self.register_buffer('tau', torch.as_tensor(tau, dtype=torch.float32))
        self.reference_E, self.reference_nu, self.nu_scale = reference_E, reference_nu, nu_scale

    def theta(self, q):
        return torch.stack((self.reference_E*q[..., 0].exp(), self.reference_nu+self.nu_scale*q[..., 1]), -1)

    def confidence(self, contexts, q):
        # B contexts; each is one ordered six-window observed interaction.
        outputs = []
        for context, coordinate in zip(contexts, q):
            state, geo = context['states'], dict(context['geometry'])
            mask = context['observed']
            geo['node_valid'] = mask
            h = self.world.encode(state, context['pads0'], context['pads1'],
                self.theta(coordinate).expand(len(state), -1), geo, context['dt'])
            mean = (h*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
            maximum = h.masked_fill(~mask[..., None], -torch.inf).amax(1)
            maximum = torch.where(torch.isfinite(maximum), maximum, 0.)
            features = torch.cat((mean, maximum, self.time_encoder(context['time_features'])), -1)
            memory = features.new_zeros(1, 128)
            values = []
            for i in range(len(features)):
                updated = self.sequence(features[i:i+1], memory)
                memory = torch.where(mask[i].any(), updated, memory)
                if bool(mask[i].any()):
                    values.append(memory[0])
            if not values:
                raise ValueError('No visible history for the joint response head')
            values = torch.stack(values)
            raw = self.joint_head(torch.cat((values.mean(0), values.amax(0), values[-1]), -1))
            strength = F.softplus(raw[:2]) * self.tau
            rho = raw[2].tanh()
            off = rho * strength.prod()
            gram = torch.stack((strength[0].square(), off, off, strength[1].square())).reshape(2, 2)
            outputs.append((strength/(strength+self.tau), rho, gram))
        return tuple(torch.stack([v[i] for v in outputs]) for i in range(3))

    def local_response(self, context, q, radii):
        # The prefix represents the nominal physical snapshot and is held
        # fixed across all parameter probes, matching teacher.restore().
        memory = burn_history(self.world, context['history'], context['base_theta'],
                              context['geometry'], self.training)
        columns = []
        for j in range(2):
            delta = torch.zeros_like(q)
            delta[j] = radii[j]
            predictions = []
            for sign in (-1, 1):
                theta = self.theta(q + sign*delta).expand(len(context['states']), -1)
                state, _ = call_step(self.world, (context['states'], context['pads0'], context['pads1'],
                    theta, context['geometry'], context['dt'], memory), self.training)
                predictions.append(state[..., :3])
            response = (predictions[1]-predictions[0])/(2*radii[j])
            response = response * context['response_scale'][:, None, None]
            columns.append(response[context['target_visible']].flatten())
        return torch.stack(columns, -1)

    def forward(self, batch, geometry, auxiliary, pair=None, local_response=False, pair_tail=16):
        predicted = forecast(self.world, batch, geometry, self.training)
        target = batch['states'][:, 1:]
        local = dynamics_loss(predicted, target, self.world, geometry['scale_m'])
        first = dynamics_loss(predicted[:, :1], target[:, :1], self.world, geometry['scale_m'])
        score, rho, _ = self.confidence(auxiliary['contexts'], auxiliary['q'])
        mask = auxiliary['valid']
        score_loss = (score-auxiliary['score']).square()[mask].mean()
        both = mask.all(-1) & auxiliary['correlation_valid']
        correlation = F.smooth_l1_loss(rho[both], auxiliary['rho'][both]) if both.any() else rho.sum()*0
        anchor = consistency = local*0
        if local_response:
            context = auxiliary['contexts'][0]
            predicted_J = self.local_response(context, auxiliary['q'][0], auxiliary['radii'][0])
            teacher_J = auxiliary['J'][0]
            energy = teacher_J.square().sum(0).clamp_min(self.tau.square())
            anchor = ((predicted_J-teacher_J).square().sum(0)/energy)[mask[0]].mean()
            js, jr, _ = response_summary(predicted_J, self.tau)
            consistency = (score[0]-js).square()[mask[0]].mean()
            if both[0]:
                consistency = consistency + .25*(rho[0]-jr).square()
        absolute = response = local*0
        if pair is not None:
            pb, pg = pair
            pp = forecast(self.world, pb, pg, self.training, tail_gradient=pair_tail)
            pp, pt = pp[:, -pair_tail:], pb['states'][:, 1:][:, -pair_tail:]
            absolute = dynamics_loss(pp, pt, self.world, pg['scale_m'])
            actual = (pt[1, ..., :3]-pt[0, ..., :3])*pg['scale_m'][0]
            estimated = (pp[1, ..., :3]-pp[0, ..., :3])*pg['scale_m'][0]
            floor = (.003*pb['parameter_distance'].clamp_min(.1)).square()
            response = (estimated-actual).square().mean()/actual.square().mean().clamp_min(floor)
        return torch.stack((local, first, score_loss, correlation, anchor, consistency, absolute, response))
