"""Causal observed-state reconstruction and independent one-step material fit."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch

from .elastic_step_objective import independent_steps
from .io import load_npz, read_json
from .shared_world_inverse import RealEpisode, phase_objective
from .shared_world_model import DT_REF
from .joint_world import interaction_context
from .teacher_contact import replace_visual_actions


def reconstruct_observed_states(initial, tracks, visible, times, projection, neighbors,
                                scale_m, smoothing=.01, ridge=1e-6, minimum_tracks=16,
                                initial_velocity_known=False):
    """Each position estimate uses tracks at that frame and the initial binding.

    Spatial regularization fills unobserved interior nodes. Velocities use only
    the current and previous position estimates. No world-model output or MPM
    intermediate state enters this estimator.
    """
    initial, tracks, times = np.asarray(initial), np.asarray(tracks), np.asarray(times)
    P = np.asarray(projection, dtype=np.float64)
    visible = np.asarray(visible, bool)
    T, M, dimensions = tracks.shape
    N = len(initial)
    if (dimensions != 3 or initial.shape != (N, 24) or P.shape != (M, N) or
        visible.shape != (T, M) or times.shape != (T,) or T < 2 or
        not np.all(np.diff(times)>0) or scale_m<=0 or smoothing<0 or ridge<=0):
        raise ValueError('Invalid independent observed-state inputs')
    if not np.isfinite(initial).all() or not np.isfinite(P).all():
        raise ValueError('Nonfinite initial state or observation binding')
    neighbors = np.asarray(neighbors)
    L = np.eye(N, dtype=np.float64)
    for i, row in enumerate(neighbors):
        if len(row):
            np.add.at(L[i], row, -1/len(row))
    regularizer = smoothing*(L.T@L)+ridge*np.eye(N)
    states = np.repeat(initial[None], T, axis=0).astype(np.float32)
    finite_initial = np.isfinite(tracks[0]).all(-1)
    valid = visible & np.isfinite(tracks).all(-1) & finite_initial[None]
    enough = valid.sum(1)>=min(minimum_tracks, M)
    # Initial geometry does not establish the real object's initial velocity.
    input_valid = enough.copy(); input_valid[0] = bool(initial_velocity_known and enough[0])
    residuals = np.full(T, np.nan)
    previous_mask = None; factor = None
    for t in range(1, T):
        if not enough[t]:
            states[t] = states[t-1]
            states[t, :, 3:6] = 0
            input_valid[t] = False
            continue
        mask = valid[t]
        if previous_mask is None or not np.array_equal(mask, previous_mask):
            A = P[mask].T@P[mask]+regularizer
            factor = cho_factor(A, lower=True, check_finite=False)
            previous_mask = mask.copy()
        observed = tracks[t, mask]-tracks[0, mask]
        displacement = cho_solve(factor, P[mask].T@observed, check_finite=False)
        states[t, :, :3] = initial[:, :3]+displacement/scale_m
        states[t, :, 3:6] = (states[t, :, :3]-states[t-1, :, :3])*DT_REF/(times[t]-times[t-1])
        input_valid[t] = bool(enough[t] and enough[t-1])
        residuals[t] = np.sqrt(np.mean((P[mask]@displacement-observed)**2))*1000
    if not np.isfinite(states).all():
        raise FloatingPointError('Observed-state reconstruction is nonfinite')
    transitions_valid = input_valid[:-1] & enough[1:] & (valid[:-1] & valid[1:]).any(1)
    finite = residuals[np.isfinite(residuals)]
    return states, transitions_valid, dict(position_source='Current observed tracks with fixed initial binding',
        velocity_source='Current minus previous observed-state estimate', smoothing=smoothing, ridge=ridge,
        prediction_feedback=False, candidate_mpm_calls=0, frames=T,
        initial_velocity_known=initial_velocity_known,
        usable_transitions=int(transitions_valid.sum()), total_transitions=T-1,
        mean_track_reconstruction_rmse_mm=float(finite.mean()) if len(finite) else None,
        max_track_reconstruction_rmse_mm=float(finite.max()) if len(finite) else None)


class ObservedStepEpisode(RealEpisode):
    """One parameter vector shared by independent observed transitions."""
    def __init__(self, root, obj, episode, device):
        super().__init__(root, obj, episode, device)
        plan = read_json(self.root/'plan.json')
        fit = read_json(plan['fit_run']+'/plan.json')
        entry = next(e for o in fit['objects'] if o['object']==obj for e in o['episodes'] if e['episode']==episode)
        row = read_json(entry['row'])
        observation = load_npz(row['observation'])
        tracks, visible = observation['tracks'], observation['track_visible']
        T = len(tracks)
        if not np.array_equal(observation['frames'], np.arange(T)) or T>self.pads.shape[1]:
            raise ValueError('Observation frames must align with recorded actions')
        states, valid, stats = reconstruct_observed_states(self.initial[0].cpu().numpy(), tracks, visible,
            self.times[0, :T].cpu().numpy(), self.projection['track_matrix'].cpu().numpy(),
            self.geometry['neighbors'][0].cpu().numpy(), self.scale)
        frames = self.frames.cpu().numpy()
        keep = (frames>0) & (frames<T)
        keep[keep] &= valid[frames[keep]-1]
        self.frames = self.frames[torch.as_tensor(keep, device=device)]
        if not len(self.frames):
            raise ValueError('No observed one-step transitions available')
        self.counts = self.counts[keep]
        self.surface = self.surface[torch.as_tensor(keep, device=device)]
        self.tracks = self.tracks[torch.as_tensor(keep, device=device)]
        self.visible = self.visible[torch.as_tensor(keep, device=device)]
        self.observed_states = torch.tensor(states[None], device=device)
        self.pads, self.times = self.pads[:, :T], self.times[:, :T]
        self.observed_tracks = torch.as_tensor(tracks, dtype=torch.float32, device=device)
        self.observed_visible = torch.as_tensor(visible, dtype=torch.bool, device=device)
        visual = load_npz(row['real'])
        visual = replace_visual_actions(visual,self.actions)
        if len(visual['times']) != T:
            raise ValueError('Confidence observations and recorded action frames must align')
        visual['positions'] = ((tracks-visual['origin_m'])/float(visual['scale_m'])).astype('f')
        visual['valid'] = np.asarray(visual['valid'], bool) & visible & np.isfinite(tracks).all(-1)
        self.confidence_context = interaction_context(visual, self.meta)
        self.state_estimation = stats
        self.state_estimation['action_provenance']=self.action_provenance
        # Require every scored phase before starting a material optimizer.
        phase_objective(torch.ones(len(self.frames), device=device), self.frames,
            self.meta['clip_start'], self.meta['clip_end'], self.config['phase_weights'])

    def predict(self, model, theta, gradients=True):
        if not model.kinematic_only:
            raise ValueError('Observed states provide x/v, not hidden MPM deformation history')
        batch = dict(states=self.observed_states, pads=self.pads, times=self.times, theta=theta[None])
        prediction = independent_steps(model, batch, self.geometry, gradients)
        return torch.cat((self.initial, prediction[0]), 0)

    def loss(self, states):
        """Predict observed material-track increments over a single interval.

        Anchoring the increment to the current observed track avoids asking E
        or nu to compensate for the spatial reconstruction residual. Only the
        next-frame observation is the target. No previous prediction is used.
        """
        frames = self.frames
        current = self.observed_states[0, frames-1, :, :3]
        displacement = (states[frames, :, :3]-current)*self.scale
        delta = torch.einsum('tnc,pn->tpc', displacement, self.projection['track_matrix'])
        predicted = self.observed_tracks[frames-1]+delta
        target = self.observed_tracks[frames]
        mask = self.observed_visible[frames-1] & self.observed_visible[frames]
        mask &= torch.isfinite(target).all(-1) & torch.isfinite(self.observed_tracks[frames-1]).all(-1)
        if not mask.any(1).all():
            raise ValueError('Scored transition lacks jointly visible tracks')
        if not torch.isfinite(predicted[mask]).all():
            raise FloatingPointError('Nonfinite prediction on a scored observed track')
        difference = torch.where(mask[..., None], predicted-target, 0.)*1000
        error = torch.linalg.vector_norm(difference, dim=-1)
        delta_mm = self.config['huber_delta_mm']
        robust = torch.where(error<=delta_mm, .5*error.square()/delta_mm, error-.5*delta_mm)
        per_frame = (robust*mask).sum(1)/mask.sum(1)
        return phase_objective(per_frame, frames, self.meta['clip_start'], self.meta['clip_end'], self.config['phase_weights'])
