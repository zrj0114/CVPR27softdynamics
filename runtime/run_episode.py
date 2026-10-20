"""Finite material/grid boundaries with nonadhesive contact and release.

Each gripper keeps its original two-sided patch construction and initial state.
Contact impulses oppose only approaching normal motion; tangential impulses
are Coulomb limited. There is no time-based release or object-track feedback.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from paths import CODE, ROOT, DATA
sys.path.insert(0, str(CODE / 'gs-layer'))
from material_contact_kernels import (wp, MPMStateStruct, MPMModelStruct,
    MPMWARPDiff, MaterialContact, scatter_material_velocity, accumulate_material_contact, material_grid_boundary)
from gaussian_io import load_gaussians
from deterministic_transfer import DeterministicTransfer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_episode(episode):
    prepared = ROOT / f'queries/episode_{episode}'
    with np.load(prepared / 'actions.npz') as a:
        poses, valid, stamps = a['poses'], a['valid'], a['source_timestamps_us']
        frames = a['source_frames']
    assert np.array_equal(np.isfinite(poses).all(axis=(-1, -2)), valid)
    assert np.array_equal(frames, np.arange(len(poses)))
    times = (stamps - stamps[0]) / 1e6
    assert (np.diff(times) > 0).all()
    meta = json.loads((prepared/'episode.json').read_text())
    if sha(prepared/'actions.npz')!=meta['source_sha256']['actions']:raise ValueError('Query action hash mismatch')
    refs = []
    for g in range(poses.shape[1]):
        candidates = np.flatnonzero(valid[:, g].all(axis=1) & (frames >= meta['clip_start']) & (frames <= meta['clip_end']))
        if not len(candidates):
            raise ValueError(f'Gripper {g}: no valid reference pair inside annotated interaction')
        refs.append(int(candidates[0]))
    if refs!=meta['reference_frames']:raise ValueError('Query contact reference mismatch')
    # Geometry is loaded exclusively from the shared asset/placed state.
    return prepared, None, poses, valid, times, refs, meta


def material_patch(xyz, reference):
    """The original grid-mode patch formula; only reference frame is variable."""
    R = Rotation.from_matrix(reference[:, :3, :3]).mean().as_matrix()
    origin = reference[:, :3, 3].mean(0)
    q = (xyz - origin) @ R
    near = np.linalg.norm(q[:, 1:] - [.09, -.02], axis=1) < .014
    if near.sum() <= 40:
        raise ValueError(f'Original patch requires >40 particles, found {near.sum()}')
    plane_centers = []
    for side in [0, 1]:
        plane_points = (xyz[near] - reference[side, :3, 3]) @ reference[side, :3, :3]
        center = (origin + R @ np.array([0., .09, -.02]) - reference[side, :3, 3]) @ reference[side, :3, :3]
        center[0] = (plane_points[:, 0].min() + .01 if side == 0 else plane_points[:, 0].max() - .01)
        plane_centers.append(center)
    near = (np.abs(q[:, 1] - .09) < .025 + .004) & (np.abs(q[:, 2] + .02) < .018 + .004)
    labels = np.full(len(xyz), -1, np.int32)
    labels[near & (q[:, 0] < np.median(q[near, 0]))] = 0
    labels[near & (q[:, 0] >= np.median(q[near, 0]))] = 1
    local = np.zeros_like(xyz)
    for side in [0, 1]:
        ids = labels == side
        local[ids] = (xyz[ids] - reference[side, :3, 3]) @ reference[side, :3, :3]
    return labels, local, np.asarray(plane_centers)


def interval_motion(p0, p1, valid0, valid1, interval, shift):
    """Never finite-difference or interpolate through missing measurements."""
    enabled = valid0 & valid1
    transforms = np.zeros((2, 7), np.float32)
    transforms[:, 6] = 1  # Identity storage for disabled sides; never activated.
    linear = np.zeros((2, 3), np.float32)
    angular = np.zeros((2, 3), np.float32)
    if enabled.any():
        a, b = p0[enabled], p1[enabled]
        transforms[enabled] = np.column_stack([a[:, :3, 3] + shift, Rotation.from_matrix(a[:, :3, :3]).as_quat()]).astype(np.float32)
        linear[enabled] = (b[:, :3, 3] - a[:, :3, 3]) / interval
        angular[enabled] = Rotation.from_matrix(b[:, :3, :3] @ a[:, :3, :3].transpose(0, 2, 1)).as_rotvec() / interval
    assert np.isfinite(transforms).all() and np.isfinite(linear).all() and np.isfinite(angular).all()
    return enabled.astype(np.int32), transforms, linear, angular


def initialize(xyz, patches, times, device, padding_cells=0, contact_friction=2.,sliding_friction=.8, young_modulus=4000., rest_volumes=None, masses=None, initial_F=None, poisson_ratio=.3, ground_friction=.8):
    # Original state, material, grid and ground initialization.
    dx = .4 / 128
    grid = 128 + 2 * padding_cells
    domain = grid * dx
    shift = np.array([.2, .1, .2]) - xyz.mean(0) + padding_cells * dx
    x = torch.tensor(xyz + shift, dtype=torch.float32, device=device)
    if rest_volumes is None:rest_volumes=np.full(len(xyz),(.003/2)**3)
    if masses is None:masses=rest_volumes*100.
    vol = torch.as_tensor(rest_volumes,dtype=torch.float32,device=device)
    states = []
    for _ in range(2):
        state = MPMStateStruct(); state.init(len(xyz), device=device)
        state.from_torch(x, vol, tensor_velocity=torch.zeros_like(x), n_grid=grid, grid_lim=domain, device=device, requires_grad=False)
        state.reset_density(torch.as_tensor(masses/rest_volumes,dtype=torch.float32,device=device), device=device, requires_grad=False, update_mass=True)
        if initial_F is not None:
            state.particle_F_trial.assign(initial_F.astype(np.float32))
            state.particle_F.assign(initial_F.astype(np.float32))
        states.append(state)
    model = MPMModelStruct(); model.init(len(xyz), device=device)
    model.init_other_params(n_grid=grid, grid_lim=domain, device=device)
    model.n_particles = len(xyz); model.gravitational_accelaration = wp.vec3(0., -9.8, 0.)
    solver = MPMWARPDiff(); solver.initialize(len(xyz), n_grid=grid, grid_lim=domain, device=device)
    solver.set_E_nu(model, young_modulus, poisson_ratio, device=device)
    solver.prepare_mu_lam(model, states[0], device=device)
    momentum = wp.zeros((grid,) * 3, dtype=wp.vec3, device=device)
    weight = wp.zeros((grid,) * 3, dtype=float, device=device)
    contacts = []
    for labels, local, planes in patches:
        c = MaterialContact()
        c.enabled = wp.zeros(2, dtype=int, device=device)
        c.labels = wp.array(labels, dtype=int, device=device)
        c.local = wp.array(local, dtype=wp.vec3, device=device)
        c.transforms = wp.zeros(2, dtype=wp.transform, device=device)
        c.linear = wp.zeros(2, dtype=wp.vec3, device=device)
        c.angular = wp.zeros(2, dtype=wp.vec3, device=device)
        c.correction = wp.zeros(2, dtype=wp.vec3, device=device)
        c.mass = wp.zeros((grid,) * 3, dtype=float, device=device)
        c.right_mass = wp.zeros((grid,) * 3, dtype=float, device=device)
        c.momentum = momentum; c.weight = weight
        c.inv_dx = model.inv_dx; c.start_time = 0.; c.end_time = float(times[-1] + 1.)
        c.gain = 35.; c.ground = float(-.01 + shift[1]); c.friction = ground_friction
        c.contact_friction = contact_friction; c.sliding_friction = sliding_friction
        c.mode = 2; c.radius = .014; c.footprint_y = .025; c.footprint_z = .018
        c.plane_centers = wp.array(planes, dtype=wp.vec3, device=device)
        contacts.append(c)
        solver.pre_p2g_operations.append(scatter_material_velocity); solver.impulse_params.append(c)
        solver.grid_postprocess.append(accumulate_material_contact)
        solver.collider_params.append(c); solver.modify_bc.append(None)
    # All sides use the same free grid velocity; apply the sum and ground once.
    solver.grid_postprocess.append(material_grid_boundary)
    solver.collider_params.append(contacts[0]); solver.modify_bc.append(None)
    return solver, model, states, contacts, shift


def stable_max_dt(young_modulus, dx, requested_dt, nu=.3, rho=100.):
    """CFL bound for the same isotropic constitutive law used by the solver."""
    if not (math.isfinite(young_modulus) and young_modulus > 0 and math.isfinite(requested_dt) and requested_dt > 0):
        raise ValueError('Young modulus and time step must be finite and positive')
    if not (math.isfinite(nu) and 0 <= nu < .5 and math.isfinite(rho) and rho > 0):
        raise ValueError('Require 0 <= nu < 0.5 and positive density')
    return min(requested_dt, .35 * dx / math.sqrt(young_modulus * (1-nu) / (rho * (1+nu) * (1-2*nu))))


def run(args, out):
    prepared, _, poses, valid, times, refs, meta = load_episode(args.episode)
    asset_folder=args.asset.resolve();asset=json.loads((asset_folder/'asset.json').read_text())
    for name,digest in asset['payload_sha256'].items():
        if sha(asset_folder/name)!=digest:raise ValueError(f'Asset payload changed: {name}')
    with np.load(asset_folder/'body.npz',allow_pickle=False) as z:body=dict(z)
    initial_path=args.initial_state or ROOT/f'initial_states/episode_{args.episode}.npz'
    initial_meta=json.loads(initial_path.with_suffix('.json').read_text())
    if initial_meta['state_sha256']!=sha(initial_path):raise ValueError('Initial state hash mismatch')
    with np.load(initial_path,allow_pickle=False) as z:initial=dict(z)
    if str(initial['asset_sha256'])!=sha(asset_folder/'asset.json'):raise ValueError('Initial state belongs to another asset')
    np.testing.assert_array_equal(initial['material_ids'],body['material_ids'])
    xyz=initial['positions_world_m'];initial_F=initial['deformation_gradient']
    if xyz.shape!=(len(body['material_ids']),3) or initial_F.shape!=(len(xyz),3,3):raise ValueError('Invalid initial state shape')
    if not np.isfinite(xyz).all() or not np.isfinite(initial_F).all() or np.linalg.det(initial_F).min()<=0:raise ValueError('Invalid initial state position/deformation')
    if args.young_modulus is None:args.young_modulus=float(asset['material']['E_pa'])
    if args.poisson_ratio is None:args.poisson_ratio=float(asset['material']['nu'])
    if args.contact_friction is None:args.contact_friction=float(asset['contact']['gripper_static_friction'])
    if args.sliding_friction is None:args.sliding_friction=float(asset['contact']['gripper_sliding_friction'])
    if args.ground_friction is None:args.ground_friction=float(asset['contact']['ground_friction'])
    if args.velocity_decay_rate is None:args.velocity_decay_rate=float(asset.get('damping',{}).get('velocity_decay_rate_s_inv',0.))
    physical_parameters=dict(E_pa=args.young_modulus,nu=args.poisson_ratio,
        gripper_static_friction=args.contact_friction,gripper_sliding_friction=args.sliding_friction,
        ground_friction=args.ground_friction,velocity_decay_rate_s_inv=args.velocity_decay_rate)
    if not all(math.isfinite(v) for v in physical_parameters.values()):raise ValueError('Non-finite physical parameter')
    if not (args.young_modulus>0 and 0<=args.poisson_ratio<.5 and 0<=args.sliding_friction<=args.contact_friction and args.ground_friction>=0 and args.velocity_decay_rate>=0):raise ValueError('Invalid physical parameter range')
    T = len(times)
    if args.last_frame is not None:T=min(T,args.last_frame+1)
    times=times[:T];poses=poses[:T];valid=valid[:T]
    # Material patch geometry uses the SAME original formula, evaluated on
    # the canonical body's placed initial positions; no fitted future shape.
    # Reference poses can be beyond a diagnostic truncation, so load them
    # separately rather than fabricate/interpolate missing poses.
    with np.load(prepared/'actions.npz') as a:reference_poses=a['poses']
    patches = [material_patch(xyz, reference_poses[f, g]) for g, f in enumerate(refs)]
    wp.config.kernel_cache_dir = str(ROOT / '.warp-cache'); wp.init()
    solver, model, states, contacts, shift = initialize(xyz, patches, times, args.device, args.grid_padding_cells,args.contact_friction,args.sliding_friction,args.young_modulus,body['rest_volumes_m3'],body['masses_kg'],initial_F,args.poisson_ratio,args.ground_friction)
    transfer = DeterministicTransfer(model, args.device) if args.transfer_mode == 'deterministic' else None
    appearance = load_gaussians(asset_folder/'appearance.npz')
    surface = appearance.xyz
    # Fixed canonical particle IDs are identical for every episode and run.
    save_ids = np.unique(np.r_[body['surface_particle_ids'],cKDTree(body['rest_positions']).query(surface)[1], np.linspace(0, len(xyz) - 1, 8192).astype(int)])
    dtmax = stable_max_dt(args.young_modulus, model.dx, args.max_dt, nu=args.poisson_ratio, rho=100.)
    labels = np.stack([p[0] for p in patches], axis=1)
    local = np.stack([p[1] for p in patches], axis=1)
    manifest = dict(episode=args.episode, frames=T, grippers=len(contacts), referenceFrames=refs,
        referenceRule='First observed-valid pair at/after annotated clip_start, separately for each gripper',
        clipStart=meta['clip_start'], clipEnd=meta['clip_end'], firstSourceFrame=0, lastSourceFrame=T-1,
        particleCount=len(xyz), storedMaterialParticles=len(save_ids), initialVelocity='zero', initialDeformationGradient='Analytic Jacobian of canonical-to-initial-state map',
        E_pa=args.young_modulus, nu=args.poisson_ratio, rho_kg_m3=100, sharedPhysicalParameters=physical_parameters, grid=128+2*args.grid_padding_cells,
        gridDomainMeters=float(model.grid_lim), gridSpacingMeters=float(model.dx),
        gridPaddingCellsPerSide=args.grid_padding_cells, groundYWorld=-.01,
        groundFriction=args.ground_friction, gripperFriction=args.contact_friction,gripperSlidingFriction=args.sliding_friction,maxDtSeconds=dtmax, boundaryFeedback=0, additionalDamping=args.velocity_decay_rate>0,
        velocityDecayRateSInv=args.velocity_decay_rate, dampingRule="Existing grid velocity decay before contact; scale=exp(-rate*dt); zero rate skips kernel",
        worldToSimulationTranslation=shift.tolist(),
        materialPatchCounts=[[int((p[0] == s).sum()) for s in [0,1]] for p in patches],
        multiplyAssociatedMaterialParticles=int(((labels >= 0).sum(1) > 1).sum()),
        multipleGripperRule='Independent side support masses; unilateral impulses from the same free grid velocity are support-weighted and averaged once',
        planeCentersInMarkerCoordinates=[c.plane_centers.numpy().tolist() for c in contacts],
        missingPoseCount=int((~valid).sum()),
        disabledIntervalsByGripperSide=[[np.flatnonzero(~(valid[:-1,g,s] & valid[1:,g,s])).tolist() for s in [0,1]] for g in range(len(contacts))],
        missingPolicy='Disable only that side on intervals with either endpoint missing; resume on first pair of consecutive observed-valid poses; raw missing poses remain NaN',
        missingIntervalIndicesReferTo='Interval i connects source frames i and i+1',
        contact='Original 50x36mm finite material-associated planes, 14mm radius and 10mm inset; approaching relative normal velocity generates a nonnegative normal impulse and Coulomb-limited tangent impulse',
        release='No attractive normal impulse or time-based gate; separating nodes receive zero contact impulse',
        frictionNote=f'Gripper static/sliding friction {args.contact_friction:g}/{args.sliding_friction:g} are explicit interface parameters; ground friction {args.ground_friction:g}',
        sourceInputSha256={'actions.npz':sha(prepared/'actions.npz')},
        codeSha256={name:sha(CODE/name) for name in ['run_episode.py','registration.py','build_asset.py','gs-layer/material_contact_kernels.py','empm/phys_sim/diff_simulator/warp_mpm/mpm_solver_diff.py']},
        unchangedUpstreamCommit='8e241057ca9ec6ad19c3fb7d75dc7407baba8de0',
        transferMode=args.transfer_mode, transferImplementationSha256=sha(CODE/'gs-layer/deterministic_transfer.py'),
        transferReductionOrder='stencil i/j/k ascending, then immutable particle index ascending' if transfer else 'original float32 atomic scatter',
        forwardOnlyTransfer=bool(transfer),
        midSequenceStateResets=False, laterObjectObservationsUsedForSimulation=False,
        compactOutput=args.compact_output, requestedMaxDtSeconds=args.max_dt,
        assetSha256=sha(asset_folder/'asset.json'),assetFolder=str(asset_folder),assetPayloadSha256=asset['payload_sha256'],
        initialStateSha256=sha(initial_path),initialStateMetadata=initial_meta,
        totalMassKg=float(body['masses_kg'].sum()),restVolumeM3=float(body['rest_volumes_m3'].sum()),
        volumeConvention='Constant reference particle volume V0 with Kirchhoff stress; fixed mass rho0*V0; current physical volume J*V0',
        materialIdentity='Same canonical material IDs/rest positions/rest volumes/masses for every episode',
        diagnosticTruncation=args.last_frame is not None,legacyInitialGeometryUsed=False)
    write(out/'run-manifest.json', manifest)
    for source, dest in [('run_episode.py','adapter-source.py'),('gs-layer/material_contact_kernels.py','contact-kernels-source.py')]:
        (out/dest).write_bytes((CODE/source).read_bytes())
    positions = [xyz[save_ids].copy()]
    gradients = [] if args.compact_output else [initial_F[save_ids].copy()]
    rows = []; current = 0; steps_total = 0; started = time.monotonic()
    for f in range(1, T):
        interval = times[f] - times[f-1]; steps = math.ceil(interval/dtmax); dt = interval/steps
        # Time-based rate: use the actual interval substep, not a fixed per-step factor.
        if args.velocity_decay_rate>0:model.grid_v_damping_scale=math.exp(-args.velocity_decay_rate*dt)
        interval_enabled = []
        for g,c in enumerate(contacts):
            enabled, transforms, linear, angular = interval_motion(poses[f-1,g], poses[f,g], valid[f-1,g], valid[f,g], interval, shift)
            c.enabled.assign(enabled); c.transforms.assign(transforms); c.linear.assign(linear); c.angular.assign(angular)
            c.frame_time = float(times[f-1]); interval_enabled.append(enabled.tolist())
        for _ in range(steps):
            if transfer:
                transfer.step(solver, model, states[current], states[1-current], dt)
            else:
                solver.p2g2p_differentiable(model, states[current], states[1-current], dt, device=args.device)
            current = 1-current
        if transfer: transfer.check()
        steps_total += steps
        p = states[current].particle_x.numpy() - shift
        F = states[current].particle_F_trial.numpy(); J = np.linalg.det(F)
        finite = bool(np.isfinite(p).all() and np.isfinite(J).all())
        positive = bool(finite and J.min() > 0)
        margin = float(np.minimum(p+shift, model.grid_lim-p-shift).min()) if finite else None
        positions.append(p[save_ids].astype(np.float32))
        if not args.compact_output: gradients.append(F[save_ids].astype(np.float32))
        counts = []; penetration = []
        for g,c in enumerate(contacts):
            cc = []; pp = []
            for side in [0,1]:
                if not valid[f,g,side]:
                    cc.append(None); pp.append(None); continue
                ids = labels[:,g] == side
                z = (p[ids] - poses[f,g,side,:3,3]) @ poses[f,g,side,:3,:3]
                pc = c.plane_centers.numpy()[side]; dist = (z[:,0]-pc[0]) * (1 if side == 0 else -1)
                active = (np.abs(z[:,1]-pc[1]) < .025) & (np.abs(z[:,2]-pc[2]) < .018) & (dist < .5/model.inv_dx) & (dist > -.03)
                cc.append(int(active.sum())); pp.append(float(np.maximum(-dist[active],0).max()*1000) if active.any() else 0.)
            counts.append(cc); penetration.append(pp)
        row = dict(frame=f, finite=finite, positiveJ=positive, minJ=float(J.min()) if finite else None,
            positionAndDeformationSha256=hashlib.sha256(p.tobytes()+F.tobytes()).hexdigest(),
            maxJ=float(J.max()) if finite else None, domainMarginMeters=margin,
            particlesAtDomainClamp=int((np.minimum(p+shift,model.grid_lim-p-shift).min(1)<=2/model.inv_dx+1.e-6).sum()) if finite else None,
            fractionJBelowPoint1=float((J<.1).mean()) if finite else None,
            centerDisplacementMM=float(np.linalg.norm(p.mean(0)-xyz.mean(0))*1000) if finite else None,
            enabledBoundarySides=interval_enabled, contactParticleCenters=counts, maximumNormalPenetrationMM=penetration,
            elapsedSeconds=time.monotonic()-started)
        rows.append(row)
        good = positive and margin > 2/model.inv_dx+1.e-6
        if f % 25 == 0 or f == T-1 or not good:
            print(json.dumps(row), flush=True)
        if f % 50 == 0 or f == T-1 or not good:
            if not args.compact_output or f == T-1 or not good:
                extra = {} if args.compact_output else dict(deformation=np.stack(gradients))
                np.savez_compressed(out/'trajectory.npz', positions=np.stack(positions),
                    times=times[:f+1], labels=labels[save_ids], marker_poses=poses[:f+1], marker_valid=valid[:f+1],
                    patch_local=local[save_ids], saved_particle_indices=save_ids, **extra)
            write(out/'progress.json', dict(completedFrames=f+1, requestedFrames=T, perFrame=rows))
        if not good:
            raise RuntimeError(f'Frame {f}: finite={finite}, positiveJ={positive}, domain margin={margin}')
    report = dict(completed=True, frames=T, substeps=steps_total, elapsedSeconds=time.monotonic()-started,
        finalParticleStateSha256={name:hashlib.sha256(getattr(states[current],name).numpy().tobytes()).hexdigest()
            for name in ['particle_x','particle_v','particle_C','particle_F_trial']},
        minimumSavedFrameJ=min(r['minJ'] for r in rows), maximumSavedFrameJ=max(r['maxJ'] for r in rows),
        minimumDomainMarginMeters=min(r['domainMarginMeters'] for r in rows), perFrame=rows)
    write(out/'validation.json', report)
    if not args.compact_output:
        np.savez_compressed(out/'final-state.npz', positions=p, deformation=F, labels=labels)
    print(json.dumps({k:v for k,v in report.items() if k!='perFrame'}), flush=True)
    return dict(episode=args.episode, status='completed', frames=T, referenceFrames=refs, elapsedSeconds=report['elapsedSeconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=int, help='Prepared episode identifier')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--transfer-mode', choices=['deterministic','atomic'], default='deterministic',
        help='Fixed-order forward P2G/contact mass reduction; atomic retains the original diagnostic baseline')
    parser.add_argument('--asset',type=Path,default=ROOT/'asset')
    parser.add_argument('--initial-state',type=Path)
    parser.add_argument('--last-frame',type=int,help='Explicit diagnostic truncation; never counted as a full episode')
    parser.add_argument('--max-dt', type=float, default=.0001)
    parser.add_argument('--young-modulus', type=float, default=None, help='Override asset uniform Young modulus for re-identification')
    parser.add_argument('--compact-output', action='store_true', help='Store material positions and diagnostics, without deformation arrays or periodic trajectory checkpoints; physics is unchanged')
    parser.add_argument('--contact-friction', type=float, default=None)
    parser.add_argument('--sliding-friction', type=float, default=None)
    parser.add_argument('--poisson-ratio', type=float, default=None)
    parser.add_argument('--ground-friction', type=float, default=None)
    parser.add_argument('--velocity-decay-rate', type=float, default=None, help='Effective grid velocity decay rate, 1/s; zero disables decay')
    parser.add_argument('--grid-padding-cells', type=int, default=0, help='Add cells to every side while preserving original dx and world-space initial state')
    parser.add_argument('--output-root', type=Path, default=ROOT/'runs')
    args = parser.parse_args()
    if not math.isfinite(args.max_dt) or args.max_dt <= 0: parser.error('--max-dt must be finite and positive')
    if args.young_modulus is not None and (not math.isfinite(args.young_modulus) or args.young_modulus <= 0): parser.error('--young-modulus must be finite and positive')
    if args.last_frame is not None and args.last_frame<1:parser.error('--last-frame must be at least 1')
    if args.grid_padding_cells < 0: parser.error('--grid-padding-cells must be nonnegative')
    out = args.output_root/f'episode_{args.episode}'; out.mkdir(parents=True,exist_ok=False)
    try:
        status = run(args, out)
    except Exception as error:
        write(out/'status.json', dict(episode=args.episode, status='failed', error=str(error), traceback=traceback.format_exc()))
        raise
    write(out/'status.json', status)


if __name__ == '__main__':
    main()
