"""Read the corrected Deform360 release into isolated, per-object workspaces.

Only initial RGB-derived Gaussians and initial masks construct material assets.
Source archives are read in place; shared datasets are never modified.
"""
import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .data import fps
from .io import load_npz, read_json, sha256, write_json


def runtime_imports(runtime):
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'gs-layer'))
    import build_asset
    from gaussian_io import load_gaussians, transform_gaussians
    from physical_asset import PhysicalAsset
    from registration import farthest_points
    return build_asset, load_gaussians, transform_gaussians, PhysicalAsset, farthest_points


def select_release(source, corrected, config, output):
    """Freeze object splits before reading motion or teacher outcomes."""
    validation = read_json(corrected / 'validation.json')
    val = set(config['dataset']['validation_objects'])
    test = set(config['dataset']['test_objects'])
    assert not val & test
    rows = []
    for r in validation['rows']:
        obj, ep = r['object'], r['episode']
        md = read_json(source / obj / f'episode_{ep}' / 'metadata.json')
        n = r['frames'] - 5
        reasons = list(r['quality_flags'])
        if md['start_frame'] < 2:
            reasons.append('no_preinteraction_window')
        if md['end_frame'] >= n - 2:
            reasons.append('no_postinteraction_window')
        split = 'validation' if obj in val else ('test' if obj in test else 'train')
        rows.append(dict(id=f'{obj}/episode_{ep}', object=obj, episode=ep,
                         split=split, candidate=not reasons, exclusion_reasons=reasons,
                         frames=r['frames'], track_frames=n, cameras=len(md['cameras']),
                         grippers=r['grippers'], clip_start=md['start_frame'],
                         clip_end=md['end_frame'], quality=r))
    selected_objects = {r['object'] for r in rows if r['candidate']}
    assert val | test <= selected_objects
    manifest = dict(schema='softdynamics.corrected-release.v1', source=str(source),
                    corrected=str(corrected), rows=rows,
                    source_validation_sha256=sha256(corrected/'validation.json'),
                    camera_corrections_sha256=sha256(corrected/'camera_corrections.json'),
                    config_sha256=sha256(output/'config.json'))
    write_json(output/'selection.json', manifest)
    return manifest


def read_splat(source, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source/'splatfacto.tar') as archive:
            dest.write_bytes(archive.extractfile('splatfacto/splat_0.ply').read())
    return dest


def carve_initial(source, cameras, splat, load_gaussians, output):
    """Existing 3 mm convex-support/silhouette carve, with corrected cameras."""
    import cv2
    import h5py
    g = load_gaussians(splat)
    surface = g.xyz[g.opacity >= .005]
    spacing = .003
    low = np.floor(surface.min(0)/spacing)*spacing
    high = surface.max(0)
    sizes = np.ceil((high-low)/spacing).astype(int)+1
    if np.prod(sizes) > 8_000_000:
        raise ValueError(f'Initial Gaussian bounds exceed carve budget: {sizes.tolist()}')
    voxels = np.stack(np.meshgrid(*[np.arange(low[k], high[k]+spacing, spacing)
                                   for k in range(3)], indexing='ij'), -1).reshape(-1, 3)
    inside = np.ones(len(voxels), bool)
    for plane in ConvexHull(surface).equations:
        inside &= voxels@plane[:3]+plane[3] <= spacing*.3
    voxels = voxels[inside]
    support, visible = np.zeros(len(voxels)), np.zeros(len(voxels))
    used = []
    with tarfile.open(source/'cameras.tar') as archive:
        for name, K, C in zip(cameras['camera_names'], cameras['K'], cameras['camera_to_world']):
            member = archive.getmember(f'{name}/mask_refined.h5')
            with h5py.File(io.BytesIO(archive.extractfile(member).read()), 'r') as f:
                mask = f['data'][0]
            if not np.any(mask):
                continue
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
            xyz = (voxels-C[:3, 3])@C[:3, :3]
            with np.errstate(divide='ignore', invalid='ignore'):
                uv_float = xyz[:, :2]/xyz[:, 2:]*[K[0, 0], K[1, 1]]+[K[0, 2], K[1, 2]]
            finite = np.isfinite(uv_float).all(1)
            uv = np.rint(np.where(finite[:, None], uv_float, -1)).astype(int)
            good = finite & (xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
            visible[good] += 1
            support[good] += mask[uv[good, 1], uv[good, 0]] > 0
            used.append(str(name))
    voxels = voxels[(visible >= 5) & (support/np.maximum(visible, 1) >= .85)]
    axis_cells = np.ptp(voxels,axis=0)/spacing if len(voxels) else np.zeros(3)
    if len(voxels) < 512 or np.min(axis_cells) < 3:
        raise ValueError(f'Insufficient resolved first-frame volume: {len(voxels)}, axis cells {axis_cells.tolist()}')
    np.savez_compressed(output, positions=voxels.astype(np.float32))
    return dict(source_frame=0, voxel_spacing_m=spacing, particles=len(voxels),
                cameras=used, silhouette_support=.85, mask_dilation_pixels=2,
                future_geometry_used=False, camera_calibration_id=str(cameras['camera_calibration_id']))


def build_canonical(workspace, obj, reference, source, corrected, runtime):
    original, load_gaussians, transform_gaussians, PhysicalAsset, farthest = runtime_imports(runtime)
    asset_dir = workspace/'asset'
    if (asset_dir/'asset.json').exists():
        return PhysicalAsset(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    for name in ['prepared', 'initial_states', 'observations', 'queries', 'checks']:
        (workspace/name).mkdir(exist_ok=True)
    prepared = workspace/'prepared'/f'episode_{reference}'
    prepared.mkdir(parents=True, exist_ok=True)
    src = source/obj/f'episode_{reference}'
    splat = read_splat(src, prepared/'splat_0.ply')
    cams = load_npz(corrected/'finger_data/cameras'/obj/f'episode_{reference}'/'cameras.npz')
    carve_path = prepared/'first-frame-volume.npz'
    evidence = carve_initial(src, cams, splat, load_gaussians, carve_path)
    coarse = load_npz(carve_path)['positions']
    axis = (np.arange(2)+.5)/2*.003-.0015
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    points = (coarse[:, None]+offsets[None]).reshape(-1, 3).astype(np.float32)
    points = points[points[:, 1] >= -.0115]
    # Conservative state/sorting estimate. Grid allocation is separately bounded.
    if len(points)*1024 > 8*1024**3:
        raise ValueError(f'Estimated particle state exceeds 8 GiB: {len(points)} particles')
    origin = points.astype(float).mean(0)
    rest = (points-origin).astype(np.float32)
    volumes = np.full(len(rest), .0015**3)
    masses = volumes*100.
    mesh, boundary = original.canonical_mesh(rest.astype(float))
    mesh.export(asset_dir/'rest_surface.ply')
    centers = farthest(rest[boundary].astype(float), 64)
    ids = np.arange(len(rest), dtype=np.int64)
    np.savez_compressed(asset_dir/'body.npz', rest_positions=rest, material_ids=ids,
                        rest_volumes_m3=volumes, masses_kg=masses,
                        surface_particle_ids=boundary, deformation_control_centers=centers)
    g = load_gaussians(splat)
    g = g.subset(g.opacity >= .005)
    pose = np.eye(4)
    pose[:3, 3] = -origin
    canonical, _ = transform_gaussians(g, pose)
    np.savez_compressed(asset_dir/'appearance.npz', **vars(canonical))
    dist, near = cKDTree(rest).query(canonical.xyz, k=8)
    weights = 1/np.maximum(dist, 1e-6)**2
    weights /= weights.sum(1, keepdims=True)
    np.savez_compressed(asset_dir/'appearance_binding.npz', particle_ids=near, weights=weights,
                        initial_offset=canonical.xyz-np.sum(rest[near]*weights[..., None], 1))
    pose[:3, 3] = origin
    write_json(asset_dir/'asset.json', dict(schema_version=1, name=obj, reference_episode=reference,
               reference_source_frame=0, reference_to_source_world=pose,
               particles=len(rest), total_mass_kg=float(masses.sum()),
               material=dict(model='fixed corotated elastic', density_kg_m3=100., physical_ground_truth=False),
               ground_y_world_m=-.01, ground_height_independently_calibrated=False,
               rest_shape_assumed_stress_free=True, numerical_particle_spacing_m=.0015,
               initial_reconstruction=evidence, shared_across_episodes=True,
               payload_sha256={n:sha256(asset_dir/n) for n in ['body.npz','appearance.npz','appearance_binding.npz','rest_surface.ply']},
               source_sha256=dict(splat=sha256(splat), cameras=sha256(corrected/'finger_data/cameras'/obj/f'episode_{reference}'/'cameras.npz'))))
    asset = PhysicalAsset(asset_dir)
    state = asset.rigid_state(pose)
    asset.save_initial_state(state, workspace/'initial_states'/f'episode_{reference}.npz',
                             dict(episode=reference, initial_surface_registration_mm=0., fit_source_frames=[0]))
    return asset


def normalize_pads(tracks, visible, initial_points, body, state, poses, valid, times,
                   point_ids, origin=None, scale=None):
    """52 features: four independent pad poses (12) plus validity (1)."""
    x = np.asarray(tracks)
    origin = initial_points.mean(0) if origin is None else np.asarray(origin)
    scale = float(np.ptp(initial_points, axis=0).max()) if scale is None else float(scale)
    if not .005 < scale < 2:
        raise ValueError(f'Invalid initial object scale {scale}')
    dist, near = cKDTree(state['positions_world_m']).query(initial_points, k=8)
    weights = 1/np.maximum(dist, 1e-6)**2
    weights /= weights.sum(1, keepdims=True)
    rest = (body['rest_positions'][near]*weights[..., None]).sum(1)/scale
    F = (state['deformation_gradient'][near]*weights[..., None, None]).sum(1)
    geometry = np.c_[(initial_points-origin)/scale, rest, F.reshape(len(x[0]), 9)]
    actions = np.zeros((len(x), 52), np.float32)
    for g in range(poses.shape[1]):
        for side in range(2):
            offset = (2*g+side)*13
            mat = poses[:, g, side, :3].copy()
            mat[:, :, 3] = (mat[:, :, 3]-origin)/scale
            good = valid[:, g, side]
            actions[:, offset:offset+12] = np.where(good[:, None], mat.reshape(len(x), 12), 0)
            actions[:, offset+12] = good
    return dict(positions=((x-origin)/scale).astype(np.float32), valid=visible.astype(bool),
                geometry=geometry.astype(np.float32), actions=actions,
                times=(times-times[0]).astype(np.float32), scale_m=np.float32(scale), origin_m=origin,
                point_ids=point_ids, action_schema=np.array('four_pad_poses_52_v1'))


def prepare_episode(workspace, row, source, corrected, asset, runtime):
    obj, ep = row['object'], row['episode']
    name = f'episode_{ep}'
    done = workspace/'prepared'/name/'complete.json'
    if done.exists():
        result = read_json(done)
        if result.get('status') == 'ready':
            return result
    _, load_gaussians, _, _, _ = runtime_imports(runtime)
    prepared = done.parent
    prepared.mkdir(parents=True, exist_ok=True)
    splat = read_splat(source/obj/name, prepared/'splat_0.ply')
    g = load_gaussians(splat)
    initial_path = workspace/'initial_states'/f'{name}.npz'
    if not initial_path.exists():
        state, registration = asset.fit_initial_surface(g.xyz[g.opacity >= .005])
        if registration['error_mm'] > 12.:
            raise ValueError(f'Initial-only registration exceeds 12 mm: {registration["error_mm"]}')
        asset.save_initial_state(state, initial_path,
                                 dict(episode=ep, fit_source_frames=[0], future_geometry_used=False, **registration))
    state, state_meta = asset.load_initial_state(initial_path)
    pads = load_npz(corrected/'finger_data'/obj/f'{name}.npz')
    camera_path = corrected/'finger_data/cameras'/obj/name/'cameras.npz'
    qdir = workspace/'queries'/name
    qdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(qdir/'actions.npz', poses=pads['world_from_pad_center'],
                        valid=pads['holder_observed_valid'], source_frames=pads['source_frames'],
                        source_timestamps_us=pads['source_timestamps_us'])
    target = qdir/'cameras.npz'
    if not target.exists():
        target.symlink_to(camera_path)
    points, visible = [], []
    max_error, previous = 0., None
    with tarfile.open(source/obj/name/'pcd_clean.tar') as archive:
        members = sorted([m for m in archive if m.isfile() and m.name.endswith('.npz')], key=lambda m:int(Path(m.name).stem))
        assert len(members) == len(pads['source_frames'])-5
        for frame, member in enumerate(members):
            assert int(Path(member.name).stem) == frame
            with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as z:
                x, v = z['pts'], z['visibility_matrix'].sum(1) >= 2
                if frame == 0:
                    initial_match = cKDTree(g.xyz).query(x)[0] < 1e-6
                    candidates = np.flatnonzero(initial_match & v & np.isfinite(x).all(1))
                    if len(candidates) < 128:
                        raise ValueError('Fewer than 128 initial visual Gaussian tracks')
                    ids = candidates[fps(x[candidates], 512)]
                if not np.isfinite(x[ids]).all():
                    raise ValueError('Selected visual tracks contain nonfinite positions')
                if previous is not None:
                    max_error = max(max_error, float(np.linalg.norm(x[ids]-previous, axis=1).max()))
                previous = x[ids]+z['vels'][ids]/30.
                points.append(x[ids]); visible.append(v[ids])
    if max_error > 1e-5:
        raise ValueError(f'Broken per-row temporal transport: {max_error}')
    tracks, visibility = np.array(points), np.array(visible)
    dist, near = cKDTree(state['positions_world_m']).query(tracks[0], k=8)
    weights = 1/np.maximum(dist, 1e-6)**2
    weights /= weights.sum(1, keepdims=True)
    observation = workspace/'observations'/f'{name}.npz'
    np.savez_compressed(observation, saved_ids=asset.body['material_ids'],
                        initial=state['positions_world_m'], surface_ids=asset.body['surface_particle_ids'],
                        tracks=tracks, track_visible=visibility, binding=near, weights=weights,
                        frames=np.arange(len(tracks)), point_ids=ids)
    good_action = pads['holder_observed_valid'].all(axis=(1, 2))
    windows = []
    for phase, weight in [('before', .1), ('interaction', .45), ('release', .45)]:
        eligible = []
        for frame in range(1, len(tracks)):
            label = 'before' if frame < row['clip_start'] else ('interaction' if frame <= row['clip_end'] else 'release')
            if label == phase and good_action[frame-1:frame+1].all() and visibility[frame].sum() >= 128:
                eligible.append(frame)
        if len(eligible) < 2:
            raise ValueError(f'Insufficient fully observed teacher endpoints in phase {phase}')
        chosen = np.unique(np.quantile(eligible, [.25, .75], method='nearest').astype(int))
        for frame in chosen:
            windows.append(dict(index=int(frame), start=int(frame)-1, end=int(frame), phase=phase, weight=weight/len(chosen)))
    meta = dict(episode=ep, object=obj, clip_start=row['clip_start'], clip_end=row['clip_end'],
                teacher_windows=sorted(windows, key=lambda w:w['end']),
                source_sha256=dict(actions=sha256(qdir/'actions.npz'), camera=sha256(camera_path)),
                action_schema='pad_center', face_size_m=[.064, .025], ground_y_world_m=-.01)
    write_json(qdir/'episode.json', meta)
    normalized = normalize_pads(tracks, visibility, tracks[0], asset.body, state,
                  pads['world_from_pad_center'][:len(tracks)], pads['holder_observed_valid'][:len(tracks)],
                  pads['source_timestamps_us'][:len(tracks)]*1e-6, ids)
    np.savez_compressed(prepared/'real.npz', **normalized)
    result = dict(**{k:row[k] for k in ['id','object','episode','split']}, status='ready',
                  workspace=str(workspace), observation=str(observation), metadata=str(qdir/'episode.json'),
                  real=str(prepared/'real.npz'), particles=len(asset.body['material_ids']),
                  selected_visual_points=len(ids), frames=len(tracks), teacher_windows=windows,
                  transport_error_m=max_error, state=state_meta, tactile_input=False,
                  point_selection='Initial Gaussian membership and >=2 visual cameras; later pointwise visibility retained',
                  upstream_tracking_refinement_provenance='Official processed trajectories; not certified free of all upstream refinement priors')
    write_json(done, result)
    return result


def prepare_object(selection_path, obj, output, runtime):
    selection = read_json(selection_path)
    rows = [r for r in selection['rows'] if r['object'] == obj and r['candidate']]
    if not rows:
        raise ValueError('No selected episodes')
    source, corrected = Path(selection['source']), Path(selection['corrected'])
    workspace = output/'objects'/obj
    workspace.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        asset = build_canonical(workspace, obj, min(r['episode'] for r in rows), source, corrected, runtime)
    except Exception as exc:
        write_json(workspace/'prepare.json', dict(object=obj, status='asset_failed', error=f'{type(exc).__name__}: {exc}', rows=[]))
        raise
    for row in rows:
        try:
            result = prepare_episode(workspace, row, source, corrected, asset, runtime)
        except Exception as exc:
            result = dict(id=row['id'], object=obj, episode=row['episode'], split=row['split'], status='rejected', error=f'{type(exc).__name__}: {exc}')
        results.append(result)
        write_json(workspace/'prepare.json', dict(object=obj, status='preparing', rows=results))
        print('PREPARED', obj, row['episode'], result['status'], result.get('error',''), flush=True)
    write_json(workspace/'prepare.json', dict(object=obj, status='complete', rows=results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path)
    p.add_argument('--object')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1]/'runtime')
    p.add_argument('--select-only', action='store_true')
    p.add_argument('--source', type=Path)
    p.add_argument('--corrected', type=Path)
    p.add_argument('--config', type=Path)
    a = p.parse_args()
    if a.select_only:
        if any(v is None for v in [a.source,a.corrected,a.config]):
            p.error('--select-only requires --source, --corrected and --config')
        a.output.mkdir(parents=True,exist_ok=True)
        destination=a.output/'config.json'
        if destination.exists() and read_json(destination)!=read_json(a.config):
            raise ValueError('Existing run uses another configuration; choose a new output')
        if not destination.exists():
            destination.write_bytes(a.config.read_bytes())
        result=select_release(a.source.resolve(),a.corrected.resolve(),read_json(a.config),a.output.resolve())
        print('SELECTED',sum(r['candidate'] for r in result['rows']))
    else:
        if a.selection is None or a.object is None:
            p.error('Preparation requires --selection and --object')
        prepare_object(a.selection, a.object, a.output, a.runtime)


if __name__ == '__main__':
    main()
