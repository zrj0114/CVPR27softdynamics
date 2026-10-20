"""Normalize tracked geometry and official EEF actions for the motion encoder."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .io import load_npz


def fps(points, count):
    points = np.asarray(points)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('Farthest point sampling needs finite, nonempty points')
    count = min(int(count), len(points))
    ids = np.empty(count, dtype=int)
    distance = np.full(len(points), np.inf)
    selected = int(np.argmax(np.sum((points - points.mean(0))**2, axis=1)))
    for i in range(count):
        ids[i] = selected
        distance = np.minimum(distance, np.sum((points - points[selected])**2, axis=1))
        selected = int(np.argmax(distance))
    return ids


def normalize_episode(tracks, visible, initial, rest, deformation, poses, openings, times,
                      point_ids=None, origin=None, scale=None):
    """Inputs are meters/seconds with fixed point IDs; no future-frame selection.

    EEF poses are [T,G,4,4], openings [T,G], with at most two grippers.
    Missing controller observations can be NaN and get an explicit validity bit.
    Occluded object tracks must retain finite reconstructed/imputed coordinates.
    """
    tracks, visible = np.asarray(tracks), np.asarray(visible, dtype=bool)
    times = np.asarray(times, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 3 or visible.shape != tracks.shape[:2]:
        raise ValueError('Expected tracks [T,N,3] and visible [T,N]')
    if not np.isfinite(tracks).all() or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Finite tracks and increasing timestamps are required')
    if point_ids is None:
        available = np.flatnonzero(visible[0])
        point_ids = available[fps(tracks[0, available], 512)]
    point_ids = np.asarray(point_ids, dtype=int)
    if len(point_ids) < 128:
        raise ValueError('The encoder needs at least 128 fixed point tracks')
    x = tracks[:, point_ids]
    origin = x[0].mean(0) if origin is None else np.asarray(origin)
    scale = float(np.ptp(x[0], axis=0).max()) if scale is None else float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive object scale required')
    distances, neighbors = cKDTree(initial).query(x[0], k=min(8, len(initial)))
    if neighbors.ndim == 1:
        neighbors, distances = neighbors[:, None], distances[:, None]
    weights = 1 / np.maximum(distances, 1e-6)**2
    weights /= weights.sum(1, keepdims=True)
    canonical = (np.asarray(rest)[neighbors] * weights[..., None]).sum(1) / scale
    F = (np.asarray(deformation)[neighbors] * weights[..., None, None]).sum(1)
    x = (x - origin) / scale
    geometry = np.concatenate([x[0], canonical, F.reshape(len(point_ids), 9)], axis=-1)
    poses, openings = np.asarray(poses), np.asarray(openings)
    if poses.ndim == 3:
        poses, openings = poses[:, None], openings[:, None]
    T = len(x)
    if poses.shape[0] != T or poses.shape[-2:] != (4, 4) or not 1 <= poses.shape[1] <= 2:
        raise ValueError('EEF poses must have shape [T,1 or 2,4,4]')
    if openings.shape != poses.shape[:2] or times.shape != (T,):
        raise ValueError('Openings and timestamps must match the trajectory')
    actions = np.zeros((T, 28), dtype=np.float32)
    for g in range(poses.shape[1]):
        matrix = poses[:, g, :3].copy()
        matrix[:, :, 3] = (matrix[:, :, 3] - origin) / scale
        good = np.isfinite(matrix).all((-1, -2)) & np.isfinite(openings[:, g])
        actions[:, g*14:g*14 + 12] = np.where(good[:, None], matrix.reshape(T, 12), 0)
        actions[:, g*14 + 12] = np.where(good, openings[:, g] / scale, 0)
        actions[:, g*14 + 13] = good
    return dict(positions=x.astype(np.float32), valid=visible[:, point_ids],
                geometry=geometry.astype(np.float32), actions=actions,
                times=(times-times[0]).astype(np.float32), scale_m=np.float32(scale),
                origin_m=origin, point_ids=point_ids)


def bind_observations(tracks, visible, initial_saved, saved_ids, surface_ids, frames):
    """Attach visual points to a runtime's saved particles for teacher/scoring.

    surface_ids are indices into initial_saved, not full material IDs. Preserve
    these arrays between nominal, perturbed and fitted simulations.
    """
    distances, neighbors = cKDTree(initial_saved).query(tracks[0], k=min(8, len(initial_saved)))
    if neighbors.ndim == 1:
        neighbors, distances = neighbors[:, None], distances[:, None]
    weights = 1 / np.maximum(distances, 1e-6)**2
    weights /= weights.sum(1, keepdims=True)
    return dict(tracks=np.asarray(tracks), track_visible=np.asarray(visible, dtype=bool),
                initial=np.asarray(initial_saved), saved_ids=np.asarray(saved_ids),
                surface_ids=np.asarray(surface_ids), binding=neighbors, weights=weights,
                frames=np.asarray(frames), point_ids=np.arange(tracks.shape[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True,
                   help='NPZ: tracks, visible, poses, openings, times; optional point_ids')
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--initial-state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data, body, state = load_npz(args.input), load_npz(args.body), load_npz(args.initial_state)
    normalized = normalize_episode(data['tracks'], data['visible'], state['positions_world_m'],
                                   body['rest_positions'], state['deformation_gradient'],
                                   data['poses'], data['openings'], data['times'],
                                   data.get('point_ids'), data.get('origin_m'), data.get('scale_m'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **normalized)


if __name__ == '__main__':
    main()
