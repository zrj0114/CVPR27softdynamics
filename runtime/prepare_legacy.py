"""Original first-frame volume reconstruction with portable repaired-data inputs.

The cameras() and object_volume() functions are extracted unchanged from the
validated October 12 baseline. This entry uses the existing measured holder
poses; it does not redo finger calibration or use later object geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

import cv2
import h5py
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from paths import CODE
sys.path.insert(0, str(CODE/'gs-layer'))
from gaussian_io import load_gaussians

SOURCE = None
OUTPUT = None


def cameras():
    meta=json.loads((SOURCE/'metadata.json').read_text());result={}
    for line in (SOURCE/'metric_params_refined_undistorted.txt').read_text().splitlines():
        p=line.split()
        if len(p)>=19 and p[11] in meta['cameras']:
            r=Rotation.from_quat(np.array(p[12:16],float)[[1,2,3,0]]).as_matrix()
            t=np.eye(4);t[:3,:3]=r.T;t[:3,3]=-r.T@np.array(p[16:19],float)
            k=np.array([[float(p[3]),0,float(p[5])],[0,float(p[4]),float(p[6])],[0,0,1]])
            result[p[11]]=(k,t)
    return result


def object_volume():
    g=load_gaussians(SOURCE/'splat_0.ply');surface=g.xyz[g.opacity>=.005]
    spacing=.003
    low=np.floor(surface.min(0)/spacing)*spacing;high=surface.max(0)
    voxels=np.stack(np.meshgrid(*[np.arange(low[k],high[k]+spacing,spacing) for k in range(3)],indexing='ij'),axis=-1).reshape(-1,3)
    hull=ConvexHull(surface)
    inside=np.ones(len(voxels),bool)
    for plane in hull.equations:inside&=(voxels@plane[:3]+plane[3]<=spacing*.3)
    voxels=voxels[inside]
    support=np.zeros(len(voxels));visible=np.zeros(len(voxels));cams=cameras()
    for name,(k,c2w) in cams.items():
        with h5py.File(SOURCE/'cameras'/name/'mask_refined.h5') as f:mask=f['data'][0]
        if not np.any(mask):continue
        mask=cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))
        xyz=(voxels-c2w[:3,3])@c2w[:3,:3]
        uv=np.rint(xyz[:,:2]/xyz[:,2:]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]).astype(int)
        valid=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<1280)&(uv[:,1]>=0)&(uv[:,1]<720)
        visible[valid]+=1;support[valid]+=mask[uv[valid,1],uv[valid,0]]>0
    voxels=voxels[(visible>=5)&(support/np.maximum(visible,1)>=.85)]
    if len(voxels)<2000:raise ValueError('Insufficient first-frame volume')
    # Uniform voxel centers carry physical mass; a deterministic subset stores
    # visualization anchors. Gaussian appearance does not define material mass.
    ids=np.linspace(0,len(voxels)-1,min(4096,len(voxels))).astype(int)
    np.savez_compressed(OUTPUT/'object-initial.npz',positions=voxels.astype(np.float32),visual_indices=ids,
                        particle_volumes=np.full(len(voxels),spacing**3,dtype=np.float32))
    (OUTPUT/'initialization.json').write_text(json.dumps({'sourceFrame':0,'sourceAppearance':'splat_0.ply',
        'construction':'Uniform voxel sampling of first-Gaussian convex support, carved by first-frame multi-view object silhouettes',
        'spacingMeters':spacing,'particles':len(voxels),'volumeM3':len(voxels)*spacing**3,
        'silhouetteSupportThreshold':.85,'maskDilationPixels':2,'visualAnchors':len(ids),
        'laterFramesUsedForInitialization':False},indent=2)+'\n')
    print('initial material particles',len(voxels),flush=True)


def carve(source, output):
    global SOURCE, OUTPUT
    SOURCE, OUTPUT = Path(source).resolve(), Path(output).resolve()
    if (OUTPUT/'first-frame-volume.npz').exists():
        raise FileExistsError('Initial volume already exists; use a new output directory')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    object_volume()
    shutil.copyfile(OUTPUT/'object-initial.npz', OUTPUT/'first-frame-volume.npz')


def prepare(capture_root, processed_root, workspace, episodes):
    """Package existing repaired actions and original first-frame visual inputs."""
    capture_root, processed_root, workspace = map(Path, (capture_root, processed_root, workspace))
    for episode in episodes:
        capture = capture_root/f'episodes/episode_{episode}'
        processed = processed_root/f'episode_{episode}'
        output = workspace/f'prepared/episode_{episode}'
        source = output/'source'
        source.mkdir(parents=True, exist_ok=False)
        with np.load(capture/'pads_world.npz', allow_pickle=False) as data:
            holder = data['smoothed_holder_poses']
            valid = data['holder_observed_valid']
            frames, stamps = data['source_frames'], data['source_timestamps_us']
        # Existing published marker registration; no newly fitted contact offset.
        holder_from_marker = np.eye(4)
        holder_from_marker[:3, 3] = [0., .029, .009]
        poses = holder @ holder_from_marker
        if not np.array_equal(np.isfinite(poses).all(axis=(-1, -2)), valid):
            raise ValueError('Holder pose validity disagrees with finite measurements')
        for name in ('metadata.json', 'metric_params_refined_undistorted.txt'):
            shutil.copyfile(capture/'source_processed'/name, source/name)
        records = []
        with tarfile.open(processed/'splatfacto.tar') as archive:
            content = archive.extractfile('splatfacto/splat_0.ply').read()
            (source/'splat_0.ply').write_bytes(content)
            records.append(dict(member='splatfacto/splat_0.ply', sha256=hashlib.sha256(content).hexdigest()))
        with tarfile.open(processed/'cameras.tar') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(('/mask_refined.h5', '/aligned_timestamps.txt')):
                    continue
                relative = Path(member.name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError(f'Invalid camera archive member: {member.name}')
                content = archive.extractfile(member).read()
                dest = source/'cameras'/relative
                dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
                records.append(dict(member=member.name, sha256=hashlib.sha256(content).hexdigest()))
        timestamps = source/'cameras/brics-odroid-001_cam0/aligned_timestamps.txt'
        camera_stamps = np.array([int(line.split('_')[1]) for line in timestamps.read_text().splitlines()])
        if not np.array_equal(camera_stamps, stamps):
            raise ValueError('Camera and repaired action timestamps disagree')
        carve(source, output)
        np.savez_compressed(output/'actions.npz', poses=poses, valid=valid,
                            source_frames=frames, source_timestamps_us=stamps)
        manifest = dict(episode=episode, source_frame=0, future_geometry_used=False,
                        holder_from_marker=holder_from_marker.tolist(), missing_poses_preserved=True,
                        source_records=records,
                        repaired_actions_sha256=hashlib.sha256((capture/'pads_world.npz').read_bytes()).hexdigest())
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    initial = sub.add_parser('carve', help='Reconstruct volume from already extracted original visual inputs')
    initial.add_argument('--source', type=Path, required=True)
    initial.add_argument('--output', type=Path, required=True)
    complete = sub.add_parser('prepare', help='Prepare initial geometry and existing measured actions')
    complete.add_argument('--capture-root', type=Path, required=True)
    complete.add_argument('--processed-root', type=Path, required=True)
    complete.add_argument('--workspace', type=Path, required=True)
    complete.add_argument('--episodes', type=int, nargs='+', default=list(range(10)))
    a = p.parse_args()
    if a.mode == 'carve': carve(a.source, a.output)
    else: prepare(a.capture_root, a.processed_root, a.workspace, a.episodes)


if __name__ == '__main__': main()
