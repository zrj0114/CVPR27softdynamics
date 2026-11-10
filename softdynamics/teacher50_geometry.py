"""Memory-bounded version of the same first-frame voxel/silhouette carve."""
import io
import hashlib
import tarfile
import numpy as np
from scipy.spatial import ConvexHull

CONSENSUS_METHOD='first_frame_point_mask_consensus.v1'

def project_mask(points,K,C,mask):
    """Return in-image and foreground membership, without treating occlusion as visibility."""
    xyz=(points-C[:3,3])@C[:3,:3]
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=xyz[:,:2]/xyz[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    finite=np.isfinite(uv).all(1);uv=np.rint(np.where(finite[:,None],uv,-1)).astype(int)
    inside=finite&(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
    foreground=np.zeros(len(points),bool)
    foreground[inside]=mask[uv[inside,1],uv[inside,0]]>0
    return inside,foreground

def surface_mask_consensus(points,masks):
    """Identify first-frame object support before using silhouettes as hard constraints.

    Source masks can describe a different foreground component or background.
    A point needs majority multiview support; a constraining camera must explain
    at least 85% of those object points that project inside its image. None of
    this reads gripper observation masks or future frames.
    """
    views=[];hits=[]
    for name,K,C,mask in masks:
        inside,hit=project_mask(points,K,C,mask);views.append(inside);hits.append(hit)
    views=np.asarray(views);hits=np.asarray(hits)
    keep=(views.sum(0)>=5)&(hits.sum(0)>=.5*views.sum(0))
    selected=[];evidence=[]
    for i,m in enumerate(masks):
        eligible=views[i]&keep;n=int(eligible.sum())
        support=float(hits[i,eligible].mean()) if n else 0.
        accepted=n>=16 and support>=.85
        evidence.append(dict(camera=m[0],visible_source_points=n,foreground_support=support,used_for_carving=accepted))
        if accepted:selected.append(m)
    if keep.sum()<4 or len(selected)<5:
        raise ValueError(f'Insufficient first-frame geometric support: {keep.sum()} points, {len(selected)} consistent cameras')
    return keep,selected,evidence

def carve_consensus(source,cameras,output,chunk_points=300000):
    """Reconstruct the same 3 mm volume using frame-zero cleaned points and consistent masks."""
    import cv2
    import h5py
    with tarfile.open(source/'pcd_clean.tar') as archive:
        first_frame_bytes=archive.extractfile('pcd_clean/000000.npz').read()
        with np.load(io.BytesIO(first_frame_bytes),allow_pickle=False) as z:
            source_points=z['pts'];observed=z['visibility_matrix'].sum(1)>=2
    ids=np.flatnonzero(observed&np.isfinite(source_points).all(1))
    annotation=source/'initial_object_point_ids.npz'
    annotation_evidence=None
    if annotation.exists():
        with np.load(annotation,allow_pickle=False) as labels:
            if str(labels['first_frame_sha256'])!=hashlib.sha256(first_frame_bytes).hexdigest():
                raise ValueError('Initial object annotation belongs to a different source frame')
            selected=labels['point_ids']
            if (selected.ndim!=1 or selected.dtype.kind not in 'iu' or
                    len(np.unique(selected))!=len(selected) or
                    np.any(selected<0) or np.any(selected>=len(source_points))):
                raise ValueError('Invalid initial object point annotation')
            annotation_evidence=dict(path=str(annotation),sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                                     reason=str(labels['reason']),selected_source_points=len(selected))
        ids=np.intersect1d(ids,selected)
    points=source_points[ids]
    masks=[];mask_hashes={}
    with tarfile.open(source/'cameras.tar') as archive:
        for name,K,C in zip(cameras['camera_names'],cameras['K'],cameras['camera_to_world']):
            with h5py.File(io.BytesIO(archive.extractfile(f'{name}/mask_refined.h5').read()),'r') as f:mask=f['data'][0]
            mask_hashes[str(name)]=hashlib.sha256(mask.tobytes()).hexdigest()
            if np.any(mask):masks.append((str(name),K,C,cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))))
    keep,masks,evidence=surface_mask_consensus(points,masks);surface=points[keep]
    spacing=.003;low=np.floor(surface.min(0)/spacing)*spacing;high=surface.max(0)
    axes=[np.arange(low[k],high[k]+spacing,spacing) for k in range(3)]
    planes=ConvexHull(surface).equations
    sample=np.stack(np.meshgrid(*[a[np.linspace(0,len(a)-1,min(len(a),8)).astype(int)] for a in axes],indexing='ij'),-1).reshape(-1,3)
    order=np.argsort(-np.mean(sample@planes[:,:3].T+planes[:,3]>.3*spacing,axis=0),kind='stable')
    kept=[];step=max(1,chunk_points//(len(axes[1])*len(axes[2])))
    for start in range(0,len(axes[0]),step):
        voxels=np.stack(np.meshgrid(axes[0][start:start+step],axes[1],axes[2],indexing='ij'),-1).reshape(-1,3)
        for plane in planes[order]:
            voxels=voxels[voxels@plane[:3]+plane[3]<=spacing*.3]
            if not len(voxels):break
        count=np.zeros(len(voxels));support=np.zeros(len(voxels))
        for name,K,C,mask in masks:
            inside,hit=project_mask(voxels,K,C,mask);count+=inside;support+=hit
        kept.append(voxels[(count>=5)&(support>=.85*count)])
    voxels=np.concatenate(kept).astype(np.float32)
    if not len(voxels):raise ValueError('No occupied first-frame voxels after consistent-mask carving')
    np.savez_compressed(output,positions=voxels)
    np.savez_compressed(output.with_name('first-frame-surface-support.npz'),positions=surface,point_ids=ids[keep])
    return dict(method=CONSENSUS_METHOD,source_frame=0,voxel_spacing_m=spacing,particles=len(voxels),
        initial_object_annotation=annotation_evidence,
        source_points=len(source_points),observed_source_points=len(points),consensus_source_points=len(surface),
        source_point_majority=.5,camera_source_point_support=.85,cameras=[m[0] for m in masks],camera_evidence=evidence,
        silhouette_support=.85,mask_dilation_pixels=2,future_geometry_used=False,gripper_quality_gates=False,
        first_frame_points_sha256=hashlib.sha256(first_frame_bytes).hexdigest(),first_frame_mask_sha256=mask_hashes,
        camera_calibration_id=str(cameras['camera_calibration_id']),memory_bounded_carve=True,
        maximum_grid_chunk_points=chunk_points,coarse_axis_cells=(np.ptp(voxels,axis=0)/spacing).tolist())

def carve_large(source,cameras,splat,load_gaussians,output,chunk_points=400000):
    import cv2
    import h5py
    g=load_gaussians(splat);surface=g.xyz[g.opacity>=.005];spacing=.003
    low=np.floor(surface.min(0)/spacing)*spacing;high=surface.max(0)
    axes=[np.arange(low[k],high[k]+spacing,spacing) for k in range(3)]
    planes=ConvexHull(surface).equations;masks=[];used=[]
    with tarfile.open(source/'cameras.tar') as archive:
        for name,K,C in zip(cameras['camera_names'],cameras['K'],cameras['camera_to_world']):
            with h5py.File(io.BytesIO(archive.extractfile(f'{name}/mask_refined.h5').read()),'r') as f:mask=f['data'][0]
            if not np.any(mask):continue
            masks.append((K,C,cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))));used.append(str(name))
    # The tests compare this output against the original full-grid carve.
    # Evaluate restrictive planes first to avoid revisiting already rejected voxels.
    sample=np.stack(np.meshgrid(*[a[np.linspace(0,len(a)-1,min(len(a),8)).astype(int)] for a in axes],indexing='ij'),-1).reshape(-1,3)
    order=np.argsort(-np.mean(sample@planes[:,:3].T+planes[:,3]>.3*spacing,axis=0),kind='stable')
    kept=[];step=max(1,chunk_points//(len(axes[1])*len(axes[2])))
    for start in range(0,len(axes[0]),step):
        voxels=np.stack(np.meshgrid(axes[0][start:start+step],axes[1],axes[2],indexing='ij'),-1).reshape(-1,3)
        for plane in planes[order]:
            voxels=voxels[voxels@plane[:3]+plane[3]<=spacing*.3]
            if not len(voxels):break
        support=np.zeros(len(voxels));visible=np.zeros(len(voxels))
        for K,C,mask in masks:
            xyz=(voxels-C[:3,3])@C[:3,:3]
            with np.errstate(divide='ignore',invalid='ignore'):
                uv_float=xyz[:,:2]/xyz[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
            finite=np.isfinite(uv_float).all(1);uv=np.rint(np.where(finite[:,None],uv_float,-1)).astype(int)
            good=finite&(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
            visible[good]+=1;support[good]+=mask[uv[good,1],uv[good,0]]>0
        kept.append(voxels[(visible>=5)&(support/np.maximum(visible,1)>=.85)])
    voxels=np.concatenate(kept);axis_cells=np.ptp(voxels,axis=0)/spacing if len(voxels) else np.zeros(3)
    if not len(voxels):raise ValueError('No occupied first-frame voxels')
    np.savez_compressed(output,positions=voxels.astype(np.float32))
    return dict(source_frame=0,voxel_spacing_m=spacing,particles=len(voxels),cameras=used,
        silhouette_support=.85,mask_dilation_pixels=2,future_geometry_used=False,
        camera_calibration_id=str(cameras['camera_calibration_id']),memory_bounded_carve=True,
        maximum_grid_chunk_points=chunk_points,occupancy_rule_unchanged=True,
        coarse_axis_cells=axis_cells.tolist(),low_coarse_voxel_count=len(voxels)<512)
