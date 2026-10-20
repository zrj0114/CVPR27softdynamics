"""One canonical material body and initial-frame-only episode placements."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

from paths import CODE, ROOT, DATA
sys.path.insert(0,str(CODE/'gs-layer'))
from gaussian_io import load_gaussians,transform_gaussians
from registration import (rigid_icp,farthest_points,fit_deformation,map_points,positive_map,subsample)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def volume_particles(episode):
    with np.load(ROOT/f'prepared/episode_{episode}/first-frame-volume.npz') as z:coarse=z['positions']
    axis=(np.arange(2)+.5)/2*.003-.0015
    offsets=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    xyz=(coarse[:,None]+offsets[None]).reshape(-1,3).astype(np.float32)
    xyz=xyz[xyz[:,1]>=-.01-.0015]
    return xyz


def initial_surface(episode):
    g=load_gaussians(ROOT/f'prepared/episode_{episode}/source/splat_0.ply')
    return g.subset(g.opacity>=.005)


def canonical_mesh(rest,spacing=.0015):
    origin=rest.min(0)-2*spacing
    ijk=np.rint((rest-origin)/spacing).astype(int)
    occupancy=np.zeros(tuple(ijk.max(0)+3),dtype=bool)
    occupancy[tuple(ijk.T)]=True
    # Do not invent missing interior material; mesh reflects actual voxel body.
    vertices,faces,_,_=marching_cubes(occupancy.astype(np.float32),level=.5,spacing=(spacing,)*3)
    vertices+=origin
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    trimesh.repair.fix_normals(mesh,multibody=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Canonical surface is not closed/oriented')
    boundary_ids=np.unique(cKDTree(rest).query(mesh.vertices)[1])
    return mesh,boundary_ids


def build(reference=5,episodes=range(10)):
    manifest_path=ROOT/'asset/asset.json'
    if manifest_path.exists():raise FileExistsError('Asset already exists; preserve it and build a new version')
    for directory in ['asset', 'initial_states', 'checks']:
        (ROOT/directory).mkdir(parents=True, exist_ok=True)
    source_particles=volume_particles(reference).astype(np.float64)
    origin=source_particles.mean(0)
    rest=(source_particles-origin).astype(np.float32)
    ids=np.arange(len(rest),dtype=np.int64)
    volumes=np.full(len(rest),.0015**3,dtype=np.float64);mass=volumes*100.
    mesh,boundary_ids=canonical_mesh(rest.astype(np.float64))
    mesh.export(ROOT/'asset/rest_surface.ply')
    g=initial_surface(reference)
    tr=np.eye(4);tr[:3,3]=-origin
    canonical_g,_=transform_gaussians(g,tr)
    np.savez_compressed(ROOT/'asset/appearance.npz',**vars(canonical_g))
    anchors=farthest_points(rest[boundary_ids].astype(float),64)
    np.savez_compressed(ROOT/'asset/body.npz',rest_positions=rest,material_ids=ids,
        rest_volumes_m3=volumes,masses_kg=mass,reference_density_kg_m3=np.full(len(rest),100.),
        young_modulus_pa=np.full(len(rest),4000.),poisson_ratio=np.full(len(rest),.3),
        surface_particle_ids=boundary_ids,mesh_vertices=mesh.vertices,mesh_faces=mesh.faces,
        deformation_control_centers=anchors)
    # Store a fixed material attachment for the same appearance in every episode.
    distances,neighbors=cKDTree(rest).query(canonical_g.xyz,k=8)
    weights=1/np.maximum(distances,1.e-6)**2;weights/=weights.sum(1,keepdims=True)
    np.savez_compressed(ROOT/'asset/appearance_binding.npz',particle_ids=neighbors,weights=weights,
        initial_offset=canonical_g.xyz-np.sum(rest[neighbors]*weights[...,None],axis=1))
    asset=dict(schema_version=1,name='121-croissant-plush',canonical_geometry_included=True,
        reference_episode=reference,reference_source_frame=0,
        units=dict(length='m',time='s',mass='kg',stress='Pa'),
        coordinate_system='Right handed source world orientation, translated to canonical particle center; +Y up',
        reference_to_source_world=np.eye(4).tolist(),
        particles=len(rest),reference_volume_m3=float(volumes.sum()),total_mass_kg=float(mass.sum()),
        material=dict(model='fixed corotated elastic, EMPM material=0',E_pa=4000,nu=.3,rho_kg_m3=100,
            E_origin='Inherited shared fit on episodes 0/5/7 before canonical geometry; no re-fit on canonical rollouts yet'),
        contact=dict(gripper_static_friction=2.,gripper_sliding_friction=.8,ground_friction=.8,
            ground_y_world_m=-.01,law='Unchanged existing unilateral material-grid contact'),
        numerical=dict(grid_spacing_m=.4/128,particle_spacing_m=.0015,max_dt_s=.0001),
        rest_shape_assumption='Episode 5 first-frame supported geometry is the single elastic reference; gravity prestress/rest-shape recovery is not independently identified',
        appearance='One episode-5 initial Gaussian appearance, fixed material binding shared by all episodes',
        source_inputs={str(p):sha(p) for p in [ROOT/f'prepared/episode_{reference}/first-frame-volume.npz',ROOT/f'prepared/episode_{reference}/source/splat_0.ply']},
        mesh=dict(watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
            volume_m3=float(abs(mesh.volume)),components=len(mesh.split(only_watertight=False)),vertices=len(mesh.vertices),faces=len(mesh.faces)),
        payload_sha256={name:sha(ROOT/'asset'/name) for name in ['body.npz','rest_surface.ply','appearance.npz','appearance_binding.npz']},
        reference_selection='Fixed training episode 5; validation/test initial frames are placement inputs only, no dynamic observations used')
    asset['reference_to_source_world']=np.eye(4).tolist()
    for i in range(3):asset['reference_to_source_world'][i][3]=float(origin[i])
    write(manifest_path,asset)
    fingerprint=sha(manifest_path);summaries=[]
    source_surface=canonical_g.xyz.astype(float)
    width=.035
    for e in episodes:
        target=initial_surface(e).xyz.astype(float)
        if e==reference:
            r=np.eye(3);t=origin.copy();rigid_error=0.;coeff=np.zeros((len(anchors),3));history=[]
            positions=source_particles;F=np.tile(np.eye(3),(len(rest),1,1));lift=0.;trials=[]
        else:
            best,alternatives=rigid_icp(source_surface,target,iterations=45)
            rigid_error,r,t=best
            coeff,history=fit_deformation(source_surface,target,anchors,r,t,width)
            chosen,trials=positive_map(source_surface,rest.astype(float),subsample(target),anchors,r,t,coeff,width)
            error,coeff,positions,F,lift=chosen
            t=t.copy();t[1]+=lift;positions[:,1]+=lift
        if e==reference:positions=source_particles.copy()
        sv=np.linalg.svd(F,compute_uv=False);J=np.linalg.det(F)
        mapped_surface=map_points(subsample(source_surface),r,t,anchors,coeff,width)
        observed=subsample(target)
        initial_error=.5*(cKDTree(observed).query(mapped_surface)[0].mean()+cKDTree(mapped_surface).query(observed)[0].mean())*1000
        transformed=np.eye(4);transformed[:3,:3]=r;transformed[:3,3]=t
        state_path=ROOT/f'initial_states/episode_{e}.npz'
        np.savez_compressed(state_path,positions_world_m=positions.astype(np.float32),
            deformation_gradient=F.astype(np.float32),velocities_world_m_s=np.zeros_like(rest),
            material_ids=ids,current_volumes_m3=volumes*J,current_density_kg_m3=mass/(volumes*J),
            world_from_canonical_rigid=transformed,deformation_coefficients=coeff,
            deformation_width_m=np.array(width),asset_sha256=np.array(fingerprint))
        report=dict(episode=e,asset_sha256=fingerprint,material_particles=len(rest),
            initial_surface_registration_mm=float(initial_error),rigid_surface_registration_mm=float(rigid_error*1000),
            min_initial_J=float(J.min()),max_initial_J=float(J.max()),
            min_initial_stretch=float(sv.min()),max_initial_stretch=float(sv.max()),ground_lift_m=float(lift),
            ground_min_particle_y_m=float(positions[:,1].min()),total_mass_kg=float(mass.sum()),
            initialization='Fixed canonical material points transformed by proper rigid pose plus bounded smooth initial deformation; F is analytic map Jacobian',
            initial_velocity='zero, preserved assumption',strain_was_reset_to_identity=False,
            correspondence='Geometric registration only; cross-episode point IDs are not assumed to match',
            reference_shape_assumed_stress_free=True,fit_source_frames=[0],future_object_observations_used=False,
            registered_initial_inputs={str(p):sha(p) for p in [ROOT/f'prepared/episode_{e}/source/splat_0.ply']},
            map_trials=trials,fit_history_mm=history,state_sha256=sha(state_path))
        write(state_path.with_suffix('.json'),report);summaries.append(report)
        print(json.dumps({k:report[k] for k in ['episode','initial_surface_registration_mm','min_initial_J','max_initial_J','ground_lift_m']}),flush=True)
    write(ROOT/'checks/registration-summary.json',summaries)
    return asset,summaries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-episode',type=int,default=5)
    p.add_argument('--episodes',type=int,nargs='+',default=list(range(10)))
    a=p.parse_args();build(a.reference_episode,a.episodes)
