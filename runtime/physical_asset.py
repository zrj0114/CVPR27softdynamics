"""Portable physical-asset loader and first-state placement API."""
from pathlib import Path
import json
import numpy as np
from build_asset import sha,write
from registration import map_points,rigid_icp,fit_deformation,positive_map,subsample


class PhysicalAsset:
    def __init__(self,folder):
        self.folder=Path(folder).resolve()
        self.manifest=json.loads((self.folder/'asset.json').read_text())
        self.fingerprint=sha(self.folder/'asset.json')
        for name,digest in self.manifest['payload_sha256'].items():
            if sha(self.folder/name)!=digest:raise ValueError(f'Asset hash mismatch: {name}')
        with np.load(self.folder/'body.npz',allow_pickle=False) as z:self.body=dict(z)
        with np.load(self.folder/'appearance.npz',allow_pickle=False) as z:self.appearance=dict(z)
        with np.load(self.folder/'appearance_binding.npz',allow_pickle=False) as z:self.appearance_binding=dict(z)

    def load_initial_state(self,path):
        path=Path(path)
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['state_sha256']!=sha(path):raise ValueError('Initial-state hash mismatch')
        with np.load(path,allow_pickle=False) as z:state=dict(z)
        if str(state['asset_sha256'])!=self.fingerprint:raise ValueError('Placement uses a different reference body')
        np.testing.assert_array_equal(state['material_ids'],self.body['material_ids'])
        if not np.isfinite(state['positions_world_m']).all() or not np.isfinite(state['deformation_gradient']).all():raise ValueError('Non-finite initial state')
        if np.linalg.det(state['deformation_gradient']).min()<=0:raise ValueError('Non-positive initial deformation')
        return state,meta

    def rigid_state(self,world_from_canonical):
        """Initialize this same body at an arbitrary proper rigid pose."""
        tr=np.asarray(world_from_canonical,dtype=float)
        if tr.shape!=(4,4) or not np.allclose(tr[3],[0,0,0,1]):raise ValueError('Expected homogeneous pose')
        r=tr[:3,:3]
        if not np.allclose(r.T@r,np.eye(3),atol=1e-7) or np.linalg.det(r)<.999999:raise ValueError('Expected proper rigid rotation')
        n=len(self.body['rest_positions'])
        return dict(positions_world_m=(self.body['rest_positions']@r.T+tr[:3,3]).astype(np.float32),
            deformation_gradient=np.broadcast_to(r,(n,3,3)).astype(np.float32).copy(),
            velocities_world_m_s=np.zeros((n,3),np.float32),material_ids=self.body['material_ids'].copy(),
            rest_volumes_m3=self.body['rest_volumes_m3'].copy(),masses_kg=self.body['masses_kg'].copy(),
            world_from_canonical_rigid=tr,asset_sha256=np.array(self.fingerprint))

    def fit_initial_surface(self,observed_initial_points):
        """Fit a new placement using ONLY supplied initial surface points."""
        target=np.asarray(observed_initial_points,dtype=float)
        if target.ndim!=2 or target.shape[1]!=3 or not np.isfinite(target).all() or len(target)<100:
            raise ValueError('Expected at least 100 finite initial surface points in meters')
        surface=self.appearance['xyz'].astype(float);centers=self.body['deformation_control_centers']
        best,_=rigid_icp(surface,target);error,r,t=best
        coeff,history=fit_deformation(surface,target,centers,r,t)
        chosen,trials=positive_map(surface,self.body['rest_positions'].astype(float),subsample(target),centers,r,t,coeff,.035)
        error,coeff,points,F,lift=chosen;points[:,1]+=lift;t=t.copy();t[1]+=lift
        pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=t
        return dict(positions_world_m=points.astype(np.float32),deformation_gradient=F.astype(np.float32),
            velocities_world_m_s=np.zeros_like(points,dtype=np.float32),material_ids=self.body['material_ids'].copy(),
            masses_kg=self.body['masses_kg'].copy(),rest_volumes_m3=self.body['rest_volumes_m3'].copy(),
            world_from_canonical_rigid=pose,deformation_coefficients=coeff,deformation_width_m=np.array(.035),
            asset_sha256=np.array(self.fingerprint)),dict(error_mm=error,ground_lift_m=lift,trials=trials)

    def placed_appearance(self,state):
        pose=state['world_from_canonical_rigid'];centers=self.body['deformation_control_centers']
        coeff=state.get('deformation_coefficients',np.zeros((len(centers),3)))
        return map_points(self.appearance['xyz'].astype(float),pose[:3,:3],pose[:3,3],centers,coeff,
                          float(state.get('deformation_width_m',.035)))

    def surface_from_particles(self,positions):
        return np.asarray(positions)[self.body['surface_particle_ids']]

    def save_initial_state(self,state,path,metadata=None):
        """Write a compatible reusable placement, including conserved mass."""
        state=dict(state);path=Path(path)
        j=np.linalg.det(state['deformation_gradient'])
        if not np.isfinite(j).all() or j.min()<=0:raise ValueError('Invalid placement Jacobian')
        np.testing.assert_array_equal(state['material_ids'],self.body['material_ids'])
        state['current_volumes_m3']=self.body['rest_volumes_m3']*j
        state['current_density_kg_m3']=self.body['masses_kg']/state['current_volumes_m3']
        state['asset_sha256']=np.array(self.fingerprint)
        if 'deformation_coefficients' not in state:
            state['deformation_coefficients']=np.zeros_like(self.body['deformation_control_centers'])
            state['deformation_width_m']=np.array(.035)
        if path.exists() or path.with_suffix('.json').exists():raise FileExistsError('Placement exists; use another output name')
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**state)
        report=dict(metadata or {},asset_sha256=self.fingerprint,state_sha256=sha(path),
            min_initial_J=float(j.min()),max_initial_J=float(j.max()),
            material_particles=len(j),total_mass_kg=float(self.body['masses_kg'].sum()))
        write(path.with_suffix('.json'),report)
        return path
