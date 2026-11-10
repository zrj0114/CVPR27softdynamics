"""The user-selected finite-pad teacher contract and shared action conventions.

This is the CURRENT teacher, including its simultaneous-correction averaging.
Do not silently replace its impulse law while aligning the learned surrogate.
The network's finite support is an action-input mask, not a hard contact solve.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONTACT_ID = 'deform360.current_teacher.finite_pad_coulomb.v1'
SUPPORT_ID = 'finite_pad_grid_swept.v1'
FACE_SIZE_M = (.064, .025)
GRID_DX_M = .003125
SLAB_CELLS = 2.
GRID_SUPPORT_CELLS = 1.5  # Quadratic B-spline particle/grid transfer support.
STATIC_FRICTION = 2.15827
SLIDING_FRICTION = .770921
DEFAULT_ACTION_ROOT = Path('/home/guangqi/data/deform360_pose_completion_source_20261103/dataset/episodes')
KERNEL_SHA256 = {
    'pad_contact_kernels.py': '42b4235079010ef63d3b826029fb8ff71340e1b859a6bfb93f1998f373345b3b',
    'material_contact_kernels.py': '4c28c2085da2a35d24e17c27252711d7c7ce4f0078e202bc55b59654b19a5629',
}

def contract():
    return dict(id=CONTACT_ID, face_size_m=list(FACE_SIZE_M), grid_dx_m=GRID_DX_M,
        slab_depth_m=SLAB_CELLS*GRID_DX_M, left_right_normal_signs=[1.,-1.],
        normal_response='approaching relative normal velocity only; no adhesion',
        tangential_response='current static/sliding Coulomb contact_delta',
        gripper_static_friction=STATIC_FRICTION, gripper_sliding_friction=SLIDING_FRICTION,
        simultaneous_contacts='current teacher average of contributing velocity corrections',
        kernels=KERNEL_SHA256, action_source=str(DEFAULT_ACTION_ROOT),
        gap_policy='source rigid interpolation / endpoint hold, no pose quality gates',
        absent_grippers='excluded by task metadata; never manufacture an absent controller',
        network_support=SUPPORT_ID, network_is_hard_contact_solver=False,
        domain_policy='include complete recorded action timeline, including withdrawal')

def contract_sha256():
    return hashlib.sha256(json.dumps(contract(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_selected_kernels(runtime):
    for name,digest in KERNEL_SHA256.items():
        if file_sha256(Path(runtime)/'gs-layer'/name)!=digest:
            raise ValueError('Selected teacher kernel changed: '+name)

def validate_contact_parameters(parameters):
    for name,expected in [('gripper_static_friction',STATIC_FRICTION),('gripper_sliding_friction',SLIDING_FRICTION)]:
        if not math.isclose(float(parameters[name]),expected,rel_tol=0,abs_tol=1e-6):
            raise ValueError('Selected teacher contract fixes '+name+' to '+str(expected))

def logical_action_sha256(actions):
    """Hash physical inputs, independent of archival metadata and NPZ encoding."""
    h=hashlib.sha256()
    for name in ('poses','valid','source_frames','source_timestamps_us'):
        a=np.asarray(actions[name]);h.update(name.encode());h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def load_completed_actions(obj, episode, prepared_path=None, root=None):
    path=Path(root or DEFAULT_ACTION_ROOT)/obj/f'episode_{episode}'/'actions.npz'
    with np.load(path,allow_pickle=False) as z:a=dict(z)
    poses,valid=a['poses'],np.asarray(a['valid'],bool)
    if poses.ndim!=5 or poses.shape[2:]!=(2,4,4) or poses.shape[1] not in (1,2):
        raise ValueError('Expected one or two recorded grippers, each with two pad poses')
    if valid.shape!=poses.shape[:3] or not valid.all() or not np.isfinite(poses).all():
        raise ValueError('Use the completed action export; source observation masks are not controller-existence masks')
    if int(a['gripper_count'])!=poses.shape[1]:raise ValueError('Gripper task metadata mismatch')
    if not np.allclose(a['face_size_m'],FACE_SIZE_M,rtol=0,atol=1e-8):raise ValueError('Different pad geometry requires its own contact contract')
    T=len(poses)
    if a['source_frames'].shape!=(T,) or a['source_timestamps_us'].shape!=(T,) or not np.all(np.diff(a['source_timestamps_us'])>0):
        raise ValueError('Action timeline shape/order mismatch')
    if prepared_path is not None:
        with np.load(prepared_path,allow_pickle=False) as old:
            for name in ('source_frames','source_timestamps_us'):
                if not np.array_equal(a[name],old[name]):raise ValueError('Prepared observations and completed actions have different '+name)
    a['valid']=valid
    return a,dict(path=str(path.resolve()),sha256=file_sha256(path),logical_sha256=logical_action_sha256(a),
        contact_id=CONTACT_ID,contact_sha256=contract_sha256(),grippers=poses.shape[1])

def pack_pads(actions,origin,scale):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive metric scale required')
    poses,valid=actions['poses'],actions['valid'];T,G=poses.shape[:2]
    pads=np.zeros((T,4,13),np.float32)
    for g in range(G):
        for side in range(2):
            k=2*g+side;good=valid[:,g,side]
            pads[good,k,:3]=(poses[good,g,side,:3,3]-origin)/scale
            pads[good,k,3:12]=poses[good,g,side,:3,:3].reshape(-1,9)
            pads[good,k,12]=1.
    return pads

def replace_visual_actions(visual,actions):
    """Confidence and dynamics read the same actions, with their own normalization."""
    T=len(visual['times']);times=(actions['source_timestamps_us']-actions['source_timestamps_us'][0])*1e-6
    if len(times)<T or not np.allclose(visual['times'],times[:T],atol=2e-6,rtol=0):
        raise ValueError('Confidence observation timeline differs from completed actions')
    p=pack_pads(actions,visual['origin_m'],float(visual['scale_m']))[:T]
    matrices=np.concatenate([p[...,3:12].reshape(T,4,3,3),p[...,:3,None]],axis=-1)
    result=dict(visual);result['actions']=np.concatenate([matrices.reshape(T,4,12),p[...,12:]],axis=-1).reshape(T,52)
    return result

def require_selected_architecture(architecture):
    if architecture.get('contact_contract')!=CONTACT_ID or architecture.get('pad_locality_geometry')!=SUPPORT_ID:
        raise ValueError('Checkpoint predates the selected teacher contact contract; retrain the aligned model before selected-policy search')

def selected_architecture(base):
    result=dict(base)
    result.pop('pad_locality_inner_m',None);result.pop('pad_locality_outer_m',None)
    result.update(contact_geometry=True,pad_locality=True,contact_contract=CONTACT_ID,pad_locality_geometry=SUPPORT_ID)
    return result
