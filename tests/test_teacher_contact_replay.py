import json
from pathlib import Path
import numpy as np
import pytest
from softdynamics.teacher_contact import (CONTACT_ID,KERNEL_SHA256,STATIC_FRICTION,SLIDING_FRICTION,
    pack_pads,validate_contact_parameters)
from softdynamics.teacher_contact_replay import audit_replay_contact

def test_stale_teacher_actions_are_identified_without_rewriting_labels(tmp_path):
    root=tmp_path/'replay';source=tmp_path/'actions';obj='fixture';episode=0
    poses=np.broadcast_to(np.eye(4),(3,1,2,4,4)).copy()
    a=dict(poses=poses,valid=np.ones((3,1,2),bool),source_frames=np.arange(3),
        source_timestamps_us=np.array([0,33333,66666]),face_size_m=np.array([.064,.025]),gripper_count=np.array(1))
    p=source/obj/'episode_0';p.mkdir(parents=True);np.savez(p/'actions.npz',**a)
    tasks=[];records=[];payloads={}
    for i in range(2):
        identity=f'{obj}/episode_0/sample_{i}';folder=root/'trajectories'/identity;folder.mkdir(parents=True)
        tasks.append(dict(id=identity,object=obj,episode=episode,randomized_actions=False,
            parameters=dict(gripper_static_friction=STATIC_FRICTION,gripper_sliding_friction=SLIDING_FRICTION)))
        (folder/'complete.json').write_text(json.dumps(dict(valid=True,full_rollout=True,source_hashes=dict(pad_kernel=KERNEL_SHA256['pad_contact_kernels.py']))))
        pads=pack_pads(a,np.zeros(3),1.)
        if i:pads[1,0,12]=0 # Historical observation gap had incorrectly removed this controller.
        np.savez(folder/'trajectory.npz',pads=pads,times=a['source_timestamps_us']*1e-6,origin_m=np.zeros(3),scale_m=1.)
        records.append(dict(id=identity));payloads[folder/'trajectory.npz']=(folder/'trajectory.npz').read_bytes()
    (root/'plan.json').write_text(json.dumps(dict(tasks=tasks)))
    report=audit_replay_contact(dict(data_root=str(root),records=records),action_root=source)
    assert report['ready_records']==1 and report['need_regeneration']==1 and not report['ready']
    assert report['records'][1]['reasons']==['completed_actions_change_teacher_controls']
    for path,before in payloads.items():assert path.read_bytes()==before

def test_reference_friction_is_fixed_but_material_parameters_can_change():
    p=dict(gripper_static_friction=STATIC_FRICTION,gripper_sliding_friction=SLIDING_FRICTION,E_pa=3500,nu=.27)
    validate_contact_parameters(p);validate_contact_parameters(dict(p,E_pa=6000,nu=.35))
    with pytest.raises(ValueError,match='fixes'):validate_contact_parameters(dict(p,gripper_static_friction=.5))
