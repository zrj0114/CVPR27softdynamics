import numpy as np
import pytest
from softdynamics.io import write_json
from softdynamics.teacher50_geometry import surface_mask_consensus,project_mask
from softdynamics.teacher50 import farm

def test_background_masks_do_not_erase_first_frame_object_or_admit_background_points():
    rng=np.random.default_rng(12)
    object_points=np.c_[rng.uniform(-.025,.025,(40,2)),np.ones(40)]
    background_points=np.c_[rng.uniform(.28,.32,20),rng.uniform(-.02,.02,20),np.ones(20)]
    points=np.r_[object_points,background_points]
    K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
    foreground=np.zeros((100,100),np.uint8);foreground[45:56,45:56]=1
    wrong=np.zeros_like(foreground);wrong[45:56,75:86]=1
    masks=[(str(i),K,np.eye(4),foreground if i<5 else wrong) for i in range(7)]
    keep,selected,evidence=surface_mask_consensus(points,masks)
    assert keep[:40].all() and not keep[40:].any()
    assert [m[0] for m in selected]==['0','1','2','3','4']
    assert all(not e['used_for_carving'] for e in evidence[5:])
    # With five good and two erroneous views the old all-view 85% test erased it.
    assert 5/7<.85

def test_projection_does_not_wrap_outside_image_or_count_points_behind_camera():
    K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
    points=np.array([[0,0,1],[0,0,-1],[-2,0,1],[0,0,0],[np.nan,0,1]])
    inside,hit=project_mask(points,K,np.eye(4),np.ones((100,100),np.uint8))
    np.testing.assert_array_equal(inside,[True,False,False,False,False])
    np.testing.assert_array_equal(hit,inside)

def test_initialization_repair_does_not_resume_stopped_teacher_collection(tmp_path):
    write_json(tmp_path/'plan.json',{})
    write_json(tmp_path/'run_control.json',dict(collection_enabled=False,initialization_repair_enabled=True))
    with pytest.raises(RuntimeError,match='stopped by the user'):farm(tmp_path,[0])
    assert not (tmp_path/'collection.lock').exists()
