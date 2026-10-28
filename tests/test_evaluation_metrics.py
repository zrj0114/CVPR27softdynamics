import numpy as np
import pytest

from softdynamics.evaluation_metrics import geometry_metrics,mask_iou,psnr_from_mse,transport_gaussians


def test_distance_direction_norm_units_and_track_visibility():
    pred=np.array([[0.,0,0],[100.,100,100]])
    ref=np.array([[.001,.002,0]])
    tracks=np.array([[0.,0,0],[10.,0,0]])
    target=np.array([[.003,.004,0],[np.nan,0,0]])
    result=geometry_metrics(pred,ref,tracks,target,[True,False])
    assert result['dist_mm']==pytest.approx(3.)  # L1, GT -> prediction only.
    assert result['track_mm']==pytest.approx(5.)  # L2, ignore invisible/NaN GT.
    assert geometry_metrics(pred,ref,tracks,target,[False,False])['track_mm'] is None
    with pytest.raises(ValueError):geometry_metrics(pred*float('nan'),ref,tracks,target,[True,False])


def test_iou_empty_and_partial_overlap():
    assert mask_iou([0,0],[0,0])==1
    assert mask_iou([1,0],[0,1])==0
    assert mask_iou([1,1,0],[0,1,1])==pytest.approx(1/3)


def test_psnr_handles_blank_views_without_silent_nan_aggregation():
    assert psnr_from_mse(.01)==pytest.approx(20.)
    assert psnr_from_mse(0.)==100.
    with pytest.raises(ValueError):psnr_from_mse(float('nan'))


def test_material_binding_reproduces_initial_and_global_affine_motion():
    initial=np.array([[0.,0,0],[1.,0,0]])
    ids=np.array([[0,1]]);weights=np.array([[.25,.75]])
    offset=np.array([[.1,.2,.3]])
    cov=np.diag([.01,.02,.03])[None]
    F0=np.broadcast_to(np.diag([2.,1.,1.]),(2,3,3)).copy()
    inverse=np.linalg.inv(F0[ids])
    g0=weights@initial+offset
    means,C=transport_gaussians(initial,F0,ids,weights,inverse,offset,cov)
    np.testing.assert_allclose(means,g0);np.testing.assert_allclose(C,cov)
    A=np.array([[0.,-2,0],[1.,0,0],[0,0,.5]]);t=np.array([3.,4,5])
    means,C=transport_gaussians(initial@A.T+t,A@F0,ids,weights,inverse,offset,cov)
    np.testing.assert_allclose(means,g0@A.T+t)
    np.testing.assert_allclose(C,A@cov@A.T)
