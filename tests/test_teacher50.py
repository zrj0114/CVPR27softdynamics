import numpy as np
import torch
from softdynamics.io import write_json,sha256
from softdynamics.teacher50 import teacher_windows
from softdynamics.joint_world import interaction_context,context_batch,JointWorld
from softdynamics.shared_world_training import Replay
from softdynamics.elastic_world_data import freeze_data
from softdynamics.shared_data import voxel_particle_quadrature

def test_small_first_frame_volume_refines_particle_quadrature_without_inflation():
    for count in (47,52,512):
        coarse=np.zeros((count,3),np.float32);coarse[:,0]=np.arange(count)*.003
        points,volume,subdivisions=voxel_particle_quadrature(coarse)
        assert len(points)>=4096
        np.testing.assert_allclose(len(points)*volume,count*.003**3,rtol=1e-14)
        offsets=points.reshape(count,-1,3)-coarse[:,None]
        assert np.abs(offsets).max()<.0015
        np.testing.assert_allclose(points.reshape(count,-1,3).mean(1,dtype=np.float64),coarse,atol=2e-7,rtol=0)
        if count==512:assert subdivisions==2

def test_partial_phase_labels_and_confidence_keep_real_available_support():
    rng=np.random.default_rng(7);T,N=10,128
    initial=rng.normal(size=(N,3)).astype('f')*.01
    tracks=np.broadcast_to(initial,(T,N,3)).copy()
    observation=dict(tracks=tracks,track_visible=np.ones((T,N),bool))
    meta=dict(clip_start=0,clip_end=5,allow_missing_phases=True)
    windows,weights=teacher_windows(observation,meta,dict(before=.1,interaction=.45,release=.45))
    assert weights==dict(interaction=.5,release=.5)
    assert sum(w['weight'] for w in windows)==1 and all(w['phase']!='before' for w in windows)
    data=dict(positions=tracks,valid=observation['track_visible'],times=np.arange(T)/30,
              scale_m=np.float32(.1),origin_m=np.zeros(3),actions=np.zeros((T,52),np.float32))
    context=interaction_context(data,meta)
    assert context['valid'].shape==(6,128)
    assert not context['valid'][:2].any() and context['valid'][2:].all()
    model=JointWorld(dict(hidden=16,layers=1,heads=4,material_dim=2),('E_pa','nu'))
    prediction=model.confidence(context_batch([context],'cpu'),torch.tensor([[3500.,.27]]))
    assert torch.isfinite(prediction).all()

def test_same_object_episode_assets_keep_distinct_geometry_and_test_data_out_of_training(tmp_path):
    root=tmp_path/'data';root.mkdir();tasks=[];excluded=['heldout']
    for index in range(3):
        obj='fixture' if index<2 else 'heldout';identity=f'{obj}/episode_{index}/sample_0'
        folder=root/'trajectories'/identity;folder.mkdir(parents=True)
        node_path=root/f'nodes_{index}.npz'
        np.savez(node_path,canonical=np.full((2,3),float(index)),neighbors=np.array([[1],[0]]),surface_mask=np.ones(2,bool))
        state=np.zeros((4,2,24),np.float32);state[...,6:15]=np.eye(3).reshape(9)
        np.savez(folder/'trajectory.npz',state=state,pads=np.zeros((4,4,13)),times=np.arange(4)/30,
                 origin_m=np.zeros(3),scale_m=1.,theta=np.array([3500.,.27]),phase=np.array([0,1,1,2]))
        split='train' if index<2 else 'test'
        tasks.append(dict(id=identity,object=obj,episode=index,split=split))
        write_json(folder/'complete.json',dict(valid=True,full_rollout=True,trajectory_sha256=sha256(folder/'trajectory.npz'),
            episode_preparation='recorded-in-episode-manifest',nodes=str(node_path),nodes_sha256=sha256(node_path)))
    write_json(root/'plan.json',dict(tasks=tasks,objects=[dict(object='fixture',split='train'),dict(object='heldout',split='test')],
        excluded_objects=excluded,collection_includes_test=True))
    replay=Replay(root,['train']);assert replay.refresh()==2
    _,geometry=replay.batch([(t['id'],0) for t in tasks[:2]],1,'cpu')
    assert torch.equal(geometry['canonical'][0],torch.zeros(2,3))
    assert torch.equal(geometry['canonical'][1],torch.ones(2,3))
    output=tmp_path/'training';output.mkdir()
    manifest=freeze_data(root,output)
    assert len(manifest['records'])==2 and all(r['split']=='train' for r in manifest['records'])
