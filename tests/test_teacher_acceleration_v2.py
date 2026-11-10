import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from softdynamics.bound_observation import BoundObservation
from softdynamics.adapters.empm import BranchRuntime


def test_observation_preserves_binding_order_duplicates_and_initial_offsets():
    rng=np.random.default_rng(19)
    positions=rng.normal(size=(1000,3)).astype('f')
    saved=rng.integers(0,1000,600)
    binding=rng.integers(0,600,(12,8))
    binding[:,3]=binding[:,0]
    weights=rng.random((12,8));weights/=weights.sum(1,keepdims=True)
    data=dict(saved_ids=saved,binding=binding,initial=positions[saved]+.123456789,
              weights=weights,tracks=rng.normal(size=(4,12,3)))
    requested=[]
    def read(ids):requested.append(len(ids));return positions[ids]
    runtime=SimpleNamespace(positions=read)
    for points in [None,np.array([7,2,7,4])]:
        dense=BoundObservation(data,points,compact=False)
        sparse=BoundObservation(data,points,compact=True)
        x=dense.capture(runtime);y=sparse.capture(runtime)
        np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8))
        assert len(sparse.particle_ids)<len(dense.particle_ids)


def test_gpu_gather_keeps_host_rounding_and_handles_changed_indices(monkeypatch):
    import warp as wp
    wp.init()
    rng=np.random.default_rng(3)
    x=rng.normal(size=(100,3)).astype('f')
    runtime=BranchRuntime.__new__(BranchRuntime)
    runtime.states=[SimpleNamespace(particle_x=wp.array(x,dtype=wp.vec3,device='cpu'))]
    runtime.current=0;runtime.n=len(x);runtime.device='cpu'
    runtime.shift=np.array([.103849184724,.98387121653,-.5736312689])
    ids=np.array([15,2,15,-1],dtype=np.int32)
    for iteration in range(7):
        ids[0]=iteration
        monkeypatch.setenv('SOFTDYNAMICS_COMPACT_OBSERVATION','0')
        reference=runtime.positions(ids)
        monkeypatch.setenv('SOFTDYNAMICS_COMPACT_OBSERVATION','1')
        result=runtime.positions(ids)
        np.testing.assert_array_equal(reference.view(np.uint8),result.view(np.uint8))
    assert len(runtime._position_index_cache)==4


def test_sparse_index_matches_full_reset_when_cells_are_left_and_revisited():
    import sys
    import warp as wp
    root=Path(__file__).resolve().parents[1]/'runtime'
    sys.path.insert(0,str(root/'gs-layer'))
    from deterministic_transfer import DeterministicTransfer,MPMModelStruct,MPMStateStruct
    wp.init()
    model=MPMModelStruct();model.n_particles=128
    model.grid_dim_x=model.grid_dim_y=model.grid_dim_z=64
    model.inv_dx=1.;model.dx=1.
    state=MPMStateStruct()
    initial=np.random.default_rng(10).uniform(4.6,6.4,(128,3)).astype('f')
    state.particle_x=wp.array(initial,dtype=wp.vec3,device='cpu')
    dense=DeterministicTransfer(model,'cpu');compact=DeterministicTransfer(model,'cpu')
    dense.active_grid=compact.active_grid=True
    dense.sparse_index_reset=False;compact.sparse_index_reset=True
    for shift in [0.,20.,40.,0.,20.,1.,40.,0.]:
        state.particle_x.assign(initial+shift)
        for transfer in [dense,compact]:transfer.prepare(state,model);transfer.check()
        for name in ['begin','marked']:
            np.testing.assert_array_equal(getattr(dense.index,name).numpy(),getattr(compact.index,name).numpy())
        occupied=dense.index.begin.numpy()>=0
        np.testing.assert_array_equal(dense.index.end.numpy()[occupied],compact.index.end.numpy()[occupied])
        count=int(dense.index.count.numpy()[0]);assert count==int(compact.index.count.numpy()[0])
        np.testing.assert_array_equal(np.sort(dense.index.nodes.numpy()[:count]),np.sort(compact.index.nodes.numpy()[:count]))


def test_collector_dispatch_resolves_revision_without_recursion(tmp_path):
    import json
    import pytest
    from softdynamics.collection_dispatch import execution_target
    root=tmp_path/'run';root.mkdir();old=tmp_path/'old';new=tmp_path/'new'
    assert execution_target(root,old) is None
    (root/'acceleration.json').write_text(json.dumps({'execution_code_root':str(new)}))
    with pytest.raises(ValueError,match='Missing collection execution revision'):execution_target(root,old)
    (new/'softdynamics').mkdir(parents=True);(new/'softdynamics/elastic_data.py').touch()
    assert execution_target(root,old)==new
    assert execution_target(root,new) is None
