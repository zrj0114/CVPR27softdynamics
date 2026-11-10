import ast
from pathlib import Path

import numpy as np

from softdynamics.elastic_data import ElasticSpace
from softdynamics.teacher import generate
from softdynamics.source_equivalence import physical_sources
from test_teacher import LinearRuntime


def test_active_node_wrappers_preserve_physics_expressions():
    root = Path(__file__).resolve().parents[1] / 'runtime'
    generated = {n.name:n for n in ast.parse((root/'gs-layer/active_grid_kernels.py').read_text()).body
                 if isinstance(n,ast.FunctionDef)}
    sources = {
        root/'empm/phys_sim/diff_simulator/warp_mpm/mpm_utils.py':
            ['zero_grid','grid_normalization_and_gravity','add_damping_via_grid'],
        root/'gs-layer/pad_contact_kernels.py':['accumulate_pad_contact'],
        root/'gs-layer/material_contact_kernels.py':['material_grid_boundary']}
    for path,names in sources.items():
        original = {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        for name in names:
            expected = ast.Module(body=original[name].body[1:],type_ignores=[])
            actual = ast.Module(body=generated['active_'+name].body[0].body[4:],type_ignores=[])
            assert ast.dump(actual)==ast.dump(expected),name


def test_cached_teacher_matches_uncached_including_refinement(tmp_path,monkeypatch):
    config = dict(baseline=dict(E_pa=3500.,nu=.27,ground_friction=.5,velocity_decay_rate_s_inv=.3),
                  reference_E_pa=3500.,reference_nu=.27,nu_scale=.2)
    space = ElasticSpace(config)

    class Nonlinear(LinearRuntime):
        def __init__(self):
            self.frame,self.times=0,np.arange(10)/30
            self.x=np.arange(12,dtype=float).reshape(4,3)*.01
            self.parameters=space.parameters(np.zeros(2))
        def advance(self,end):
            q=space.coordinate(self.parameters)
            # Radius-sensitive response forces the existing refinement branch.
            self.x+=(end-self.frame)*(np.exp(20*q[0])+.01*np.exp(3*q[1]))*.0001
            self.frame=end
            return dict(valid=True)

    initial=Nonlinear().x.copy()
    data=dict(tracks=np.broadcast_to(initial,(10,4,3)),track_visible=np.ones((10,4),bool),
              initial=initial,saved_ids=np.arange(4),binding=np.arange(4)[:,None],
              weights=np.ones((4,1)),frames=np.arange(10),point_ids=np.arange(4))
    spec=dict(windows_per_phase=2,points=4,workpoints=[[0.,0.],[.05,-.05],[-.05,.05]],
              radii=[.05,.025],refinement_radii=[.0125,.00625],relative_column_tolerance=.01)
    records=[];stats=[]
    for enabled in ['0','1']:
        monkeypatch.setenv('SOFTDYNAMICS_TEACHER_CACHE',enabled)
        timing={};out=tmp_path/enabled
        rows=generate(Nonlinear(),data,dict(clip_start=3,clip_end=6),space,spec,
                      dict(before=.1,interaction=.45,release=.45),out,timings=timing)
        records.append(rows);stats.append(timing)
    assert stats[1]['reused_perturbation_rollouts']>0
    assert stats[1]['perturbation_rollouts']<stats[0]['perturbation_rollouts']
    assert stats[1]['independent_nominal_repeats']==stats[0]['independent_nominal_repeats']==18
    assert any(row['refined_axes'] for row in records[0])
    for i in range(3):
        with np.load(tmp_path/'0'/f'workpoint_{i}.npz') as x,np.load(tmp_path/'1'/f'workpoint_{i}.npz') as y:
            assert x.files==y.files
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        assert records[0][i]['windows']==records[1][i]['windows']


def test_execution_equivalence_never_aliases_physical_inputs():
    import pytest
    old=dict(source_hashes=dict(transfer='dense',asset='body',actions='poses',adapter='pad'))
    new=dict(source_hashes=dict(old['source_hashes'],transfer='active'),
             source_equivalence=dict(schema='softdynamics.transfer-equivalence.v1',validation='pinned-report',
                 aliases=dict(transfer=dict(executed='active',reference='dense'))))
    assert physical_sources(old)==physical_sources(new)
    new['source_hashes']['actions']='different'
    assert physical_sources(old)!=physical_sources(new)
    new['source_hashes']['transfer']='unvalidated'
    with pytest.raises(ValueError,match='actual transfer'):physical_sources(new)
    new['source_equivalence']['aliases']['actions']=dict(executed='different',reference='poses')
    with pytest.raises(ValueError,match='Invalid execution'):physical_sources(new)


def test_material_pairs_accept_validated_execution_and_reject_domain_change():
    import copy
    from softdynamics.elastic_world_data import MaterialPairs
    source=dict(minimum_start=0,pads=np.zeros((2,4,13)),times=np.arange(2),phase=np.zeros(2),
                origin_m=np.zeros(3),scale_m=1.,state=np.zeros((2,3,24)),theta=np.array([3500.,.27]),
                record=dict(source_hashes={'initial':'same','transfer':'dense'},domain={'grid_size':160}),
                task=dict(parameters={'E_pa':3500.,'nu':.27,'friction':1.},dtmax=.001))
    other=copy.deepcopy(source);other['theta'][0]=5000.
    other['record']['source_hashes']['transfer']='active'
    assert MaterialPairs.mismatch(source,other)=='Different physical sources'
    other['record']['source_equivalence']=dict(schema='softdynamics.transfer-equivalence.v1',
        validation='pinned-report',aliases=dict(transfer=dict(executed='active',reference='dense')))
    assert MaterialPairs.mismatch(source,other) is None
    other['record']['domain']['grid_size']=176
    assert MaterialPairs.mismatch(source,other)=='Different simulation domain'
