import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from softdynamics.shared_world_model import (SharedWorldModel,GoalPolicy,GoalValue,apply_gripper_action,initialize_goal_module,
    observation_loss,confidence_gradient,physical_parameters,rollout,pad_contact_features,policy_action_step)
from softdynamics.shared_world_data import randomize_actions,object_split,stable_seed,task_available
from softdynamics.objective import score
from softdynamics.shared_world_inverse import phase_objective,sparse_projection,project_observations
from softdynamics.shared_world_control import choose_control_start,expanded_actions,action_pads,control_rollout,bounded_pose_sequence,anchored_positions,bounded_reference_commands
from softdynamics.shared_world_control_search import search_protocol,update_search
from softdynamics.shared_world_material_windows import window_rollout
from softdynamics.shared_world_policy_benchmark import actor_digest,validate_reference_case
from softdynamics.shared_world_policy import validate_policy_initialization,eligible_windows,bounded_experience_start
from softdynamics.shared_world_evaluation import validate_manifest
from softdynamics.shared_world_material_benchmark import select_support,bounded_export
from softdynamics.shared_world_observation import padded_simulation_grid
from softdynamics.shared_world_training import Replay,sample_recent_mixture
from softdynamics.io import sha256,write_json


def example(batch=2,nodes=7,dtype=torch.float64):
    torch.manual_seed(32)
    state=torch.randn(batch,nodes,24,dtype=dtype)*.02
    state[...,1]+=.2
    pads=torch.zeros(batch,4,13,dtype=dtype)
    pads[...,:3]=torch.randn(batch,4,3,dtype=dtype)*.05
    pads[...,3:12]=torch.eye(3,dtype=dtype).flatten()
    pads[...,12]=1
    geometry=dict(canonical=torch.randn(batch,nodes,3,dtype=dtype)*.1,
        neighbors=torch.stack([torch.stack([torch.tensor([(j+1)%nodes,(j+2)%nodes]) for j in range(nodes)]) for _ in range(batch)]),
        ground_y=torch.zeros(batch,dtype=dtype),scale_m=torch.tensor([.1]*batch,dtype=dtype),
        surface_mask=torch.ones(batch,nodes,dtype=torch.bool))
    theta=torch.tensor([[3500.,.27,.8,.4]]*batch,dtype=dtype)
    return state,pads,theta,geometry


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_deformation_memory_initialization_preserves_rollout_and_input_gradients(dtype):
    state,pads,theta,geometry=example(batch=1,dtype=dtype)
    source=SharedWorldModel(16,1,4,direct_displacement=True,contact_geometry=True).to(dtype=dtype)
    model=SharedWorldModel(16,1,4,direct_displacement=True,contact_geometry=True,deformation_memory=True).to(dtype=dtype)
    model.initialize_deformation_memory(source.state_dict())
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in source.state_dict().items())
    source.requires_grad_(False);model.requires_grad_(False);theta.requires_grad_()
    action=torch.full((1,2,7),.03,dtype=dtype,requires_grad=True)
    next_pads=apply_gripper_action(pads,action,geometry['scale_m'])
    poses=torch.stack([pads,next_pads,next_pads,next_pads],1)
    times=torch.arange(4,dtype=dtype)[None]/30
    before=rollout(source,state,poses,theta,geometry,times)
    after=rollout(model,state,poses,theta,geometry,times)
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    gradients=[torch.autograd.grad(value[...,:3].square().sum(),(theta,action),retain_graph=True)
               for value in (before,after)]
    for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_deformation_memory_recurrence_uses_F_ignores_C_and_has_material_action_gradients(tmp_path):
    from softdynamics.shared_world_training import load_model
    state,pads,theta,geometry=example(batch=1)
    model=SharedWorldModel(16,1,4,direct_displacement=True,deformation_memory=True).double()
    with torch.no_grad():
        model.deformation_encoder[-1].weight.normal_(std=.1)
        model.deformation_transition[-1].weight.normal_(std=.1)
    model.requires_grad_(False)
    q=torch.zeros(1,4,dtype=torch.float64,requires_grad=True)
    action=torch.full((1,2,7),.03,dtype=torch.float64,requires_grad=True)
    scales=theta.new_tensor([3500.,.1,.5,.5]);times=torch.arange(5,dtype=theta.dtype)[None]/30
    def predict(q,action,start=state):
        end=apply_gripper_action(pads,action,geometry['scale_m'])
        poses=torch.stack([pads,end,end,end,end],1)
        return rollout(model,start,poses,theta+q*scales,geometry,times)
    prediction=predict(q,action)
    changed_C=state.clone();changed_C[...,15:]+=100
    torch.testing.assert_close(predict(q,action,changed_C)[...,:15],prediction[...,:15],rtol=0,atol=0)
    changed_F=state.clone();changed_F[...,6]+=.1
    assert (predict(q,action,changed_F)[...,:3]-prediction[...,:3]).abs().max()>1e-10
    F=prediction[...,6:15].reshape(1,4,7,3,3)+torch.eye(3,dtype=theta.dtype)
    assert torch.isfinite(F).all() and torch.linalg.det(F).min()>0
    objective=prediction[...,:15].square().sum()
    gradients=torch.autograd.grad(objective,(q,action))
    directions=(torch.full_like(q,.2),torch.full_like(action,-.3))
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in gradients)
    expected=sum((g*d).sum() for g,d in zip(gradients,directions))
    eps=1e-4
    positive=predict(q+eps*directions[0],action+eps*directions[1])[...,:15].square().sum()
    negative=predict(q-eps*directions[0],action-eps*directions[1])[...,:15].square().sum()
    torch.testing.assert_close((positive-negative)/(2*eps),expected,atol=1e-8,rtol=1e-5)
    path=tmp_path/'deformation.pt';torch.save(dict(model=model.state_dict(),architecture=model.architecture),path)
    restored,_=load_model(path,'cpu')
    assert restored.deformation_memory and not restored.kinematic_only
    assert restored.architecture==model.architecture
    for key,value in restored.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key].float())


def test_deformation_memory_learns_from_ordinary_F_targets_without_C_supervision():
    from softdynamics.shared_world_model import dynamics_loss
    state,pads,theta,geometry=example(batch=1)
    model=SharedWorldModel(16,1,4,direct_displacement=True,deformation_memory=True).double()
    prediction=model(state,pads,pads,theta,geometry,theta.new_full((1,),1/30))[:,None]
    target=prediction.detach().clone();target[...,6:15]+=.02
    loss=dynamics_loss(prediction,target,model,geometry['scale_m'])
    changed=target.clone();changed[...,15:]+=100
    torch.testing.assert_close(dynamics_loss(prediction,changed,model,geometry['scale_m']),loss,atol=0,rtol=0)
    loss.backward()
    assert model.deformation_transition[-1].weight.grad.abs().max()>0


@pytest.mark.parametrize('extra',[{},dict(kinematic_only=True,direct_displacement=True),
    dict(affine_material=True,direct_displacement=True),dict(residual_material=True,direct_displacement=True)])
def test_deformation_memory_rejects_incompatible_state_representations(extra):
    with pytest.raises(ValueError,match='Deformation memory'):
        SharedWorldModel(deformation_memory=True,**extra)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('factory',[GoalPolicy,GoalValue])
def test_pointwise_goal_initialization_preserves_behavior_and_can_learn(dtype,factory):
    state,pads,theta,geometry=example(batch=1,dtype=dtype)
    goal=state[...,:3]+torch.randn_like(state[...,:3])*.1
    old=factory(16).to(dtype=dtype);new=factory(16,pointwise_goal=True).to(dtype=dtype)
    initialize_goal_module(new,old.state_dict(),old.architecture)
    def apply(model):
        return model(state,geometry,goal,pads,theta) if factory is GoalPolicy else model(state,geometry,goal,theta,pads)
    torch.testing.assert_close(apply(new),apply(old),rtol=0,atol=0)
    assert 'goal_condition.weight' not in old.state_dict()
    apply(new).square().sum().backward()
    assert new.goal_condition.weight.grad.abs().max()>0
    with pytest.raises(ValueError,match='discard'):
        initialize_goal_module(old,new.state_dict(),new.architecture)


@pytest.mark.parametrize('factory',[GoalPolicy,GoalValue])
def test_pointwise_goal_represents_correspondence_and_keeps_joint_permutation_invariance(factory):
    state,pads,theta,geometry=example(batch=1)
    goal=state[...,:3].clone();permutation=torch.arange(state.shape[1]).roll(1)
    other=goal[:,permutation]
    old=factory(16).double();new=factory(16,pointwise_goal=True).double()
    initialize_goal_module(new,old.state_dict(),old.architecture)
    def apply(model,state,geometry,goal):
        return model(state,geometry,goal,pads,theta) if factory is GoalPolicy else model(state,geometry,goal,theta,pads)
    torch.testing.assert_close(apply(old,state,geometry,goal),apply(old,state,geometry,other),atol=1e-14,rtol=0)
    # A learned nonzero projection can distinguish the same target set with
    # different material-point assignments. This is not a reachability claim.
    with torch.no_grad():new.goal_condition.weight.normal_(std=2.)
    assert (apply(new,state,geometry,goal)-apply(new,state,geometry,other)).abs().max()>1e-10
    reordered={**geometry,'canonical':geometry['canonical'][:,permutation],
        'surface_mask':geometry['surface_mask'][:,permutation]}
    torch.testing.assert_close(apply(new,state,geometry,other),
        apply(new,state[:,permutation],reordered,other[:,permutation]),atol=1e-14,rtol=0)


def test_pointwise_goal_policy_checkpoint_restores_representation(tmp_path):
    from softdynamics.shared_world_policy import load_policy
    actor=GoalPolicy(16,pointwise_goal=True)
    with torch.no_grad():actor.goal_condition.weight.normal_(std=.1)
    path=tmp_path/'policy.pt';torch.save(dict(actor=actor.state_dict(),
        manifest=dict(architecture=actor.architecture,bounded_actions=True)),path)
    restored,_=load_policy(path,'cpu')
    assert restored.pointwise_goal and restored.bounded_actions
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in actor.state_dict().items())


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_material_residual_initialization_preserves_every_material_and_input_gradient(dtype):
    state,pads,theta,geometry=example(batch=2,dtype=dtype)
    theta=torch.tensor([[2500.,.18,.2,0.],[5000.,.36,1.5,1.2]],dtype=dtype,requires_grad=True)
    source=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True).to(dtype=dtype)
    model=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True,residual_material=True).to(dtype=dtype)
    model.initialize_residual_material(source.state_dict())
    source.requires_grad_(False);model.requires_grad_(False)
    action=torch.full((2,2,7),.02,dtype=dtype,requires_grad=True)
    next_pads=apply_gripper_action(pads,action,geometry['scale_m']);dt=theta.new_full((2,),1/30)
    before=source(state,pads,next_pads,theta,geometry,dt);after=model(state,pads,next_pads,theta,geometry,dt)
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    ga=torch.autograd.grad(after[...,:3].square().sum(),(theta,action),retain_graph=True)
    gb=torch.autograd.grad(before[...,:3].square().sum(),(theta,action))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_material_residual_training_preserves_parent_weights_and_reference_rollout(tmp_path):
    from softdynamics.shared_world_training import load_model
    state,pads,theta,geometry=example(batch=1)
    source=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True).double()
    model=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True,residual_material=True).double()
    model.initialize_residual_material(source.state_dict());model.train_material_residual_only()
    theta=model.material_reference[None].clone()+theta.new_tensor([[400.,.02,.2,.1]])
    dt=theta.new_full((1,),1/30);target=source(state,pads,pads,theta,geometry,dt).detach()[...,:3]+.01
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        optimizer.zero_grad();loss=(model(state,pads,pads,theta,geometry,dt)[...,:3]-target).square().mean()
        loss.backward();optimizer.step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items())
    assert model.material_residual[-1].weight.abs().max()>0
    assert all(p.requires_grad==k.startswith('material_residual.') for k,p in model.named_parameters())
    poses=pads[:,None].repeat(1,4,1,1);times=torch.arange(4,dtype=torch.float64)[None]/30
    reference=model.material_reference[None].clone().requires_grad_()
    before=rollout(source,state,poses,reference,geometry,times);after=rollout(model,state,poses,reference,geometry,times)
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    ga=torch.autograd.grad(after[...,:3].square().sum(),reference)[0]
    gb=torch.autograd.grad(before[...,:3].square().sum(),reference)[0]
    assert torch.linalg.vector_norm(ga-gb)>0
    path=tmp_path/'residual.pt';torch.save(dict(model=model.state_dict(),architecture=model.architecture),path)
    restored,_=load_model(path,'cpu');assert restored.residual_material
    assert restored.architecture==model.architecture


def test_material_residual_rejects_incompatible_representations():
    with pytest.raises(ValueError,match='one material'):
        SharedWorldModel(direct_displacement=True,affine_material=True,residual_material=True)
    with pytest.raises(ValueError,match='observed-velocity'):
        SharedWorldModel(residual_material=True)


def test_affine_material_conversion_preserves_reference_predictions_and_action_gradients(tmp_path):
    from softdynamics.shared_world_training import load_model
    state,pads,_,geometry=example(batch=1)
    source=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True,contact_geometry=True).double()
    model=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True,contact_geometry=True,
        affine_material=True,material_reference=[3547.0699,.278514,.853391,.359543]).double()
    model.initialize_affine_material(source.state_dict())
    theta=model.material_reference[None].clone().requires_grad_()
    action=torch.full((1,2,7),.02,dtype=torch.float64,requires_grad=True)
    end=apply_gripper_action(pads,action,geometry['scale_m'])
    dt=torch.tensor([1/30],dtype=torch.float64)
    expected=source(state,pads,end,theta,geometry,dt)
    actual=model(state,pads,end,theta,geometry,dt)
    assert torch.allclose(actual,expected,atol=1e-14,rtol=1e-12)
    source_grad=torch.autograd.grad(expected[...,:3].sum(),action,retain_graph=True)[0]
    action_grad,material_grad=torch.autograd.grad(actual[...,:3].sum(),(action,theta))
    assert action_grad.abs().max()>0
    assert torch.allclose(action_grad,source_grad,atol=1e-14,rtol=1e-12)
    assert torch.equal(material_grad,torch.zeros_like(material_grad))
    # A saved model restores the new architecture/reference without migration.
    path=tmp_path/'model.pt'
    torch.save(dict(architecture=model.architecture,model=model.float().state_dict()),path)
    loaded,_=load_model(path,'cpu')
    assert loaded.affine_material
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    assert 'affine_material' not in source.architecture
    assert 'material_reference' not in source.state_dict()


def test_affine_material_rollout_has_correct_material_and_action_gradients():
    state,pads,theta,geometry=example(batch=1)
    model=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True,
        contact_geometry=True,affine_material=True).double()
    model.requires_grad_(False)
    first=theta.clone();second=theta*theta.new_tensor([1.2,.9,1.1,.8])
    midpoint=.5*(first+second);midpoint[:,0]=(first[:,0]*second[:,0]).sqrt()
    dt=theta.new_tensor([1/30])
    a=model(state,pads,pads,first,geometry,dt)
    b=model(state,pads,pads,second,geometry,dt)
    middle=model(state,pads,pads,midpoint,geometry,dt)
    assert torch.allclose(middle,.5*(a+b),atol=1e-14,rtol=1e-12)
    q=theta.new_tensor([[.01,.02,-.01,.03]],requires_grad=True)
    action=torch.full((1,2,7),.02,dtype=theta.dtype,requires_grad=True)
    scales=theta.new_tensor([3500.,.4,.8,1.])
    def future(q,action):
        material=theta+q*scales
        end=apply_gripper_action(pads,action,geometry['scale_m'])
        next_state=model(state,pads,end,material,geometry,dt)
        return model(next_state,end,end,material,geometry,dt)[...,:3]
    assert torch.autograd.gradcheck(future,(q,action),fast_mode=True,eps=1e-5,atol=1e-8,rtol=1e-4)
    grads=torch.autograd.grad(future(q,action).square().sum(),(q,action))
    assert torch.all(grads[0].abs()>1e-12)
    assert grads[1].abs().max()>1e-12


def test_affine_material_fields_learn_from_ordinary_position_loss():
    state,pads,theta,geometry=example(batch=1)
    source=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True).double()
    model=SharedWorldModel(hidden=16,layers=1,heads=4,direct_displacement=True,affine_material=True).double()
    model.initialize_affine_material(source.state_dict())
    theta=theta*theta.new_tensor([1.2,.9,1.1,.8])
    prediction=model(state,pads,pads,theta,geometry,theta.new_tensor([1/30]))
    loss=(prediction[...,:3]-(state[...,:3]+.01)).square().mean()
    loss.backward()
    assert torch.all(model.output[-1][-1].weight.grad[3:].abs().sum(-1)>0)
    with pytest.raises(ValueError,match='observed-velocity'):
        SharedWorldModel(affine_material=True)


def test_policy_refinement_preserves_action_state_semantics_and_training_split():
    saved=dict(manifest=dict(data_plan_sha256='plan',observed_velocity=True,bounded_actions=True,horizon=16),
        training_objects=['train'])
    objects={'train':{'split':'train'},'heldout':{'split':'test'}}
    validate_policy_initialization(saved,'plan',objects,True,True,16)
    with pytest.raises(ValueError,match='action domain'):validate_policy_initialization(saved,'plan',objects,True,False,16)
    with pytest.raises(ValueError,match='velocity'):validate_policy_initialization(saved,'plan',objects,False,True,16)
    with pytest.raises(ValueError,match='horizon'):validate_policy_initialization(saved,'plan',objects,True,True,32)
    with pytest.raises(ValueError,match='data plan'):validate_policy_initialization(saved,'other',objects,True,True,16)
    saved['training_objects'].append('heldout')
    with pytest.raises(ValueError,match='Held-out'):validate_policy_initialization(saved,'plan',objects,True,True,16)


def test_bounded_policy_goals_use_the_original_bound_origin_and_command_rate():
    from types import SimpleNamespace
    import copy
    pads=np.zeros((33,4,13));pads[:,:2,12]=1
    ordinary=dict(pads=pads,phase=np.ones(33),task={'object':'a'},minimum_start=0,record={})
    bounded=copy.deepcopy(ordinary)
    bounded['record']=dict(schema='softdynamics.control-experience.v1',response_labels_used=False,
        total_translation_bound_m_per_axis=.012,total_rotation_bound_rad_per_axis=.08,total_aperture_bound_m=.006,
        random_seed=42,knot_std=.15)
    fast=copy.deepcopy(bounded);fast['record']['knot_std']=10.
    policy=copy.deepcopy(ordinary);policy['record']=dict(schema='softdynamics.policy-experience.v1',response_labels_used=False,bounded_actions=True)
    legacy=copy.deepcopy(policy);legacy['record']['bounded_actions']=False
    no_history=copy.deepcopy(policy);no_history['minimum_start']=1
    replay=SimpleNamespace(trajectories={'raw':ordinary,'bounded':bounded,'fast':fast,'policy':policy,'legacy':legacy,'no_history':no_history})
    selected=eligible_windows(replay,16,'bounded_experience')['a']
    assert {identity for identity,_ in selected}=={'bounded','policy'}
    assert all(starts.tolist()==[0] for _,starts in selected)
    assert len(eligible_windows(replay,16)['a'])==6
    assert not bounded_experience_start(fast,16)


def test_material_experience_enters_only_world_replay_and_preserves_object_split(tmp_path):
    nodes=tmp_path/'nodes.npz';np.savez(nodes,canonical=np.zeros((3,3)),neighbors=np.array([[1,2],[0,2],[0,1]]),surface_mask=np.ones(3,bool))
    plan=dict(objects=[dict(object='o',split='train',nodes=str(nodes),nodes_sha256=sha256(nodes))],tasks=[])
    write_json(tmp_path/'plan.json',plan)
    folder=tmp_path/'experience/material_0/o/episode_0/sample_000';folder.mkdir(parents=True)
    pads=np.zeros((5,4,13));pads[:,:2,12]=1
    np.savez(folder/'trajectory.npz',state=np.zeros((5,3,24)),pads=pads,times=np.arange(5)*.04,
        phase=np.ones(5),theta=np.array([3500,.27,.8,.4]),origin_m=np.zeros(3),scale_m=.2,
        initial_observed_velocity_m_s=np.zeros((3,3)))
    identity='experience/material_0/o/episode_0/sample_000'
    write_json(folder/'complete.json',dict(schema='softdynamics.material-experience.v1',valid=True,full_rollout=True,
        source_data_plan_sha256=sha256(tmp_path/'plan.json'),nodes_sha256=sha256(nodes),trajectory_sha256=sha256(folder/'trajectory.npz'),
        task=dict(id=identity,object='o',split='train',variant=0),policy_training_eligible=False))
    replay=Replay(tmp_path,['train'],True);assert replay.refresh()==1
    assert identity in replay.trajectories and not eligible_windows(replay,2)
    plan['objects'][0]['split']='test';write_json(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='Held-out'):Replay(tmp_path,['train'],True).refresh()


def test_physical_search_counts_failures_and_rejects_worse_actions():
    protocol=search_protocol(8,0)
    assert protocol['amplitude_scales']==[1,.5,.25,.125,.0625,.03125,.015625,.0078125]
    assert protocol['cma_seed']!=0
    rows=[]
    update_search(rows,5.,True,3.,2.)
    update_search(rows,None,False,3.,4.)
    update_search(rows,2.,True,3.,1.)
    update_search(rows,float('nan'),True,3.,3.)
    assert [r['best_terminal_mm'] for r in rows]==[3.,3.,2.,2.]
    assert rows[-1]['candidate_rollouts']==4
    assert rows[-1]['physical_rollouts_including_baseline']==5
    assert rows[-1]['elapsed_seconds']==10.
    assert rows[-1]['valid'] is False
    with pytest.raises(ValueError):search_protocol(3,0)


def test_material_windows_cover_remainder_and_reset_gradient_at_fixed_starts():
    class Step(torch.nn.Module):
        def forward(self,state,p0,p1,theta,geometry,dt):return 2*state+theta
    starts=torch.arange(10.,70.,10.,dtype=torch.float64).reshape(1,6,1,1).requires_grad_(True)
    pads=torch.zeros(1,6,1,dtype=torch.float64);times=torch.arange(6.,dtype=torch.float64)[None]
    theta=torch.ones(1,dtype=torch.float64,requires_grad=True)
    y=window_rollout(Step(),starts,pads,theta,{},times,2)
    torch.testing.assert_close(y.flatten(),torch.tensor([10.,21.,43.,61.,123.,101.],dtype=torch.float64))
    gradient,initial_gradient=torch.autograd.grad(y.sum(),(theta,starts),allow_unused=True)
    torch.testing.assert_close(gradient,torch.tensor([9.],dtype=torch.float64))
    assert initial_gradient is None
    torch.testing.assert_close(y,window_rollout(Step(),starts,pads,theta,{},times,2,gradients=False))
    assert torch.autograd.gradcheck(lambda p:window_rollout(Step(),starts,pads,p,{},times,2),(theta,))


def test_feedback_commands_remain_in_whole_window_domain_and_differentiable():
    class Actor(torch.nn.Module):
        bounded_actions=True
        def __init__(self):
            super().__init__();self.value=torch.nn.Parameter(torch.ones(1,2,7,dtype=torch.float64)*.2)
        def forward(self,*args):return self.value
    state,pads,theta,g=example(1);actor=Actor();current=pads;command=torch.zeros_like(actor.value)
    for _ in range(32):
        current,_,command=policy_action_step(actor,state,g,state[...,:3],current,theta,pads,command)
    expected=apply_gripper_action(pads,(actor.value*8).tanh(),g['scale_m'],translation_m=.012,rotation_rad=.08,aperture_m=.006)
    torch.testing.assert_close(current,expected)
    gradient=torch.autograd.grad(current[...,:3].square().sum(),actor.value)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    with torch.no_grad():
        for _ in range(100):current,_,command=policy_action_step(actor,state,g,state[...,:3],current,theta,pads,command)
        centers=current.reshape(1,2,2,13)[...,:3].mean(2)
        initial_centers=pads.reshape(1,2,2,13)[...,:3].mean(2)
        assert float(((centers-initial_centers)*g['scale_m'][:,None,None]).abs().max())<=.012+1e-10


def test_policy_comparison_rejects_changed_physics_and_distinguishes_action_semantics():
    from types import SimpleNamespace
    physical=SimpleNamespace(obj='held-out',episode=3,horizon=32,start=10,end=42,
        parameters={'E_pa':3500.},snapshot_sha256='original')
    case=dict(object=physical.obj,episode=3,horizon=32,start=10,end=42,
        parameters=physical.parameters,snapshot_sha256='original')
    validate_reference_case(case,physical)
    with pytest.raises(ValueError,match='parameters'):
        validate_reference_case({**case,'parameters':{'E_pa':3600.}},physical)
    with pytest.raises(ValueError,match='snapshot'):
        validate_reference_case({**case,'snapshot_sha256':'different'},physical)
    validate_reference_case(case,physical,{'source':'recorded'})
    with pytest.raises(ValueError,match='goal specification'):
        validate_reference_case(case,physical,{'source':'bounded_knots','seed':0})
    actor=GoalPolicy(16);before=actor_digest(actor)
    actor.bounded_actions=True
    assert actor_digest(actor)!=before
    same=GoalPolicy(16);same.load_state_dict(actor.state_dict());same.bounded_actions=True
    assert actor_digest(same)==actor_digest(actor)


@pytest.mark.parametrize('knot_std',[.15,10.])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_bounded_goal_reference_is_reproducible_through_actual_policy_commands(knot_std,dtype):
    state,pads,theta,g=example(2,dtype=dtype)
    reference,actions,raw=bounded_reference_commands(pads,g['scale_m'],16,7,knot_std)
    assert float(actions.abs().max())<1
    torch.testing.assert_close(reference[:,0],pads)
    repeated=bounded_reference_commands(pads,g['scale_m'],16,7,knot_std)
    for first,second in zip((reference,actions,raw),repeated):torch.testing.assert_close(first,second,atol=0,rtol=0)

    class ReferenceActor:
        bounded_actions=True
        index=0
        def __call__(self,*args):
            action=actions[:,self.index];self.index+=1;return action

    actor=ReferenceActor();current=pads;command=torch.zeros_like(actions[:,0])
    for index in range(16):
        current,_,command=policy_action_step(actor,state,g,state[...,:3],current,theta,pads,command)
        torch.testing.assert_close(current,reference[:,index+1],rtol=0,atol=0)


def test_requested_exploration_start_still_requires_valid_complete_interaction():
    pads=np.zeros((40,4,13));pads[...,12]=1;pads[20,0,12]=0
    assert choose_control_start(pads,5,35,8,10)==10
    with pytest.raises(ValueError,match='Requested start'):choose_control_start(pads,5,35,8,4)
    with pytest.raises(ValueError,match='Requested start'):choose_control_start(pads,5,35,8,18)


def test_recent_replay_mixture_preserves_history_and_never_samples_missing_group():
    replay=Replay.__new__(Replay);replay.trajectories={}
    for prefix,obj in [('old/','a'),('old/','b'),('experience/round_1/','a'),('experience/round_1/','c')]:
        replay.trajectories[prefix+obj]=dict(state=np.zeros((40,1,24)),minimum_start=1,
            phase=np.ones(40),task=dict(object=obj))
    rng=np.random.default_rng(4)
    choices=sample_recent_mixture(replay,4,16,rng,'experience/round_1/',.5)
    assert sum(identity.startswith('experience/round_1/') for identity,_ in choices)==2
    assert all(1<=start<24 for _,start in choices)
    latest={'experience/round_1/c'}
    choices=sample_recent_mixture(replay,4,16,rng,fraction=.5,identities=latest)
    assert sum(identity in latest for identity,_ in choices)==2
    with pytest.raises(ValueError,match='No eligible'):
        sample_recent_mixture(replay,4,16,rng,'experience/missing/',.5)
    with pytest.raises(ValueError):sample_recent_mixture(replay,4,16,rng,'experience/round_1/',1.)


def test_one_checkpoint_handles_distinct_geometry_in_same_batch():
    state,pads,theta,g=example()
    model=SharedWorldModel(16,1,2).double()
    dt=torch.ones(2,dtype=torch.float64)/30
    together=model(state,pads,pads,theta,g,dt)
    separate=torch.cat([model(state[i:i+1],pads[i:i+1],pads[i:i+1],theta[i:i+1],
        {k:v[i:i+1] for k,v in g.items()},dt[i:i+1]) for i in range(2)])
    torch.testing.assert_close(together,separate)
    assert not any('canonical' in k or 'neighbors' in k for k in model.state_dict())


def test_contact_geometry_distinguishes_physically_opposite_pad_sides():
    state,pads,theta,g=example(1);dt=torch.tensor([1/30],dtype=torch.float64)
    swapped=pads[:,[1,0,2,3]]
    old=SharedWorldModel(16,1,2,direct_displacement=True).double()
    torch.testing.assert_close(old(state,pads,pads,theta,g,dt),old(state,swapped,swapped,theta,g,dt))
    corrected=SharedWorldModel(16,1,2,direct_displacement=True,contact_geometry=True).double()
    a=corrected(state,pads,pads,theta,g,dt);b=corrected(state,swapped,swapped,theta,g,dt)
    assert float((a[...,:6]-b[...,:6]).detach().abs().max())>1e-8


def test_contact_features_keep_metric_geometry_and_transmit_action_gradient():
    state,pads,theta,g=example(1);dt=torch.tensor([1/30],dtype=torch.float64)
    f=pad_contact_features(state,pads,pads,g['scale_m'],dt)
    other_state=state.clone();other_state[...,:6]/=2
    other_pads=pads.clone();other_pads[...,:3]/=2
    torch.testing.assert_close(f,pad_contact_features(other_state,other_pads,other_pads,g['scale_m']*2,dt))
    model=SharedWorldModel(16,1,2,direct_displacement=True,contact_geometry=True).double().requires_grad_(False)
    action=torch.full((1,2,7),.1,dtype=torch.float64,requires_grad=True)
    def predict(a):
        nxt=apply_gripper_action(pads,a,g['scale_m'])
        return model(state,pads,nxt,theta,g,dt)[...,:3]
    assert torch.autograd.gradcheck(predict,(action,),eps=1e-5,atol=1e-6,rtol=1e-3)
    gradient=torch.autograd.grad(predict(action).square().sum(),action)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>1e-10


def test_frozen_model_transmits_material_and_action_gradients():
    state,pads,theta,g=example(1)
    model=SharedWorldModel(16,1,2).double().requires_grad_(False)
    q=torch.zeros(1,4,dtype=torch.float64,requires_grad=True)
    action=torch.full((1,2,7),.1,dtype=torch.float64,requires_grad=True)
    scales=torch.tensor([3500.,.4,.8,1.],dtype=torch.float64)
    def f(q,a):
        next_pad=apply_gripper_action(pads,a,g['scale_m'])
        return model(state,pads,next_pad,physical_parameters(q,theta,scales),g,
                     torch.ones(1,dtype=torch.float64)/30)[...,:6]
    assert torch.autograd.gradcheck(f,(q,action),eps=1e-5,atol=1e-6,rtol=1e-3)
    gradients=torch.autograd.grad(f(q,action).square().sum(),(q,action))
    assert torch.all(gradients[0].abs()>1e-12)
    assert torch.isfinite(gradients[1]).all() and gradients[1].abs().sum()>1e-10
    assert all(p.grad is None for p in model.parameters())


def test_policy_receives_gradient_through_multistep_world():
    state,pads,theta,g=example(1)
    model=SharedWorldModel(16,1,2).double().requires_grad_(False)
    actor=GoalPolicy(16).double();goal=state[...,:3]+.05
    for _ in range(3):
        action=actor(state,g,goal,pads,theta)
        next_pads=apply_gripper_action(pads,action,g['scale_m'])
        state=model(state,pads,next_pads,theta,g,torch.ones(1,dtype=torch.float64)/30)
        pads=next_pads
    (state[...,:3]-goal).square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None)>0


def test_actor_and_value_do_not_read_hidden_state_or_future_interior_goal():
    state,pads,theta,g=example(1);g['surface_mask'][:,4:]=False;goal=state[...,:3]+.05
    actor=GoalPolicy(16).double();value=GoalValue(16).double()
    action=actor(state,g,goal,pads,theta);estimate=value(state,g,goal,theta,pads)
    changed=state.clone();changed[...,6:]+=100;changed[:,4:,:6]+=50
    other_goal=goal.clone();other_goal[:,4:]+=200
    torch.testing.assert_close(action,actor(changed,g,other_goal,pads,theta))
    torch.testing.assert_close(estimate,value(changed,g,other_goal,theta,pads))


def test_four_confidence_columns_remain_distinct():
    q=torch.tensor([1.,2.,3.,4.],requires_grad=True)
    losses=torch.stack([q.square().sum(),(q*torch.tensor([1.,4.,9.,16.])).sum()])
    w=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    result=confidence_gradient(losses,q,w)
    torch.testing.assert_close(result,torch.tensor([2.,4.,6.,16.]))


def test_material_acceptance_requires_strict_valid_support_improvement():
    assert select_support([3.,5.],[2.,5.])
    assert not select_support([3.,5.],[3.,5.])
    assert not select_support([3.,5.],[1.,9.])
    assert not select_support([3.,5.],[1.,None])
    assert not select_support([None,5.],[1.,2.])


def test_material_export_only_corrects_float32_clamp_rounding():
    q=bounded_export(np.array([.025,-.025,.003,0.],np.float32))
    assert q[:2]==[.025,-.025]
    assert q[2]==float(np.float32(.003))
    with pytest.raises(ValueError,match='outside'):bounded_export([.025001,0,0,0])


def test_torch_observation_objective_matches_existing_numpy_definition():
    rng=np.random.default_rng(9)
    surface=rng.normal(size=(6,7,3))*.05;target=rng.normal(size=(6,9,3))*.05
    tracks=rng.normal(size=(6,4,3))*.05;truth=tracks+rng.normal(size=(6,4,3))*.007
    visible=np.ones((6,4),bool);visible[:,0]=False
    cfg=dict(surface_weight=.5,track_weight=.5,huber_delta_mm=5.,phase_weights=dict(before=.1,interaction=.45,release=.45))
    obs=dict(surface=target,trees=[cKDTree(x) for x in target],tracks=truth,visible=visible,
             meta=dict(frames=list(range(6)),clip_start=2,clip_end=3))
    reference=score(dict(surface=surface,tracks=tracks),obs,cfg,return_curves=True)
    expected=reference['per_frame_loss_mm']
    tensors=[torch.tensor(x) for x in (surface,target,tracks,truth,visible)]
    tensors[0].requires_grad_(True)
    got=observation_loss(*tensors)
    np.testing.assert_allclose(got.detach(),expected,rtol=1e-12,atol=1e-12)
    total=phase_objective(got,torch.arange(6),2,3,cfg['phase_weights'])
    np.testing.assert_allclose(total.detach(),reference['objective_mm'],rtol=1e-12,atol=1e-12)
    got.sum().backward();assert torch.isfinite(tensors[0].grad).all()


def test_observation_distance_resolves_small_motion_at_metric_coordinates():
    surface=np.zeros((1,40,3),np.float32)
    surface[0,:,0]=.3+np.arange(40)*.01;surface[0,:,1]=.2;surface[0,:,2]=.1
    target=surface+np.array([.00001,-.00002,.00001],np.float32)
    distances=np.linalg.norm(target.astype(np.float64)-surface.astype(np.float64),axis=-1)*1000
    expected=.5*distances.mean()+.5*(.5*distances[:,:4]**2/5).mean()
    value=observation_loss(torch.tensor(surface),torch.tensor(target),torch.tensor(surface[:,:4]),
        torch.tensor(target[:,:4]),torch.ones((1,4),dtype=torch.bool))
    np.testing.assert_allclose(value.item(),expected,rtol=3e-6,atol=1e-8)


def test_sparse_observation_projection_preserves_rigid_translation_and_gradient():
    initial=np.array([[0.,0,0],[1.,0,0],[.25,.5,0],[.75,-.5,0]],dtype=np.float32)
    nodes=dict(material_indices=np.array([0,1]),dense_binding=np.array([[0,1]]*4),
               dense_weights=np.array([[1.,0.],[0.,1.],[.75,.25],[.25,.75]],np.float32))
    obs=dict(saved_ids=np.arange(4),binding=np.array([[2,3]]),weights=np.array([[.3,.7]]),
             tracks=np.array([[[.6,-.2,.4]]],np.float32))
    projection={k:torch.tensor(v) for k,v in sparse_projection(nodes,obs,initial,np.arange(4)).items()}
    delta=torch.tensor([.05,-.1,.2],requires_grad=True)
    surface,tracks=project_observations(torch.tensor(initial[:2])[None]+delta,projection)
    torch.testing.assert_close(surface[0],torch.tensor(initial)+delta)
    torch.testing.assert_close(tracks[0],torch.tensor(obs['tracks'][0])+delta)
    grad=torch.autograd.grad(surface.sum()+tracks.sum(),delta)[0]
    torch.testing.assert_close(grad,torch.full((3,),5.))


def test_action_randomization_preserves_initial_state_rigid_rotations_and_validity():
    poses=np.tile(np.eye(4),(9,1,2,1,1));poses[:,:,0,0,3]=-.02;poses[:,:,1,0,3]=.02
    valid=np.ones((9,1,2),bool);valid[8,0,1]=False;poses[8,0,1]=np.nan
    cfg=dict(action_translation_m=.01,action_aperture_m=.005,action_rotation_rad=.1)
    result=randomize_actions(poses,valid,np.arange(9)/30,1,7,123,cfg)
    np.testing.assert_equal(result[0],poses[0]);np.testing.assert_equal(result[8],poses[8])
    np.testing.assert_array_equal(np.isfinite(result).all(axis=(-1,-2)),valid)
    R=result[valid][:,:3,:3]
    np.testing.assert_allclose(R@R.transpose(0,2,1),np.broadcast_to(np.eye(3),R.shape),atol=1e-12)
    assert np.max(np.abs(result[4]-poses[4]))>1e-5
    np.testing.assert_equal(result,randomize_actions(poses,valid,np.arange(9)/30,1,7,123,cfg))


def test_object_split_keeps_all_material_action_variants_together():
    cfg={'dataset':{'validation_objects':['b'],'test_objects':['c']}}
    assert [object_split(o,cfg) for o in ['a','b','c']]==['train','validation','test']
    assert stable_seed(4,'a/0/1')==stable_seed(4,'a/0/1')
    assert stable_seed(4,'a/0/1')!=stable_seed(4,'a/0/2')


def test_collection_scheduler_does_not_claim_a_running_or_completed_trajectory(tmp_path):
    import fcntl
    key='object/episode_0/sample_0';folder=tmp_path/'trajectories'/key
    assert task_available(tmp_path,key)
    with (folder/'task.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not task_available(tmp_path,key)
        assert not (folder/'complete.json').exists()
    assert task_available(tmp_path,key)
    write_json(folder/'complete.json',{'valid':True})
    assert not task_available(tmp_path,key)


def test_checkpointed_control_preserves_action_gradient_and_initial_pad_pose():
    state,pads,theta,g=example(1);model=SharedWorldModel(16,1,2).double().requires_grad_(False)
    raw=torch.full((1,14,2),.03,dtype=torch.float64,requires_grad=True)
    actions=expanded_actions(raw,4);times=torch.arange(5,dtype=torch.float64)[None]/30
    direct,poses=control_rollout(model,state,pads,actions,theta,g,times)
    checked,_=control_rollout(model,state,pads,actions,theta,g,times,use_checkpoint=True)
    torch.testing.assert_close(checked,direct)
    grad1=torch.autograd.grad(direct.square().sum(),raw,retain_graph=True)[0]
    grad2=torch.autograd.grad(checked.square().sum(),raw)[0]
    torch.testing.assert_close(grad1,grad2);assert torch.isfinite(grad2).all() and grad2.abs().sum()>0
    torch.testing.assert_close(poses[:,0],pads)
    torch.testing.assert_close(action_pads(pads,actions,g['scale_m']),poses)


def test_control_window_does_not_cross_missing_tool_observations():
    pads=np.zeros((20,4,13));pads[:,0:2,12]=1;pads[8:10,1,12]=0
    start=choose_control_start(pads,2,18,4)
    assert start==10
    assert np.all(pads[start:start+5,:2,12]==1)


def test_control_translation_bound_applies_to_whole_horizon():
    _,pads,_,g=example(1)
    raw=torch.full((1,14,4),50.,dtype=torch.float64,requires_grad=True)
    poses,commands=bounded_pose_sequence(pads,raw,64,g['scale_m'])
    centers=poses.reshape(1,65,2,2,13)[...,:3].mean(3)
    displacement=(centers-centers[:,0:1])*g['scale_m'][:,None,None,None]
    assert float(displacement.detach().abs().max())<=.012+1e-10
    torch.testing.assert_close(poses[:,0],pads)


def test_simulator_anchor_removes_baseline_bias_without_gradient_to_measurement():
    action=torch.tensor([.2],dtype=torch.float64,requires_grad=True)
    baseline_action=torch.tensor([.2],dtype=torch.float64,requires_grad=True)
    measured=torch.tensor([3.4],dtype=torch.float64,requires_grad=True)
    corrected=anchored_positions(10+2*action,10+2*baseline_action,measured)
    torch.testing.assert_close(corrected,measured)
    gradients=torch.autograd.grad(corrected.sum(),(action,baseline_action,measured),allow_unused=True)
    torch.testing.assert_close(gradients[0],torch.tensor([2.],dtype=torch.float64))
    assert gradients[1] is None and gradients[2] is None
    changed=anchored_positions(10+2*(action+.1),10+2*baseline_action,measured)
    torch.testing.assert_close(changed,torch.tensor([3.6],dtype=torch.float64))


def test_evaluation_manifest_rejects_changed_or_missing_targets(tmp_path):
    from types import SimpleNamespace
    write_json(tmp_path/'plan.json',{'fixed':'data plan'})
    replay=SimpleNamespace(root=tmp_path,trajectories={'a':dict(state=np.zeros((4,3,24)),record={'trajectory_sha256':'old'})})
    manifest=dict(data_plan_sha256=sha256(tmp_path/'plan.json'),trajectories={'a':dict(trajectory_sha256='old',frames=4)})
    validate_manifest(replay,manifest)
    replay.trajectories['b']={'new':'data collected after evaluation freeze'}
    validate_manifest(replay,manifest)
    replay.trajectories['a']['record']['trajectory_sha256']='replacement'
    with pytest.raises(ValueError,match='Changed frozen'):validate_manifest(replay,manifest)
    del replay.trajectories['a']
    with pytest.raises(ValueError,match='Missing frozen'):validate_manifest(replay,manifest)


def test_kinematic_ablation_removes_auxiliary_feedback_but_preserves_material_gradient():
    state,pads,theta,g=example(1);theta.requires_grad_(True)
    model=SharedWorldModel(16,1,2,kinematic_only=True).double().requires_grad_(False)
    dt=torch.tensor([1/30],dtype=torch.float64)
    prediction=model(state,pads,pads,theta,g,dt)
    altered=state.clone();altered[...,6:]+=100
    other=model(altered,pads,pads,theta,g,dt)
    torch.testing.assert_close(prediction[...,:6],other[...,:6])
    torch.testing.assert_close(prediction[...,6:],state[...,6:])
    grad=torch.autograd.grad(prediction[...,:6].square().sum(),theta)[0]
    assert torch.isfinite(grad).all() and torch.all(grad.abs()>1e-12)


def test_grid_padding_preserves_cell_alignment_and_restores_domain_function():
    from softdynamics.adapters import pad_empm
    original=pad_empm.simulation_domain
    initial=np.array([[0.,0.,0.],[.04,.05,.03]])
    poses=np.tile(np.eye(4),(2,1,2,1,1));valid=np.ones((2,1,2),bool);dx=.003125
    grid,shift=original(initial,poses,valid,dx)
    with padded_simulation_grid(32):
        new_grid,new_shift=pad_empm.simulation_domain(initial,poses,valid,dx)
        assert new_grid==grid+64
        np.testing.assert_allclose((new_shift-shift)/dx,[32,32,32],atol=1e-12)
    assert pad_empm.simulation_domain is original
    with padded_simulation_grid(32,[0,0,0]):
        new_grid,new_shift=pad_empm.simulation_domain(initial,poses,valid,dx)
        assert new_grid==grid+64
        np.testing.assert_array_equal(new_shift,shift)
        np.testing.assert_array_equal((initial+new_shift).astype('f'),(initial+shift).astype('f'))
    with padded_simulation_grid(32,[32,32,64]):
        new_grid,new_shift=pad_empm.simulation_domain(initial,poses,valid,dx)
        assert new_grid==grid+64
        np.testing.assert_allclose((new_shift-shift)/dx,[32,32,64],atol=1e-12)
        np.testing.assert_allclose(new_grid*dx-new_shift[2],grid*dx-shift[2],atol=1e-12)
    with pytest.raises(ValueError,match='Lower-side'):
        with padded_simulation_grid(32,[32,32,65]):pass


def test_displacement_world_has_exact_interval_velocity_consistency():
    state,pads,theta,g=example(1);theta.requires_grad_(True)
    model=SharedWorldModel(16,1,2,direct_displacement=True).double().requires_grad_(False)
    dt=torch.tensor([.041],dtype=torch.float64)
    prediction=model(state,pads,pads,theta,g,dt)
    torch.testing.assert_close(prediction[...,:3]-state[...,:3],prediction[...,3:6]*(dt/(1/30))[:,None,None])
    gradients=torch.autograd.grad(prediction[...,:3].square().sum(),theta)[0]
    assert torch.isfinite(gradients).all() and torch.all(gradients.abs()>1e-12)


def test_observed_velocity_uses_only_past_positions_and_rejects_missing_history(tmp_path):
    def build(root,future_shift,bounded=False):
        root.mkdir();folder=root/'trajectories'/'o/episode_0/sample_0';folder.mkdir(parents=True)
        nodes=root/'nodes.npz';np.savez(nodes,canonical=np.zeros((3,3)),neighbors=np.array([[1,2],[0,2],[0,1]]),surface_mask=np.ones(3,bool))
        state=np.zeros((4,3,24),np.float32);state[:, :, 0]=np.arange(4)[:,None]*.003
        state[:,:,3:6]=999;state[2:,:,:3]+=future_shift
        np.savez(folder/'trajectory.npz',state=state,times=np.arange(4)*.04,pads=np.zeros((4,4,13)),theta=np.array([3500,.27,.8,.4]),
                 origin_m=np.zeros(3),scale_m=.2,phase=np.ones(4))
        task=dict(id='o/episode_0/sample_0',object='o',split='train',variant=0)
        write_json(root/'plan.json',dict(objects=[dict(object='o',split='train',nodes=str(nodes),nodes_sha256=sha256(nodes))],tasks=[task]))
        write_json(folder/'complete.json',dict(valid=True,full_rollout=True,trajectory_sha256=sha256(folder/'trajectory.npz'),
                   trajectory_kind='complete_bounded_policy_episode' if bounded else 'full_source_episode'))
        replay=Replay(root,['train'],observed_velocity=True);replay.refresh();return replay
    first=build(tmp_path/'a',0);changed=build(tmp_path/'b',50)
    key='o/episode_0/sample_0';a=first.trajectories[key]['state'];b=changed.trajectories[key]['state']
    np.testing.assert_array_equal(a[1,:,3:6],b[1,:,3:6])
    np.testing.assert_allclose(a[1,:,3],.003/.04*(1/30)/.2)
    bounded=build(tmp_path/'c',0,True)
    with pytest.raises(ValueError,match='past-only'):bounded.batch([(key,0)],2,'cpu')
