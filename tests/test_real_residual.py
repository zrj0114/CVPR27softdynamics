import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from test_coupled_world import fixture
from softdynamics.coupled_world_model import forecast, load_coupled
from softdynamics.real_residual_data import prepare, RealReplay, validate_packet
from softdynamics.real_residual_objective import (
    causal_batch, real_rollout, observation_objective, residual_regularizers)
from softdynamics.real_residual_train import initialize, fit_object, prepare_training, train
from softdynamics.io import read_json, write_json


ARCHITECTURE = dict(hidden=16,layers=1,heads=2,kinematic_only=True,
    direct_displacement=True,contact_geometry=True,material_dim=2)
RESIDUAL = dict(hidden=16,max_velocity_m_s=.02)


def setup_case(tmp_path):
    base,context,*_ = fixture()
    initial = context['states'][0].numpy().copy()
    n,t,scale = len(initial),8,np.float32(.25)
    x0 = initial[:,:3]*scale
    times = np.arange(t,dtype='f')/30
    tracks = x0[None]+times[:,None,None]*np.array([.012,-.003,.001],'f')
    binding = np.arange(n)[:,None]
    data = dict(state0=initial,x0=x0,origin=np.zeros(3,'f'),scale=scale,
        canonical=context['geometry']['canonical'][0].numpy(),
        neighbors=context['geometry']['neighbors'][0].numpy(),surface_mask=np.ones(n,bool),
        pads=np.repeat(context['pads0'][0].numpy()[None],t,axis=0),times=times,
        P=np.eye(n,dtype='f'),track0=x0.copy(),tracks=tracks.astype('f'),visible=np.ones((t,n),bool),
        surface0=x0.copy(),surface_binding=binding,surface_weights=np.ones((n,1),'f'),
        reference=tracks.astype('f'),reference_valid=np.ones((t,n),bool),
        phase=np.array([0,0,1,1,1,2,2,2]))
    validate_packet(data)
    bundle = dict(schema='softdynamics.coupled-temporal-checkpoint.v1',model=base.state_dict(),
        architecture=ARCHITECTURE,tau=[.1,.2],step=16500,statistics={},
        material_coordinates=dict(reference_E_pa=3500.,reference_nu=.27,nu_scale=.2),
        protocol=dict(training_objects=['train-object'],validation_objects=['val-object'],
                      test_objects_excluded=['held-out']))
    path = tmp_path/'base-source.pt'; torch.save(bundle,path)
    model,_ = initialize(path,RESIDUAL)
    return base,model,context,data,path


def test_zero_initialization_preserves_old_rollout_and_sensitivity(tmp_path):
    base,model,context,data,path = setup_case(tmp_path)
    q = torch.zeros(2)
    batch,geometry = causal_batch(data,model.theta(q),'cpu')
    expected = forecast(base.world,batch,geometry,gradients=False)
    actual = forecast(model.world,batch,geometry,gradients=False)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for before,after in zip(base.confidence([context],q[None]),model.confidence([context],q[None])):
        torch.testing.assert_close(before,after,atol=0,rtol=0)
    for name,value in base.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0)
    assert all(p.requires_grad == ('velocity_residual' in name) for name,p in model.named_parameters())
    loaded,_ = load_coupled(path)
    assert loaded.residual_config is None


@pytest.mark.parametrize('scale,dt',[(.1,1/30),(.4,1/60)])
def test_correction_uses_meters_per_second_and_updates_velocity_channel(tmp_path,scale,dt):
    _,model,_,data,_ = setup_case(tmp_path)
    data['scale'] = np.float32(scale)
    batch,geo = causal_batch(data,model.theta(torch.zeros(2)),'cpu')
    args = (batch['states'][:,0],batch['pads'][:,0],batch['pads'][:,1],batch['theta'],geo,
            torch.tensor([dt]),None)
    intended = torch.tensor([.01,-.005,.0025])
    with torch.no_grad():
        model.world.velocity_residual[-1][-1].bias.copy_(torch.atanh(intended/.02))
        uncorrected,_ = model.world.base_step(*args)
        corrected,_,dv = model.world.step_with_correction(*args)
    torch.testing.assert_close(dv,intended.expand_as(dv),atol=1e-8,rtol=1e-6)
    torch.testing.assert_close((corrected[...,:3]-uncorrected[...,:3])*scale,
        (intended*dt).expand_as(dv),atol=1e-8,rtol=1e-4)
    torch.testing.assert_close((corrected[...,3:6]-uncorrected[...,3:6])*scale/(1/30),
        intended.expand_as(dv),atol=1e-8,rtol=1e-5)


def test_corrected_state_reenters_base_and_future_gt_is_not_an_input(tmp_path,monkeypatch):
    _,model,_,data,_ = setup_case(tmp_path)
    with torch.no_grad():
        model.world.velocity_residual[-1][-1].bias.fill_(.5)
    seen = []
    encode = model.world.encode
    def remember_input(state,*args,**kwargs):
        seen.append(state.detach().clone())
        return encode(state,*args,**kwargs)
    monkeypatch.setattr(model.world,'encode',remember_input)
    theta = model.theta(torch.zeros(2))
    positions,_,_ = real_rollout(model,data,theta,'cpu')
    torch.testing.assert_close(seen[1][0,:,:3]*float(data['scale']),positions[1],atol=0,rtol=0)
    corrupt = copy.deepcopy(data)
    corrupt['tracks'][1:] = 1e6; corrupt['reference'][1:] = -1e6
    corrupt['visible'][1:] = False
    other,_,_ = real_rollout(model,corrupt,theta,'cpu')
    torch.testing.assert_close(positions,other,atol=0,rtol=0)


def test_teacher_losses_and_secants_ignore_real_correction(tmp_path):
    base,model,c,data,_ = setup_case(tmp_path)
    batch,geo = causal_batch(data,model.theta(torch.zeros(2)),'cpu')
    batch['states'] = batch['states'].expand(-1,len(data['times']),-1,-1).clone()
    aux = dict(contexts=[c],q=torch.zeros(1,2),valid=torch.ones(1,2,dtype=torch.bool),
        score=torch.full((1,2),.2),rho=torch.zeros(1),correlation_valid=torch.ones(1,dtype=torch.bool),
        radii=torch.full((1,2),.025),J=[torch.zeros(int(c['target_visible'].sum())*3,2)])
    pb = {k:({j:v.expand(2,*v.shape[1:]) for j,v in val.items()} if isinstance(val,dict)
             else val.expand(2,*val.shape[1:])) for k,val in batch.items()}
    pb['theta'] = model.theta(torch.tensor([[0.,0.],[.2,.1]]))
    pb['parameter_distance'] = torch.tensor(.2236)
    pg = {k:v.expand(2,*v.shape[1:]) for k,v in geo.items()}
    base.eval(); model.eval()
    with torch.no_grad():
        model.world.velocity_residual[-1][-1].bias.fill_(3.)
        expected = base(batch,geo,aux,pair=(pb,pg),local_response=True)
        actual = model(batch,geo,aux,pair=(pb,pg),local_response=True)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert len(actual)==8


def test_hidden_nans_do_not_poison_gt_loss_or_gradient(tmp_path):
    _,_,_,data,_ = setup_case(tmp_path)
    data['visible'][2:,0] = False; data['tracks'][2:,0] = np.nan
    data['reference_valid'][1:,1] = False; data['reference'][1:,1] = np.nan
    data['visible'][-1] = False; data['reference_valid'][-1] = False
    validate_packet(data)
    x = torch.tensor(data['x0'][None].repeat(len(data['times']),axis=0),requires_grad=True)
    loss,metrics = observation_objective(x,data)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert metrics['scored_frames']==len(data['times'])-2
    assert x.grad[-1].abs().sum()==0


def test_residual_only_learning_and_material_derivatives(tmp_path):
    base,model,c,data,_ = setup_case(tmp_path)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.world.velocity_residual.parameters(),lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        x,dv,first = real_rollout(model,data,model.theta(torch.zeros(2)),'cpu',gradients=True,gradient_tail=4)
        loss,_ = observation_objective(x,data,first)
        amp,smooth = residual_regularizers(dv,data['neighbors'],.02,first-1)
        (loss+.1*amp+.05*smooth).backward()
        assert all(p.grad is None for name,p in model.named_parameters() if 'velocity_residual' not in name)
        optimizer.step()
    changed = [k for k,v in model.state_dict().items() if not torch.equal(before[k],v)]
    assert changed and all('velocity_residual' in k for k in changed)
    for before_score,after_score in zip(base.confidence([c],torch.zeros(1,2)),model.confidence([c],torch.zeros(1,2))):
        torch.testing.assert_close(before_score,after_score,atol=0,rtol=0)
    q = torch.zeros(2,requires_grad=True)
    x,_,_ = real_rollout(model,data,model.theta(q),'cpu',gradients=True)
    derivative = torch.autograd.grad(x[-1].square().sum(),q)[0]
    assert torch.isfinite(derivative).all() and (derivative.abs()>0).all()


def make_dataset(tmp_path):
    _,_,_,data,path = setup_case(tmp_path)
    packet = tmp_path/'observed.npz'; np.savez_compressed(packet,**data)
    index = tmp_path/'observations.json'
    write_json(index,dict(episodes=[dict(object=obj,episode=ep,data=str(packet))
        for obj in ('train-object','val-object') for ep in (0,1)]))
    root = tmp_path/'data'
    prepare(index,path,root,fit_episodes=[0],query_episodes=[1])
    return root


def test_dataset_freezes_arrays_and_rejects_heldout_objects(tmp_path):
    root = make_dataset(tmp_path)
    replay = RealReplay(root)
    assert len(replay.records(split='train',role='fit'))==1
    assert len(replay.records(split='validation',role='query'))==1
    index = read_json(tmp_path/'observations.json'); index['episodes'][0]['object']='held-out'
    write_json(tmp_path/'bad.json',index)
    with pytest.raises(ValueError,match='outside frozen'):
        prepare(tmp_path/'bad.json',tmp_path/'base-source.pt',tmp_path/'bad-data',[0],[1])
    packet = root/replay.records()[0]['path']
    packet.write_bytes(packet.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Changed real GT'):
        RealReplay(root)


def test_cma_never_accesses_query_and_keeps_weights_fixed(tmp_path):
    root = make_dataset(tmp_path); replay = RealReplay(root)
    model,_ = initialize(root/'base.pt',RESIDUAL)
    original = replay.data
    class SupportOnly(dict):
        def __getitem__(self,key):
            assert key[1]==0, 'query GT accessed during fitting'
            return original[key]
    replay.data = SupportOnly()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    result = fit_object(model,replay,'train-object','cpu',generations=1,population=4,surface_points=12)
    assert result['support_episodes']==[0] and not result['query_used']
    assert len(result['history'])==5 and result['objective_mm']<=result['initial_objective_mm']
    assert 1000<=result['E_pa']<=12000 and .08<=result['nu']<=.42
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_cma_rejects_candidate_if_any_support_rollout_fails(tmp_path,monkeypatch):
    import softdynamics.real_residual_train as trainer
    root = make_dataset(tmp_path); replay = RealReplay(root)
    model,_ = initialize(root/'base.pt',RESIDUAL)
    def fail(*args,**kwargs):
        raise FloatingPointError('invalid trajectory')
    monkeypatch.setattr(trainer,'evaluate_episode',fail)
    with pytest.raises(FloatingPointError,match='No valid material'):
        fit_object(model,replay,'train-object','cpu',generations=1,population=4)


def test_resume_at_material_refit_boundary_is_exact(tmp_path,monkeypatch):
    import softdynamics.real_residual_train as trainer
    root = make_dataset(tmp_path)
    # CMA itself is exercised above. A deterministic cheap fitter isolates
    # checkpoint/RNG/refit scheduling across an optimizer-step boundary here.
    def cheap_fit(model,replay,obj,device,generations,population,seed,surface_points,initial_q):
        q = np.zeros(2) if initial_q is None else np.asarray(initial_q)+.01
        return dict(object=obj,q=q.tolist(),support_episodes=[0],query_used=False)
    monkeypatch.setattr(trainer,'fit_object',cheap_fit)
    for name in ('continuous','interrupted'):
        out = tmp_path/name
        cfg = prepare_training(root,out,steps=2,hidden=16)
        cfg.update(material_refit_interval=1,validation_interval=1,save_interval=1,surface_points=12)
        write_json(out/'config.json',cfg)
    train(tmp_path/'continuous',device='cpu')
    original_save = trainer.atomic_checkpoint
    def pause_after_step(path,payload):
        original_save(path,payload)
        if Path(path).name=='latest.pt' and payload['step']==1:
            write_json(tmp_path/'interrupted'/'pause.request.json',dict(reason='resume test'))
    monkeypatch.setattr(trainer,'atomic_checkpoint',pause_after_step)
    train(tmp_path/'interrupted',device='cpu')
    assert read_json(tmp_path/'interrupted/training/status.json')['status']=='paused'
    monkeypatch.setattr(trainer,'atomic_checkpoint',original_save)
    train(tmp_path/'interrupted',resume=True,device='cpu')
    full = torch.load(tmp_path/'continuous/training/latest.pt',weights_only=False)
    resumed = torch.load(tmp_path/'interrupted/training/latest.pt',weights_only=False)
    assert full['step']==resumed['step']==2
    for k,v in full['model'].items():
        torch.testing.assert_close(v,resumed['model'][k],atol=0,rtol=0)
    assert full['materials']==resumed['materials']
    model,_ = load_coupled(tmp_path/'interrupted/training/latest.pt')
    assert model.residual_config['hidden']==16
