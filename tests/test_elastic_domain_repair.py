import copy
from pathlib import Path

import numpy as np
import pytest

from softdynamics import elastic_domain_repair as repair
from softdynamics.io import read_json,write_json,sha256
from softdynamics.parameters import ParameterSpace
from softdynamics.physical_work import executions
from softdynamics.physics_protocol import read_protocol,cache_compatible
from test_elastic_downstream import PHYSICS


def setup_run(tmp_path):
    root=tmp_path/'run';workspace=tmp_path/'workspace';actions=workspace/'queries/episode_0/actions.npz'
    actions.parent.mkdir(parents=True);np.savez(actions,source_timestamps_us=np.arange(3)*20000)
    nodes=tmp_path/'nodes.npz';np.savez(nodes,material_indices=np.arange(2))
    data=tmp_path/'data';write_json(data/'plan.json',dict(objects=[dict(object='test',nodes=str(nodes),nodes_sha256=sha256(nodes))]))
    entry=dict(episode=0,workspace=str(workspace),frames=2,dtmax=.00005,grid_padding_cells=32)
    plan=dict(config=PHYSICS,runtime='/fixed/runtime',objects=[dict(object='test',support=[0,1],query=[2],
        episodes=[entry,dict(entry,episode=1),dict(entry,episode=2)])],conditions=['real'])
    write_json(root/'physics/plan.json',plan)
    spec=dict(physics_plan_sha256=sha256(root/'physics/plan.json'),data_root=str(data))
    write_json(root/'protocol.json',spec)
    failed_folder=root/'physics/trajectories/test/episode_0/nominal'
    failed=dict(valid=False,object='test',episode=0,check=dict(failure_kind='domain_boundary'),source_hashes=dict(actions='same',kernel='same'))
    write_json(failed_folder/'complete.json',failed)
    write_json(failed_folder/'execution.json',dict(pid=10,elapsed_s=7.))
    write_json(failed_folder/'elastic_receipt.json',dict(execution_pid=10,complete_sha256=sha256(failed_folder/'complete.json'),
        requester='initial/test',physics_plan_sha256=spec['physics_plan_sha256']))
    probes=[]
    for index,name in enumerate(['origin_preserved','symmetric_larger']):
        probe=root/'diagnostics/domain_187_episode0'/name;new_plan=copy.deepcopy(plan)
        new_plan['objects'][0]['episodes'][0]['grid_padding_cells']=32+index*32
        new_plan['objects'][0]['episodes'][0]['grid_lower_cells']=[0,0,0] if index==0 else None
        write_json(probe/'physics/plan.json',new_plan)
        write_json(probe/'protocol.json',dict(spec,physics_plan_sha256=sha256(probe/'physics/plan.json')))
        folder=probe/'physics/trajectories/test/episode_0/nominal';folder.mkdir(parents=True)
        payload={}
        for key in ('tracks','surface'):
            np.save(folder/(key+'.npy'),np.zeros((2,2,3)));payload[key]=sha256(folder/(key+'.npy'))
        record=dict(valid=True,object='test',episode=0,smoke=False,q=[0.]*4,
            parameters=ParameterSpace.from_config(PHYSICS).parameters(np.zeros(4)),source_hashes=failed['source_hashes'],
            dtmax=entry['dtmax'],domain=dict(dx_m=.003125,grid_size=208+64*index),frames=2,payload_sha256=payload)
        write_json(folder/'complete.json',record)
        np.savez(folder/'state.npz',state=np.zeros((3,2,3)),times=np.arange(3))
        write_json(folder/'state.json',dict(state_sha256=sha256(folder/'state.npz'),nodes_sha256=sha256(nodes),
            full_rollout=True,query_observations_used=False,physics_complete_sha256=sha256(folder/'complete.json'),frames=3))
        write_json(folder/'validation.json',[dict(frame=f,valid=True) for f in [1,2]])
        write_json(folder/'scores.json',dict(valid=True,conditions=dict(real=dict(objective_mm=5.-index*4))))
        write_json(folder/'execution.json',dict(pid=11+index,elapsed_s=11.+index*2))
        write_json(folder/'elastic_receipt.json',dict(execution_pid=11+index,complete_sha256=sha256(folder/'complete.json'),
            requester='domain_diagnostic/'+name,physics_plan_sha256=sha256(probe/'physics/plan.json')))
        probes.append(dict(id=name,root=str(probe)))
    write_json(root/'numerical_repair_plan.json',dict(object='test',episode=0,failed_initial=str(failed_folder/'complete.json'),probes=probes))
    return root,failed_folder,probes,spec['physics_plan_sha256']


def test_domain_adoption_preserves_physics_and_failures_and_ignores_probe_loss(tmp_path):
    root,folder,probes,old_digest=setup_run(tmp_path)
    result=repair.publish(root)
    assert result['selected_probe']=='origin_preserved'  # The other probe has lower fitting loss.
    assert result['numerical_preflight_seconds']==20.
    assert result['numerical_preflight_episode_rollouts']==2
    assert read_json(Path(result['retired_initial'])/'complete.json')['valid'] is False
    assert read_json(folder/'complete.json')['parameters']==ParameterSpace.from_config(PHYSICS).parameters(np.zeros(4))
    assert cache_compatible(root,old_digest,'test',1)
    assert not cache_compatible(root,old_digest,'test',0)
    assert repair.publish(root)==result
    attempts=executions(root)
    assert len(attempts)==3  # Adopted probe cache is a copy, not another execution.
    assert sum(a['seconds'] for a in attempts)==31.
    assert sum(not a['valid'] for a in attempts)==1
    assert sum(a['retired'] for a in attempts)==1
    archive=Path(result['original_plan']);archive.write_text(archive.read_text()+' ')
    with pytest.raises(ValueError,match='Archived physics protocol changed'):
        cache_compatible(root,old_digest,'test',1)


def test_incomplete_probe_is_never_promoted(tmp_path):
    root,folder,probes,old_digest=setup_run(tmp_path)
    probe_folder=Path(probes[0]['root'])/'physics/trajectories/test/episode_0/nominal'
    checks=read_json(probe_folder/'validation.json');write_json(probe_folder/'validation.json',checks[:-1])
    with pytest.raises(ValueError,match='validity checks'):
        repair.publish(root)
    assert sha256(root/'physics/plan.json')==old_digest
    assert read_json(folder/'complete.json')['valid'] is False
    assert not (root/'domain_repair_pending.json').exists()


def test_interrupted_publication_resumes_without_losing_original_attempt(tmp_path,monkeypatch):
    root,folder,probes,old_digest=setup_run(tmp_path);original=repair.write_json;crashed=False
    def interrupted(path,value):
        nonlocal crashed
        if Path(path)==root/'protocol.json' and not crashed:
            crashed=True
            raise SystemExit('simulated exit between physics plan and protocol publication')
        return original(path,value)
    monkeypatch.setattr(repair,'write_json',interrupted)
    with pytest.raises(SystemExit,match='simulated exit'):
        repair.publish(root)
    assert not (root/'domain_repair_complete.json').exists()
    result=repair.publish(root)
    assert read_protocol(root)[0]['physics_plan_sha256']==result['new_physics_plan_sha256']
    assert cache_compatible(root,old_digest,'test',1)
    assert len(executions(root))==3


def test_changed_physics_never_reuses_an_archived_cache(tmp_path):
    root,folder,probes,old_digest=setup_run(tmp_path);repair.publish(root)
    spec,plan=read_protocol(root);plan['config']['baseline']['E_pa']+=1.
    write_json(root/'physics/plan.json',plan);spec['physics_plan_sha256']=sha256(root/'physics/plan.json')
    write_json(root/'protocol.json',spec)
    assert not cache_compatible(root,old_digest,'test',1)
