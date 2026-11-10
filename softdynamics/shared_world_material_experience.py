"""Ordinary state trajectories with independently sampled simulation materials.

Each rollout starts from the same original physical snapshot and uses the same
recorded controls. Only absolute next-state trajectories enter replay; no
response differences, derivative labels, or held-out objects are used.
"""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from .io import read_json,sha256,write_json
from .parameters import PARAMETERS
from .shared_world_control import PhysicalControl
from .shared_world_data import stable_seed


def collect(root,obj,episode,samples=12,horizon=32,round_no=0,start_frame=None,device='cuda:0'):
    root=Path(root);plan=read_json(root/'plan.json');item=next(o for o in plan['objects'] if o['object']==obj)
    if item['split']!='train':raise ValueError('Material experience must exclude held-out objects')
    if samples<2:raise ValueError('Need the baseline and independent material samples')
    prefix=f'material_{round_no}/{obj}/episode_{episode}'
    if start_frame is not None:prefix+=f'_start_{start_frame}'
    folder=root/'experience'/prefix;folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(source_data_plan_sha256=sha256(root/'plan.json'),object=obj,episode=episode,
        samples=samples,horizon=horizon,start_frame=start_frame,round=round_no,
        material_lower=plan['config']['material_lower'],material_upper=plan['config']['material_upper'])
    with (folder/'collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'protocol.json').exists():
            if read_json(folder/'protocol.json')!=protocol:raise ValueError('Material collection protocol changed')
        else:write_json(folder/'protocol.json',protocol)
        pending=[i for i in range(samples) if not (folder/f'sample_{i:03d}'/'complete.json').exists()]
        if pending:
            torch.set_num_threads(2)
            physical=PhysicalControl(root,obj,episode,horizon,device,'baseline',start_frame=start_frame)
            prefix_parameters=dict(physical.parameters);initial=physical.initial.copy()
            commands=physical.pads[physical.start:physical.end+1].copy()
            for index in pending:
                identity=prefix+f'/sample_{index:03d}';output=root/'experience'/identity
                seed=stable_seed(plan['config']['seed'],identity)
                theta=np.array([prefix_parameters[k] for k in PARAMETERS]) if index==0 else np.random.default_rng(seed).uniform(
                    protocol['material_lower'],protocol['material_upper'])
                physical.parameters={**prefix_parameters,**dict(zip(PARAMETERS,theta.tolist()))}
                physical.theta=theta.astype(np.float32)
                states,result=physical.rollout(commands)
                np.testing.assert_array_equal(states[0],initial)
                physical.save(output,states,commands,result)
                result.update(schema='softdynamics.material-experience.v1',
                    task=dict(id='experience/'+identity,object=obj,episode=episode,split='train',variant=0 if index==0 else 3000+index),
                    source_data_plan_sha256=protocol['source_data_plan_sha256'],nodes_sha256=item['nodes_sha256'],
                    source_start_frame=physical.start,source_end_frame=physical.end,
                    original_recorded_prefix=True,prefix_parameters=prefix_parameters,
                    material_parameters=physical.parameters,material_sample_index=index,random_seed=seed,
                    identical_initial_physical_state=True,identical_recorded_controls=True,
                    full_source_episode=False,trajectory_kind='complete_material_randomized_episode',
                    full_rollout=bool(result['valid'] and len(states)==horizon+1),
                    policy_training_eligible=False,response_labels_used=False,
                    world_model_targets='Absolute actual simulation state at every step',created_at=time.time())
                write_json(output/'complete.json',result)
                write_json(folder/'progress.json',dict(last_sample=index,samples=samples,valid=result['valid'],updated_at=time.time()))
                print('MATERIAL_EXPERIENCE',identity,result['valid'],flush=True)
        records=[read_json(folder/f'sample_{i:03d}'/'complete.json') for i in range(samples)]
        write_json(folder/'collection_complete.json',dict(samples=samples,valid=sum(r['valid'] for r in records),
            completed_at=time.time(),training_targets='Ordinary state sequences only'))
        return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--object',required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--samples',type=int,default=12);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--round',type=int,default=0);p.add_argument('--start-frame',type=int)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    collect(a.root,a.object,a.episode,a.samples,a.horizon,a.round,a.start_frame,a.device)


if __name__=='__main__':main()
