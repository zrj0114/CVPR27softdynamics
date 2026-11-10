"""Ordinary simulated experience in the downstream planner's action domain.

Every independent trajectory contributes next-state targets. There are no
response differences, Jacobian labels, or held-out objects in this replay.
"""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np
import torch

from .io import read_json, sha256, write_json
from .shared_world_control import PhysicalControl, bounded_pose_sequence
from .shared_world_data import stable_seed


def collect(root,obj,episode,variants=(0,1,2,3),samples=8,horizon=32,round_no=0,device='cuda:0',start_frame=None):
    root=Path(root);plan=read_json(root/'plan.json');item=next(o for o in plan['objects'] if o['object']==obj)
    if item['split']!='train':raise ValueError('Exploration replay must exclude all held-out objects')
    prefix=f'exploration_{round_no}/{obj}/episode_{episode}'
    if start_frame is not None:prefix+=f'_start_{start_frame}'
    folder=root/'experience'/prefix;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'exploration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);rows=[]
        for variant in variants:
            task=next(t for t in plan['tasks'] if t['object']==obj and t['episode']==episode and t['variant']==variant)
            pending=[i for i in range(samples) if not (folder/f'material_{variant}_sample_{i}'/'complete.json').exists()]
            if not pending:continue
            try:physical=PhysicalControl(root,obj,episode,horizon,device,'baseline',task['parameters'],start_frame)
            except (ValueError,RuntimeError) as exc:
                write_json(folder/f'material_{variant}_failure.json',dict(error=repr(exc),created_at=time.time()))
                continue
            _,pads0,_,geometry,_=physical.tensors('cpu',observed_velocity=True)
            for index in pending:
                identity=prefix+f'/material_{variant}_sample_{index}';out=root/'experience'/identity
                seed=stable_seed(plan['config']['seed'],identity);rng=np.random.default_rng(seed)
                spread=(.15,.35,.7)[index%3]
                # A stationary command is also an ordinary control trajectory.
                raw=np.zeros((1,14,4)) if index==0 else rng.normal(0,spread,(1,14,4))
                with torch.no_grad():
                    pads=bounded_pose_sequence(pads0,torch.tensor(raw,dtype=torch.float32),horizon,geometry['scale_m'])[0][0].numpy()
                states,result=physical.rollout(pads)
                physical.save(out,states,pads,result)
                result.update(schema='softdynamics.control-experience.v1',
                    task=dict(id='experience/'+identity,object=obj,episode=episode,split='train',variant=2000+variant),
                    source_data_plan_sha256=sha256(root/'plan.json'),nodes_sha256=item['nodes_sha256'],
                    source_start_frame=physical.start,source_end_frame=physical.end,
                    requested_start_frame=start_frame,
                    source_initial_state_preserved=True,original_recorded_prefix=True,
                    full_source_episode=False,trajectory_kind='complete_bounded_policy_episode',
                    full_rollout=bool(result['valid'] and len(states)==horizon+1),
                    control_policy='Independent bounded random knots around the current pad pose',
                    random_seed=seed,knot_std=0. if index==0 else spread,
                    total_translation_bound_m_per_axis=.012,total_rotation_bound_rad_per_axis=.08,total_aperture_bound_m=.006,
                    material_parameters=physical.parameters,response_labels_used=False,created_at=time.time())
                write_json(out/'complete.json',result)
                rows.append(dict(id=identity,valid=result['valid'],elapsed_s=result['elapsed_s']))
                write_json(folder/'progress.json',dict(rows=rows,updated_at=time.time()))
                print('EXPLORATION',identity,result['valid'],round(result['elapsed_s'],2),flush=True)
            del physical
        write_json(folder/'exploration_complete.json',dict(new_trajectories=rows,created_at=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--object',required=True)
    p.add_argument('--episode',type=int,required=True);p.add_argument('--variants',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--samples',type=int,default=8);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--round',type=int,default=0);p.add_argument('--device',default='cuda:0')
    p.add_argument('--start-frame',type=int);a=p.parse_args()
    collect(a.root,a.object,a.episode,a.variants,a.samples,a.horizon,a.round,a.device,a.start_frame)


if __name__=='__main__':main()
