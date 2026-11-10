"""Evaluate the frozen joint checkpoint on the three declared held-out objects."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .io import read_json,write_json,sha256
from .joint_world import load_joint,ConfidenceData,confidence_evaluation
from .shared_world_training import Replay,validate


def evaluate(run,device):
    run=Path(run);p=read_json(run/'protocol.json');path=run/'training/best.pt'
    model,bundle=load_joint(path,device);tau=np.array(bundle['protocol']['tau'])
    if set(p['test_objects'])&set(bundle['training_objects']):raise ValueError('Test overlap')
    data=ConfidenceData(p['pretrain'],('test',),set(p['test_objects']))
    training=ConfidenceData(p['pretrain'],('train',),set(bundle['training_objects']))
    indices=model.target_indices
    constant={j:np.mean([r['S'][j]/(r['S'][j]+tau[j]) for r in training.rows if r['valid'][j]]) for j in indices}
    per_object={}
    all_ids=data.indices['test'].copy()
    for obj in p['test_objects']:
        data.indices['test']=np.array([i for i in all_ids if data.rows[i]['object']==obj])
        metric=confidence_evaluation(model,data,'test',device,tau)
        metric['constant_mae']=[float(np.mean([abs(constant[j]-r['S'][j]/(r['S'][j]+tau[j]))
            for r in data.rows if r['object']==obj and r['valid'][j]])) for j in indices]
        per_object[obj]=metric
    replay=Replay(p['data_root'],['test'],True);replay.refresh()
    replay.trajectories={i:t for i,t in replay.trajectories.items() if t['task']['object'] in p['test_objects']}
    prediction={}
    for horizon in [8,32]:
        choices=[]
        for obj in p['test_objects']:
            available=sorted(i for i,t in replay.trajectories.items() if t['task']['object']==obj and len(t['state'])>horizon+1)
            for identity in available[:2]:
                t=replay.trajectories[identity]
                for start in np.linspace(t['minimum_start'],len(t['state'])-horizon-1,4).astype(int):choices.append((identity,int(start)))
        prediction[str(horizon)]=validate(model.world,replay,choices,horizon,2,device)
    write_json(run/'evaluation/model.json',dict(checkpoint_sha256=sha256(path),test_objects=p['test_objects'],
        confidence=per_object,dynamics=prediction,train_only_calibration=True,
        use='Evaluation only; these results do not select the checkpoint or downstream parameters.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(2);evaluate(a.run,a.device)
