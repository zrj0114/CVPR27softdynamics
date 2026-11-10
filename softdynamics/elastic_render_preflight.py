"""Check the exact downstream renderer on support-frame-zero assets only."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import render_evaluation as evaluation
from .io import load_npz,read_json,sha256,write_json


def check(run,device):
    run=Path(run);spec=read_json(run/'protocol.json');physics=read_json(run/'physics/plan.json')
    source=read_json(Path(spec['fit_run'])/'plan.json')['source_data'];results={}
    for item in physics['objects']:
        obj=item['object'];episode=item['support'][0]
        root=run/'diagnostics/renderer'/obj;key='initial_support_frame';out=root/'cases'/key
        out.mkdir(parents=True,exist_ok=True)
        entry=next(e for e in item['episodes'] if e['episode']==episode)
        workspace=Path(entry['workspace']);camera_path=workspace/'queries'/f'episode_{episode}'/'cameras.npz'
        if not (out/'first_frame_check.json').exists():
            appearance,body,_,map_points=evaluation.appearance_initial(workspace,Path(physics['runtime']))
            initial=load_npz(workspace/'initial_states'/f'episode_{episode}.npz');pose=initial['world_from_canonical_rigid']
            means,G=map_points(appearance['xyz'].astype(float),pose[:3,:3],pose[:3,3],
                body['deformation_control_centers'],initial['deformation_coefficients'],
                float(initial['deformation_width_m']),gradient=True)
            rotation=Rotation.from_quat(appearance['quats'][:,[1,2,3,0]]).as_matrix()
            basis=G@(rotation*appearance['scales'][:,None,:]);cov=basis@basis.transpose(0,2,1)
            packed=np.concatenate([means,evaluation.pack_covariance(cov)],axis=1).astype('f')[None]
            np.save(out/'gaussians.npy',packed)
            write_json(out/'replay.json',dict(schema='initial-appearance-diagnostic.v1',frames=1,
                full_physics_rollout=False,gaussians_sha256=sha256(out/'gaussians.npy')))
            plan=dict(run=str(run/'physics'),source_data=source,empm='/home/rongjin/empm-study/empm',
                protocol=dict(image_metric_profile='empm-script',image_size=[1280,720]),
                cases={key:dict(object=obj,episode=episode,entry=entry,frames=[0],cameras=[0],
                    camera_path=str(camera_path),camera_sha256=sha256(camera_path))})
            write_json(root/'plan.json',plan)
            evaluation.render(root,plan,key,device,only_first=True)
        result=read_json(out/'first_frame_check.json');metrics=result['metrics']
        if not all(np.isfinite(metrics[k]) for k in ('iou_percent','psnr_db','ssim','lpips','lpips_standard')):
            raise ValueError('Invalid first-frame image metrics')
        if metrics['predicted_mask_pixels']<=0 or metrics['reference_mask_pixels']<=0:
            raise ValueError('Empty object render or ground-truth mask')
        results[obj]=dict(episode=episode,metrics=metrics,path=str(out/'first_frame_check.json'))
    write_json(run/'diagnostics/renderer/complete.json',dict(status='passed',results=results,
        scope='Frozen initial appearance, exact image-metric implementation, and camera alignment on support frames only',
        query_observations_used=False,full_physics_replay_validated=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args();check(args.run,args.device)
