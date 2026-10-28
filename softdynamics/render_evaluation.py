"""Reproducible six-metric evaluation of frozen real CMA-ES query selections.

Only appearance states are newly recorded. Every replay must match the existing
surface and track cache; the physical search and its observation split are fixed.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .adapters.pad_empm import PadRuntime
from .evaluation_metrics import geometry_metrics,mask_iou,psnr_from_mse,transport_gaussians,pack_covariance
from .io import load_npz,read_json,sha256,write_json
from .parameters import ParameterSpace
from .recovery_data import find_entry,trajectory_folder,qkey


def prepare(root,run,empm):
    root,run,empm=map(lambda p:Path(p).resolve(),[root,run,empm])
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():return read_json(root/'plan.json')
    original=read_json(run/'plan.json');rows=[];unavailable=[];cases={}
    for item in original['objects']:
        obj=item['object']
        for method in original['methods']:
            result=run/'results'/obj/'real_start0'/'96'/(method+'.json')
            if not result.exists():
                unavailable.append(dict(object=obj,method=method,reason='No completed frozen real selection at budget 96'))
                continue
            selected=read_json(result)
            for episode in item['query']:
                key=f'{obj}/episode_{episode}/{qkey(selected["selected_q"])}'
                row=dict(object=obj,episode=episode,method=method,budget=96,case=key,
                         selection=str(result),selection_sha256=sha256(result),query_valid=selected['query_valid'])
                rows.append(row)
                entry=find_entry(original,obj,episode);workspace=Path(entry['workspace'])
                camera_path=workspace/'queries'/f'episode_{episode}'/'cameras.npz'
                cameras=load_npz(camera_path)
                frames=np.load(Path(entry['input'])/'score_frames.npy');frames=frames[frames>0]
                folder=trajectory_folder(run,obj,episode,selected['selected_q'])
                cases[key]=dict(object=obj,episode=episode,q=selected['selected_q'],entry=entry,
                    trajectory=str(folder),complete_sha256=sha256(folder/'complete.json'),
                    frames=frames,cameras=[int(i) for i in cameras['validation_cameras']],
                    camera_path=str(camera_path),camera_sha256=sha256(camera_path))
    plan=dict(schema='softdynamics.six-metrics.v1',created_at=time.time(),run=str(run),
        run_plan_sha256=sha256(run/'plan.json'),empm=str(empm),rows=rows,cases=cases,unavailable=unavailable,
        protocol=dict(selection='Frozen real_start0, 96 candidates, held-out interaction; no query fitting',
            frames='Every original score frame after frame zero; frame zero is diagnostic only',
            dist='Mean GT->predicted-surface L1 nearest-neighbor distance, millimeters; original 1500-point surface sampling',
            track='Mean visible fixed-correspondence L2 error, millimeters; original 512 visual tracks and fixed material binding',
            image_size=[1280,720],cameras='Pre-existing validation_cameras indices, no outcome-based view selection; these are not novel-view claims',
            RGB='Original undistorted video RGB times refined binary object mask on black; prediction rendered on black without GT masking',
            occlusion='No separate hand/robot occlusion mask is available; occluded object pixels remain a limitation',
            IoU='Intersection/union percent, predicted alpha >= 0.5; additionally report alpha rounded to uint8 > 0',
            PSNR='10 log10(1/max(MSE,1e-10)), RGB in [0,1], full image; mean of per-image dB, capped at 100 dB for perfect images',
            SSIM='11x11 Gaussian sigma1.5, C1=.01^2 C2=.03^2, original EMPM implementation',
            LPIPS='AlexNet v0.1, pretrained official weights, standard RGB [-1,1] input; original EMPM script passes [0,1], so not numerically identical to that script',
            appearance='Frozen canonical first-frame 3DGS and eight-particle binding; means and covariance transported by relative deformation gradient; SH coefficients and opacity frozen',
            aggregation='Frames equally within camera, cameras equally within object, objects equally; failures retained explicitly',
            distinction='These are reporting metrics, not phase-weighted or confidence-weighted fitting losses'),
        source_sha256={str(Path(__file__)):sha256(__file__),
            str(Path(__file__).with_name('evaluation_metrics.py')):sha256(Path(__file__).with_name('evaluation_metrics.py'))})
    write_json(root/'plan.json',plan)
    return read_json(root/'plan.json')


def progress(root,stage,**fields):
    write_json(root/'status.json',dict(pid=os.getpid(),updated_at=time.time(),status='running',stage=stage,**fields))
    print(stage,fields,flush=True)


def geometry(root,plan,key):
    case=plan['cases'][key];out=root/'cases'/key;out.mkdir(parents=True,exist_ok=True)
    dest=out/'geometry.json'
    if dest.exists():return read_json(dest)
    folder=Path(case['trajectory']);complete=read_json(folder/'complete.json')
    assert sha256(folder/'complete.json')==case['complete_sha256']
    if not complete['valid']:
        result=dict(valid=False,reason='Invalid selected query simulation');write_json(dest,result);return result
    for name,digest in complete['payload_sha256'].items():
        assert sha256(folder/(name+'.npy'))==digest
    entry=case['entry'];prepared=read_json(entry['row']);inp=Path(entry['input'])
    assert sha256(prepared['observation'])==entry['observation_sha256']
    assert sha256(inp/'real_surface.npy')==entry['real_surface_sha256']
    observation=load_npz(prepared['observation'])
    reference=np.load(inp/'real_surface.npy',mmap_mode='r');counts=np.load(inp/'real_surface_counts.npy')
    surface=np.load(folder/'surface.npy',mmap_mode='r');tracks=np.load(folder/'tracks.npy',mmap_mode='r')
    rows=[dict(frame=f,**geometry_metrics(surface[f],reference[f,:counts[f]],tracks[f],
            observation['tracks'][f],observation['track_visible'][f])) for f in case['frames']]
    assert all(r['dist_mm'] is not None and r['track_mm'] is not None for r in rows)
    result=dict(valid=True,frames=len(rows),dist_mm=float(np.mean([r['dist_mm'] for r in rows])),
                track_mm=float(np.mean([r['track_mm'] for r in rows])),per_frame=rows)
    write_json(dest,result);return result


def appearance_initial(workspace,runtime):
    sys.path[:0]=[str(runtime),str(runtime/'gs-layer')]
    from registration import map_points
    appearance=load_npz(workspace/'asset/appearance.npz')
    body=load_npz(workspace/'asset/body.npz');binding=load_npz(workspace/'asset/appearance_binding.npz')
    return appearance,body,binding,map_points


def replay(root,plan,key,device):
    case=plan['cases'][key];out=root/'cases'/key;dest=out/'replay.json'
    out.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        result=read_json(dest)
        assert sha256(out/'gaussians.npy')==result['gaussians_sha256']
        return result
    original=read_json(Path(plan['run'])/'plan.json');entry=case['entry'];workspace=Path(entry['workspace'])
    prepared=read_json(entry['row']);obs=load_npz(prepared['observation']);T=len(obs['tracks'])
    appearance,body,binding,map_points=appearance_initial(workspace,Path(original['runtime']))
    initial=load_npz(workspace/'initial_states'/f'episode_{case["episode"]}.npz')
    pose=initial['world_from_canonical_rigid']
    means0,G0=map_points(appearance['xyz'].astype(float),pose[:3,:3],pose[:3,3],
        body['deformation_control_centers'],initial['deformation_coefficients'],float(initial['deformation_width_m']),gradient=True)
    rotation=Rotation.from_quat(appearance['quats'][:,[1,2,3,0]]).as_matrix()
    basis=G0@(rotation*appearance['scales'][:,None,:]);cov0=basis@basis.transpose(0,2,1)
    ids=binding['particle_ids'];weights=binding['weights']
    offset=means0-np.einsum('gn,gni->gi',weights,initial['positions_world_m'][ids])
    inverse=np.linalg.inv(initial['deformation_gradient'][ids].astype(float))
    nbytes=T*len(means0)*9*4
    if shutil.disk_usage(root).free<nbytes+1024**3:raise RuntimeError('Insufficient evaluation output disk space')
    packed=np.lib.format.open_memmap(out/'gaussians.npy',mode='w+',dtype=np.float32,shape=(T,len(means0),9))
    space=ParameterSpace.from_config(original['config'])
    sim=PadRuntime(case['episode'],space.parameters(case['q']),workspace,original['runtime'],entry['dtmax'],device,entry['domain_margin'])
    cached=read_json(Path(case['trajectory'])/'complete.json')
    assert sim.source_hashes==cached['source_hashes']
    assert sim.domain==cached['domain'] and sim.dtmax==cached['dtmax']
    ref_surface=np.load(Path(case['trajectory'])/'surface.npy',mmap_mode='r')
    ref_tracks=np.load(Path(case['trajectory'])/'tracks.npy',mmap_mode='r')
    surface_ids=np.load(Path(entry['input'])/'surface_ids.npy');max_error=0.;started=time.time()
    for frame in range(len(sim.times)):
        if frame:
            check=sim.advance(frame)
            if not check['valid']:raise RuntimeError(f'Replay invalid at frame {frame}: {check}')
        if frame<T:
            xyz=sim.positions(np.arange(sim.n));x=xyz[obs['saved_ids']]
            tracks=obs['tracks'][0]+np.sum((x[obs['binding']]-obs['initial'][obs['binding']])*obs['weights'][...,None],axis=1)
            err=max(float(abs(x[surface_ids]-ref_surface[frame]).max()),float(abs(tracks-ref_tracks[frame]).max()))
            max_error=max(max_error,err)
            if err>1e-6:raise RuntimeError(f'Replay differs from frozen output by {err} meters at frame {frame}')
            gradients=sim.states[sim.current].particle_F_trial.numpy().astype(float)
            means,cov=transport_gaussians(xyz,gradients,ids,weights,inverse,offset,cov0)
            if frame==0:
                np.testing.assert_allclose(means,means0,atol=1e-7,rtol=0)
                np.testing.assert_allclose(cov,cov0,atol=1e-10,rtol=1e-5)
            if not np.isfinite(means).all() or not np.isfinite(cov).all():raise ValueError('Nonfinite appearance transport')
            packed[frame]=np.concatenate([means,pack_covariance(cov)],axis=1)
        if frame%25==0:progress(root,'replay',case=key,frame=frame,total=len(sim.times)-1,seconds=time.time()-started)
    packed.flush();del packed
    result=dict(valid=True,frames=T,gaussians=len(means0),gaussians_sha256=sha256(out/'gaussians.npy'),
        max_replay_coordinate_error_m=max_error,elapsed_s=time.time()-started,
        source_hashes=sim.source_hashes,domain=sim.domain,dtmax=sim.dtmax)
    write_json(dest,result);return result


class Renderer:
    def __init__(self,appearance,K,C,width,height,device):
        import torch
        from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
        self.torch=torch;self.device=device
        def tensor(x):return torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32).contiguous()
        self.tensor=tensor;view=np.linalg.inv(C);proj=np.zeros((4,4))
        proj[0,0]=2*K[0,0]/width;proj[1,1]=2*K[1,1]/height
        proj[0,2]=2*(K[0,2]+.5)/width-1;proj[1,2]=2*(K[1,2]+.5)/height-1
        proj[2,2]=10/(10-.003);proj[2,3]=-10*.003/(10-.003);proj[3,2]=1
        settings=GaussianRasterizationSettings(image_height=height,image_width=width,
            tanfovx=width/(2*K[0,0]),tanfovy=height/(2*K[1,1]),bg=tensor([0,0,0]),
            scale_modifier=1,depth_threshold=0,viewmatrix=tensor(view.T),projmatrix=tensor((proj@view).T),
            sh_degree=int(np.sqrt(appearance['sh'].shape[1])-1),campos=tensor(C[:3,3]),
            prefiltered=False,debug=False,antialiasing=False)
        self.raster=GaussianRasterizer(settings);self.opacity=tensor(appearance['opacity'][:,None])
        self.sh=tensor(appearance['sh']);self.screen=tensor(np.zeros_like(appearance['xyz']))
        self.white=tensor(np.ones_like(appearance['xyz']))

    def render(self,packed):
        with self.torch.inference_mode():
            args=dict(means3D=self.tensor(packed[:,:3]),means2D=self.screen,
                      opacities=self.opacity,cov3D_precomp=self.tensor(packed[:,3:]))
            image=self.raster(**args,shs=self.sh)[0].clamp(0,1)
            alpha=self.raster(**args,colors_precomp=self.white)[0][0].clamp(0,1)
        return image,alpha


def camera_source(root,plan,case,index):
    import cv2,h5py
    cameras=load_npz(case['camera_path']);name=str(cameras['camera_names'][index])
    original=read_json(Path(plan['run'])/'plan.json')
    source=Path(original['source_data'])/case['object']/f'episode_{case["episode"]}'/'cameras.tar'
    out=root/'reference'/case['object']/f'episode_{case["episode"]}'/name
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'source.json'
    if not manifest.exists():
        hashes={}
        with tarfile.open(source) as archive:
            for member in ['undistorted.mp4','mask_refined.h5','aligned_timestamps.txt']:
                path=out/member;temp=path.with_suffix(path.suffix+'.tmp')
                with archive.extractfile(f'{name}/{member}') as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
                temp.replace(path);hashes[member]=sha256(path)
        actions=load_npz(Path(case['entry']['workspace'])/'queries'/f'episode_{case["episode"]}'/'actions.npz')
        timestamps=[int(line.split('_')[1]) for line in (out/'aligned_timestamps.txt').read_text().splitlines()]
        np.testing.assert_array_equal(timestamps,actions['source_timestamps_us'])
        cap=cv2.VideoCapture(str(out/'undistorted.mp4'))
        if not cap.isOpened():raise ValueError('Cannot decode source video')
        width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        with h5py.File(out/'mask_refined.h5','r') as f:maskshape=f['data'].shape
        assert maskshape==(count,height,width) and count==len(timestamps)
        assert [width,height]==plan['protocol']['image_size']
        write_json(manifest,dict(camera=name,index=index,source_archive=str(source),payload_sha256=hashes,
            width=width,height=height,frames=count,timestamps_match=True))
    return out,read_json(manifest)


def render(root,plan,key,device,only_first=False):
    import cv2,h5py,torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    sys.path.insert(0,plan['empm'])
    from third_party.gaussian_splatting.utils.loss_utils import ssim
    from third_party.gaussian_splatting.lpipsPyTorch import LPIPS
    case=plan['cases'][key];out=root/'cases'/key;record=read_json(out/'replay.json')
    assert sha256(out/'gaussians.npy')==record['gaussians_sha256']
    assert sha256(case['camera_path'])==case['camera_sha256']
    packed=np.load(out/'gaussians.npy',mmap_mode='r');cameras=load_npz(case['camera_path'])
    appearance=load_npz(Path(case['entry']['workspace'])/'asset/appearance.npz')
    metric=LPIPS('alex','0.1').to(device).eval();metric.requires_grad_(False)
    selected=set(case['frames']);chosen={0,case['frames'][len(case['frames'])//2],case['frames'][-1]}
    for index in case['cameras']:
        name=str(cameras['camera_names'][index]);dest=out/'cameras'/(name+'.json')
        if dest.exists() and not only_first:continue
        source,source_info=camera_source(root,plan,case,index)
        for name_,digest in source_info['payload_sha256'].items():
            assert sha256(source/name_)==digest
        renderer=Renderer(appearance,cameras['K'][index],cameras['camera_to_world'][index],
                          source_info['width'],source_info['height'],device)
        cap=cv2.VideoCapture(str(source/'undistorted.mp4'));rows=[];started=time.time();diagnostic=None
        try:
            with h5py.File(source/'mask_refined.h5','r') as masks,torch.inference_mode():
                for frame in range(len(packed)):
                    ok,bgr=cap.read()
                    if not ok:raise ValueError(f'Video decode failed: {name}/{frame}')
                    if frame not in selected and frame!=0:continue
                    reference_mask=np.asarray(masks['data'][frame])>0
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
                    rgb*=reference_mask[...,None]
                    target=torch.from_numpy(rgb.transpose(2,0,1).copy()).to(device)
                    prediction,alpha=renderer.render(packed[frame])
                    mse=float(torch.mean((prediction-target)**2))
                    alpha_np=alpha.cpu().numpy()
                    row=dict(frame=frame,psnr_db=psnr_from_mse(mse),
                        perfect_rgb=bool(mse==0),ssim=float(ssim(prediction[None],target[None])),
                        lpips=float(metric(prediction[None]*2-1,target[None]*2-1)),
                        iou_percent=100*mask_iou(alpha_np>=.5,reference_mask),
                        iou_nonzero_uint8_percent=100*mask_iou(np.rint(alpha_np*255)>0,reference_mask),
                        reference_mask_pixels=int(reference_mask.sum()),predicted_mask_pixels=int((alpha_np>=.5).sum()))
                    if frame==0:diagnostic=row
                    else:rows.append(row)
                    if frame in chosen:
                        preview=np.concatenate([rgb,prediction.permute(1,2,0).cpu().numpy(),
                            np.stack([reference_mask,alpha_np>=.5,np.zeros_like(reference_mask)],axis=-1).astype(float)],axis=1)
                        folder=out/'previews';folder.mkdir(exist_ok=True)
                        cv2.imwrite(str(folder/f'{name}-f{frame:04d}.png'),np.rint(preview[:,:,::-1]*255).astype(np.uint8))
                    if frame%50==0:progress(root,'render',case=key,camera=name,frame=frame,total=len(packed)-1,seconds=time.time()-started)
                    if only_first:
                        write_json(out/'first_frame_check.json',dict(camera=name,metrics=row,source=source_info))
                        return
        finally:cap.release()
        assert [r['frame'] for r in rows]==case['frames']
        metric_names=['psnr_db','ssim','lpips','iou_percent','iou_nonzero_uint8_percent']
        if any(r[k] is None or not np.isfinite(r[k]) for r in rows for k in metric_names):
            raise ValueError('Nonfinite metric requires explicit aggregation handling')
        result=dict(camera=name,index=index,frames=len(rows),source=source_info,
            mean={k:float(np.mean([r[k] for r in rows])) for k in metric_names},
            initial_frame_diagnostic=diagnostic,per_frame=rows,elapsed_s=time.time()-started)
        write_json(dest,result)
        report(root,plan)


def report(root,plan):
    rows=[]
    for item in plan['rows']:
        out=root/'cases'/item['case'];row=dict(item)
        geom=read_json(out/'geometry.json') if (out/'geometry.json').exists() else {}
        row.update({k:geom.get(k) for k in ['dist_mm','track_mm']})
        case=plan['cases'][item['case']];cameras=load_npz(case['camera_path']);available=[]
        for i in case['cameras']:
            p=out/'cameras'/(str(cameras['camera_names'][i])+'.json')
            if p.exists():available.append(read_json(p))
        row.update(completed_cameras=len(available),total_cameras=len(case['cameras']),frames=len(case['frames']))
        row['complete']=geom.get('valid',False) and len(available)==len(case['cameras'])
        for k in ['iou_percent','psnr_db','ssim','lpips']:
            row[k]=float(np.mean([r['mean'][k] for r in available])) if row['complete'] else None
        rows.append(row)
    macro={}
    for method in ['uniform','learned']:
        group=[r for r in rows if r['method']==method]
        if group and all(r['complete'] for r in group):
            macro[method]={k:float(np.mean([r[k] for r in group])) for k in ['dist_mm','track_mm','iou_percent','psnr_db','ssim','lpips']}
    result=dict(updated_at=time.time(),status='complete' if rows and all(r['complete'] for r in rows) else 'partial',
        rows=rows,object_equal_macro=macro,unavailable=plan['unavailable'],protocol=plan['protocol'])
    write_json(root/'report.json',result)
    columns=['object','episode','method','budget','dist_mm','track_mm','iou_percent','psnr_db','ssim','lpips','complete','completed_cameras','total_cameras','frames']
    with (root/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path)
    p.add_argument('--empm',type=Path,help='External EMPM checkout containing Gaussian rendering and image metrics; required for prepare')
    p.add_argument('--mode',choices=['prepare','geometry','replay','render','first-frame','report','all'],required=True)
    p.add_argument('--case');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();root=a.root.resolve()
    if a.mode=='prepare' and (a.run is None or a.empm is None):
        p.error('--mode prepare requires --run and --empm')
    plan=prepare(root,a.run,a.empm) if a.mode=='prepare' else read_json(root/'plan.json')
    if a.mode=='prepare':return
    try:
        if a.mode=='report':report(root,plan);return
        keys=[a.case] if a.case else sorted(plan['cases'])
        for key in keys:
            if a.mode in ['geometry','all']:geometry(root,plan,key)
            if a.mode in ['replay','all']:replay(root,plan,key,a.device)
            if a.mode in ['render','first-frame','all']:render(root,plan,key,a.device,a.mode=='first-frame')
            report(root,plan)
        write_json(root/'status.json',dict(pid=os.getpid(),updated_at=time.time(),status='complete',mode=a.mode))
    except Exception as error:
        write_json(root/'status.json',dict(pid=os.getpid(),updated_at=time.time(),status='failed',mode=a.mode,
            error=dict(type=type(error).__name__,message=str(error))))
        raise


if __name__=='__main__':main()
