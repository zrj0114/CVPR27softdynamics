"""Copy minimal independent action/camera inputs for the ten asset scenarios."""
import json
import shutil
import numpy as np
from build_asset import ROOT,DATA,sha,write
from paths import CODE


def main():
    (ROOT/'checks').mkdir(parents=True, exist_ok=True)
    for folder in sorted((ROOT/'prepared').glob('episode_*')):
        e = int(folder.name.split('_')[-1])
        out=ROOT/f'queries/episode_{e}';out.mkdir(parents=True,exist_ok=False)
        action=ROOT/f'prepared/episode_{e}/actions.npz'
        camera=DATA/f'episodes/episode_{e}/cameras.npz'
        shutil.copy2(action,out/'actions.npz');shutil.copy2(camera,out/'cameras.npz')
        source=DATA/f'episodes/episode_{e}/repair_evidence/manifest.json'
        metadata=json.loads(source.read_text())
        with np.load(action) as z:valid=z['valid'];poses=z['poses'];frames=z['source_frames']
        refs=[]
        for g in range(poses.shape[1]):
            candidates=np.flatnonzero(valid[:,g].all(1)&(frames>=metadata['clip_start'])&(frames<=metadata['clip_end']))
            if not len(candidates):raise ValueError(f'No valid contact reference for episode {e} gripper {g}')
            refs.append(int(candidates[0]))
        write(out/'episode.json',dict(episode=e,clip_start=metadata['clip_start'],clip_end=metadata['clip_end'],
            reference_frames=refs,full_frames=len(frames),camera_count=33,
            source_sha256=dict(actions=sha(action),cameras=sha(camera),annotation=sha(source)),
            contact_reference_rule='Original first valid pair within annotated interaction, no future object observation',
            missing_policy='Only enable a side when both consecutive action poses are valid'))
    write(ROOT/'checks/build-provenance.json',dict(sources={name:sha(CODE/name) for name in
        ['build_asset.py','registration.py','run_episode.py','physical_asset.py','build_queries.py']},
        input_source='Prepared actions copied byte-for-byte; first frame asset registration only'))


if __name__=='__main__':main()
