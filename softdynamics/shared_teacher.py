"""Generate offline teacher vectors and their matching synthetic motion input."""
import argparse
import os
import time
from pathlib import Path

import numpy as np

from .adapters.pad_empm import PadRuntime
from .io import load_npz, read_json, sha256, write_json
from .parameters import ParameterSpace
from .shared_data import normalize_pads
from .teacher import generate


def run(row_path, config_path, runtime_path, truth_name, output, device='cuda:0', smoke_frames=None, domain_margin=.075):
    started = time.monotonic()
    row, config = read_json(row_path), read_json(config_path)
    config_digest = sha256(config_path)
    workspace = Path(row['workspace'])
    spec = config['teacher']
    space = ParameterSpace.from_config(config)
    truth = next(t for t in config['truths'] if t['name']==truth_name)
    candidates = [space.parameters(truth['q'])]
    for q in spec['workpoints']:
        for j in range(4):
            for sign in [-1,1]:
                v = np.array(q); v[j] += sign*max(spec['radii'])
                candidates.append(space.parameters(v))
    # One step size across truth and every perturbation; no parameter-dependent dt noise.
    dtmax = min(config['physics']['max_dt_s'], min(.35*.003125/np.sqrt(
                 p['E_pa']*(1-p['nu'])/(100.*(1+p['nu'])*(1-2*p['nu']))) for p in candidates))
    sim = PadRuntime(row['episode'], space.parameters(truth['q']), workspace, runtime_path, dtmax, device,domain_margin)
    observation = load_npz(row['observation'])
    meta = read_json(row['metadata'])
    if smoke_frames is not None:
        for frame in range(1,min(smoke_frames+1,len(sim.times))):
            check = sim.advance(frame)
            print('SMOKE', frame, check, flush=True)
            if not check['valid']:
                raise ValueError(f'Invalid smoke: {check}')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'smoke.json', dict(valid=True, frames=sim.frame, domain=sim.domain,
                   particles=sim.n, dtmax=dtmax, elapsed_s=time.monotonic()-started))
        return
    synthetic = np.empty_like(observation['tracks'])
    binding, weights = observation['binding'], observation['weights']
    initial = observation['initial']
    def save_motion(runtime):
        frame = runtime.frame
        if frame < len(synthetic):
            x = runtime.positions(observation['saved_ids'])
            synthetic[frame] = observation['tracks'][0]+np.sum((x[binding]-initial[binding])*weights[...,None],axis=1)
        if frame % 25 == 0:
            write_json(output/'status.json',dict(pid=os.getpid(),phase='nominal',frame=frame,
                       total_frames=len(runtime.times),updated_at=time.time(),elapsed_s=time.monotonic()-started))
            print('NOMINAL',row['id'],truth_name,frame,len(runtime.times),round(time.monotonic()-started,1),flush=True)
    records = generate(sim, observation, meta, space, spec, config['fitting']['phase_weights'],
                       output, nominal_observer=save_motion)
    body = load_npz(workspace/'asset/body.npz')
    state = load_npz(workspace/'initial_states'/f'episode_{row["episode"]}.npz')
    real = load_npz(row['real'])
    normalized = normalize_pads(synthetic, observation['track_visible'], observation['tracks'][0],
                   body, state, sim.poses[:len(synthetic)], sim.valid[:len(synthetic)],
                   sim.times[:len(synthetic)], observation['point_ids'],real['origin_m'],float(real['scale_m']))
    np.savez_compressed(output/'input.npz', **normalized)
    manifest = dict(id=row['id'],object=row['object'],episode=row['episode'],split=row['split'],
                    truth=truth_name, truth_q=truth['q'], status='complete', records=records,
                    input=str(output/'input.npz'), real=row['real'], source_hashes=sim.source_hashes,
                    config_sha256=config_digest, observation_sha256=sha256(row['observation']),
                    domain=sim.domain, particles=sim.n, dtmax=dtmax,
                    elapsed_s=time.monotonic()-started, action_dim=52,
                    synthetic_motion_matches_truth=True, tactile_input=False,
                    derivative='Full-state-conditioned one-observation-interval partial derivative; not full-trajectory total derivative')
    write_json(output/'manifest.json',manifest)
    print('TEACHER_COMPLETE',row['id'],truth_name,round(manifest['elapsed_s'],1),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--row', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[1]/'runtime')
    p.add_argument('--truth', default='center')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke-frames', type=int)
    p.add_argument('--domain-margin', type=float, default=.075)
    a = p.parse_args()
    try:
        run(a.row,a.config,a.runtime,a.truth,a.output,a.device,a.smoke_frames,a.domain_margin)
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        message=f'{type(exc).__name__}: {exc}'
        numerical=any(s in message for s in ['Invalid nominal','outside allocated grid','Branch state contamination','Invalid smoke'])
        boundary='domain_boundary' in message or 'outside allocated grid' in message
        write_json(a.output/'failure.json',dict(status='failed',error=message,
                   classification='domain_boundary' if boundary else ('numerical' if numerical else 'infrastructure_or_contract'),pid=os.getpid(),updated_at=time.time(),domain_margin_m=a.domain_margin))
        raise


if __name__=='__main__':
    main()
