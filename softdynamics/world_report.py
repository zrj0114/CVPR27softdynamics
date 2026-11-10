"""Standalone report for the connected fixed-asset dynamics pilot."""
import argparse
from pathlib import Path
import numpy as np
from .io import read_json


def write_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    report = read_json(output/'evaluation_test.json')
    manifest = read_json(output/'manifest.json')
    cfg = manifest['config']; handoff = manifest.get('first_contribution')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in ('persistence', 'constant_velocity', 'learned'):
        rows = [r for r in report['aggregates'] if r['method'] == method]
        axes[0].plot([r['horizon'] for r in rows], [r['mean_mm'] for r in rows], 'o-', label=method)
    axes[0].set(xlabel='Autoregressive horizon (control steps)', ylabel='Material-node error against teacher (mm)',
                title='Held-out nominal source episodes')
    axes[0].legend(); axes[0].grid(alpha=.25)
    for path in sorted(output.glob('*_nominal_test_prediction.npz')):
        with np.load(path) as data:
            error = np.linalg.norm(data['predicted'][..., :3]-data['teacher'][..., :3],axis=-1).mean(1)*1000
            axes[1].plot(data['times'],error,label=path.name.split('_nominal')[0])
    axes[1].set(xlabel='Time (s)',ylabel='Material-node error against teacher (mm)',title='Full nominal episode; initialized once')
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output/'prediction_errors.png',dpi=160); plt.close(fig)
    lines=['# Connected learned dynamics pilot', '']
    if handoff:
        lines += [f"Object: `{handoff['object']}`. First contribution: `{handoff['method']} / {handoff['context']} / {handoff['budget']}`.", '',
                  'The frozen confidence checkpoint and its four independent interaction weights produced the accepted '
                  'CMA selection. The second module inherits exactly that selection, shared asset and corrected-pad runtime.', '',
                  f"Weighted trace audit: {handoff['weighted_trace_audit']['checked_candidates']} candidates; "
                  f"maximum cost difference {handoff['weighted_trace_audit']['max_abs_cost_difference_mm']:.3g} mm.", '',
                  f"Parameters: `{handoff['parameters']}`.", '']
    else:
        lines += ['Synthetic software fixture; not evidence about a real asset.', '']
    lines += [f"Variant: `{manifest.get('variant')}`. Selected checkpoint: step {report['checkpoint_step']}; "
              f"validation {max(cfg['evaluation_horizons'])}-step mean {report['selection_validation_mm']:.4f} mm.", '',
              f"Source-episode split: `{cfg['splits']}`. Variants of one episode stay in its split. "
              'Nominal trajectories are the primary metric; translated actions are paired interventions, not independent episodes.', '',
              '**Error reference:** corresponding sampled material nodes in the fixed physics teacher. '
              'These are surrogate errors, not distances to real camera observations; they cannot be compared directly '
              'with historical real-to-sim reconstruction errors.', '',
              '| Horizon | Persistence mm | Constant velocity mm | Learned mm |',
              '|---:|---:|---:|---:|']
    def value(v): return 'unavailable' if v is None else f'{v:.4f}'
    for horizon in sorted({r['horizon'] for r in report['aggregates']}):
        rows={r['method']:r for r in report['aggregates'] if r['horizon']==horizon}
        lines.append(f"| {horizon} | "+' | '.join(value(rows[m]['mean_mm']) for m in ('persistence','constant_velocity','learned'))+' |')
    lines += ['', '| Full episode / action variant | Method | Mean mm | Final-frame mm | Nonpositive F fraction |',
              '|---|---|---:|---:|---:|']
    for r in report['full_rollouts']:
        lines.append(f"| {r['id']} | {r['method']} | {value(r.get('mean_mm'))} | {value(r.get('endpoint_mm'))} | {value(r.get('nonpositive_F_fraction'))} |")
    lines += ['', 'Paired action response uses half the difference between the +4 mm and -4 mm teacher rollouts. '
              'Only frames with mean teacher response above 0.1 mm are scored. '
              'This is a finite action intervention, not an infinitesimal derivative at contact switches.', '',
              '| Episode | Teacher response mm | Learned response error mm | Zero-response error mm | Cosine |',
              '|---|---:|---:|---:|---:|']
    for r in report.get('action_response',{}).get('rows',[]):
        if r['valid']:
            lines.append(f"| {r['episode']} | "+' | '.join(value(r[k]) for k in
                         ('teacher_response_mm','response_error_mm','zero_response_baseline_mm','cosine'))+' |')
        else:
            lines.append(f"| {r['episode']} | unavailable: {r['reason']} | | | |")
    speed=report['speed']; gradient=report['action_gradient']
    lines += ['', f"Batch-one latency: {speed['learned_device_resident_ms']:.3f} ms with device-resident tensors; "
              f"{speed['learned_roundtrip_ms']:.3f} ms including CPU/GPU transfers. Teacher: "
              f"{speed['teacher_control_interval_ms']:.3f} ms per control interval.", '', speed['caveat'], '',
              f"Network autograd finite-difference check: {gradient['valid']}. "
              'This checks software differentiation. Teacher action agreement is reported separately above; '
              'closed-loop control success is not established.', '',
              'Scope: one historically inspected object and one fitted parameter set. '
              'Training validation is held out from dynamics fitting but was available to the earlier parameter fit. '
              'The test episode remains the first contribution\'s original query. '
              'This is a development pilot, not a blind or cross-object benchmark. '
              'The short and long variants differ in integration, rollout training horizon and training budget; '
              'their comparison does not isolate one of those factors.', '',
              '![Prediction error curves](prediction_errors.png)', '']
    (output/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    write_report(p.parse_args().output)
