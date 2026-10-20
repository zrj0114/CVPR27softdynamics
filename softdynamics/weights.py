"""Freeze per-parameter interaction weights before physical search."""
import argparse
from pathlib import Path

import numpy as np

from .inference import ConfidenceModel
from .io import load_npz, read_json, relative_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True,
                   help='Ordered rows: episode, input, optional teacher (at the same query)')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--start', type=float, nargs=4, required=True,
                   help='Normalized q coordinates used by the teacher and search')
    p.add_argument('--teacher-mask', action='store_true',
                   help='Explicit oracle reliability mask, for controlled comparisons only')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = read_json(args.manifest)
    model = ConfidenceModel(args.checkpoint, args.device)
    parameters = model.bundle['base'].numpy()+np.asarray(args.start)*model.bundle['scales'].numpy()
    prediction = np.stack([model.predict(load_npz(relative_path(args.manifest, r['input'])),
                           parameters)['confidence'] for r in rows])
    valid = np.isfinite(prediction)
    teachers = []
    have_teachers = all('teacher' in r for r in rows)
    if args.teacher_mask and not have_teachers:
        p.error('--teacher-mask requires a teacher file for every episode')
    if have_teachers:
        for row in rows:
            teacher = load_npz(relative_path(args.manifest, row['teacher']))
            np.testing.assert_allclose(teacher['q'], args.start, atol=1e-9, rtol=0)
            teachers.append(teacher)
        if args.teacher_mask:
            valid &= np.stack([r['valid'] for r in teachers])
    constant = -np.expm1(-model.bundle['heads'][0]['target_mean'].numpy())
    methods = ['uniform', 'constant', 'learned']
    values = [np.ones_like(prediction), np.broadcast_to(constant, prediction.shape).copy(), prediction]
    if have_teachers:
        if not args.teacher_mask:
            p.error('Teacher comparisons require --teacher-mask to share the same eligible episodes')
        sensitivity = np.stack([r['S'] for r in teachers])
        teacher = sensitivity/(sensitivity+model.bundle['tau'].numpy())
        shuffled = teacher.copy()
        for j in range(4):
            ids = np.flatnonzero(valid[:, j])
            permutation = np.random.default_rng(7300+j).permutation(len(ids))
            if len(ids) > 1 and np.array_equal(permutation, np.arange(len(ids))):
                permutation = np.roll(permutation, 1)
            shuffled[ids, j] = teacher[ids[permutation], j]
        methods += ['teacher', 'shuffled']
        values += [teacher, shuffled]
    confidence = np.stack([np.where(valid, value, np.nan) for value in values])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, methods=np.asarray(methods), confidence=confidence,
                        episodes=np.array([r['episode'] for r in rows]), start=args.start,
                        shared_teacher_mask=np.bool_(args.teacher_mask))


if __name__ == '__main__':
    main()
