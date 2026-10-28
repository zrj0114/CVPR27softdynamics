# Frozen downstream selections: six reporting metrics

`softdynamics.render_evaluation` evaluates the real held-out interaction for each
available object at the same 96-candidate budget. It does not optimize parameters,
appearance, camera calibration, or evaluation masks. Missing selections remain in
the report's `unavailable` list.

Install `.[physics,deform360,report]` and provide an external EMPM checkout via
`--empm` when preparing a run. That checkout must include
`third_party/gaussian_splatting/utils/loss_utils.py` and `lpipsPyTorch`, their
dependencies and pretrained LPIPS weights. Build its compatible
`diff_gaussian_rasterization` CUDA extension separately. These upstream rendering
sources, extensions and weights are not bundled with this repository.

```bash
python -m softdynamics.render_evaluation --mode prepare \
  --root /path/to/evaluation --run /path/to/cma-run --empm /path/to/original-empm
python -m softdynamics.render_evaluation --mode all \
  --root /path/to/evaluation --device cuda:0
```

Both methods use the existing score frames, excluding the initialization frame,
and the pre-existing `validation_cameras` indices. These cameras were defined
before this evaluation, but initial asset reconstruction uses multiple views;
this evaluation does not establish novel-view generalization. The output records
the selected parameter vectors and hashes of the frozen selections.

| Metric | Definition |
|---|---|
| Dist, mm | Mean nearest-neighbor **L1** distance from observed points to predicted surface, following EMPM's `evaluate_chamfer.py`. Retains our existing 1500-point surface samples. |
| Track, mm | Mean **L2** distance for visible corresponding 3D tracks. Uses our existing 512 visual tracks and fixed material interpolation. No Huber transform. |
| IoU, % | Intersection / union of predicted alpha >= 0.5 and the refined observed object mask. Also stores the original-script-like nonzero uint8 alpha result. |
| PSNR, dB | Full-image RGB MSE in [0,1], converted to dB per frame with an MSE floor of 1e-10 (100 dB cap for exact matches). |
| SSIM | Original EMPM Gaussian-window implementation: 11x11, sigma 1.5. |
| LPIPS | Pretrained AlexNet, version 0.1, standard [-1,1] RGB normalization. |

Images retain the original 1280x720 resolution. Ground truth is decoded from the
undistorted recording and composited onto black using its refined object mask.
Predictions render onto black and are **not** multiplied by the ground-truth mask.
Separate hand/robot occlusion masks are unavailable. Full-image appearance scores
include the common black background and should be interpreted alongside IoU.
The original EMPM LPIPS caller passes [0,1] into a network expecting [-1,1]; this
implementation uses the standard normalization, so it is not an exact reproduction
of that caller. Different data, point sampling, track binding and camera/frame
splits also prevent direct comparison against paper table values.

The physical simulation is replayed with the unchanged frozen solver, initial
state, actions, time step and domain. At every recorded frame, surface positions
and tracks must agree with the existing cache within 1 micrometer. The first-frame
Gaussian appearance and existing eight-particle binding remain fixed. Gaussian
means and covariance are transported using the weighted relative material
deformation gradients. Opacity and spherical-harmonic coefficients remain fixed;
there is no appearance refinement using future observations.

Each camera saves per-frame scores and initial/middle/final preview images.
Frames are averaged equally within a camera, cameras equally within an object,
and objects equally for the final aggregate. A partial camera set is not reported
as a completed object's metric. Identical parameter vectors share exactly the same
rendered result. Metrics, state caches, raw selected camera videos and masks are
stored in the evaluation directory; original experimental outputs are untouched.

`report.json` and `metrics.csv` hold the comparison. `plan.json`, replay audits,
camera source manifests and metric implementation checks document provenance.
