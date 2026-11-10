Selected teacher contact — 2026-11-03

User decision: 教师接触是最好的，现在统一按这个来。
Reference: B / current_teacher in the three-way contact visualization.

Active code:
  /home/guangqi/zrj/CVPR27softdynamics_teacher_contact_20261103
Selection and architecture:
  configs/selected_teacher_pipeline.json
  /home/guangqi/data/deform360_contact_current.json
Implementation and verification:
  /home/guangqi/analysis/teacher-contact-unification-20261103

Primary entrypoints in this code directory:
  Teacher labels: python -m softdynamics.elastic_data prepare / collect
  World model:   python -m softdynamics.elastic_world_train prepare / train
  E/nu search:   python -m softdynamics.elastic_world_search
Use --help for the required arguments. New teacher labels and training runs
belong in fresh output directories. Archived experiment directories and their
frozen code are historical records; this change does not rewrite them.

The selected teacher uses 64 x 25 mm finite pad faces, a two-cell slab at
dx = 3.125 mm, unilateral approaching-normal contact, and the existing static
and sliding Coulomb response (2.15827 / 0.770921). Both kernel sources remain
bitwise identical to the selected visualization. Simultaneous velocity
corrections are still averaged. That previously identified constraint issue
is deliberately not changed in this unification.

Teacher generation, dynamics, and search use the completed controller export:
  /home/guangqi/data/deform360_pose_completion_source_20261103/dataset/episodes
50 objects / 500 episodes / 13,794 completed pose slots were checked. Source
interpolation and endpoint hold remain unchanged. Observation gaps never turn
off an existing controller. Rabbit episode_5 has only G0; G1 is not invented.
No pose quality screening or quality-weighted contact gate is introduced.

The new network replaces the broad sphere action gate with a finite swept pad
mask expanded by quadratic MPM grid-interpolation support. It considers pad
translation, rotation, and linearly extrapolated particle motion. Both heads
share the encoder. The boolean mask is piecewise constant; local physical
features and material conditioning remain differentiable. This aligns the
surrogate's geometry and inputs, but does not turn the NN into an exact MPM
contact solver or guarantee nonpenetration. Contact response must be learned
from the chosen teacher. Existing weights have not been converted or retrained.

The solver domain now includes the complete recorded action timeline, including
withdrawal. Full-clip geometry can require more grid memory. A two-frame CUDA
integration passed; no new full-dataset physical rollout has been run here.

Old replay: 103 of 269 trajectories have matching selected pad controls; 166
have stale controls after pose completion and require regeneration. Training
preparation rejects this mismatch instead of replacing only the input pads
while retaining incompatible target motion. This is label/input consistency,
not pose quality screening. Data assets for all 500 episodes are not thereby
declared ready for training. No new training or batch regeneration was started.

The training schedule is unchanged in this implementation. Agree on the new
training plan separately before starting a run; the previous one-step/40k
configuration has not been launched again.
