# Reproducing the drawer-pulling pipeline at the authors' scale

`training.md` gives the commands but not the scale. The scale is recoverable from the
tensorboard files shipped inside `quickstart_ckpt/`: teachers ran at **1024** environments,
students at **256**, students to ~165M agent steps.

```
stage1_teacher.sh              no perturbation              ~25 min on an L4
stage2_teacher.sh              force perturbation           70M steps, ~4 h, resumes stage 1
stage3_student_hydrofots.sh    HydroFots student (H0)       24 h cap
```

Every script sources `common.sh`, is bounded by wall clock (`WALL=`), writes to
`$RL_OUTPUT_PATH`, runs headless with wandb off, and accepts extra hydra overrides as
arguments. Checkpoints are chosen by the number in `best_sr_X.XX.pth`, not by mtime.
Requires PRs `fix/stage2-arrowgeometry` and `fix/student-use-shear-3d` (or the `++` override
in stage 3, which is harmless once the key exists).

## Two things that will mislead you if nobody says them

**The environment count is NOT adjustable from the command line.** Both entry points overwrite
it after hydra composes: `train_teacher.py` sets `numEnvs = 1024` unconditionally, and
`train_student_aacd.py` sets `numEnvs = 128 if use_tactile_rgb else 256`. Both lines are
upstream. The `NENV` variable in these scripts is therefore **inert**; it is passed through in
case upstream ever removes the hardcode, and the scripts print a warning if you set it to
anything else. To really change the count you must edit those two lines.

**Stage 2 has a trough near 10M steps that looks exactly like failure.** Success climbs to about
0.39 by 8M, falls to ~0.00 between 10M and 14M, and only then recovers: 0.71 at 17M, 0.91 at 25M,
0.98 at 37M, 1.00 at 70M. Two of our runs shared an identical config apart from `max_agent_steps`
(20M vs 70M) and produced numerically identical curves up to 11.3M; the short one was killed
inside the trough and looked like a collapse. **Do not stop a stage-2 teacher before ~17M steps.**

## Budgets are set by WALL, not by MAX_STEPS

Measured on an L4: teacher ~11k agent-steps/s at 1024 envs, student ~800 agent-steps/s overall at
256 envs (the tactile backend and the two convolutional preprocessors dominate). So the student's
nominal `MAX_STEPS=165000000` would need about 57 hours and cannot be reached inside the 24 h cap:
our H0 run stopped at 38M. Treat `MAX_STEPS` as a ceiling and `WALL` as the real terminator, and
read the realised step count out of the log rather than assuming the budget was spent.
