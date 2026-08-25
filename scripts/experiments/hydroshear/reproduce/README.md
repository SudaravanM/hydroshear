# Reproducing the drawer-pulling pipeline at the authors' scale

`training.md` gives the commands but not the scale. The scale is recoverable from the
tensorboard files shipped inside `quickstart_ckpt/`: teachers ran at **1024** environments,
students at **256**, students to ~165M agent steps. Env count is decisive: the stage-2 teacher
collapses at the repo default of 128 envs and trains to 1.00 at 1024.

```
stage1_teacher.sh              no perturbation             ~25 min on an L4, 1024 envs
stage2_teacher.sh              force perturbation           ~4 h, 70M steps, resumes stage 1
stage3_student_hydrofots.sh    HydroFots student (H0)       ~1,200 env-steps/s; 24 h cap
```

Every script sources `common.sh`, is bounded by wall clock (`WALL=`), writes to
`$RL_OUTPUT_PATH`, runs headless with wandb off, and accepts extra hydra overrides as
arguments. Checkpoints are chosen by the number in `best_sr_X.XX.pth`, not by mtime.
Requires PRs `fix/stage2-arrowgeometry` and `fix/student-use-shear-3d` (or the `++` override
in stage 3, which is harmless once the key exists).
