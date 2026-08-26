#!/bin/bash
# Drawer teacher, stage 2 (random force perturbation), resumed from stage 1.
# THE TROUGH: success climbs to ~0.39 by 8M, drops to ~0.00 between 10M and 14M, then recovers:
# 0.71 at 17M, 0.91 at 25M, 0.98 at 37M, 1.00 at 70M. A run killed inside that window looks like
# a collapse and is not one. Two of our runs differed only in max_agent_steps (20M vs 70M) and
# were numerically identical up to 11.3M. Do not stop this before ~17M steps.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# NENV is inert: both entry points overwrite numEnvs after hydra composes (see README)
NENV="${NENV:-1024}"; MAX_STEPS="${MAX_STEPS:-70000000}"; WALL="${WALL:-8h}"; RUN="${RUN:-drawer_teacher_stage2}"
CKPT="${CKPT:-$(best_ckpt drawer_teacher_stage1)}"; [ -z "$CKPT" ] && { echo "no stage-1 checkpoint"; exit 1; }
warn_inert_nenv 1024
echo "stage2: envs=$NENV max_steps=$MAX_STEPS wall=$WALL from=$CKPT"
timeout "$WALL" python3 scripts/experiments/hydroshear/train_teacher.py \
    train=hydroshear/drawer_pulling/teacher_lstm task=DrawerTaskPulling \
    wandb_name="$RUN" task.env.numEnvs="$NENV" wandb_activate=False headless=True \
    task.randomize.use_force_perturb=True task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.randomize.use_randomized_force_perturb=True task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    train.ppo.max_agent_steps="$MAX_STEPS" ckpt_path="$CKPT" "$@"
echo "STAGE2_EXIT=$?"
