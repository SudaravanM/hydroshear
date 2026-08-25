#!/bin/bash
# Drawer teacher, stage 1 (no force perturbation). Command from training.md; wandb off,
# headless, and bounded by WALL (wall clock, not agent steps: throughput varies 10x by stage).
# Measured: 1024 envs reach success 1.00 in ~112 epochs.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NENV="${NENV:-1024}"; WALL="${WALL:-2h}"; RUN="${RUN:-drawer_teacher_stage1}"
timeout "$WALL" python3 scripts/experiments/hydroshear/train_teacher.py \
    train=hydroshear/drawer_pulling/teacher_lstm task=DrawerTaskPulling \
    wandb_name="$RUN" task.env.numEnvs="$NENV" wandb_activate=False headless=True \
    task.randomize.use_force_perturb=False task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 "$@"
echo "STAGE1_EXIT=$?"
