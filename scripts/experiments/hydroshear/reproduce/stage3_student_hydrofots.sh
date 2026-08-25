#!/bin/bash
# Drawer student, stage 3, HydroFots backend (the H0 reference), distilled from stage 2.
# 256 envs as in the authors' Table VIII; their students ran to ~165M steps and reached 0.9
# at 37M (bin packing) and 85M (peg). Measured here: ~1,200 env-steps/s, so 165M is ~38 h.
# GATE: refuses to distil a teacher below GATE (a collapsed teacher makes a useless student).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NENV="${NENV:-256}"; MAX_STEPS="${MAX_STEPS:-165000000}"; WALL="${WALL:-24h}"; GATE="${GATE:-0.80}"; RUN="${RUN:-drawer_student_hydrofots}"
CKPT="${CKPT:-$(best_ckpt drawer_teacher_stage2)}"; [ -z "$CKPT" ] && { echo "no stage-2 checkpoint"; exit 1; }
SR="$(basename "$CKPT" | sed -E 's/best_sr_([0-9.]+)\.pth/\1/')"
python3 -c "import sys; sys.exit(0 if float('$SR') >= float('$GATE') else 1)" || { echo "GATE FAILED: teacher sr=$SR < $GATE"; exit 2; }
echo "stage3: envs=$NENV max_steps=$MAX_STEPS wall=$WALL teacher=$CKPT (sr=$SR)"
timeout "$WALL" python3 scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/drawer_pulling/student_lstm task=DrawerTaskPullingStudent \
    wandb_name="$RUN" task.env.numEnvs="$NENV" wandb_activate=False headless=True \
    task.randomize.use_force_perturb=True task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.rl.dense_gripper_post_perturb_reward_scale=1.0 \
    task.randomize.use_randomized_force_perturb=False task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 task.randomize.dof_padding_fail_preperturb=0.005 \
    task.env.friction_elastomer=1.0 task.sensor.hydroshear.randomize_coefficients=False \
    task.env.use_hydrosoft_model=True ++task.env.use_shear_3d=False \
    train.ppo.max_agent_steps="$MAX_STEPS" ckpt_path="$CKPT" "$@"
echo "STAGE3_EXIT=$?"
