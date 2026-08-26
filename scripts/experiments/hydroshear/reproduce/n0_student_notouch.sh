#!/bin/bash
# N0: the no-touch student. Same teacher, reward, algorithm, env count and steps as the
# HydroFots student (stage3_student_hydrofots.sh); only the tactile observation is removed.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NENV="${NENV:-256}"; MAX_STEPS="${MAX_STEPS:-165000000}"; WALL="${WALL:-24h}"; GATE="${GATE:-0.80}"; RUN="${RUN:-drawer_student_notouch}"
CKPT="${CKPT:-$(best_ckpt drawer_teacher_stage2)}"; [ -z "$CKPT" ] && { echo "no stage-2 checkpoint"; exit 1; }
SR="$(basename "$CKPT" | sed -E 's/best_sr_([0-9.]+)\.pth/\1/')"
python3 -c "import sys; sys.exit(0 if float('$SR') >= float('$GATE') else 1)" || { echo "GATE FAILED: teacher sr=$SR < $GATE"; exit 2; }
echo "N0: envs=$NENV max_steps=$MAX_STEPS wall=$WALL teacher=$CKPT (sr=$SR)"
timeout "$WALL" python3 scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/drawer_pulling/student_lstm_notouch task=DrawerTaskPullingStudentNoTouch \
    wandb_name="$RUN" task.env.numEnvs="$NENV" wandb_activate=False headless=True \
    task.randomize.use_force_perturb=True task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.rl.dense_gripper_post_perturb_reward_scale=1.0 \
    task.randomize.use_randomized_force_perturb=False task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 task.randomize.dof_padding_fail_preperturb=0.005 \
    task.env.friction_elastomer=1.0 \
    train.ppo.max_agent_steps="$MAX_STEPS" ckpt_path="$CKPT" "$@"
echo "N0_EXIT=$?"
