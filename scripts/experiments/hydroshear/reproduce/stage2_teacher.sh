#!/bin/bash
# Drawer teacher, stage 2 (random force perturbation), resumed from stage 1.
# Env count matters: at 128 envs success peaked 0.53 then collapsed to 0 while reward rose;
# at 1024 envs (the authors' Table VII) it reached 1.00 at 70M steps, 0.8 at 17M, 0.99 at 37M.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
NENV="${NENV:-1024}"; MAX_STEPS="${MAX_STEPS:-70000000}"; WALL="${WALL:-8h}"; RUN="${RUN:-drawer_teacher_stage2}"
CKPT="${CKPT:-$(best_ckpt drawer_teacher_stage1)}"; [ -z "$CKPT" ] && { echo "no stage-1 checkpoint"; exit 1; }
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
