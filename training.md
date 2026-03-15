# Training Commands

## Peg Insertion Training

### Stage 1: Teacher training without contact penalty
```bash
python scripts/experiments/hydroshear/train_teacher.py train=hydroshear/peg_insertion/teacher_lstm task=TacSLTaskInsertion wandb_name=peg_insertion_teacher_stage_1 rl.contact_penalty_scale=0.0 wandb_activate=True
```

### Stage 2: Teacher training with contact penalty
```bash
python scripts/experiments/hydroshear/train_teacher.py train=hydroshear/peg_insertion/teacher_lstm task=TacSLTaskInsertion wandb_name=peg_insertion_teacher_stage_2 wandb_activate=True ckpt=/path/to/ckpt_stage1.pt
```

### Stage 3: Student training with AACD
```bash
# TacSL Grayscale student 
# ! Set use_tactile_rgb = True and use_tactile_shear = False in the script
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/peg_insertion/student_lstm_w_tactile_gray_only task=TacSLTaskInsertionGrayOnly wandb_name=peg_insertion_student_tacsl_grayscale wandb_activate=True ckpt=/path/to/ckpt_stage2.pt
```

```bash
# TacSL Shear student
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/peg_insertion/student_lstm task=TacSLTaskInsertion wandb_name=peg_insertion_student_tacsl_shear wandb_activate=True task.env.use_tacsl_shear=True ckpt=/path/to/ckpt_stage2.pt
```

```bash
# FOTS Shear student
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/peg_insertion/student_lstm task=TacSLTaskInsertion wandb_name=peg_insertion_student_fots_shear wandb_activate=True task.env.use_fots_model=True ckpt=/path/to/ckpt_stage2.pt
```

```bash
# HydroShear student
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/peg_insertion/student_lstm task=TacSLTaskInsertion wandb_name=peg_insertion_student_hydroshear wandb_activate=True task.env.use_hydrosoft_model=True ckpt=/path/to/ckpt_stage2.pt
```

## Bin Packing Training

### Stage 1: Teacher training without contact penalty
```bash
python scripts/experiments/hydroshear/train_teacher.py task=BinTaskPackingSquishGoalRandomizationNoContactPenaltyNoPackedReward train=hydroshear/bin_packing/teacher_lstm.yaml wandb_name=bin_packing_teacher_stage1 wandb_activate=True headless=True
```

### Stage 2: Teacher training with contact penalty
```bash
python scripts/experiments/hydroshear/train_teacher.py task=BinTaskPackingSquishGoalRandomizationWithContactPenaltyNoPackedReward train=hydroshear/bin_packing/teacher_lstm.yaml wandb_name=bin_packing_teacher_stage2 wandb_activate=True headless=True ckpt_path=/path/to/ckpt_stage1.pt
```

### Stage 3: Student training with AACD

```bash
# TacSL Grayscale student
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/bin_packing/student_lstm_w_tactile_gray_only task=BinTaskPackingSquishGoalRandomizationWithContactPenaltyNoPackedRewardStudentTactileGrayOnly wandb_name=bin_packing_student_tacsl_grayscale wandb_activate=True ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# TacSL Shear student
# Set use_tactile_rgb = False and use_tactile_shear = True in the script
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/bin_packing/student_lstm task=BinTaskPackingSquishGoalRandomizationWithContactPenaltyNoPackedRewardStudent wandb_name=bin_packing_student_tacsl_shear wandb_activate=True ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# FOTS Shear student
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/bin_packing/student_lstm task=BinTaskPackingSquishGoalRandomizationWithContactPenaltyNoPackedRewardStudent wandb_name=bin_packing_student_fots_shear wandb_activate=True task.env.use_fots_model=True ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# HydroShear student
python scripts/experiments/hydroshear/train_student_aacd.py train=hydroshear/bin_packing/student_lstm task=BinTaskPackingSquishGoalRandomizationWithContactPenaltyNoPackedRewardStudent wandb_name=bin_packing_student_hydroshear wandb_activate=True ckpt_path=/path/to/ckpt_stage2.pt
```

## Book Shelving Training

### Stage 1: Teacher training without contact penalty
```bash
python scripts/experiments/hydroshear/train_teacher.py \
    train=hydroshear/book_shelving/teacher_lstm \
    task=BookTaskShelvingSquishRandomization \
    wandb_name=book_shelving_teacher_stage1 \
    wandb_activate=True \
    headless=True
```

### Stage 2: Teacher training with contact penalty
```bash
python scripts/experiments/hydroshear/train_teacher.py \
    train=hydroshear/book_shelving/teacher_lstm \
    task=BookTaskShelvingSquishStage2 \
    wandb_name=book_shelving_teacher_stage2 \
    wandb_activate=True \
    headless=True \
    ckpt_path=/path/to/ckpt_stage1.pt
```

### Stage 3: Student training with AACD

```bash
# TacSL Grayscale Student Curriculum Part 1
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm_w_tactile_gray_only \
    task=BookTaskShelvingSquishStudentTactileGrayOnly \
    wandb_name=book_shelving_student_tacsl_grayscale_curriculum_p1 \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[0.0,0.01] \
    task.rl.close_error_thresh=0.02 \
    wandb_activate=True \
    headless=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# TacSL Grayscale Student Curriculum Part 2
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm_w_tactile_gray_only \
    task=BookTaskShelvingSquishStudentTactileGrayOnly \
    wandb_name=book_shelving_student_tacsl_grayscale_curriculum_p2 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[-0.005,0.0] \
    task.rl.close_error_thresh=0.02 \
    ckpt_path=/path/to/ckpt_stage3.pt
```

```bash
# TacSL Shear Student Curriculum Part 1
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm \
    task=BookTaskShelvingSquishStudent \
    wandb_name=book_shelving_student_tacsl_shear_curriculum_p1 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[0.0,0.01] \
    task.rl.close_error_thresh=0.02 \
    task.env.use_tacsl_shear=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# TacSL Shear Student Curriculum Part 2
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm \
    task=BookTaskShelvingSquishStudent \
    wandb_name=book_shelving_student_tacsl_shear_curriculum_p2 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[-0.005,0.0] \
    task.rl.close_error_thresh=0.02 \
    task.env.use_tacsl_shear=True \
    ckpt_path=/path/to/ckpt_stage3.pt
```

```bash
# FOTS Student Curriculum Part 1
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm \
    task=BookTaskShelvingSquishStudent \
    wandb_name=book_shelving_student_fots_shear_curriculum_p1 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[0.0,0.01] \
    task.rl.close_error_thresh=0.02 \
    task.env.use_fots_model=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# FOTS Student Curriculum Part 2
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm \
    task=BookTaskShelvingSquishStudent \
    wandb_name=book_shelving_student_fots_shear_curriculum_p2 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[-0.005,0.0] \
    task.rl.close_error_thresh=0.02 \
    task.env.use_fots_model=True \
    ckpt_path=/path/to/ckpt_stage3.pt
```

```bash
# HydroShear Student Curriculum Part 1
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm \
    task=BookTaskShelvingSquishStudent \
    wandb_name=book_shelving_student_hydroshear_curruculum_p1 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[0.0,0.01] \
    task.rl.close_error_thresh=0.02 \
    task.sensor.hydroshear.randomize_coefficients=False \
    task.env.use_hydrosoft_model=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# HydroShear Student Curriculum Part 2
python scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/book_shelving/student_lstm \
    task=BookTaskShelvingSquishStudent \
    wandb_name=book_shelving_student_hydroshear_curruculum_p2 \
    wandb_activate=True \
    headless=True \
    task.rl.contact_penalty_scale=0.001 \
    task.randomize.preset_object_squish_noise_range=[-0.005,0.0] \
    task.rl.close_error_thresh=0.02 \
    task.sensor.hydroshear.randomize_coefficients=False \
    task.env.use_hydrosoft_model=True \
    ckpt_path=/path/to/ckpt_stage3.pt
```

### Drawer Pulling Training


### Stage 1: Teacher training without force perturbance
```bash
python3 scripts/experiments/hydroshear/train_teacher.py \
    train=hydroshear/drawer_pulling/teacher_lstm \
    task=DrawerTaskPulling \
    wandb_name=drawer_pulling_teacher_stage1 \
    task.randomize.use_force_perturb=False \
    task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 \
    task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    wandb_activate=True headless=True
```

### Stage 2: Teacher training with force perturbance
```bash
python3 scripts/experiments/hydroshear/train_teacher.py \
    train=hydroshear/drawer_pulling/teacher_lstm \
    task=DrawerTaskPulling \
    wandb_name=drawer_pulling_teacher_stage2 \
    wandb_activate=True headless=True \
    task.randomize.use_force_perturb=True \
    task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.randomize.use_randomized_force_perturb=True \
    task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] \
    task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 \
    task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    ckpt_path=/path/to/ckpt_stage1.pt
```

### Stage 3: Student training with AACD

```bash
# TacSL Grayscale Student
python3 scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/drawer_pulling/student_lstm_w_tactile_gray_only \
    task=DrawerTaskPullingStudentTactileGrayOnly \ wandb_name=drawer_pulling_student_tacsl_grayscale \
    wandb_activate=True headless=True \
    task.randomize.use_force_perturb=True \
    task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.rl.dense_gripper_post_perturb_reward_scale=1.0 \
    task.randomize.use_randomized_force_perturb=False \
    task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] \
    task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 \
    task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    task.randomize.dof_padding_fail_preperturb=0.005 \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# TacSL Shear Student
python3 scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/drawer_pulling/student_lstm \
    task=DrawerTaskPullingStudent \
    wandb_name=drawer_pulling_student_tacsl_shear \
    wandb_activate=True headless=True \
    task.randomize.use_force_perturb=True \
    task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.rl.dense_gripper_post_perturb_reward_scale=1.0 \
    task.randomize.use_randomized_force_perturb=False \
    task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] \
    task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 \
    task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    task.randomize.dof_padding_fail_preperturb=0.005 \
    task.env.use_tacsl_shear=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# FOTS Student
python3 scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/drawer_pulling/student_lstm \
    task=DrawerTaskPullingStudent \
    wandb_name=drawer_pulling_student_fots_shear \
    wandb_activate=True headless=True \
    task.randomize.use_force_perturb=True \
    task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.rl.dense_gripper_post_perturb_reward_scale=1.0 \
    task.randomize.use_randomized_force_perturb=False \
    task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] \
    task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 \
    task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    task.randomize.dof_padding_fail_preperturb=0.005 \
    task.env.use_fots_model=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```

```bash
# HydroShear Student
python3 scripts/experiments/hydroshear/train_student_aacd.py \
    train=hydroshear/drawer_pulling/student_lstm \
    task=DrawerTaskPullingStudent \
    wandb_name=drawer_pulling_student_hydroshear \
    wandb_activate=True headless=True \
    task.randomize.use_force_perturb=True \
    task.rl.gripper_force_perturb_penalty_scale=1.0 \
    task.rl.dense_gripper_post_perturb_reward_scale=1.0 \
    task.randomize.use_randomized_force_perturb=False \
    task.randomize.force_perturb_amount=-2 \
    task.randomize.force_perturb_range=[-2.0,0.0] \
    task.rl.alignment_keypoint_reward_scale=0.0 \
    task.rl.gripper_action_gradient_penalty_scale=0.0 \
    task.rl.goal_drawer_dof=-0.125 \
    task.rl.goal_drawer_dof_success_padding=0.01 \
    task.randomize.dof_padding_fail_preperturb=0.005 \
    task.env.friction_elastomer=1.0 \
    task.sensor.hydroshear.randomize_coefficients=False \
    task.env.use_hydrosoft_model=True \
    ckpt_path=/path/to/ckpt_stage2.pt
```