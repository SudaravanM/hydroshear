#!/bin/bash
# Build a readable TensorBoard view of the real experiments.
#
# TensorBoard names each run by its path relative to --logdir, so the fix is a symlink tree with
# names that say what the run IS. Non-destructive: nothing is moved or renamed on disk.
#
#   bash scripts/experiments/hydroshear/tb_view.sh                 rebuild ~/tb
#   tensorboard --logdir ~/tb --port 6006 --bind_all
#
# Naming:  <TAG>__<role>__<envs>env__sr<best>__<MM-DD>
#   H0   the HydroShear reference student (tactile)      N0   the no-touch control
#   T1   teacher stage 1        T2  teacher stage 2 (kept)   T2x  stage 2 aborted
#   _scratch/  smoke, probe, viewer, envtest      _other/  evals, plays, recorder agents
OUT="$HOME/hydroshear/outputs"
TB="$HOME/tb"
rm -rf "$TB"; mkdir -p "$TB/_scratch" "$TB/_other"

best_sr () {  # highest best_sr_X.XX.pth in a run dir, or "na"
  ls "$1"/nn/best_sr_*.pth 2>/dev/null \
    | sed -E 's/.*best_sr_([0-9.]+)\.pth/\1/' | sort -rn | head -1 || true
}
cfg_get () { grep -m1 -oE "^\s*$2: .*" "$1/config.yaml" 2>/dev/null | awk '{print $2}'; }

n_real=0; n_other=0
while IFS= read -r tbdir; do
  run="$(dirname "$tbdir")"; base="$(basename "$run")"
  date="$(echo "$base" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1 | cut -c6-)"
  envs="$(cfg_get "$run" numEnvs)"; sr="$(best_sr "$run")"; sr="${sr:-na}"
  case "$base" in
    *student_hydroshear*)   tag=H0;  role=student-hydrofots ;;
    *student_notouch*)      tag=N0;  role=student-notouch ;;
    *teacher_stage1*)       tag=T1;  role=teacher-stage1 ;;
    *teacher_stage2_full*)  tag=T2;  role=teacher-stage2 ;;
    *teacher_stage2*)       tag=T2x; role=teacher-stage2-aborted ;;
    *probe*|*smoke*|*viewer*|*envtest*) tag=_scratch; role="$base" ;;
    *) tag=_other; role="$base" ;;
  esac
  if [ "$tag" = "_scratch" ] || [ "$tag" = "_other" ]; then
    ln -sfn "$tbdir" "$TB/$tag/$(echo "$role" | cut -c1-48)"; n_other=$((n_other+1))
  else
    name="${tag}__${role}__${envs:-?}env__sr${sr}__${date}"
    ln -sfn "$tbdir" "$TB/$name"; n_real=$((n_real+1)); echo "  $name"
  fi
done < <(find "$OUT" -maxdepth 4 -type d -name tb 2>/dev/null | sort)
echo
echo "$n_real real experiments linked, $n_other parked under _scratch/ and _other/"
echo "point tensorboard at: $TB"
