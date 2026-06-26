#!/usr/bin/env bash
set -uo pipefail

# CVS-RFFI-only satellite-channel augmentation comparison.
#
# Compares two ways of using the same satellite-ground channel simulator:
#   A) current CVS-RFFI auxiliary satellite loss: late, weak sat CE
#   B) 拼接星地信道增强: baseline-style clean+sat concatenated supervised batch
#
# Examples:
#   bash scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh --dry-run
#   GPU_IDS=3,4,5,6 bash scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh
#   PLAN=FULL GPU_IDS=3,4,5,6,7 bash scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh
#   PLAN=BACKBONE_ABL GPU_IDS=0,1,2,3,4,5,6 bash scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh --ratio 0.1
#   PLAN=BACKBONE_DSQ_FOLLOWUP GPU_IDS=0,1,2,3,4,5,6 bash scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh --ratio 0.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-3,4,5,6}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-}"
FEWSHOT_RATIO="${FEWSHOT_RATIO:-0.2}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/cvs_rffi_sat_aug_compare}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/cvs_rffi_sat_aug_compare}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"

usage() {
  sed -n '1,16p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV       GPUs to use, default 3,4,5,6
  --plan NAME         CORE, NOMIX, CEONLY, BACKBONE_ABL, BACKBONE_DSQ_FOLLOWUP, or FULL; default CORE
  --ratio FLOAT       WiSig train ratio, default 0.2
  --python PATH       Python executable
  --run-root PATH     Output checkpoint root
  --log-root PATH     Log root
  --no-skip-done      Re-run even when best_primary_ood_model.pth exists
  --stop-on-fail      Stop after first failed job
  --stream-logs       Stream each job log to scheduler stdout
  --dry-run           Print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --ratio) FEWSHOT_RATIO="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${HOME}/.conda/envs/ssr-gpu/bin/python" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/ssr-gpu/bin/python"
  elif [ -x "${HOME}/.conda/envs/CVS-RFFI/bin/python" ]; then
    PYTHON_BIN="${HOME}/.conda/envs/CVS-RFFI/bin/python"
  else
    for candidate in python3 python python.exe py; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        PYTHON_BIN="${candidate}"
        break
      fi
    done
  fi
fi

if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no Python executable found. Pass --python /path/to/python." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${PLAN}_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

append_rows_for_plan() {
  local plan_name
  plan_name="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  case "${plan_name}" in
    CORE)
      cat <<'EOF' >> "${QUEUE_FILE}"
SA01_cvs_loss_mixed|CORE|Current CVS-RFFI method: late weak auxiliary satellite CE on mixed_orbit.|--sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00
SA02_concat_sat_mixed|CORE|拼接星地信道增强: clean+sat concatenated supervised batch on mixed_orbit.|--sat_train_scenario mixed_orbit --use_concat_sat_channel_aug --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00
SA03_cvs_loss_all5|CORE|Current CVS-RFFI method with five-scenario cycle; tests whether coverage alone helps.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00
SA04_concat_sat_all5|CORE|拼接星地信道增强 with five-scenario clean+sat concatenated supervised batches.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00
EOF
      ;;
    NOMIX)
      cat <<'EOF' >> "${QUEUE_FILE}"
SA05_cvs_loss_mixed_nomix|NOMIX|Current CVS-RFFI satellite loss without MixStyle; isolates MixStyle conflict.|--sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --no_use_mixstyle
SA06_concat_sat_mixed_nomix|NOMIX|拼接星地信道增强 without MixStyle on mixed_orbit.|--sat_train_scenario mixed_orbit --use_concat_sat_channel_aug --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --no_use_mixstyle
SA07_cvs_loss_all5_nomix|NOMIX|Current CVS-RFFI five-scenario satellite loss without MixStyle.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --no_use_mixstyle
SA08_concat_sat_all5_nomix|NOMIX|拼接星地信道增强 five-scenario clean+sat batches without MixStyle.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --no_use_mixstyle
EOF
      ;;
    CEONLY)
      cat <<'EOF' >> "${QUEUE_FILE}"
SA09_concat_sat_ceonly_mixed|CEONLY|RA-baseline-inspired central CVS run: clean losses unchanged; mixed_orbit satellite views add TX CE only.|--sat_train_scenario mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00
SA10_concat_sat_ceonly_all5|CEONLY|RA-baseline-inspired central CVS run: clean losses unchanged; five-scenario satellite cycle adds TX CE only.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00
EOF
      ;;
    BACKBONE_ABL)
      cat <<'EOF' >> "${QUEUE_FILE}"
SA11_ceonly_backbone_anchor_r010|BACKBONE_ABL|Central anchor: Lite-D no-DAC with five-scenario CE-only satellite views and no optional stability stems.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00
SA12_ceonly_id_phase_delta_r010|BACKBONE_ABL|Optional direction 1: ID backbone complex phase-delta time-stability cues on the CE-only anchor.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --id_time_stability_mode phase_delta
SA13_ceonly_id_dsq_r010|BACKBONE_ABL|Optional direction 2: ID backbone differential spectral-quotient frequency-stability cues on the CE-only anchor.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --id_freq_stability_mode dsq
SA14_ceonly_id_phase_dsq_r010|BACKBONE_ABL|Combined ID-backbone phase-delta time stability and DSQ frequency stability.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --id_time_stability_mode phase_delta --id_freq_stability_mode dsq
SA15_ceonly_domain_phase_delta_r010|BACKBONE_ABL|Domain-backbone-only phase-delta probe; keeps the ID backbone on the mature anchor.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_time_stability_mode phase_delta
SA16_ceonly_domain_dsq_r010|BACKBONE_ABL|Domain-backbone-only DSQ frequency-stability probe; keeps the ID backbone on the mature anchor.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_freq_stability_mode dsq
SA17_ceonly_all_phase_dsq_r010|BACKBONE_ABL|Full optional backbone probe: ID phase+DSQ and domain backbone mirrors those stability cues.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --id_time_stability_mode phase_delta --id_freq_stability_mode dsq --domain_time_stability_mode same --domain_freq_stability_mode same
EOF
      ;;
    BACKBONE_DSQ_FOLLOWUP)
      cat <<'EOF' >> "${QUEUE_FILE}"
SA18_domain_dsq_ch2_r010|BACKBONE_DSQ_FOLLOWUP|SA16 follow-up: domain-backbone DSQ with lower frequency-stability capacity; tests whether the clean/UDU gain is over-capacity-sensitive.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_freq_stability_mode dsq --freq_stability_channels 2
SA19_domain_dsq_ch8_r010|BACKBONE_DSQ_FOLLOWUP|SA16 follow-up: domain-backbone DSQ with higher frequency-stability capacity; tests whether extra DSQ capacity improves or destabilizes primary OOD.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_freq_stability_mode dsq --freq_stability_channels 8
SA20_domain_phase_dsq_r010|BACKBONE_DSQ_FOLLOWUP|Domain-backbone phase+DSQ combination without ID-backbone perturbation; tests whether SA15 and SA16 gains compose cleanly.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_time_stability_mode phase_delta --domain_freq_stability_mode dsq
SA21_id_domain_dsq_r010|BACKBONE_DSQ_FOLLOWUP|Pure double-DSQ without phase-delta; tests whether SA17's clean drop came from phase cues or from mirroring DSQ into both backbones.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --id_freq_stability_mode dsq --domain_freq_stability_mode same
SA22_domain_dsq_satce_w0p7_r010|BACKBONE_DSQ_FOLLOWUP|SA16 with weaker CE-only satellite pressure; tests whether clean/strict UDU improves when satellite views act as lighter regularization.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 0.7 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_freq_stability_mode dsq
SA23_domain_dsq_satce_w1p5_r010|BACKBONE_DSQ_FOLLOWUP|SA16 with stronger CE-only satellite pressure; tests whether satellite robustness can rise without losing the primary-route advantage.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 1.5 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --domain_freq_stability_mode dsq
SA24_id_phase_dsq_satce_w0p7_r010|BACKBONE_DSQ_FOLLOWUP|SA14 robustness route with weaker CE-only satellite pressure; tests whether the high satellite-average variant can recover clean/strict UDU.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --use_concat_sat_channel_aug --concat_sat_ce_only --concat_sat_ce_weight 0.7 --concat_sat_start_epoch 1 --sat_view_prob 1.00 --no_use_sat_consistency --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --id_time_stability_mode phase_delta --id_freq_stability_mode dsq
EOF
      ;;
    FULL|ALL)
      append_rows_for_plan CORE
      append_rows_for_plan NOMIX
      append_rows_for_plan CEONLY
      append_rows_for_plan BACKBONE_ABL
      append_rows_for_plan BACKBONE_DSQ_FOLLOWUP
      ;;
    *)
      echo "ERROR: unknown plan '${plan_name}'. Use CORE, NOMIX, CEONLY, BACKBONE_ABL, BACKBONE_DSQ_FOLLOWUP, or FULL." >&2
      exit 2
      ;;
  esac
}

: > "${QUEUE_FILE}"
append_rows_for_plan "${PLAN}"
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: selected plan produced an empty queue." >&2
  exit 2
fi

BASE_ARGS=(
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio "${FEWSHOT_RATIO}"
  --primary_udu_weight 0.65
  --epochs 170
  --eval_sat_channel
  --eval_sat_on test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --slim_group none
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --use_sat_consistency
  --lambda_fishr 0.02
  --fishr_min_domains 4
  --seed 1337
)

RUNNING_PIDS=()
RUNNING_TAGS=()
RUNNING_GPUS=()
FREE_GPUS=("${GPU_LIST[@]}")
NEXT_INDEX=0
STATUS=0

queue_line_at() {
  sed -n "$(($1 + 1))p" "${QUEUE_FILE}"
}

launch_one() {
  local gpu_id="$1" exp_id="$2" group="$3" purpose="$4" extra_args="$5"
  local out_dir="${RUN_ROOT}/${exp_id}"
  local mixstyle_flag="--use_mixstyle"
  if [[ "${extra_args}" == *"--no_use_mixstyle"* ]]; then
    mixstyle_flag=""
    extra_args="${extra_args/--no_use_mixstyle/}"
  fi
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/best_primary_ood_model.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  mkdir -p "${out_dir}"
  local log="${LOG_ROOT}/${exp_id}_${STAMP}.log"
  local run_base_args=("${BASE_ARGS[@]}")
  if [[ "${extra_args}" == *"--use_concat_sat_channel_aug"* ]]; then
    local filtered_base_args=()
    local base_arg
    for base_arg in "${run_base_args[@]}"; do
      if [ "${base_arg}" = "--use_sat_consistency" ]; then
        continue
      fi
      filtered_base_args+=("${base_arg}")
    done
    run_base_args=("${filtered_base_args[@]}")
    extra_args="${extra_args/--no_use_sat_consistency/}"
  fi
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u train.py "${run_base_args[@]}" \
    ${mixstyle_flag} \
    --run_name "${exp_id}" \
    --latest_save_path "${out_dir}/latest_model.pth" \
    --best_save_path "${out_dir}/best_val_model.pth" \
    --best_primary_save_path "${out_dir}/best_primary_ood_model.pth" \
    --best_unseen_day_unseen_rx_save_path "${out_dir}/best_strict_udu_model.pth")"
  cmd="${cmd}${extra_args}"

  {
    echo "EXP_ID=${exp_id}"
    echo "GROUP=${group}"
    echo "PURPOSE=${purpose}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
  } > "${log}"

  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} group=${group} cmd=${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  if [ "${STREAM_LOGS}" = "1" ]; then
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" 2>&1
      status="$?"
      echo "__EXIT_STATUS__=${status}"
      exit "${status}"
    ) | sed -u "s/^/[${exp_id}|GPU${gpu_id}] /" | tee -a "${log}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" >> "${log}" 2>&1 &
  fi
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} group=${group} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id group purpose extra_args
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS='|' read -r exp_id group purpose extra_args <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${group}" "${purpose}" "${extra_args}"
  done
}

reap_one() {
  wait -n
  local status=$?
  local i
  for i in "${!RUNNING_PIDS[@]}"; do
    if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      local gpu="${RUNNING_GPUS[$i]}" tag="${RUNNING_TAGS[$i]}"
      unset 'RUNNING_PIDS[i]' 'RUNNING_TAGS[i]' 'RUNNING_GPUS[i]'
      RUNNING_PIDS=("${RUNNING_PIDS[@]}")
      RUNNING_TAGS=("${RUNNING_TAGS[@]}")
      RUNNING_GPUS=("${RUNNING_GPUS[@]}")
      FREE_GPUS+=("${gpu}")
      if [ "${status}" -ne 0 ]; then
        STATUS=1
      fi
      log_msg "[FINISHED] exp=${tag} status=${status} freed_gpu=${gpu}"
      return
    fi
  done
}

log_msg "CVS-RFFI satellite augmentation comparison launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "QUEUE_FILE=${QUEUE_FILE}"
log_msg "DRY_RUN=${DRY_RUN} SKIP_DONE=${SKIP_DONE} STOP_ON_FAIL=${STOP_ON_FAIL} STREAM_LOGS=${STREAM_LOGS}"

start_until_full
while [ "${#RUNNING_PIDS[@]}" -gt 0 ] || [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
  if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
    reap_one
    if [ "${STATUS}" -ne 0 ] && [ "${STOP_ON_FAIL}" = "1" ]; then
      log_msg "Stopping early because a job failed and STOP_ON_FAIL=1."
      exit 1
    fi
  fi
  start_until_full
done

log_msg "CVS-RFFI satellite augmentation comparison queue finished status=${STATUS}"
exit "${STATUS}"
