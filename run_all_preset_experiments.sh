#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for all root-level preset experiments added in this project:
# SGC, model slimming, and SSDG semi-supervised presets.
#
# Examples:
#   bash run_all_preset_experiments.sh
#   PRESET_GROUPS=slim,ssdg bash run_all_preset_experiments.sh
#   GPU_IDS=0,1,2,3 STOP_ON_FAIL=1 bash run_all_preset_experiments.sh
#   SGC_STAGES=source,augment,adapt PRESET_GROUPS=sgc bash run_all_preset_experiments.sh
#   DRY_RUN=1 bash run_all_preset_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PRESET_GROUPS_CSV="${PRESET_GROUPS:-sgc}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

BASE_ARGS="${BASE_ARGS:---dataset wisig --wisig_domain rx_day --batch_size 256 --primary_udu_weight 0.65}"
SLIM_BASE_ARGS="${SLIM_BASE_ARGS:-${BASE_ARGS} --wisig_train_ratio 0.2}"
SSDG_BASE_ARGS="${SSDG_BASE_ARGS:-${BASE_ARGS}}"
SGC_BASE_ARGS="${SGC_BASE_ARGS:-${BASE_ARGS} --wisig_train_ratio 0.2}"

EPOCHS_SLIM="${EPOCHS_SLIM:-200}"
EPOCHS_SSDG="${EPOCHS_SSDG:-200}"
EPOCHS_SGC_SOURCE="${EPOCHS_SGC_SOURCE:-200}"
EPOCHS_SGC_AUGMENT="${EPOCHS_SGC_AUGMENT:-100}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-50}"
SAT_SCENARIO="${SAT_SCENARIO:-mixed_orbit}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.85}"
ENABLE_SAT_TARGET_EVAL="${ENABLE_SAT_TARGET_EVAL:-1}"
SAT_TARGET_ON="${SAT_TARGET_ON:-test_unseen_day_unseen_rx}"
SAT_TARGET_SCENARIOS="${SAT_TARGET_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_TARGET_MAX_BATCHES="${SAT_TARGET_MAX_BATCHES:--1}"

SGC_STAGES_CSV="${SGC_STAGES:-source}"

SGC_PRESETS="${SGC_PRESETS:-sgc_lite_b_no_dac,sgc_lite_b_no_dac_no_amp,sgc_lite_b_no_dac_no_freq,sgc_lite_b_no_dac_no_spec,sgc_lite_b_no_dac_no_res,sgc_lite_b_no_dac_no_amp_freq,sgc_lite_b_no_dac_residual_only,sgc_lite_b_no_dac_light,sgc_lite_d_no_dac,sgc_lite_d_no_dac_light,sgc_baseline_no_adapter}"
SLIM_PRESETS="${SLIM_PRESETS:-slim_r19_anchor,slim_r25_compact,slim_r19_groupce006,slim_r19_fishr002,slim_r25_fishr002,slim_no_domain_enhancer,slim_lite_d_lowmix,slim_lite_e_no_dac_probe,slim_no_dac_no_pa_probe,slim_no_dac_no_stats_guard,slim_full_upper_bound}"
SSDG_PRESETS="${SSDG_PRESETS:-ssdg_r19_pseudo_cons,ssdg_r19_pseudo_cons_strict,ssdg_r25_pseudo_cons,ssdg_r19_pseudo_cons_fishr}"

mkdir -p logs all_preset_runs sgc_runs slimming_runs ssdg_runs
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a PRESET_GROUP_LIST <<< "${PRESET_GROUPS_CSV}"

if [ "${ENABLE_SAT_TARGET_EVAL}" = "1" ]; then
  SGC_BASE_ARGS="${SGC_BASE_ARGS} --eval_sat_channel --eval_sat_on ${SAT_TARGET_ON} --eval_sat_scenarios ${SAT_TARGET_SCENARIOS} --sat_eval_max_batches ${SAT_TARGET_MAX_BATCHES}"
fi

if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "GPU_IDS is empty." >&2
  exit 1
fi

trim() {
  echo "$1" | xargs
}

run_or_print() {
  local gpu_id="$1"
  shift
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY-RUN][GPU ${gpu_id}] $*"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 "$@"
}

launch_single_preset() {
  local group="$1"
  local preset="$2"
  local gpu_id="$3"
  local run_dir
  local log_path
  local epochs
  local base_args
  LAUNCHED_PID=""

  case "${group}" in
    slim)
      run_dir="slimming_runs/${preset}"
      log_path="logs/all_slim_${preset}_$(date +%Y%m%d_%H%M%S).log"
      epochs="${EPOCHS_SLIM}"
      base_args="${SLIM_BASE_ARGS}"
      ;;
    ssdg)
      run_dir="ssdg_runs/${preset}"
      log_path="logs/all_ssdg_${preset}_$(date +%Y%m%d_%H%M%S).log"
      epochs="${EPOCHS_SSDG}"
      base_args="${SSDG_BASE_ARGS}"
      ;;
    *)
      echo "launch_single_preset only supports slim/ssdg, got ${group}" >&2
      return 2
      ;;
  esac

  mkdir -p "${run_dir}"
  echo "[ALL][${group}] launch preset=${preset} gpu=${gpu_id} log=${log_path}"
  if [ "${DRY_RUN}" = "1" ]; then
    run_or_print "${gpu_id}" "${PYTHON_BIN}" -u train.py ${base_args} --preset "${preset}" --epochs "${epochs}" --latest_save_path "${run_dir}/latest_model.pth" --best_save_path "${run_dir}/best_model.pth"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 \
    nohup "${PYTHON_BIN}" -u train.py \
      ${base_args} \
      --preset "${preset}" \
      --epochs "${epochs}" \
      --latest_save_path "${run_dir}/latest_model.pth" \
      --best_save_path "${run_dir}/best_model.pth" \
      > "${log_path}" 2>&1 &
    LAUNCHED_PID="$!"
  fi
}

run_sgc_stage() {
  local preset="$1"
  local stage="$2"
  local gpu_id="$3"
  local run_dir="sgc_runs/${preset}/${stage}"
  local log_path="logs/all_sgc_${preset}_${stage}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${run_dir}"

  case "${stage}" in
    source)
      echo "[ALL][sgc] preset=${preset} stage=source gpu=${gpu_id} log=${log_path}"
      run_or_print "${gpu_id}" "${PYTHON_BIN}" -u train.py \
        ${SGC_BASE_ARGS} \
        --preset "${preset}" \
        --stage source \
        --epochs "${EPOCHS_SGC_SOURCE}" \
        --latest_save_path "${run_dir}/latest_model.pth" \
        --best_save_path "${run_dir}/best_model.pth" \
        2>&1 | tee "${log_path}"
      ;;
    augment|sgc_augment)
      echo "[ALL][sgc] preset=${preset} stage=augment gpu=${gpu_id} log=${log_path}"
      run_or_print "${gpu_id}" "${PYTHON_BIN}" -u train.py \
        ${SGC_BASE_ARGS} \
        --preset "${preset}" \
        --stage sgc_augment \
        --source_ckpt "sgc_runs/${preset}/source/best_model.pth" \
        --train_sat_channel \
        --train_sat_scenario "${SAT_SCENARIO}" \
        --sat_view_source main \
        --lambda_feat 1.0 \
        --lambda_res 0.01 \
        --epochs "${EPOCHS_SGC_AUGMENT}" \
        --latest_save_path "${run_dir}/latest_model.pth" \
        --best_save_path "${run_dir}/best_model.pth" \
        2>&1 | tee "${log_path}"
      ;;
    adapt|sgc_adapt)
      if [ "${preset}" = "sgc_baseline_no_adapter" ]; then
        echo "[ALL][sgc] skip adapter-only adapt for ${preset}: no SGC-Adapter parameters by design."
        return 0
      fi
      echo "[ALL][sgc] preset=${preset} stage=adapt gpu=${gpu_id} log=${log_path}"
      run_or_print "${gpu_id}" "${PYTHON_BIN}" -u train.py \
        ${SGC_BASE_ARGS} \
        --preset "${preset}" \
        --stage sgc_adapt \
        --source_ckpt "sgc_runs/${preset}/augment/best_model.pth" \
        --pseudo_label_threshold "${PSEUDO_THRESHOLD}" \
        --lambda_proto 1.0 \
        --lambda_cons 0.5 \
        --lambda_ent 0.01 \
        --lambda_res 0.01 \
        --adapt_lr 1e-4 \
        --adapt_epochs "${ADAPT_EPOCHS}" \
        --latest_save_path "${run_dir}/latest_model.pth" \
        --best_save_path "${run_dir}/best_model.pth" \
        2>&1 | tee "${log_path}"
      ;;
    *)
      echo "Unknown SGC stage: ${stage}" >&2
      return 2
      ;;
  esac
}

launch_sgc_chain() {
  local preset="$1"
  local gpu_id="$2"
  local chain_log="logs/all_sgc_${preset}_chain_$(date +%Y%m%d_%H%M%S).log"
  LAUNCHED_PID=""
  IFS=',' read -r -a SGC_STAGE_LIST <<< "${SGC_STAGES_CSV}"
  if [ "${DRY_RUN}" = "1" ]; then
    for raw_stage in "${SGC_STAGE_LIST[@]}"; do
      run_sgc_stage "${preset}" "$(trim "${raw_stage}")" "${gpu_id}"
    done
  else
    echo "[ALL][sgc] launch chain preset=${preset} gpu=${gpu_id} stages=${SGC_STAGES_CSV} chain_log=${chain_log}" >&2
    (
      set -euo pipefail
      for raw_stage in "${SGC_STAGE_LIST[@]}"; do
        run_sgc_stage "${preset}" "$(trim "${raw_stage}")" "${gpu_id}"
      done
    ) > "${chain_log}" 2>&1 &
    LAUNCHED_PID="$!"
  fi
}

wait_batch() {
  local status_ref=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[ALL] done ${tags[$i]} pid=${pids[$i]}"
    else
      echo "[ALL] failed ${tags[$i]} pid=${pids[$i]}" >&2
      status_ref=1
      [ "${STOP_ON_FAIL}" = "1" ] && exit 1
    fi
  done
  pids=()
  tags=()
  slot=0
  if [ "${status_ref}" -ne 0 ]; then
    status=1
  fi
}

pids=()
tags=()
slot=0
status=0

queue_job() {
  local tag="$1"
  local pid="$2"
  if [ "${DRY_RUN}" = "1" ]; then
    slot=$((slot + 1))
    if [ "$slot" -ge "${#GPU_LIST[@]}" ]; then
      slot=0
    fi
    return 0
  fi
  pids+=("${pid}")
  tags+=("${tag}")
  slot=$((slot + 1))
  if [ "$slot" -ge "${#GPU_LIST[@]}" ]; then
    wait_batch
  fi
}

for raw_group in "${PRESET_GROUP_LIST[@]}"; do
  group="$(trim "${raw_group}")"
  [ -z "${group}" ] && continue

  case "${group}" in
    sgc)
      IFS=',' read -r -a PRESETS <<< "${SGC_PRESETS}"
      for raw_preset in "${PRESETS[@]}"; do
        preset="$(trim "${raw_preset}")"
        [ -z "${preset}" ] && continue
        gpu="${GPU_LIST[$slot]}"
        launch_sgc_chain "${preset}" "${gpu}"
        queue_job "sgc:${preset}:${SGC_STAGES_CSV}" "${LAUNCHED_PID}"
      done
      ;;
    slim)
      IFS=',' read -r -a PRESETS <<< "${SLIM_PRESETS}"
      for raw_preset in "${PRESETS[@]}"; do
        preset="$(trim "${raw_preset}")"
        [ -z "${preset}" ] && continue
        gpu="${GPU_LIST[$slot]}"
        launch_single_preset slim "${preset}" "${gpu}"
        queue_job "slim:${preset}" "${LAUNCHED_PID}"
      done
      ;;
    ssdg)
      IFS=',' read -r -a PRESETS <<< "${SSDG_PRESETS}"
      for raw_preset in "${PRESETS[@]}"; do
        preset="$(trim "${raw_preset}")"
        [ -z "${preset}" ] && continue
        gpu="${GPU_LIST[$slot]}"
        launch_single_preset ssdg "${preset}" "${gpu}"
        queue_job "ssdg:${preset}" "${LAUNCHED_PID}"
      done
      ;;
    *)
      echo "Unknown group: ${group}. Use PRESET_GROUPS=sgc,slim,ssdg" >&2
      status=1
      [ "${STOP_ON_FAIL}" = "1" ] && exit 1
      ;;
  esac
done

if [ "${DRY_RUN}" != "1" ] && [ "${#pids[@]}" -gt 0 ]; then
  wait_batch
fi

exit "${status}"
