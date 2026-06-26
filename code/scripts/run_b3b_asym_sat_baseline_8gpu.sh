#!/usr/bin/env bash
set -uo pipefail

# Dynamic 8-GPU launcher for the next B3b-centered baseline search.
# Each GPU pulls the next queued experiment as soon as its current run exits.
#
# Examples:
#   bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh
#   bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh --plan CORE --dry-run
#   bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh --plan FULL --dry-run
#   STREAM_LOGS=1 bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh --plan FULL
#   GPU_IDS=0,1,2,3,4,5,6,7 PYTHON_BIN=python3 bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/b3b_asym_sat_baseline}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/b3b_asym_sat_baseline}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"

usage() {
  sed -n '1,20p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python python.exe py; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN//,/}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

append_rows_for_plan() {
  local plan_name
  plan_name="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  case "${plan_name}" in
    CORE)
      cat <<'EOF' >> "${QUEUE_FILE}"
N01_b3b_manual_repro|CORE|B3b manual R25 baseline reproduction; verifies the hand-expanded preset.| 
N02_fishr001_cls010|CORE|B3b + weak Fishr; low-risk strict_udu gain probe.|--lambda_fishr 0.01 --fishr_min_domains 4
N03_fishr0015_cls010|CORE|B3b + mid Fishr; expected best balance between B2 and B3b.|--lambda_fishr 0.015 --fishr_min_domains 4
N04_fishr002_cls010|CORE|B3b + Fishr=0.02; direct B2+B3b fusion.|--lambda_fishr 0.02 --fishr_min_domains 4
N05_fishr002_cls010_nomix|CORE|B3b + Fishr=0.02 without MixStyle; tests satellite/MixStyle conflict.|--lambda_fishr 0.02 --fishr_min_domains 4 --no_use_mixstyle
N06_fishr001_cls010_nomix|CORE|Weak Fishr without MixStyle; conservative satellite-first candidate.|--lambda_fishr 0.01 --fishr_min_domains 4 --no_use_mixstyle
N07_fishr0015_cls012|CORE|Mid Fishr with stronger satellite CE; targets storm/rain headroom.|--lambda_sat_cls 0.12 --lambda_fishr 0.015 --fishr_min_domains 4
N08_fishr0015_cls008|CORE|Mid Fishr with weaker satellite CE; checks if cls=0.10 is over-forcing.|--lambda_sat_cls 0.08 --lambda_fishr 0.015 --fishr_min_domains 4
EOF
      ;;
    MULTISAT)
      cat <<'EOF' >> "${QUEUE_FILE}"
N09_multisat_rain_storm_cls|MULTISAT|Two-view hard weather cls-only cycle: rain_leo,storm_mp.|--sat_train_scenarios rain_leo,storm_mp
N10_multisat_rain_storm_fishr0015|MULTISAT|Two-view hard weather with mid Fishr.|--sat_train_scenarios rain_leo,storm_mp --lambda_fishr 0.015 --fishr_min_domains 4
N11_multisat_low_rain_storm_fishr0015|MULTISAT|Three-view hard LEO cycle; broad satellite robustness probe.|--sat_train_scenarios low_elev_leo,rain_leo,storm_mp --lambda_fishr 0.015 --fishr_min_domains 4
N12_multisat_all5_fishr0015|MULTISAT|All five satellite scenarios cycled per batch; widest satellite-view training.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_fishr 0.015 --fishr_min_domains 4
N13_multisat_mixed_rain_fishr002|MULTISAT|Mixed-orbit plus rain, preserving the strict_udu-correlated rain view.|--sat_train_scenarios mixed_orbit,rain_leo --lambda_fishr 0.02 --fishr_min_domains 4
N14_multisat_mixed_storm_fishr002|MULTISAT|Mixed-orbit plus storm, targeted at the known satellite bottleneck.|--sat_train_scenarios mixed_orbit,storm_mp --lambda_fishr 0.02 --fishr_min_domains 4
N15_multisat_low_rain_storm_nomix|MULTISAT|Three hard views with no MixStyle; satellite-first stress test.|--sat_train_scenarios low_elev_leo,rain_leo,storm_mp --lambda_fishr 0.015 --fishr_min_domains 4 --no_use_mixstyle
N16_multisat_all5_nomix|MULTISAT|All five views with no MixStyle; strongest multi-sat/no-MixStyle contrast.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_fishr 0.01 --fishr_min_domains 4 --no_use_mixstyle
N17_multisat_all5_cls012|MULTISAT|All five views with stronger satellite CE; checks whether multi-view needs more CE.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_sat_cls 0.12 --lambda_fishr 0.01 --fishr_min_domains 4
N18_multisat_all5_tiny_cons|MULTISAT|All five views with tiny feature consistency; tests if multi-view makes consistency useful again.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_sat_cons 0.005 --lambda_fishr 0.01 --fishr_min_domains 4
EOF
      ;;
    SCENARIO)
      cat <<'EOF' >> "${QUEUE_FILE}"
N19_train_rain_fishr0015|SCENARIO|Single rain_leo training; rain had highest correlation with strict_udu.|--sat_train_scenario rain_leo --lambda_fishr 0.015 --fishr_min_domains 4
N20_train_storm_fishr0015|SCENARIO|Single storm_mp training; direct bottleneck optimization.|--sat_train_scenario storm_mp --lambda_fishr 0.015 --fishr_min_domains 4
N21_train_low_fishr0015|SCENARIO|Single low_elev_leo training; tests strongest easy-scenario satellite score.|--sat_train_scenario low_elev_leo --lambda_fishr 0.015 --fishr_min_domains 4
N22_train_clear_fishr0015|SCENARIO|Single clear_leo control; should not be final unless global metrics win.|--sat_train_scenario clear_leo --lambda_fishr 0.015 --fishr_min_domains 4
N23_train_rain_cls_only|SCENARIO|Pure B3b cls-only on rain view; isolates scenario effect from Fishr.|--sat_train_scenario rain_leo
N24_train_storm_cls_only|SCENARIO|Pure B3b cls-only on storm view; isolates storm scenario effect from Fishr.|--sat_train_scenario storm_mp
EOF
      ;;
    ASYM)
      cat <<'EOF' >> "${QUEUE_FILE}"
N25_asym_rcn020_all5|ASYM|Default ID no-DAC/domain no-stats, weaker RCN enhancer, all satellite views.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_enhancer_strength 0.20 --lambda_fishr 0.015 --fishr_min_domains 4
N26_asym_rcn050_all5|ASYM|Default asymmetric branches with stronger RCN enhancer; tests over-domainization.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_enhancer_strength 0.50 --lambda_fishr 0.015 --fishr_min_domains 4
N27_asym_rcn_off_all5|ASYM|Remove RCN enhancer while keeping asymmetric branches; channel-feature ablation.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_enhancer off --lambda_fishr 0.015 --fishr_min_domains 4
N28_sym_domain_nodac_all5|ASYM|Make domain backbone mirror ID no-DAC; tests whether asymmetry itself helps.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_branch_ablation same --lambda_fishr 0.015 --fishr_min_domains 4
N29_swap_id_nostats_domain_nodac|ASYM|Swap branch bias: ID no-stats, domain no-DAC; checks identity-vs-domain attribution.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --branch_ablation no_stats --domain_branch_ablation no_dac --lambda_fishr 0.015 --fishr_min_domains 4
N30_id_nodac_nostats_domain_nostats|ASYM|More compact ID branch plus no-stats domain branch; strong compression probe.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --branch_ablation no_dac,no_stats --domain_branch_ablation no_stats --lambda_fishr 0.015 --fishr_min_domains 4
N31_domain_full_all5|ASYM|ID no-DAC with full domain backbone; tests if domain path needs more capacity.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_branch_ablation none --lambda_fishr 0.015 --fishr_min_domains 4
N32_id_nostats_domain_full_all5|ASYM|ID no-stats with full domain backbone; wider asymmetric contrast.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --branch_ablation no_stats --domain_branch_ablation none --lambda_fishr 0.015 --fishr_min_domains 4
EOF
      ;;
    SEED)
      cat <<'EOF' >> "${QUEUE_FILE}"
N33_seed2027_fishr0015|SEED|Seed validation for N03-style balanced candidate.|--seed 2027 --lambda_fishr 0.015 --fishr_min_domains 4
N34_seed42_fishr0015|SEED|Seed validation for N03-style balanced candidate.|--seed 42 --lambda_fishr 0.015 --fishr_min_domains 4
N35_seed2027_multisat_all5|SEED|Seed validation for multi-satellite all-view candidate.|--seed 2027 --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_fishr 0.015 --fishr_min_domains 4
N36_seed42_multisat_all5|SEED|Seed validation for multi-satellite all-view candidate.|--seed 42 --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_fishr 0.015 --fishr_min_domains 4
EOF
      ;;
    *)
      echo "ERROR: unknown plan '${plan_name}'. Use CORE,MULTISAT,SCENARIO,ASYM,SEED,FULL." >&2
      exit 2
      ;;
  esac
}

generate_queue() {
  : > "${QUEUE_FILE}"
  local plan_upper
  plan_upper="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
  if [ "${plan_upper}" = "FULL" ] || [ "${plan_upper}" = "ALL" ]; then
    append_rows_for_plan CORE
    append_rows_for_plan MULTISAT
    append_rows_for_plan SCENARIO
    append_rows_for_plan ASYM
    append_rows_for_plan SEED
  else
    IFS=',' read -r -a plans <<< "${PLAN}"
    for p in "${plans[@]}"; do
      append_rows_for_plan "${p}"
    done
  fi
}

generate_queue
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
  --wisig_train_ratio 0.2
  --primary_udu_weight 0.65
  --epochs 200
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
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
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
  fi
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/best_primary_ood_model.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  mkdir -p "${out_dir}"
  local log="${LOG_ROOT}/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u train.py "${BASE_ARGS[@]}" \
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

log_msg "B3b asymmetric satellite baseline launcher"
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

log_msg "B3b asymmetric satellite queue finished status=${STATUS}"
exit "${STATUS}"
