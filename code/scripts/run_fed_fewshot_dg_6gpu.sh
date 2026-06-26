#!/usr/bin/env bash
set -uo pipefail

# Few-shot federated domain-generalization launcher for CVS-RFFI.
#
# Setting:
#   - --wisig_train_ratio defaults to 0.1, so the train/val split is 10%/90%.
#   - Federated clients default to receiver_day; receiver-agnostic plans use receiver clients.
#   - Domain generalization is measured on named WiSig day/rx held-out splits,
#     especially test_unseen_day_unseen_rx.
#
# Examples:
#   bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan SMOKE --dry-run
#   bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FED_BASE --gpu-ids 0,1,2,3,4,5
#   bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FED_DG --gpu-ids 0,1,2
#   bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan CENTRAL --gpu-ids 0,1
#   FEWSHOT_RATIO=0.1 FL_ROUNDS=170 FL_LOCAL_EPOCHS=2 bash code/scripts/run_fed_fewshot_dg_6gpu.sh --plan FULL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Repo layout guard: this launcher is sometimes placed at <repo>/scripts/* (not <repo>/code/scripts/*).
# When that's the case, Dataset_WigSig lives under CODE_ROOT, so treat CODE_ROOT as the workspace root.
if [ -d "${CODE_ROOT}/Dataset_WigSig" ] || [ -d "${CODE_ROOT}/Dataset_ORALCE" ]; then
  WORKSPACE_ROOT="${CODE_ROOT}"
fi
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-${WORKSPACE_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/fed_fewshot_dg}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/fed_fewshot_dg}"
FEWSHOT_RATIO="${FEWSHOT_RATIO:-0.1}"
EPOCHS="${EPOCHS:-170}"
FL_ROUNDS="${FL_ROUNDS:-170}"
FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
AUTO_IDLE_GPUS="${AUTO_IDLE_GPUS:-1}"
GPU_IDLE_MAX_USED_MB="${GPU_IDLE_MAX_USED_MB:-1024}"
GPU_IDLE_MAX_UTIL="${GPU_IDLE_MAX_UTIL:-5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
CPU_THREADS="${CPU_THREADS:-${CVSRFFI_CPU_THREADS:-4}}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-${CVSRFFI_CPU_INTEROP_THREADS:-1}}"
THREAD_ENV=(
  "CVSRFFI_CPU_THREADS=${CPU_THREADS}"
  "CVSRFFI_CPU_INTEROP_THREADS=${CPU_INTEROP_THREADS}"
  "OMP_NUM_THREADS=${CPU_THREADS}"
  "MKL_NUM_THREADS=${CPU_THREADS}"
  "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
  "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
)

usage() {
  sed -n '1,16p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV        GPUs to use, default 0,1,2,3,4,5
  --plan NAME          SMOKE, FED_BASE, CORE, FED_DG, CENTRAL, CLIENTS, or FULL
  --wisig-pkl PATH     Dataset_WigSig/ManySig.pkl path
  --python PATH        Python executable
  --run-root PATH      Output checkpoint root
  --log-root PATH      Log root
  --ratio FLOAT        WiSig train ratio, default 0.1
  --rounds N           Federated communication rounds, default 170
  --local-epochs N     Federated local epochs, default 2
  --epochs N           Centralized epochs, default 170
  --no-skip-done       Re-run even when a completion artifact exists
  --stop-on-fail       Stop queue after first failure
  --stream-logs        Stream job logs to scheduler stdout
  --no-auto-idle-gpus  Disable nvidia-smi based idle-GPU assignment
  --gpu-idle-max-used-mb N  Treat GPU as idle only when used memory <= N MB, default 1024
  --gpu-idle-max-util N     Treat GPU as idle only when utilization <= N %, default 5
  --gpu-poll-seconds N      Seconds between idle-GPU polls, default 30
  --dry-run            Print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --ratio) FEWSHOT_RATIO="$2"; shift 2 ;;
    --rounds) FL_ROUNDS="$2"; shift 2 ;;
    --local-epochs) FL_LOCAL_EPOCHS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --auto-idle-gpus) AUTO_IDLE_GPUS=1; shift ;;
    --no-auto-idle-gpus) AUTO_IDLE_GPUS=0; shift ;;
    --gpu-idle-max-used-mb) GPU_IDLE_MAX_USED_MB="$2"; shift 2 ;;
    --gpu-idle-max-util) GPU_IDLE_MAX_UTIL="$2"; shift 2 ;;
    --gpu-poll-seconds) GPU_POLL_SECONDS="$2"; shift 2 ;;
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

if [ "${DRY_RUN}" != "1" ] && { [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; }; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
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
    SMOKE)
      cat <<'EOF' >> "${QUEUE_FILE}"
FSDG00_centralized_ce_smoke|SMOKE|Centralized 10% CE sanity control.|--train_mode centralized --epochs 3 --no_use_aug --no_use_mixstyle --lambda_dom 0 --lambda_adv 0 --lambda_orth 0 --lambda_cons 0 --lambda_group_ce 0 --lambda_fishr 0 --eval_max_batches 2
FSDG01_fedavg_rxday_smoke|SMOKE|FedAvg receiver_day smoke test with tiny rounds.|--train_mode fedavg --fl_client_key receiver_day --fl_rounds 2 --fl_local_epochs 1 --eval_max_batches 2
EOF
      ;;
    CENTRAL)
      cat <<'EOF' >> "${QUEUE_FILE}"
FSDG02_centralized_ce|CENTRAL|Centralized CE-only 10% baseline; isolates data scarcity without DG losses.|--train_mode centralized --no_use_aug --no_use_mixstyle --lambda_dom 0 --lambda_adv 0 --lambda_orth 0 --lambda_cons 0 --lambda_group_ce 0 --lambda_fishr 0
FSDG03_centralized_base_aug|CENTRAL|Centralized backbone recipe with augmentation enabled but no explicit DG losses.|--train_mode centralized --lambda_dom 0 --lambda_adv 0 --lambda_orth 0 --lambda_cons 0 --lambda_group_ce 0 --lambda_fishr 0
FSDG04_centralized_mixstyle|CENTRAL|Centralized MixStyle domain-generalization control at 10%.|--train_mode centralized --use_mixstyle --no_use_sat_consistency --lambda_fishr 0 --lambda_sat_cls 0 --lambda_sat_cons 0
FSDG05_centralized_fishr|CENTRAL|Centralized Fishr-only DG control at 10%.|--train_mode centralized --no_use_mixstyle --no_use_sat_consistency --lambda_fishr 0.02 --fishr_min_domains 4 --lambda_sat_cls 0 --lambda_sat_cons 0
FSDG06_centralized_sat|CENTRAL|Centralized satellite-consistency route without Fishr; isolates star-ground channel training signal.|--train_mode centralized --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0
FSDG07_centralized_strong_dg|CENTRAL|Current strong centralized DG recipe at 10%; main non-FL reference.|--train_mode centralized --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    FED_BASE|CORE)
      cat <<'EOF' >> "${QUEUE_FILE}"
FSDG12_fedavg_rxday|FED_BASE|Main few-shot FL baseline: receiver_day clients, sample-weighted FedAvg.|--train_mode fedavg --fl_client_key receiver_day
FSDG12A_fedavg_rxday_local3|FED_BASE|FedAvg with three local epochs; controls for the extra local-drift budget used by FedProx local3.|--train_mode fedavg --fl_client_key receiver_day --fl_local_epochs 3
FSDG13_fedprox_rxday_mu001|FED_BASE|FedProx weak proximal control for client drift under 10% data.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.001
FSDG14_fedprox_rxday_mu01|FED_BASE|FedProx medium proximal control for client drift under 10% data.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.01
FSDG14A_fedprox_rxday_mu01_local3|FED_BASE|FedProx medium proximal control with three local epochs; checks stronger local drift than the default FL schedule.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.01 --fl_local_epochs 3
FSDG14B_fedprox_rxday_mu1|FED_BASE|FedProx strong proximal-control probe from the common mu sweep; should visibly separate from FedAvg if proximal regularization matters.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 1.0
FSDG15_fedavg_rx|FED_BASE|Coarser receiver clients; tests whether receiver_day is too fragmented for few-shot.|--train_mode fedavg --fl_client_key receiver
FSDG16_fedavg_rxday_frac05|FED_BASE|Partial participation control; tests client sampling stability.|--train_mode fedavg --fl_client_key receiver_day --fl_clients_per_round 0.5
FSDG17_fedprox_rxday_frac05_mu01|FED_BASE|FedProx with partial participation; tests proximal control and client sampling together.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.01 --fl_clients_per_round 0.5
EOF
      ;;
    FED_DG)
      cat <<'EOF' >> "${QUEUE_FILE}"
FSDG18_fedavg_rxday_bex02dg|FED_DG|Federated BEX02_fishr002_mixed_e170 local objective: augmentation + MixStyle + Fishr=0.02 + mixed-orbit satellite CE inside each client.|--train_mode fedavg --fl_client_key receiver_day --fl_local_objective bex02_dg --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FSDG19_fedprox_rxday_bex02dg_mu001|FED_DG|FedProx weak proximal BEX02 local DG objective; low-drift regularization under the strong recipe.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.001 --fl_local_objective bex02_dg --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FSDG1A_fedprox_rxday_bex02dg_mu01|FED_DG|FedProx medium proximal BEX02 local DG objective; main proximal-control strong DG setting.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.01 --fl_local_objective bex02_dg --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FSDG1B_fedprox_rxday_bex02dg_mu1|FED_DG|FedProx strong proximal BEX02 local DG objective; diagnostic mu-sweep point for visible FedAvg separation.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 1.0 --fl_local_objective bex02_dg --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FSDG49_fedprox_receiver_ra_bex02_cvs_sat|FED_DG|FedProx + CVS-RFFI receiver-agnostic BEX02: each receiver is a client, GRL removes receiver information, CVS satellite consistency retained.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FSDG50_fedprox_receiver_ra_bex02_baseline_sat|FED_DG|FedProx + CVS-RFFI receiver-agnostic BEX02 with baseline supervised clean+satellite view expansion migrated into CVS-RFFI.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios mixed_orbit --sat_view_prob 1.0 --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FSDG51_fedprox_receiver_proto_stats|FED_DG|FedProx receiver clients with federated prototype/statistic exchange; lets single-domain clients interact through global class prototypes without sharing samples.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --fl_rounds 170 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --use_fed_proto_stats --lambda_fed_proto 0.10 --fed_proto_min_count 2 --fed_proto_momentum 0.20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    CLIENTS)
      cat <<'EOF' >> "${QUEUE_FILE}"
FSDG30_fedavg_receiver|CLIENTS|Client granularity: receiver.|--train_mode fedavg --fl_client_key receiver
FSDG31_fedavg_receiver_day|CLIENTS|Client granularity: receiver_day.|--train_mode fedavg --fl_client_key receiver_day
FSDG32_fedprox_receiver|CLIENTS|FedProx with receiver clients.|--train_mode fedprox --fl_client_key receiver --fedprox_mu 0.01
FSDG33_fedprox_receiver_day|CLIENTS|FedProx with receiver_day clients.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.01
EOF
      ;;
    *)
      echo "ERROR: unknown plan '${plan_name}'. Use SMOKE,FED_BASE,CORE,FED_DG,CENTRAL,CLIENTS,FULL." >&2
      exit 2
      ;;
  esac
}

generate_queue() {
  : > "${QUEUE_FILE}"
  local plan_upper
  plan_upper="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
  if [ "${plan_upper}" = "FULL" ] || [ "${plan_upper}" = "ALL" ]; then
    append_rows_for_plan FED_BASE
    append_rows_for_plan FED_DG
    append_rows_for_plan CENTRAL
  else
    IFS=',' read -r -a plans <<< "${PLAN}"
    local p
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
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_domain rx_day
  --wisig_train_ratio "${FEWSHOT_RATIO}"
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --fl_rounds "${FL_ROUNDS}"
  --fl_local_epochs "${FL_LOCAL_EPOCHS}"
  --fl_clients_per_round 1.0
  --fl_agg_weight num_samples
  --primary_udu_weight 0.65
  --test_eval_policy every_epoch
  --eval_max_batches 0
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
  --sat_cons_start_epoch 20
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --seed 1337
)

RUNNING_PIDS=()
RUNNING_TAGS=()
RUNNING_GPUS=()
ACQUIRED_GPU=""
FREE_GPUS=()
if [ "${AUTO_IDLE_GPUS}" != "1" ]; then
  FREE_GPUS=("${GPU_LIST[@]}")
fi
FAILURES=0
COMPLETED=0

join_by_space() {
  printf '%q ' "$@"
}

gpu_in_running_set() {
  local needle="$1" gpu
  for gpu in "${RUNNING_GPUS[@]}"; do
    if [ "${gpu}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

gpu_in_free_set() {
  local needle="$1" gpu
  for gpu in "${FREE_GPUS[@]}"; do
    if [ "${gpu}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

gpu_is_idle() {
  local gpu="$1"
  local pids line used util
  if [ "${AUTO_IDLE_GPUS}" != "1" ]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print; exit}')"
  if [ -n "${pids}" ]; then
    return 1
  fi
  line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
  if [ -z "${line}" ]; then
    return 1
  fi
  used="${line%%,*}"
  util="${line##*,}"
  used="${used:-999999}"
  util="${util:-999}"
  [ "${used}" -le "${GPU_IDLE_MAX_USED_MB}" ] && [ "${util}" -le "${GPU_IDLE_MAX_UTIL}" ]
}

refresh_free_gpus() {
  local gpu
  if [ "${AUTO_IDLE_GPUS}" != "1" ]; then
    return 0
  fi
  for gpu in "${GPU_LIST[@]}"; do
    if gpu_in_running_set "${gpu}" || gpu_in_free_set "${gpu}"; then
      continue
    fi
    if gpu_is_idle "${gpu}"; then
      FREE_GPUS+=("${gpu}")
    fi
  done
}

acquire_gpu_for_job() {
  local gpu
  while true; do
    poll_jobs
    refresh_free_gpus
    if [ "${#FREE_GPUS[@]}" -gt 0 ]; then
      gpu="${FREE_GPUS[0]}"
      FREE_GPUS=("${FREE_GPUS[@]:1}")
      ACQUIRED_GPU="${gpu}"
      return 0
    fi
    log_msg "[WAIT-GPU] no idle GPU among ${GPU_IDS_CSV}; polling again in ${GPU_POLL_SECONDS}s" >&2
    sleep "${GPU_POLL_SECONDS}"
  done
}

launch_job() {
  local gpu="$1" tag="$2" plan="$3" desc="$4" extra_args="$5"
  local out_dir="${RUN_ROOT}/${tag}"
  local log_file="${LOG_ROOT}/${tag}_${STAMP}.log"
  local done_files=(
    "${out_dir}/summary.json"
    "${out_dir}/metrics.csv"
    "${out_dir}/metrics.json"
    "${out_dir}/logs.jsonl"
  )

  if [ "${SKIP_DONE}" = "1" ]; then
    local f
    for f in "${done_files[@]}"; do
      if [ -f "${f}" ]; then
        log_msg "[SKIP] ${tag}: completion artifact exists (${f})."
        log_msg "       To re-run intentionally, pass --no-skip-done or set SKIP_DONE=0."
        return 0
      fi
    done
    if [ -d "${out_dir}" ] && [ "$(ls -A "${out_dir}" 2>/dev/null | wc -l | tr -d ' ')" != "0" ]; then
      log_msg "[SKIP] ${tag}: output dir exists and is non-empty (${out_dir})."
      log_msg "       To re-run intentionally, pass --no-skip-done or set SKIP_DONE=0."
      return 0
    fi
  fi

  mkdir -p "${out_dir}"
  read -r -a EXTRA <<< "${extra_args}"
  local cmd=(
    "${PYTHON_BIN}" train.py
    "${BASE_ARGS[@]}"
    "${EXTRA[@]}"
    --device cuda:0
    --output_dir "${out_dir}"
    --best_save_path "${out_dir}/best_model.pth"
    --latest_save_path "${out_dir}/latest_model.pth"
  )

  log_msg "[LAUNCH] gpu=${gpu} tag=${tag} plan=${plan}"
  log_msg "         ${desc}"
  log_msg "         CUDA_VISIBLE_DEVICES=${gpu} $(join_by_space "${THREAD_ENV[@]}") $(join_by_space "${cmd[@]}")"

  if [ "${DRY_RUN}" = "1" ]; then
    if ! gpu_in_free_set "${gpu}"; then
      FREE_GPUS+=("${gpu}")
    fi
    return 0
  fi

  if [ "${STREAM_LOGS}" = "1" ]; then
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" "${cmd[@]}" 2>&1 | tee "${log_file}" &
  else
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" "${cmd[@]}" > "${log_file}" 2>&1 &
  fi
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${tag}")
  RUNNING_GPUS+=("${gpu}")
  return 0
}

poll_jobs() {
  local i pid tag gpu status
  for i in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$i]}"
    tag="${RUNNING_TAGS[$i]}"
    gpu="${RUNNING_GPUS[$i]}"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      continue
    fi
    wait "${pid}"
    status="$?"
    if [ "${status}" -ne 0 ]; then
      FAILURES=$((FAILURES + 1))
      log_msg "[FAIL] ${tag} exit=${status}"
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        exit "${status}"
      fi
    else
      COMPLETED=$((COMPLETED + 1))
      log_msg "[DONE] ${tag}"
    fi
    if [ "${AUTO_IDLE_GPUS}" != "1" ]; then
      FREE_GPUS+=("${gpu}")
    elif gpu_is_idle "${gpu}" && ! gpu_in_free_set "${gpu}"; then
      FREE_GPUS+=("${gpu}")
    fi
    unset 'RUNNING_PIDS[i]'
    unset 'RUNNING_TAGS[i]'
    unset 'RUNNING_GPUS[i]'
  done
  RUNNING_PIDS=("${RUNNING_PIDS[@]}")
  RUNNING_TAGS=("${RUNNING_TAGS[@]}")
  RUNNING_GPUS=("${RUNNING_GPUS[@]}")
}

wait_for_phase_jobs() {
  while [ "${#RUNNING_PIDS[@]}" -gt 0 ]; do
    poll_jobs
    sleep 5
  done
}

run_queue_phase() {
  local phase_filter="$1"
  local phase_label="$2"
  local phase_jobs=0
  log_msg "[PHASE-BEGIN] ${phase_label}"
  while IFS='|' read -r tag plan desc extra_args; do
    if [ "${phase_filter}" != "__ALL__" ] && [ "${plan}" != "${phase_filter}" ]; then
      continue
    fi
    phase_jobs=$((phase_jobs + 1))
    acquire_gpu_for_job
    gpu="${ACQUIRED_GPU}"
    launch_job "${gpu}" "${tag}" "${plan}" "${desc}" "${extra_args}"
  done < "${QUEUE_FILE}"
  wait_for_phase_jobs
  log_msg "[PHASE-END] ${phase_label} jobs=${phase_jobs} completed=${COMPLETED} failures=${FAILURES}"
}

VAL_RATIO="$(awk -v r="${FEWSHOT_RATIO}" 'BEGIN { printf "%.1f", 1.0 - r }')"
log_msg "[SCHED] plan=${PLAN} jobs=${TOTAL_JOBS} gpus=${GPU_IDS_CSV} ratio=${FEWSHOT_RATIO} train/val=${FEWSHOT_RATIO}/${VAL_RATIO}"
log_msg "[SCHED] auto_idle_gpus=${AUTO_IDLE_GPUS} max_used_mb=${GPU_IDLE_MAX_USED_MB} max_util=${GPU_IDLE_MAX_UTIL} poll_seconds=${GPU_POLL_SECONDS}"
log_msg "[SCHED] queue=${QUEUE_FILE}"

PLAN_UPPER="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
if [ "${PLAN_UPPER}" = "FULL" ] || [ "${PLAN_UPPER}" = "ALL" ]; then
  log_msg "[SCHED] phased execution enabled: FED_BASE -> FED_DG -> CENTRAL"
  run_queue_phase FED_BASE "1/3 FED_BASE pure federated baselines"
  run_queue_phase FED_DG "2/3 FED_DG federated BEX02 strong DG"
  run_queue_phase CENTRAL "3/3 CENTRAL centralized baselines"
else
  run_queue_phase __ALL__ "selected plans"
fi

log_msg "[SCHED] completed=${COMPLETED} failures=${FAILURES} dry_run=${DRY_RUN}"
exit "${FAILURES}"
