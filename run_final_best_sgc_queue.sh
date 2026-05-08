#!/usr/bin/env bash
set -euo pipefail

# Queue launcher for the post-Phase-A experiment plan.
# Default order is B,C,D,E:
#   B: stability / label-smoothing calibration proxy
#   C: new ECC-SAT training mode
#   D: second-domain-backbone disentanglement validation
#   E: SGC / residual adapter validation, always last
#
# Example:
#   PHASES=B,C,D,E GPU_IDS=0,1,2,3,4,5,6,7 nohup bash run_final_best_sgc_queue.sh \
#     > logs/next_round_$(date +%Y%m%d_%H%M%S).nohup.log 2>&1 &

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PHASES_CSV="${PHASES:-B,C,D,E}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"
GLOBAL_SEED="${SEED:-1337}"
MERGE_PRE_SGC_PHASES="${MERGE_PRE_SGC_PHASES:-1}"

DATASET="${DATASET:-wisig}"
WISIG_DOMAIN="${WISIG_DOMAIN:-rx_day}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_RATIO="${TRAIN_RATIO:-0.2}"
PRIMARY_UDU_WEIGHT="${PRIMARY_UDU_WEIGHT:-0.65}"
MAIN_EPOCHS="${MAIN_EPOCHS:-200}"
SGC_EPOCHS="${SGC_EPOCHS:-60}"
SGC_FT_LR="${SGC_FT_LR:-5e-5}"
SGC_LAMBDA_RES="${SGC_LAMBDA_RES:-}"
SAT_SCENARIO="${SAT_SCENARIO:-mixed_orbit}"
SAT_EVAL_ON="${SAT_EVAL_ON:-test_unseen_day_unseen_rx}"
SAT_EVAL_SCENARIOS="${SAT_EVAL_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"
RUN_SGC_EXTENDED="${RUN_SGC_EXTENDED:-0}"

MIN_SGC_SAT_PRIMARY="${MIN_SGC_SAT_PRIMARY:-87.50}"
ROOT_DIR="$(pwd)"
RUN_ROOT="${RUN_ROOT:-finalist_runs}"
LOG_DIR="${LOG_DIR:-logs}"
QUEUE_DIR="${QUEUE_DIR:-${RUN_ROOT}/queue_state_next_$(date +%Y%m%d_%H%M%S)_$$}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${QUEUE_DIR}"
: > "${QUEUE_DIR}/main_manifest.tsv"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a PHASE_LIST <<< "${PHASES_CSV}"

if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "[QUEUE] GPU_IDS is empty." >&2
  exit 1
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

append_extra_args() {
  local array_name="$1"
  local extra="${2:-}"
  local -n target_array="${array_name}"
  local extra_args=()
  if [ -n "$(trim "${extra}")" ]; then
    read -r -a extra_args <<< "${extra}"
    target_array+=("${extra_args[@]}")
  fi
}

sat_eval_args=(
  --eval_sat_channel
  --eval_sat_on "${SAT_EVAL_ON}"
  --eval_sat_scenarios "${SAT_EVAL_SCENARIOS}"
  --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}"
)

common_args=(
  --dataset "${DATASET}"
  --wisig_domain "${WISIG_DOMAIN}"
  --batch_size "${BATCH_SIZE}"
  --wisig_train_ratio "${TRAIN_RATIO}"
  --primary_udu_weight "${PRIMARY_UDU_WEIGHT}"
)

run_cmd() {
  local gpu="$1"
  local log_path="$2"
  shift 2
  echo "[QUEUE][GPU ${gpu}] log=${log_path}"
  echo "[QUEUE][GPU ${gpu}] $*"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log_path}"
}

run_main_job() {
  local gpu="$1"
  local name="$2"
  local seed="$3"
  local epochs="$4"
  local sat_cls="$5"
  local sat_cons="$6"
  local sat_start="$7"
  local fishr="$8"
  local use_sat="$9"
  local extra="${10:-}"

  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}_seed${seed}.log"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -u train.py
    "${common_args[@]}"
    --slim_group rxrobust_lite_b_no_dac_mix015
    --epochs "${epochs}"
    --seed "${seed}"
    --lambda_fishr "${fishr}"
    --fishr_min_domains 4
    "${sat_eval_args[@]}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_model_val.pth"
    --best_primary_save_path "${run_dir}/best_model_primary_ood.pth"
    --best_test_save_path "${run_dir}/best_model_test_overall.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_model_strict_udu.pth"
    --best_worst_rx_save_path "${run_dir}/best_model_worst_rx.pth"
  )

  if [ "${use_sat}" = "1" ]; then
    cmd+=(
      --use_sat_consistency
      --sat_train_scenario "${SAT_SCENARIO}"
      --sat_cons_start_epoch "${sat_start}"
      --lambda_sat_cls "${sat_cls}"
      --lambda_sat_cons "${sat_cons}"
    )
  fi
  append_extra_args cmd "${extra}"

  printf '%s\t%s\t%s\n' "${name}" "${log_path}" "${run_dir}/best_model_primary_ood.pth" >> "${QUEUE_DIR}/main_manifest.tsv"
  run_cmd "${gpu}" "${log_path}" "${cmd[@]}"
}

run_continue_job() {
  local gpu="$1"
  local name="$2"
  local source_ckpt="$3"
  local epochs="$4"
  local sat_cls="$5"
  local sat_cons="$6"
  local sat_start="$7"
  local extra="${8:-}"

  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -u train.py
    "${common_args[@]}"
    --slim_group rxrobust_lite_b_no_dac_mix015
    --source_ckpt "${source_ckpt}"
    --epochs "${epochs}"
    --seed "${GLOBAL_SEED}"
    --lr "${SGC_FT_LR}"
    --use_sat_consistency
    --sat_train_scenario "${SAT_SCENARIO}"
    --sat_cons_start_epoch "${sat_start}"
    --lambda_sat_cls "${sat_cls}"
    --lambda_sat_cons "${sat_cons}"
    --lambda_fishr 0.02
    --fishr_min_domains 4
    "${sat_eval_args[@]}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_model_val.pth"
    --best_primary_save_path "${run_dir}/best_model_primary_ood.pth"
    --best_test_save_path "${run_dir}/best_model_test_overall.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_model_strict_udu.pth"
    --best_worst_rx_save_path "${run_dir}/best_model_worst_rx.pth"
  )
  append_extra_args cmd "${extra}"
  run_cmd "${gpu}" "${log_path}" "${cmd[@]}"
}

run_sgc_job() {
  local gpu="$1"
  local name="$2"
  local source_ckpt="$3"
  local epochs="$4"
  local sat_cls="$5"
  local sat_cons="$6"
  local sat_start="$7"
  local adapter_json="$8"
  local extra="${9:-}"

  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -u train.py
    "${common_args[@]}"
    --slim_group rxrobust_lite_b_no_dac_mix015
    --stage sgc_augment
    --source_ckpt "${source_ckpt}"
    --sgc_adapter_kwargs "${adapter_json}"
    --train_sat_channel
    --train_sat_scenario "${SAT_SCENARIO}"
    --sat_view_source main
    --epochs "${epochs}"
    --seed "${GLOBAL_SEED}"
    --lr "${SGC_FT_LR}"
    --sat_cons_start_epoch "${sat_start}"
    --lambda_sat_cls "${sat_cls}"
    --lambda_sat_cons "${sat_cons}"
    --lambda_fishr 0.02
    --fishr_min_domains 4
    "${sat_eval_args[@]}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_model_val.pth"
    --best_primary_save_path "${run_dir}/best_model_primary_ood.pth"
    --best_test_save_path "${run_dir}/best_model_test_overall.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_model_strict_udu.pth"
    --best_worst_rx_save_path "${run_dir}/best_model_worst_rx.pth"
  )
  if [ -n "${SGC_LAMBDA_RES}" ] && [[ " ${extra} " != *" --lambda_res "* ]]; then
    cmd+=(--lambda_res "${SGC_LAMBDA_RES}")
  fi
  append_extra_args cmd "${extra}"
  run_cmd "${gpu}" "${log_path}" "${cmd[@]}"
}

claim_next_job() {
  local queue_file="$1"
  local lock_file="$2"
  local abort_file="$3"
  local line
  (
    flock -x 9
    if [ -f "${abort_file}" ]; then
      exit 1
    fi
    if [ ! -s "${queue_file}" ]; then
      exit 1
    fi
    line="$(head -n 1 "${queue_file}")"
    tail -n +2 "${queue_file}" > "${queue_file}.tmp"
    mv "${queue_file}.tmp" "${queue_file}"
    printf '%s\n' "${line}"
  ) 9>"${lock_file}"
}

worker_loop() {
  local phase="$1"
  local gpu="$2"
  local queue_file="$3"
  local lock_file="$4"
  local abort_file="$5"
  local line kind rc

  while line="$(claim_next_job "${queue_file}" "${lock_file}" "${abort_file}")"; do
    IFS='|' read -r kind f1 f2 f3 f4 f5 f6 f7 f8 f9 <<< "${line}"
    echo "[QUEUE][phase ${phase}][GPU ${gpu}] start ${kind}:${f1}"
    set +e
    case "${kind}" in
      main)
        run_main_job "${gpu}" "${f1}" "${f2}" "${f3}" "${f4}" "${f5}" "${f6}" "${f7}" "${f8}" "${f9:-}"
        rc="$?"
        ;;
      cont)
        run_continue_job "${gpu}" "${f1}" "${f2}" "${f3}" "${f4}" "${f5}" "${f6}" "${f7:-}"
        rc="$?"
        ;;
      sgc)
        run_sgc_job "${gpu}" "${f1}" "${f2}" "${f3}" "${f4}" "${f5}" "${f6}" "${f7}" "${f8:-}"
        rc="$?"
        ;;
      *)
        echo "[QUEUE][phase ${phase}][GPU ${gpu}] unknown job kind: ${kind}" >&2
        rc=2
        ;;
    esac
    set -e
    if [ "${rc}" -ne 0 ]; then
      echo "[QUEUE][phase ${phase}][GPU ${gpu}] failed ${kind}:${f1} rc=${rc}" >&2
      echo "${kind}:${f1}:rc=${rc}" >> "${QUEUE_DIR}/failures.log"
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        touch "${abort_file}"
        return "${rc}"
      fi
    else
      echo "[QUEUE][phase ${phase}][GPU ${gpu}] done ${kind}:${f1}"
      echo "${kind}:${f1}" >> "${QUEUE_DIR}/completed.log"
    fi
  done
}

run_phase_queue() {
  local phase="$1"
  local queue_file="$2"
  local lock_file="${QUEUE_DIR}/phase_${phase}.lock"
  local abort_file="${QUEUE_DIR}/phase_${phase}.abort"
  local pids=()
  local gpu

  if [ ! -s "${queue_file}" ]; then
    echo "[QUEUE][phase ${phase}] no jobs."
    return 0
  fi

  echo "[QUEUE][phase ${phase}] starting with GPUs: ${GPU_IDS_CSV}"
  for gpu in "${GPU_LIST[@]}"; do
    worker_loop "${phase}" "$(trim "${gpu}")" "${queue_file}" "${lock_file}" "${abort_file}" &
    pids+=("$!")
  done

  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  return "${rc}"
}

write_phase_a_queue() {
  local queue_file="$1"
  : > "${queue_file}"
  cat >> "${queue_file}" <<EOF_JOBS
main|A0_fishr_only_ref|1337|${MAIN_EPOCHS}|0|0|0|0.02|0|
main|A1_fishr_sat_mild_v1|1337|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|
main|A2_fishr_sat_light_v2|1337|${MAIN_EPOCHS}|0.05|0.02|20|0.02|1|
main|A3_fishr_sat_mid_v3|1337|${MAIN_EPOCHS}|0.12|0.06|20|0.02|1|
main|A4_fishr_sat_delayed_v4|1337|${MAIN_EPOCHS}|0.08|0.04|60|0.02|1|
main|A5_sat_mild_no_fishr_ablation|1337|${MAIN_EPOCHS}|0.08|0.04|20|0.00|1|
main|A6_fishr_sat_mild_seed2026|2026|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|
main|A7_fishr_sat_mild_seed3407|3407|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|
EOF_JOBS
}

write_phase_b_queue() {
  local queue_file="$1"
  : > "${queue_file}"
  cat >> "${queue_file}" <<EOF_JOBS
main|B1_A1_mild_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|
main|B2_A2_light_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.05|0.02|20|0.02|1|
main|B3_A1_ls002_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--label_smoothing 0.02
main|B4_A2_ls002_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.05|0.02|20|0.02|1|--label_smoothing 0.02
EOF_JOBS
}

write_phase_c_queue() {
  local queue_file="$1"
  local ecc_sat_002="--lambda_ecc 0.02 --ecc_apply_to sat --ecc_tau_start 0.65 --ecc_tau_end 0.95 --ecc_epochs 60 --ecc_start_epoch 1"
  local ecc_satmain_003="--lambda_ecc 0.03 --ecc_apply_to sat_main --ecc_tau_start 0.65 --ecc_tau_end 0.95 --ecc_epochs 60 --ecc_start_epoch 1"
  local ecc_satmain_cons="--lambda_ecc 0.03 --ecc_apply_to sat_main --ecc_tau_start 0.70 --ecc_tau_end 0.95 --ecc_epochs 40 --ecc_start_epoch 1"
  : > "${queue_file}"
  cat >> "${queue_file}" <<EOF_JOBS
main|C1_A1_ecc002_sat_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|${ecc_sat_002}
main|C2_A1_ecc003_satmain_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|${ecc_satmain_003}
main|C3_A1_ecc003_conservative_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|${ecc_satmain_cons}
main|C4_A2_ecc003_satmain_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.05|0.02|20|0.02|1|${ecc_satmain_003}
EOF_JOBS
}

write_phase_d_queue() {
  local queue_file="$1"
  : > "${queue_file}"
  cat >> "${queue_file}" <<EOF_JOBS
main|D1_domain_enhancer_off_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--force_domain_enhancer off
main|D2_domain_rcn020_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--force_domain_enhancer_strength 0.20
main|D3_domain_branch_same_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--force_domain_branch_ablation same
main|D4_domain_no_pa_no_stats_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--force_domain_branch_ablation no_pa,no_stats
main|D5_no_grl_adv_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--force_lambda_adv 0.0
main|D6_no_orth_seed${GLOBAL_SEED}|${GLOBAL_SEED}|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1|--force_lambda_orth 0.0
EOF_JOBS
}

select_sgc_sources() {
  local primary_file="${QUEUE_DIR}/selected_sgc_primary_ckpt.txt"
  local sat_file="${QUEUE_DIR}/selected_sgc_sat_ckpt.txt"
  local forced_primary="${PHASE_E_SOURCE_CKPT:-}"
  local forced_sat="${PHASE_E_SAT_SOURCE_CKPT:-}"

  if [ -n "${forced_primary}" ]; then
    printf '%s\n' "${forced_primary}" > "${primary_file}"
    printf '%s\n' "${forced_sat:-${forced_primary}}" > "${sat_file}"
    echo "[QUEUE] using forced PHASE_E_SOURCE_CKPT=${forced_primary}"
    return 0
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' "${RUN_ROOT}/A1_fishr_sat_mild_v1/best_model_primary_ood.pth" > "${primary_file}"
    printf '%s\n' "${RUN_ROOT}/A3_fishr_sat_mid_v3/best_model_primary_ood.pth" > "${sat_file}"
    echo "[QUEUE][DRY-RUN] selected placeholder SGC sources."
    return 0
  fi

  "${PYTHON_BIN}" - "${QUEUE_DIR}/main_manifest.tsv" "${LOG_DIR}" "${primary_file}" "${sat_file}" "${RUN_ROOT}/A1_fishr_sat_mild_v1/best_model_primary_ood.pth" "${MIN_SGC_SAT_PRIMARY}" <<'PY'
import glob
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
primary_file = Path(sys.argv[3])
sat_file = Path(sys.argv[4])
fallback_ckpt = Path(sys.argv[5])
min_sat_primary = float(sys.argv[6])

items = {}

def add_item(name, log_path, ckpt_path):
    log = Path(log_path)
    if not log.exists():
        return
    key = str(log.resolve())
    items[key] = {"name": name, "log": log, "ckpt": Path(ckpt_path)}

if manifest.exists():
    for line in manifest.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            add_item(parts[0], parts[1], parts[2])

for pattern in ("A*.log", "B*.log", "C*.log", "D*.log"):
    for raw in glob.glob(str(log_dir / pattern)):
        log = Path(raw)
        text = log.read_text(errors="ignore")
        m = re.search(r"Training finished\. best_primary_ood_score=.*? -> (\S+)", text)
        if m:
            add_item(log.stem, str(log), m.group(1))

rows = []
for item in items.values():
    text = item["log"].read_text(errors="ignore")
    m = re.search(
        r"\[FINAL-PRIMARY\] val_tx=([0-9.]+)% \| test_overall_tx=([0-9.]+)% \| strict_udu=([0-9.]+)% \| score=([0-9.]+)",
        text,
    )
    if not m:
        continue
    sat_vals = [
        float(x)
        for x in re.findall(r"\[FINAL-PRIMARY\] \[SAT-TEST\].*?strict_udu=([0-9.]+)%", text)
    ]
    rows.append({
        "name": item["name"],
        "log": str(item["log"]),
        "ckpt": str(item["ckpt"]),
        "ckpt_exists": item["ckpt"].exists(),
        "overall": float(m.group(2)),
        "udu": float(m.group(3)),
        "primary": float(m.group(4)),
        "sat_avg": sum(sat_vals) / len(sat_vals) if sat_vals else -1.0,
    })

eligible = [r for r in rows if r["ckpt_exists"]]
if not eligible and fallback_ckpt.exists():
    primary_file.write_text(str(fallback_ckpt) + "\n")
    sat_file.write_text(str(fallback_ckpt) + "\n")
    print(f"[QUEUE] no parsed B/C/D candidates; fallback SGC source={fallback_ckpt}")
    raise SystemExit(0)
if not eligible:
    raise SystemExit("No valid checkpoint found for Phase E. Set PHASE_E_SOURCE_CKPT=/path/to/best_model_primary_ood.pth.")

best_primary = max(eligible, key=lambda r: (r["primary"], r["udu"], r["overall"], r["sat_avg"]))
sat_pool = [r for r in eligible if r["primary"] >= min_sat_primary and r["sat_avg"] >= 0.0]
best_sat = max(sat_pool, key=lambda r: (r["sat_avg"], r["primary"], r["udu"])) if sat_pool else best_primary

primary_file.write_text(best_primary["ckpt"] + "\n")
sat_file.write_text(best_sat["ckpt"] + "\n")

print("[QUEUE] SGC source candidates:")
for r in sorted(eligible, key=lambda x: x["primary"], reverse=True):
    print(
        f"  {r['name']}: Primary={r['primary']:.2f} UDU={r['udu']:.2f} "
        f"Overall={r['overall']:.2f} SATAvg={r['sat_avg']:.2f} ckpt={r['ckpt']}"
    )
print(f"[QUEUE] Selected SRC-P: {best_primary['ckpt']} ({best_primary['name']})")
print(f"[QUEUE] Selected SRC-S: {best_sat['ckpt']} ({best_sat['name']})")
PY
}

append_sgc_suite() {
  local queue_file="$1"
  local prefix="$2"
  local source_ckpt="$3"
  local residual_only_std='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'
  local no_res_control='{"use_amp_norm":true,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":false,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35}'
  local full_sgc_mild='{"use_amp_norm":true,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":true,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'

  cat >> "${queue_file}" <<EOF_JOBS
cont|${prefix}0_no_adapter_continue|${source_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|
sgc|${prefix}1_residual_only_std|${source_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${residual_only_std}|
sgc|${prefix}2_residual_only_std_res001|${source_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${residual_only_std}|--lambda_res 0.01
sgc|${prefix}3_no_res_control|${source_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${no_res_control}|
sgc|${prefix}4_full_sgc_mild_res001|${source_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${full_sgc_mild}|--lambda_res 0.01
EOF_JOBS
}

write_phase_e_queue() {
  local queue_file="$1"
  local source_primary="$2"
  local source_sat="$3"
  : > "${queue_file}"
  append_sgc_suite "${queue_file}" "E" "${source_primary}"
  if [ "${RUN_SGC_EXTENDED}" = "1" ] && [ "${source_sat}" != "${source_primary}" ]; then
    append_sgc_suite "${queue_file}" "ES" "${source_sat}"
  else
    echo "[QUEUE][phase E] extended SGC suite skipped: RUN_SGC_EXTENDED=${RUN_SGC_EXTENDED}, SRC-P=${source_primary}, SRC-S=${source_sat}"
  fi
}

phase_enabled() {
  local wanted="$1"
  local raw phase
  for raw in "${PHASE_LIST[@]}"; do
    phase="$(trim "${raw}")"
    if [ "${phase}" = "${wanted}" ]; then
      return 0
    fi
  done
  return 1
}

echo "[QUEUE] root=${ROOT_DIR}"
echo "[QUEUE] queue_state=${QUEUE_DIR}"
echo "[QUEUE] GPUs=${GPU_IDS_CSV} phases=${PHASES_CSV} seed=${GLOBAL_SEED} merge_pre_sgc=${MERGE_PRE_SGC_PHASES} dry_run=${DRY_RUN}"

if phase_enabled "A"; then
  phase_a_queue="${QUEUE_DIR}/phase_A.queue"
  write_phase_a_queue "${phase_a_queue}"
  run_phase_queue "A" "${phase_a_queue}"
fi

if [ "${MERGE_PRE_SGC_PHASES}" = "1" ]; then
  phase_pre_queue="${QUEUE_DIR}/phase_PRE.queue"
  : > "${phase_pre_queue}"
  if phase_enabled "B"; then
    phase_b_queue="${QUEUE_DIR}/phase_B.queue"
    write_phase_b_queue "${phase_b_queue}"
    cat "${phase_b_queue}" >> "${phase_pre_queue}"
  fi
  if phase_enabled "C"; then
    phase_c_queue="${QUEUE_DIR}/phase_C.queue"
    write_phase_c_queue "${phase_c_queue}"
    cat "${phase_c_queue}" >> "${phase_pre_queue}"
  fi
  if phase_enabled "D"; then
    phase_d_queue="${QUEUE_DIR}/phase_D.queue"
    write_phase_d_queue "${phase_d_queue}"
    cat "${phase_d_queue}" >> "${phase_pre_queue}"
  fi
  if [ -s "${phase_pre_queue}" ]; then
    run_phase_queue "PRE" "${phase_pre_queue}"
  fi
else
  if phase_enabled "B"; then
    phase_b_queue="${QUEUE_DIR}/phase_B.queue"
    write_phase_b_queue "${phase_b_queue}"
    run_phase_queue "B" "${phase_b_queue}"
  fi

  if phase_enabled "C"; then
    phase_c_queue="${QUEUE_DIR}/phase_C.queue"
    write_phase_c_queue "${phase_c_queue}"
    run_phase_queue "C" "${phase_c_queue}"
  fi

  if phase_enabled "D"; then
    phase_d_queue="${QUEUE_DIR}/phase_D.queue"
    write_phase_d_queue "${phase_d_queue}"
    run_phase_queue "D" "${phase_d_queue}"
  fi
fi

if phase_enabled "E"; then
  select_sgc_sources
  source_primary="$(cat "${QUEUE_DIR}/selected_sgc_primary_ckpt.txt")"
  source_sat="$(cat "${QUEUE_DIR}/selected_sgc_sat_ckpt.txt")"
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${source_primary}" ]; then
    echo "[QUEUE] selected primary SGC checkpoint does not exist: ${source_primary}" >&2
    exit 1
  fi
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${source_sat}" ]; then
    echo "[QUEUE] selected SAT SGC checkpoint does not exist: ${source_sat}" >&2
    exit 1
  fi
  phase_e_queue="${QUEUE_DIR}/phase_E.queue"
  write_phase_e_queue "${phase_e_queue}" "${source_primary}" "${source_sat}"
  run_phase_queue "E" "${phase_e_queue}"
fi

echo "[QUEUE] all requested phases finished. Logs are in ${LOG_DIR}; queue state is in ${QUEUE_DIR}."
