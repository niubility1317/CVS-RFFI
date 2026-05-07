#!/usr/bin/env bash
set -euo pipefail

# Queue launcher for the 2026-05-07 best-model + SGC residual experiment set.
# GPU0-7 are used by default. By default this runs Phase A only, so the first
# pass explores the best-model candidates before any SGC/residual jobs.
# Use PHASES=A,B when you want fully automatic best-model exploration first,
# then SGC/residual jobs last.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PHASES_CSV="${PHASES:-A}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"

DATASET="${DATASET:-wisig}"
WISIG_DOMAIN="${WISIG_DOMAIN:-rx_day}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_RATIO="${TRAIN_RATIO:-0.2}"
PRIMARY_UDU_WEIGHT="${PRIMARY_UDU_WEIGHT:-0.65}"
MAIN_EPOCHS="${MAIN_EPOCHS:-200}"
SGC_EPOCHS="${SGC_EPOCHS:-60}"
SGC_FT_LR="${SGC_FT_LR:-5e-5}"
SGC_LAMBDA_RES="${SGC_LAMBDA_RES:-0.02}"
SAT_SCENARIO="${SAT_SCENARIO:-mixed_orbit}"
SAT_EVAL_ON="${SAT_EVAL_ON:-test_unseen_day_unseen_rx}"
SAT_EVAL_SCENARIOS="${SAT_EVAL_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"

MIN_PRIMARY="${MIN_PRIMARY:-87.80}"
MIN_UDU="${MIN_UDU:-86.20}"
MIN_OVERALL="${MIN_OVERALL:-90.50}"
MIN_SAT_AVG="${MIN_SAT_AVG:-41.50}"

ROOT_DIR="$(pwd)"
RUN_ROOT="${RUN_ROOT:-finalist_runs}"
LOG_DIR="${LOG_DIR:-logs}"
QUEUE_DIR="${QUEUE_DIR:-${RUN_ROOT}/queue_state_$(date +%Y%m%d_%H%M%S)_$$}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${QUEUE_DIR}"

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

  printf '%s\t%s\t%s\n' "${name}" "${log_path}" "${run_dir}/best_model_primary_ood.pth" >> "${QUEUE_DIR}/phase_a_manifest.tsv"
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

  local run_dir="${RUN_ROOT}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  mkdir -p "${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" -u train.py
    "${common_args[@]}"
    --slim_group rxrobust_lite_b_no_dac_mix015
    --source_ckpt "${source_ckpt}"
    --epochs "${epochs}"
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
    --lr "${SGC_FT_LR}"
    --sat_cons_start_epoch "${sat_start}"
    --lambda_sat_cls "${sat_cls}"
    --lambda_sat_cons "${sat_cons}"
    --lambda_res "${SGC_LAMBDA_RES}"
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
  local line kind

  while line="$(claim_next_job "${queue_file}" "${lock_file}" "${abort_file}")"; do
    IFS='|' read -r kind f1 f2 f3 f4 f5 f6 f7 f8 <<< "${line}"
    echo "[QUEUE][phase ${phase}][GPU ${gpu}] start ${kind}:${f1}"
    set +e
    case "${kind}" in
      main)
        run_main_job "${gpu}" "${f1}" "${f2}" "${f3}" "${f4}" "${f5}" "${f6}" "${f7}" "${f8}"
        rc="$?"
        ;;
      cont)
        run_continue_job "${gpu}" "${f1}" "${f2}" "${f3}" "${f4}" "${f5}" "${f6}"
        rc="$?"
        ;;
      sgc)
        run_sgc_job "${gpu}" "${f1}" "${f2}" "${f3}" "${f4}" "${f5}" "${f6}" "${f7}"
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
  : > "${QUEUE_DIR}/phase_a_manifest.tsv"

  cat >> "${queue_file}" <<EOF_JOBS
main|A0_fishr_only_ref|1337|${MAIN_EPOCHS}|0|0|0|0.02|0
main|A1_fishr_sat_mild_v1|1337|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1
main|A2_fishr_sat_light_v2|1337|${MAIN_EPOCHS}|0.05|0.02|20|0.02|1
main|A3_fishr_sat_mid_v3|1337|${MAIN_EPOCHS}|0.12|0.06|20|0.02|1
main|A4_fishr_sat_delayed_v4|1337|${MAIN_EPOCHS}|0.08|0.04|60|0.02|1
main|A5_sat_mild_no_fishr_ablation|1337|${MAIN_EPOCHS}|0.08|0.04|20|0.00|1
main|A6_fishr_sat_mild_seed2026|2026|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1
main|A7_fishr_sat_mild_seed3407|3407|${MAIN_EPOCHS}|0.08|0.04|20|0.02|1
EOF_JOBS
}

select_phase_a_checkpoint() {
  local forced="${PHASE_B_SOURCE_CKPT:-}"
  local selected_file="${QUEUE_DIR}/selected_phase_a_ckpt.txt"
  if [ -n "${forced}" ]; then
    printf '%s\n' "${forced}" > "${selected_file}"
    echo "[QUEUE] using forced PHASE_B_SOURCE_CKPT=${forced}"
    return 0
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    local dry_ckpt="${RUN_ROOT}/A1_fishr_sat_mild_v1/best_model_primary_ood.pth"
    printf '%s\n' "${dry_ckpt}" > "${selected_file}"
    echo "[QUEUE][DRY-RUN] selected ${dry_ckpt}"
    return 0
  fi

  "${PYTHON_BIN}" - "${QUEUE_DIR}/phase_a_manifest.tsv" "${selected_file}" "${MIN_PRIMARY}" "${MIN_UDU}" "${MIN_OVERALL}" "${MIN_SAT_AVG}" <<'PY'
import re
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
selected_file = Path(sys.argv[2])
min_primary = float(sys.argv[3])
min_udu = float(sys.argv[4])
min_overall = float(sys.argv[5])
min_sat = float(sys.argv[6])

rows = []
for line in manifest.read_text(errors="ignore").splitlines():
    if not line.strip():
        continue
    name, log_path, ckpt_path = line.split("\t")
    log = Path(log_path)
    ckpt = Path(ckpt_path)
    if not log.exists():
        continue
    text = log.read_text(errors="ignore")
    m = re.search(
        r"\[FINAL-PRIMARY\] val_tx=([0-9.]+)% \| test_overall_tx=([0-9.]+)% \| strict_udu=([0-9.]+)% \| score=([0-9.]+)",
        text,
    )
    if not m:
        continue
    overall = float(m.group(2))
    udu = float(m.group(3))
    primary = float(m.group(4))
    sat_vals = [
        float(x)
        for x in re.findall(
            r"\[FINAL-PRIMARY\] \[SAT-TEST\].*?strict_udu=([0-9.]+)%",
            text,
        )
    ]
    sat_avg = sum(sat_vals) / len(sat_vals) if sat_vals else -1.0
    skipped = None
    sm = re.search(r"skipped_backward_batches=([0-9]+)", text)
    if sm:
        skipped = int(sm.group(1))
    passed = (
        primary >= min_primary
        and udu >= min_udu
        and overall >= min_overall
        and sat_avg >= min_sat
        and (skipped is None or skipped <= 50)
        and ckpt.exists()
    )
    rows.append(
        {
            "name": name,
            "log": log_path,
            "ckpt": ckpt_path,
            "primary": primary,
            "udu": udu,
            "overall": overall,
            "sat_avg": sat_avg,
            "skipped": skipped,
            "passed": passed,
            "ckpt_exists": ckpt.exists(),
        }
    )

if not rows:
    raise SystemExit("No parsable Phase A logs; cannot select Phase B checkpoint.")

passed_rows = [r for r in rows if r["passed"]]
pool = passed_rows if passed_rows else [r for r in rows if r["ckpt_exists"]]
if not pool:
    raise SystemExit("No Phase A checkpoint exists; cannot select Phase B checkpoint.")

best = max(pool, key=lambda r: (r["primary"], r["udu"], r["overall"], r["sat_avg"]))
selected_file.write_text(best["ckpt"] + "\n")

print("[QUEUE] Phase A candidates:")
for r in sorted(rows, key=lambda x: x["primary"], reverse=True):
    status = "PASS" if r["passed"] else "FALLBACK-CANDIDATE" if r["ckpt_exists"] else "NO-CKPT"
    print(
        f"  {status} {r['name']}: Primary={r['primary']:.2f} "
        f"UDU={r['udu']:.2f} Overall={r['overall']:.2f} "
        f"SATAvg={r['sat_avg']:.2f} skipped={r['skipped']} ckpt={r['ckpt']}"
    )
print(f"[QUEUE] Selected Phase B source checkpoint: {best['ckpt']} ({best['name']})")
PY
}

write_phase_b_queue() {
  local queue_file="$1"
  local selected_ckpt="$2"
  : > "${queue_file}"

  local residual_only_std='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'
  local residual_only_small='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_channels":16,"residual_blocks":1,"residual_kernel_size":5,"residual_init_gamma":0.0}'
  local residual_only_wide='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":false,"use_residual_comp":true,"residual_channels":48,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'
  local no_res_control='{"use_amp_norm":true,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":false,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35}'
  local no_amp_residual_full='{"use_amp_norm":false,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":true,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'
  local no_amp_no_res_control='{"use_amp_norm":false,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":false,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35}'
  local full_sgc_mild='{"use_amp_norm":true,"use_freq_comp":true,"use_spectral_suppressor":true,"use_residual_comp":true,"freq_hidden_dim":32,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'
  local no_amp_freq_probe='{"use_amp_norm":false,"use_freq_comp":false,"use_spectral_suppressor":true,"use_residual_comp":true,"spectral_hidden_dim":32,"spectral_residual_alpha":0.35,"residual_channels":32,"residual_blocks":2,"residual_kernel_size":5,"residual_init_gamma":0.0}'

  cat >> "${queue_file}" <<EOF_JOBS
cont|B0_no_adapter_sat_continue|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20
sgc|B1_residual_only_std|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${residual_only_std}
sgc|B2_residual_only_small|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${residual_only_small}
sgc|B3_residual_only_wide|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${residual_only_wide}
sgc|B4_no_res_control|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${no_res_control}
sgc|B5_no_amp_residual_full|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${no_amp_residual_full}
sgc|B6_no_amp_no_res_control|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${no_amp_no_res_control}
sgc|B7_full_sgc_mild|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${full_sgc_mild}
sgc|B8_no_amp_freq_sat_probe|${selected_ckpt}|${SGC_EPOCHS}|0.08|0.04|20|${no_amp_freq_probe}
EOF_JOBS
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
echo "[QUEUE] GPUs=${GPU_IDS_CSV} phases=${PHASES_CSV} dry_run=${DRY_RUN}"

if phase_enabled "A"; then
  phase_a_queue="${QUEUE_DIR}/phase_A.queue"
  write_phase_a_queue "${phase_a_queue}"
  run_phase_queue "A" "${phase_a_queue}"
fi

if phase_enabled "B"; then
  select_phase_a_checkpoint
  selected_ckpt="$(cat "${QUEUE_DIR}/selected_phase_a_ckpt.txt")"
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${selected_ckpt}" ]; then
    echo "[QUEUE] selected checkpoint does not exist: ${selected_ckpt}" >&2
    exit 1
  fi
  phase_b_queue="${QUEUE_DIR}/phase_B.queue"
  write_phase_b_queue "${phase_b_queue}" "${selected_ckpt}"
  run_phase_queue "B" "${phase_b_queue}"
fi

echo "[QUEUE] all requested phases finished. Logs are in ${LOG_DIR}; queue state is in ${QUEUE_DIR}."
