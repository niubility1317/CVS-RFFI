#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_advb02_crra_mixed_orbit_20260819}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU="${GPU:-0}"
SEED="${SEED:-392033}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATES="${ONLY_CANDIDATES:-}"
LAUNCH_STABILIZE_SEC="${LAUNCH_STABILIZE_SEC:-20}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    *) echo "[CRRA-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

candidate_enabled() {
  local cid="$1"
  [[ -z "${ONLY_CANDIDATES}" || ",${ONLY_CANDIDATES}," == *",${cid},"* ]]
}

build_command() {
  local cid="$1"
  local crra_flag="--no_use_crra"
  local sat_start="17"
  if [[ "${cid}" == "ADVB02_CRRA_MIXED_ORBIT_E200" ]]; then
    crra_flag="--use_crra"
  fi
  CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_val_ratio 0.30
    --source_cal_ratio 0.15
    --source_select_ratio 0.15
    --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --output_dir "${RUNS_ROOT}/${cid}"
    --run_id "${RUN_ID}"
    --candidate_id "${cid}"
    --base_candidate ADV3B02_MIXED_ORBIT_E200
    --epochs 200
    --label_epochs 130
    --pseudo_epochs 70
    --from_scratch true
    --model_variant lite_d
    --id_feature_key feat_joint
    --dom_feature_key feat_imp
    --domain_enhancer rcn_stats
    --phase1_source_val_selection_only true
    --checkpoint_selection final_only
    --best_metric source_val_sat_hmean
    --use_unlabeled true
    --no_use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_training_mode concat_masked
    --concat_sat_ce_weight 1.0
    --sat_train_scenario mixed_orbit
    --sat_train_scenarios mixed_orbit
    --sat_cons_start_epoch "${sat_start}"
    --sat_view_prob 1.0
    --sat_view_seed "${SEED}"
    --lambda_sat_cls 0.50
    --lambda_sat_cons 0.0
    --lambda_domain 1.0
    --lambda_adv 0.35
    --lambda_orth 0.05
    --lambda_cons 0.08
    --lambda_group_ce 0.16
    --lambda_fishr 0.04
    "${crra_flag}"
    --crra_scenario mixed_orbit
    --crra_rank 8
    --crra_alpha_max 0.25
    --crra_shrinkage 0.10
    --crra_condition_dim 32
    --crra_nuisance_dim 9
    --crra_start_epoch 17
    --crra_ramp_epochs 30
    --crra_target_adapter false
    --lambda_crra_pair 0.0
    --lambda_crra_sat_kl 0.0
    --lambda_crra_sat_shell 0.15
    --crra_sat_shell_width_deg 12.0
    --lambda_crra_energy 0.001
    --lambda_crra_gate_l1 0.001
    --lambda_crra_nuisance 0.02
    --lambda_crra_condition_tx_adv 0.02
    --eval_sat_channel true
    --eval_sat_scenarios mixed_orbit
    --sat_eval_max_batches -1
    --use_ema_teacher true
    --device cuda:0
    --seed "${SEED}")
}

run_candidate() {
  local cid="$1"
  if ! candidate_enabled "${cid}"; then
    echo "[CRRA-SKIP] candidate=${cid} reason=only-filter"
    return 0
  fi
  build_command "${cid}"
  echo "[CRRA-CANDIDATE] run_id=${RUN_ID} candidate=${cid} gpu=${GPU} seed=${SEED} channel=mixed_orbit"
  printf "[CRRA-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  if [[ -e "${RUNS_ROOT}/${cid}" || -e "${LOG_ROOT}/${cid}.out" ]]; then
    echo "[CRRA-ERROR] refusing to overwrite ${cid}" >&2
    return 3
  fi
  mkdir -p "${RUNS_ROOT}/${cid}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  local pid=$!
  echo "${pid}" > "${LOG_ROOT}/${cid}.pid"
  echo "running pid=${pid} gpu=${GPU} started=$(date -Is)" > "${LOG_ROOT}/${cid}.status"
  sleep "${LAUNCH_STABILIZE_SEC}"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[CRRA-ERROR] candidate=${cid} exited during startup; inspect ${LOG_ROOT}/${cid}.out" >&2
    wait "${pid}" || true
    return 4
  fi
  echo "[CRRA-LAUNCHED] candidate=${cid} pid=${pid} log=${LOG_ROOT}/${cid}.out"
  set +e
  wait "${pid}"
  local status=$?
  set -e
  echo "exit=${status} finished=$(date -Is)" >> "${LOG_ROOT}/${cid}.status"
  echo "[CRRA-FINISHED] candidate=${cid} pid=${pid} exit=${status}"
  return "${status}"
}

echo "[CRRA-RUN] run_id=${RUN_ID} dry_run=${DRY_RUN} gpu=${GPU} seed=${SEED} ratios=0.07/0.63/0.15/0.15 roles=L_s/U_s/V_cal/V_select channel=mixed_orbit sat_training_mode=concat_masked kl=0 pair=0 shell=0.15"
run_candidate ADV3B02_MIXED_ORBIT_E200
run_candidate ADVB02_CRRA_MIXED_ORBIT_E200
echo "[CRRA-DONE] run_id=${RUN_ID}"
