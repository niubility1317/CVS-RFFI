#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
SEED="${SEED:-392005}"
SPLIT_SEED="${SPLIT_SEED:-392005}"
GPU_MAP="${GPU_MAP:-0,1,2,3,4,5,6,7}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
LAUNCH_CHECK_SEC="${LAUNCH_CHECK_SEC:-8}"
DRY_RUN=0
ONLY_ROWS=""

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_ROWS="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<< "${GPU_MAP}"
if [[ "${#GPUS[@]}" -ne 8 ]]; then
  echo "[ERROR] GPU_MAP must contain exactly eight comma-separated GPU ids" >&2
  exit 2
fi

row_enabled() {
  local row="$1"
  [[ -z "${ONLY_ROWS}" || ",${ONLY_ROWS}," == *",${row},"* ]]
}

gpu_active_count() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="${gpu}" '$1 == gpu && $3 == "C" { count++ } END { print count + 0 }'
}

check_gpu_slot() {
  local gpu="$1"
  local active
  active="$(gpu_active_count "${gpu}")"
  echo "[NMFDU-GPU] gpu=${gpu} active_compute=${active} cap=${MAX_ACTIVE_PER_GPU}"
  if [[ "${active}" -ge "${MAX_ACTIVE_PER_GPU}" ]]; then
    echo "[ERROR] gpu=${gpu} has no registered launch slot" >&2
    return 3
  fi
}

build_command() {
  local row="$1"
  local variant="$2"
  local mode="$3"
  local gpu="$4"
  local candidate="ADV3B02_GATE8_${row}_${mode^^}_E200_S${SEED}"
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --wisig_equalized 1
    --wisig_train_rxs 1,3,4,6,8
    --wisig_test_rxs 0,2,5,7,9,10,11
    --wisig_train_days 1,2,3
    --wisig_test_days 0,1,2,3
    --wisig_split_seed "${SPLIT_SEED}"
    --allow_source_target_day_overlap_by_disjoint_rx true
    --wisig_max_day123_per_combo 0
    --wisig_max_test_per_combo 0
    --split_mode tx_rx_day_1_7_2
    --phase1_source_role_protocol l_s_u_s_v
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_val_ratio 0.30
    --source_cal_ratio 0
    --source_select_ratio 0
    --output_dir "${RUNS_ROOT}/${row}"
    --run_id "${RUN_ID}"
    --candidate_id "${candidate}"
    --base_candidate ADV3B02_CORE90_SOFT_E200
    --epochs 200
    --label_epochs 130
    --pseudo_epochs 70
    --from_scratch true
    --best_metric source_val_sat_hmean
    --enable_joint_safe_guard false
    --checkpoint_selection final_only
    --use_feature_masks true
    --use_txrx_geometry_losses true
    --use_tx_rx_balanced_sampler false
    --phase1_distribution_audit_only true
    --lambda_tx_proto 0
    --lambda_rx_proto 0
    --lambda_mask_aux 0
    --lambda_tx_supcon_masked 0
    --lambda_rx_supcon_masked 0
    --lambda_txrx_rect 0
    --use_proto_memory true
    --lambda_proto 0.0032
    --proto_domain_align_weight 0.10
    --proto_margin 0.15
    --proto_push_weight 0.10
    --proto_min_count 2
    --lambda_open_world_feat 0.0024
    --ow_feat_start_epoch 12
    --ow_feat_warmup_epochs 25
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight 0
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.14
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight 0.40
    --ow_feat_vacuum_width_deg 6
    --ow_feat_vacuum_hard_k 3
    --lambda_zid_compact 0.032
    --zid_compact_start_epoch 8
    --zid_compact_warmup_epochs 25
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.35
    --zid_compact_cvar_weight 0.35
    --zid_compact_cvar_alpha 0.95
    --zid_compact_radius_deg 40
    --zid_compact_domain_aware true
    --lambda_proxy_unknown 0.0045
    --proxy_unknown_start_epoch 45
    --proxy_unknown_warmup_epochs 25
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 48
    --proxy_unknown_virtual_mode hard
    --proxy_unknown_energy_margin 0.0
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_virtual_detach false
    --lambda_soft_unknown_mixup 0.0045
    --soft_unknown_mixup_start_epoch 25
    --soft_unknown_mixup_warmup_epochs 25
    --soft_unknown_mixup_count 24
    --soft_unknown_mixup_order 3
    --soft_unknown_mixup_alpha 0.5
    --soft_unknown_mixup_energy_margin 1.0
    --soft_unknown_mixup_ce_weight 0.60
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_detach false
    --lambda_source_episode 0.0035
    --source_episode_start_epoch 20
    --source_episode_warmup_epochs 25
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg 33
    --source_episode_mixup_weight 0.75
    --source_episode_mixup_hard_k 3
    --test_eval_policy interval_final
    --test_eval_start_epoch 1
    --test_eval_interval 10
    --test_eval_final_window 20
    --test_eval_final_interval 2
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_view_schedule '1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak'
    --sat_cons_start_epoch 80
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0
    --lambda_zid_channel_invariance 0
    --zid_channel_pair_weight 1.0
    --lambda_u 0.16
    --lambda_ent 0.01
    --lambda_domain 1
    --lambda_adv 0.35
    --lambda_group_ce 0.16
    --lambda_fishr 0.04
    --tau_min 0.92
    --tau_max 0.97
    --pseudo_quantile 0.86
    --use_ema_teacher true
    --eval_sat_channel true
    --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_eval_max_batches -1
    --physical_gate_variant "${variant}"
    --device cuda:0
    --seed "${SEED}")
  if [[ "${variant}" == "nmfdu_v1" ]]; then
    CMD+=(
      --nmfdu_ablation_mode "${mode}"
      --nmfdu_stage1_end 80
      --nmfdu_stage2_end 120
      --nmfdu_stage3_end 200
      --nmfdu_gate_lr_scale 0.5
      --nmfdu_joint_backbone_lr_scale 0.1
      --lambda_nmfdu_branch_aux 0.2
      --lambda_nmfdu_route 0.1
      --lambda_nmfdu_phys 0.1
      --lambda_nmfdu_fused_pair 0.2
      --lambda_nmfdu_branch_pair 0.1
      --lambda_nmfdu_null_cal 0.05
      --lambda_nmfdu_balance 0.01
      --nmfdu_oracle_temperature 0.5)
  fi
}

launch_row() {
  local row="$1"
  local variant="$2"
  local mode="$3"
  local gpu="$4"
  if ! row_enabled "${row}"; then
    echo "[NMFDU-SKIP] row=${row} reason=only-filter"
    return 0
  fi
  build_command "${row}" "${variant}" "${mode}" "${gpu}"
  printf '[NMFDU-CMD] row=%s gpu=%s ' "${row}" "${gpu}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  check_gpu_slot "${gpu}"
  if [[ -e "${RUNS_ROOT}/${row}" || -e "${LOG_ROOT}/${row}.out" || -e "${LOG_ROOT}/${row}.pid" ]]; then
    echo "[ERROR] refusing to overwrite row=${row}" >&2
    return 4
  fi
  mkdir "${RUNS_ROOT}/${row}"
  "${CMD[@]}" > "${LOG_ROOT}/${row}.out" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${LOG_ROOT}/${row}.pid"
  sleep "${LAUNCH_CHECK_SEC}"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[ERROR] row=${row} exited before launch verification" >&2
    return 5
  fi
  local cwd cmdline
  cwd="$(readlink -f "/proc/${pid}/cwd")"
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  echo "[NMFDU-LANDED] row=${row} pid=${pid} gpu=${gpu} cwd=${cwd} cmdline=${cmdline} log=${LOG_ROOT}/${row}.out"
  wait "${pid}"
  local status=$?
  echo "[NMFDU-FINISHED] row=${row} pid=${pid} exit=${status}"
  return "${status}"
}

ROWS=(E1 E2 E3 E4 E5 E6 E7 E8)
VARIANTS=(nmfdu_v1 nmfdu_v1 nmfdu_v1 nmfdu_v1 nmfdu_v1 nmfdu_v1 nmfdu_v1 nmfdu_v1)
MODES=(equal i_only i_d i_d_s physical_fixed physical_full full_no_null full)

echo "[NMFDU-RUN] run_id=${RUN_ID} seed=${SEED} split_seed=${SPLIT_SEED} matrix=E1-E8 dry_run=${DRY_RUN}"
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "[ERROR] missing WiSig data: ${WISIG_PKL}" >&2; exit 6; }
  [[ ! -e "${RUNS_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "[ERROR] refusing existing run/log root" >&2; exit 4; }
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  for index in "${!ROWS[@]}"; do
    launch_row "${ROWS[$index]}" "${VARIANTS[$index]}" "${MODES[$index]}" "${GPUS[$index]}"
  done
else
  pids=()
  for index in "${!ROWS[@]}"; do
    launch_row "${ROWS[$index]}" "${VARIANTS[$index]}" "${MODES[$index]}" "${GPUS[$index]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  exit "${status}"
fi
