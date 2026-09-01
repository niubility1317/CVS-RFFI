#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_nmfdu_gate_v1_s392002_20260901_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
CORE90_CKPT="${CORE90_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
SEED="${SEED:-392002}"
GPU_MAP="${GPU_MAP:-4,5,6,7}"
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
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "[ERROR] GPU_MAP must contain exactly four comma-separated GPU ids" >&2
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
  local mode="$2"
  local gpu="$3"
  local candidate="ADV3B02_NMFDU_${row}_${mode^^}_E200_S${SEED}"
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_val_ratio 0.30
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
    --proxy_unknown_placeholder_weight 0.0
    --proxy_unknown_virtual_detach false
    --proxy_unknown_vacuum_weight 0.55
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 3
    --proxy_unknown_vacuum_radius_deg 40
    --proxy_unknown_core_quantile 0.90
    --proxy_unknown_accept_quantile 0.85
    --proxy_unknown_tail_quantile 0.92
    --proxy_unknown_overflow_quantile 0.97
    --proxy_unknown_vaccept_weight 1.00
    --proxy_unknown_core_accept_weight 0.45
    --proxy_unknown_component_gate_weight 0.65
    --proxy_unknown_tail_quarantine_weight 0.20
    --proxy_unknown_source_safe_weight 0.20
    --proxy_unknown_vaccept_cvar_alpha 0.30
    --proxy_unknown_unknown_margin 0.08
    --proxy_unknown_known_margin 0.05
    --proxy_unknown_energy_softplus_temperature 0.04
    --proxy_unknown_component_temperature_deg 3.0
    --proxy_unknown_component_margin_deg 4.0
    --proxy_unknown_component_margin_temperature_deg 3.0
    --proxy_unknown_shell_width_deg 4.0
    --lambda_soft_unknown_mixup 0.0045
    --soft_unknown_mixup_start_epoch 25
    --soft_unknown_mixup_warmup_epochs 25
    --soft_unknown_mixup_count 24
    --soft_unknown_mixup_order 3
    --soft_unknown_mixup_alpha 0.5
    --soft_unknown_mixup_energy_margin 1.0
    --soft_unknown_mixup_ce_weight 0.60
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_vacuum_weight 0.35
    --soft_unknown_mixup_vacuum_width_deg 6
    --soft_unknown_mixup_vacuum_hard_k 3
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
    --physical_gate_variant nmfdu_v1
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
    --nmfdu_oracle_temperature 0.5
    --device cuda:0
    --seed "${SEED}")
}

launch_row() {
  local row="$1"
  local mode="$2"
  local gpu="$3"
  if ! row_enabled "${row}"; then
    echo "[NMFDU-SKIP] row=${row} reason=only-filter"
    return 0
  fi
  build_command "${row}" "${mode}" "${gpu}"
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

echo "[NMFDU-RUN] run_id=${RUN_ID} seed=${SEED} matrix=M0-M4 dry_run=${DRY_RUN}"
if row_enabled M0; then
  echo "[NMFDU-M0] execution=historical_checkpoint_eval_only checkpoint=${CORE90_CKPT}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "[ERROR] missing WiSig source data: ${WISIG_PKL}" >&2; exit 6; }
  [[ -f "${CORE90_CKPT}" ]] || { echo "[ERROR] missing M0 checkpoint: ${CORE90_CKPT}" >&2; exit 6; }
  [[ ! -e "${RUNS_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "[ERROR] refusing existing run/log root" >&2; exit 4; }
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  launch_row M1 equal "${GPUS[0]}"
  launch_row M2 i_only "${GPUS[1]}"
  launch_row M3 physical_full "${GPUS[2]}"
  launch_row M4 full "${GPUS[3]}"
else
  launch_row M1 equal "${GPUS[0]}" & p1=$!
  launch_row M2 i_only "${GPUS[1]}" & p2=$!
  launch_row M3 physical_full "${GPUS[2]}" & p3=$!
  launch_row M4 full "${GPUS[3]}" & p4=$!
  status=0
  wait "${p1}" || status=1
  wait "${p2}" || status=1
  wait "${p3}" || status=1
  wait "${p4}" || status=1
  exit "${status}"
fi
