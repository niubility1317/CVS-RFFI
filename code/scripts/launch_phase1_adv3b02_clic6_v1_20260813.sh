#!/usr/bin/env bash
set -euo pipefail

# Six-fold, source-only, config-equivalent retraining of the historical
# ADV3B02_CORE90_SOFT_E200 mechanism.  This launcher deliberately has no
# target cache/package/truth/query/scorer input.  It is a training entry only.

RUN_ID="${RUN_ID:-phase1_adv3b02_clic6_20260813_v1}"
METHOD_ID="ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL"
HISTORICAL_PROFILE_ID="ADV3B02_CORE90_SOFT_E200"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/SSDG/train_ssdg.py}"
DRY_RUN=0
PRINT_CONTRACT=0
VALIDATE_CONTRACT_FILE=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --print-contract)
      PRINT_CONTRACT=1
      shift
      ;;
    --validate-contract-file)
      [[ "$#" -ge 2 ]] || { echo "--validate-contract-file requires a path" >&2; exit 2; }
      VALIDATE_CONTRACT_FILE="$2"
      shift 2
      ;;
    --validate-contract-file=*)
      VALIDATE_CONTRACT_FILE="${1#*=}"
      shift
      ;;
    *)
      echo "invalid argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ "${RUN_ID}" == "phase1_adv3b02_clic6_20260813_v1" ]] || {
  echo "RUN_ID is immutable for this launcher: ${RUN_ID}" >&2
  exit 2
}

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20"
  "14-10,20-19,6-15,8-20"
  "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20"
  "14-10,14-7,20-15,20-19"
  "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")
FOLD_GPUS=(0 1 2 3 4 5)

emit_frozen_contract() {
  cat <<'EOF'
schema=cvs.phase1.adv3b02_clic6_training_contract.v1
run_id=phase1_adv3b02_clic6_20260813_v1
method_id=ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL
historical_profile_id=ADV3B02_CORE90_SOFT_E200
training_scope=source_only
split_mode=tx_rx_day_1_6_3
labeled_ratio=0.07
unlabeled_ratio=0.63
source_val_ratio=0.30
seed=392002
epochs=200
label_epochs=130
pseudo_epochs=70
from_scratch=true
checkpoint_selection=final_only
profile.lambda_open_world_feat=0.0024
profile.lambda_zid_compact=0.032
profile.lambda_proxy_unknown=0.0045
profile.proxy_unknown_core_quantile=0.90
profile.proxy_unknown_accept_quantile=0.85
profile.proxy_unknown_vaccept_cvar_alpha=0.30
profile.lambda_soft_unknown_mixup=0.0045
profile.lambda_source_episode=0.0035
profile.lambda_sat_cls=0.68
profile.lambda_sat_cons=0
profile.lambda_u=0.16
profile.lambda_ent=0.01
profile.lambda_domain=1
profile.lambda_adv=0.35
profile.lambda_group_ce=0.16
profile.lambda_fishr=0.04
fold.1.train=20-15,20-19,6-15,8-20
fold.1.known_validation=14-7
fold.1.proxy_unknown=14-10
fold.1.gpu=0
fold.2.train=14-10,20-19,6-15,8-20
fold.2.known_validation=20-15
fold.2.proxy_unknown=14-7
fold.2.gpu=1
fold.3.train=14-10,14-7,6-15,8-20
fold.3.known_validation=20-19
fold.3.proxy_unknown=20-15
fold.3.gpu=2
fold.4.train=14-10,14-7,20-15,8-20
fold.4.known_validation=6-15
fold.4.proxy_unknown=20-19
fold.4.gpu=3
fold.5.train=14-10,14-7,20-15,20-19
fold.5.known_validation=8-20
fold.5.proxy_unknown=6-15
fold.5.gpu=4
fold.6.train=14-7,20-15,20-19,6-15
fold.6.known_validation=14-10
fold.6.proxy_unknown=8-20
fold.6.gpu=5
EOF
}

if [[ "${PRINT_CONTRACT}" == "1" ]]; then
  emit_frozen_contract
  exit 0
fi

if [[ -n "${VALIDATE_CONTRACT_FILE}" ]]; then
  [[ -f "${VALIDATE_CONTRACT_FILE}" ]] || {
    echo "contract file is missing: ${VALIDATE_CONTRACT_FILE}" >&2
    exit 2
  }
  if ! cmp -s "${VALIDATE_CONTRACT_FILE}" <(emit_frozen_contract); then
    echo "frozen contract mismatch: ${VALIDATE_CONTRACT_FILE}" >&2
    exit 3
  fi
  exit 0
fi

assert_tx_disjoint() {
  local fold="$1"
  local train_csv="$2"
  local known_csv="$3"
  local proxy_csv="$4"
  local train_item known_item proxy_item
  IFS=',' read -r -a train_items <<<"${train_csv}"
  IFS=',' read -r -a known_items <<<"${known_csv}"
  IFS=',' read -r -a proxy_items <<<"${proxy_csv}"
  [[ "${#train_items[@]}" -eq 4 && "${#known_items[@]}" -eq 1 && "${#proxy_items[@]}" -eq 1 ]] || {
    echo "invalid frozen role count for fold ${fold}" >&2
    exit 2
  }
  for train_item in "${train_items[@]}"; do
    for known_item in "${known_items[@]}"; do
      [[ "${train_item}" != "${known_item}" ]] || { echo "role overlap fold=${fold}" >&2; exit 2; }
    done
    for proxy_item in "${proxy_items[@]}"; do
      [[ "${train_item}" != "${proxy_item}" ]] || { echo "role overlap fold=${fold}" >&2; exit 2; }
    done
  done
  [[ "${known_items[0]}" != "${proxy_items[0]}" ]] || { echo "role overlap fold=${fold}" >&2; exit 2; }
}

for fold_index in 0 1 2 3 4 5; do
  assert_tx_disjoint "$((fold_index + 1))" \
    "${FOLD_TRAIN_TX[fold_index]}" \
    "${FOLD_KNOWN_VAL_TX[fold_index]}" \
    "${FOLD_PROXY_TX[fold_index]}"
done

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }

COMMON=(
  --wisig_pkl "${WISIG_PKL}"
  --split_mode tx_rx_day_1_6_3
  --labeled_ratio 0.07
  --unlabeled_ratio 0.63
  --source_val_ratio 0.30
  --base_candidate "${METHOD_ID}"
  --epochs 200
  --label_epochs 130
  --pseudo_epochs 70
  --from_scratch true
  --best_metric joint_safe
  --checkpoint_selection final_only
  --phase1_source_val_selection_only true
  --enable_joint_safe_guard true
  --one_epoch_drop_guard_pp 2.0
  --paic_guard_enabled true
  --paic_guard_sat_ce_delta 0.12
  --paic_guard_grad_delta 3.0
  --paic_guard_reliable_drop 0.01
  --paic_guard_cooldown_epochs 1
  --paic_guard_sat_scale 0.75
  --use_phase2_ground_prototypes true
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
  --phase2_export_prototypes true
  --phase2_export_feature_key z_id
  --phase2_export_split train
  --phase2_fuse_prototypes true
  --phase2_fuse_max_components 6
  --phase2_fuse_merge_angle_deg 2.5
  --phase2_fuse_radius_cap_deg 15.0
  --phase2_fuse_tail_abs_deg 24
  --phase2_fuse_accept_policy local_component
  --phase2_fuse_accept_radius_key p95
  --phase2_fuse_max_p95_increase_deg 2.0
  --phase2_fuse_keep_tail_sentinel true
  --phase2_fuse_global_ball_accept false
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
  --device cuda:0
  --seed 392002
)

build_command() {
  local fold_index="$1"
  local candidate="F$((fold_index + 1))_ADV3B02_CLIC"
  local output_dir="${RUN_ROOT}/${candidate}"
  COMMAND=("${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON[@]}"
    --run_id "${RUN_ID}"
    --candidate_id "${candidate}"
    --output_dir "${output_dir}"
    --phase2_export_path "${output_dir}/phase2_zid_prototypes.pt"
    --phase1_source_train_tx_ids "${FOLD_TRAIN_TX[fold_index]}"
    --phase1_source_known_validation_tx_ids "${FOLD_KNOWN_VAL_TX[fold_index]}"
    --phase1_source_proxy_unknown_tx_ids "${FOLD_PROXY_TX[fold_index]}")
}

for fold_index in 0 1 2 3 4 5; do
  build_command "${fold_index}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' "${FOLD_GPUS[fold_index]}" "${CODE_ROOT}"
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
  fi
done

[[ "${DRY_RUN}" == "1" ]] && exit 0

# Check roots before the dataset so a collision is always a no-mutation failure.
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || {
  echo "refusing to overwrite run/log root" >&2
  exit 3
}
[[ -f "${WISIG_PKL}" ]] || { echo "missing WiSig dataset: ${WISIG_PKL}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${LOG_ROOT}/status"
emit_frozen_contract >"${LOG_ROOT}/frozen_contract.txt"
printf '%s\n' "$$" >"${LOG_ROOT}/outer.pid"
printf 'kind|pid|fold|candidate|physical_gpu|output_dir|log_path|state\n' >"${LOG_ROOT}/pids.tsv"
printf 'outer|%s|-|-|-|%s|%s|running\n' "$$" "${RUN_ROOT}" "${LOG_ROOT}" >>"${LOG_ROOT}/pids.tsv"

declare -a PIDS FOLDS CANDIDATES GPUS OUTPUTS LOGS
launch_fold() {
  local fold_index="$1"
  local fold="$((fold_index + 1))"
  local candidate="F${fold}_ADV3B02_CLIC"
  local gpu="${FOLD_GPUS[fold_index]}"
  local output_dir="${RUN_ROOT}/${candidate}"
  local log_path="${LOG_ROOT}/${candidate}.out"
  build_command "${fold_index}"
  [[ ! -e "${output_dir}" && ! -e "${log_path}" ]] || {
    echo "refusing to overwrite fold output: ${candidate}" >&2
    exit 3
  }
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${COMMAND[@]}" >"${log_path}" 2>&1 &
  local pid="$!"
  printf '%s\n' "${pid}" >"${LOG_ROOT}/${candidate}.pid"
  printf 'child|%s|%s|%s|%s|%s|%s|running\n' \
    "${pid}" "${fold}" "${candidate}" "${gpu}" "${output_dir}" "${log_path}" >>"${LOG_ROOT}/pids.tsv"
  printf 'running pid=%s gpu=%s started=%s\n' "${pid}" "${gpu}" "$(date -Is)" >"${LOG_ROOT}/status/${candidate}.status"
  PIDS+=("${pid}")
  FOLDS+=("${fold}")
  CANDIDATES+=("${candidate}")
  GPUS+=("${gpu}")
  OUTPUTS+=("${output_dir}")
  LOGS+=("${log_path}")
}

for fold_index in 0 1 2 3 4 5; do
  launch_fold "${fold_index}"
done

exception_fingerprint() {
  local log_path="$1"
  grep -m1 -E 'Traceback|RuntimeError:|ValueError:|AssertionError:|Error:' "${log_path}" 2>/dev/null \
    | sed -E 's/[0-9]+/<n>/g' \
    | cut -c1-240 || true
}

has_protocol_hash_or_overwrite_fault() {
  local log_path="$1"
  grep -Eqi 'protocol.*(violation|error)|hash.*(mismatch|error)|overwrite' "${log_path}" 2>/dev/null
}

terminate_live_run_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}

declare -A FINGERPRINT_COUNT
status=0
systemic_stop=0
for child_index in "${!PIDS[@]}"; do
  pid="${PIDS[child_index]}"
  candidate="${CANDIDATES[child_index]}"
  output_dir="${OUTPUTS[child_index]}"
  log_path="${LOGS[child_index]}"
  if wait "${pid}"; then
    printf 'exit=0 finished=%s\n' "$(date -Is)" >>"${LOG_ROOT}/status/${candidate}.status"
    continue
  fi
  child_status="$?"
  status=1
  printf 'exit=%s finished=%s\n' "${child_status}" "$(date -Is)" >>"${LOG_ROOT}/status/${candidate}.status"
  if [[ ! -f "${output_dir}/final_ssdg.pth" ]]; then
    fingerprint="$(exception_fingerprint "${log_path}")"
    if [[ -n "${fingerprint}" ]]; then
      FINGERPRINT_COUNT["${fingerprint}"]=$(( ${FINGERPRINT_COUNT["${fingerprint}"]:-0} + 1 ))
      if [[ "${FINGERPRINT_COUNT["${fingerprint}"]}" -ge 2 ]]; then
        systemic_stop=1
      fi
    fi
    if has_protocol_hash_or_overwrite_fault "${log_path}"; then
      systemic_stop=1
    fi
  fi
  if [[ "${systemic_stop}" == "1" ]]; then
    printf 'systemic_technical_stop=%s trigger_child=%s\n' "$(date -Is)" "${candidate}" >"${LOG_ROOT}/systemic_stop.status"
    terminate_live_run_children
  fi
done

exit "${status}"
