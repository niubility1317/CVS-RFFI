#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_PREFIX="${SOURCE_RUN_PREFIX:-stage2_short195s3_otherdomains_finaltest_20260630_1525}"
RUN_ID="${RUN_ID:-stage2_short195s3_tailgate_diagnostic_20260630_1605}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
OA_MSE_UNKNOWN_TX_IDS="${OA_MSE_UNKNOWN_TX_IDS:-10-1,10-10}"
DRY_RUN="${DRY_RUN:-0}"

DOMAINS=("3-19" "7-14" "7-7" "8-8")

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

common_eval_args() {
  local feature_npz="$1"
  printf '%q ' \
    --protocol ftrc \
    --feature_npz "${feature_npz}" \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
    --target_old_support_per_tx 10 \
    --target_old_query_per_tx 30 \
    --unknown_tx_ids "${OA_MSE_UNKNOWN_TX_IDS}" \
    --shots 0 \
    --source_proto_per_tx 48 \
    --source_query_per_tx 40 \
    --query_per_tx 30 \
    --openmax_tail_size 20 \
    --openmax_quantile 1.0 \
    --openmax_min_threshold 0.1 \
    --oa_mse_adapter_rank 2 \
    --oa_mse_adapter_kind low_rank \
    --oa_mse_adapter_steps 80 \
    --oa_mse_adapter_selection_policy final \
    --oa_mse_source_anchor_weight 0.05 \
    --oa_mse_source_ce_weight 1.10 \
    --oa_mse_unknown_moat_weight 0.05 \
    --oa_mse_unknown_moat_margin 0.30 \
    --pseudo_unknown_samples_per_pair 4 \
    --pseudo_unknown_offset_scale 0.15 \
    --pseudo_unknown_source_boundary_samples_per_pair 8 \
    --pseudo_unknown_source_boundary_offset_scale 0.18 \
    --pseudo_unknown_target_shift_samples_per_class 4 \
    --pseudo_unknown_target_shift_offset_scale 0.22 \
    --pseudo_unknown_target_halo_samples_per_class 4 \
    --pseudo_unknown_target_halo_offset_scale 0.32 \
    --pseudo_unknown_target_ring_samples_per_class 6 \
    --pseudo_unknown_target_ring_offset_scale 0.38 \
    --oa_mse_old_bridge_weight 0.18 \
    --old_bridge_samples_per_class 4 \
    --old_bridge_max_mix 0.78 \
    --oa_mse_support_contrast_weight 0.05 \
    --oa_mse_support_center_ce_weight 0.18 \
    --support_center_temperature 0.35 \
    --oa_mse_soft_proto_weight 0.12 \
    --soft_proto_topk 3 \
    --soft_proto_temperature 0.20 \
    --oa_mse_soft_proto_boundary_weight 0.05 \
    --soft_proto_boundary_margin 0.10 \
    --old_acc_target 0.80 \
    --seen_new_acc_target 0.75 \
    --seed 362001
}

strict_gate_args() {
  printf '%q ' \
    --unknown_threshold 0.96 \
    --gate_mode oa_mse \
    --min_margin 0.02 \
    --oa_mse_anchor_density_gate \
    --anchor_density_topk 3 \
    --anchor_density_temperature 0.10 \
    --anchor_density_min_quantile 0.10 \
    --anchor_density_margin_quantile 0.10 \
    --anchor_density_gate_action reject \
    --oa_mse_class_envelope_gate \
    --class_envelope_evidence_quantile 0.10 \
    --class_envelope_residual_quantile 0.90 \
    --class_envelope_score_quantile 0.10 \
    --class_envelope_margin_quantile 0.10 \
    --class_envelope_evidence_slack 0.02 \
    --class_envelope_residual_slack 0.02 \
    --class_envelope_score_slack 0.03 \
    --class_envelope_margin_slack 0.02 \
    --class_envelope_min_failures 1 \
    --class_envelope_gate_action reject
}

void_gate_args() {
  printf '%q ' \
    --oa_mse_void_gate \
    --oa_mse_void_gate_min_score 0.55 \
    --oa_mse_void_gate_min_margin 0.02 \
    --oa_mse_old_unknown_acceptance_guard \
    --oa_mse_old_unknown_guard_min_old_support_anchor_margin 0.00 \
    --oa_mse_old_unknown_guard_min_margin 0.02 \
    --oa_mse_old_unknown_guard_min_failures 1
}

run_eval() {
  local domain_label="$1"
  local source_candidate="$2"
  local out_candidate="$3"
  local multiproto="$4"
  local variant="$5"
  local domain_slug="${domain_label//-/_}"
  local source_dir="${RUNS_ROOT}/${SOURCE_RUN_PREFIX}_${domain_slug}/${source_candidate}"
  local feature_npz="${source_dir}/features.npz"
  local out_dir="${RUNS_ROOT}/${RUN_ID}_${domain_slug}/${out_candidate}"
  local log_path="${LOG_ROOT}/${domain_slug}_${out_candidate}.out"
  local multi_args=""
  local common_args
  local strict_args
  local extra_args=""

  if [[ "${multiproto}" == "1" ]]; then
    printf -v multi_args ' %q' \
      --oa_mse_multiproto_score \
      --multiproto_topk 5 \
      --multiproto_temperature 0.12 \
      --multiproto_score_weight 0.65 \
      --oa_mse_mixture_consistency_gate \
      --mixture_consistency_min_cos 0.25 \
      --mixture_consistency_max_residual 1.20 \
      --mixture_consistency_min_margin -0.08 \
      --mixture_consistency_action reject
  fi
  common_args="$(common_eval_args "${feature_npz}")"
  strict_args="$(strict_gate_args)"
  if [[ "${variant}" == "rdm_void" ]]; then
    extra_args="$(void_gate_args)"
  fi

  local cmd="mkdir -p '${out_dir}'; '${PYTHON}' -u '${ROOT}/code/eval_spaceborne_fewshot.py' --output_json '${out_dir}/metrics.json' --manifest_json '${out_dir}/manifest.json' --score_table_csv '${out_dir}/score_table.csv' ${common_args} ${strict_args} ${extra_args} ${multi_args}"
  echo "[TAILGATE-CMD] domain=${domain_label} candidate=${out_candidate} variant=${variant}"
  echo "${cmd}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" bash -lc "${cmd}" > "${log_path}" 2>&1
    echo "[TAILGATE-DONE] domain=${domain_label} candidate=${out_candidate} log=${log_path}"
  fi
}

echo "[TAILGATE] run_id=${RUN_ID} source=${SOURCE_RUN_PREFIX} dry_run=${DRY_RUN}"
for domain in "${DOMAINS[@]}"; do
  run_eval "${domain}" "SHORT195S3_STAGE2B_FINALTEST_BASELINE_K10" "SHORT195S3_TAILGATE_RDM_BASELINE_K10" "0" "rdm"
  run_eval "${domain}" "SHORT195S3_STAGE2B_FINALTEST_MULTIPROTO_K10" "SHORT195S3_TAILGATE_RDM_MULTIPROTO_K10" "1" "rdm"
  run_eval "${domain}" "SHORT195S3_STAGE2B_FINALTEST_BASELINE_K10" "SHORT195S3_TAILGATE_RDMVOID_BASELINE_K10" "0" "rdm_void"
  run_eval "${domain}" "SHORT195S3_STAGE2B_FINALTEST_MULTIPROTO_K10" "SHORT195S3_TAILGATE_RDMVOID_MULTIPROTO_K10" "1" "rdm_void"
done
echo "[TAILGATE-DONE] run_id=${RUN_ID}"
