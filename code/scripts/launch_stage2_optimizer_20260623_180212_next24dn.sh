#!/usr/bin/env bash
# Generated locally for next24dn Phase2 OA-MSE source-boundary repair: Phase1 disabled by user request; 24 Phase2 rows at 3/GPU to improve N607 utilization while staying within control-plane capacity.
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-stage2_spaceborne_next24dn_20260623_180212}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-3}"
PHASE1_MAX_ACTIVE_PER_GPU="${PHASE1_MAX_ACTIVE_PER_GPU:-1}"
PHASE2_MAX_ACTIVE_PER_GPU="${PHASE2_MAX_ACTIVE_PER_GPU:-3}"
COMBINED_MAX_ACTIVE_PER_GPU="${COMBINED_MAX_ACTIVE_PER_GPU:-4}"
STAGE2_MAX_SCHEDULER_SECONDS="${STAGE2_MAX_SCHEDULER_SECONDS:-5400}"
STAGE2_EXPECTED_CANDIDATE_MAX_SECONDS="${STAGE2_EXPECTED_CANDIDATE_MAX_SECONDS:-1200}"
PHASE1_LANE_ACTIVE="${PHASE1_LANE_ACTIVE:-1}"
PHASE2_LANE_ACTIVE="${PHASE2_LANE_ACTIVE:-0}"
PHASE2_LOCAL_PATCH_REQUIRED="${PHASE2_LOCAL_PATCH_REQUIRED:-0}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

SCHED_LOG="${LOG_ROOT}/scheduler.out"
EVENTS_TSV="${LOG_ROOT}/scheduler_events.tsv"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  : > "${SCHED_LOG}"
  : > "${EVENTS_TSV}"
fi

log_msg() { if [[ "${DRY_RUN}" == "1" ]]; then echo "$@"; else echo "$@" | tee -a "${SCHED_LOG}"; fi; }
event_row() { local row; row="$(printf "%s	%s	%s	%s	%s	%s" "$(date -Is)" "$1" "$2" "$3" "$4" "$5")"; if [[ "${DRY_RUN}" == "1" ]]; then echo "${row}"; else echo "${row}" | tee -a "${EVENTS_TSV}" | tee -a "${SCHED_LOG}"; fi; }

declare -a CAND_ID=(S2N133_GPU0_A_OA_MSE_7X14_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K1_RADIUS_D S2N133_GPU0_B_OA_MSE_20X1_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K2_REGISTRATION_G S2N133_GPU0_C_OA_MSE_20X1_MSE_SUBSPACE_SOURCE_BOUNDARY_K5_SB6_O0p18 S2N133_GPU1_A_OA_MSE_8X8_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K2_RADIUS_E S2N133_GPU1_B_OA_MSE_7X14_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K5_QUALITY_DEFER_H S2N133_GPU1_C_OA_MSE_7X14_MSE_SUBSPACE_SOURCE_BOUNDARY_K2_SB4_O0p14 S2N133_GPU2_A_OA_MSE_3X19_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K5_DENSITY_F S2N133_GPU2_B_OA_MSE_7X14_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K2_REGISTRATION_G S2N133_GPU2_C_OA_MSE_3X19_MSE_SUBSPACE_SOURCE_BOUNDARY_K5_SB6_O0p16 S2N133_GPU3_A_OA_MSE_20X1_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K1_RADIUS_D S2N133_GPU3_B_OA_MSE_8X8_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K5_QUALITY_DEFER_H S2N133_GPU3_C_OA_MSE_8X8_MSE_SUBSPACE_SOURCE_BOUNDARY_K2_SB4_O0p2 S2N133_GPU4_A_OA_MSE_7X14_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K2_RADIUS_E S2N133_GPU4_B_OA_MSE_8X8_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K2_REGISTRATION_G S2N133_GPU4_C_OA_MSE_7X7_OA_MSE_HEAD_SOURCE_BOUNDARY_K5_SB5_O0p18 S2N133_GPU5_A_OA_MSE_8X8_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K5_DENSITY_F S2N133_GPU5_B_OA_MSE_3X19_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K5_QUALITY_DEFER_H S2N133_GPU5_C_OA_MSE_20X1_OA_MSE_HEAD_SOURCE_BOUNDARY_K2_SB4_O0p22 S2N133_GPU6_A_OA_MSE_20X1_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K5_DENSITY_F S2N133_GPU6_B_OA_MSE_3X19_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K2_REGISTRATION_G S2N133_GPU6_C_OA_MSE_3X19_OA_MSE_HEAD_SOURCE_BOUNDARY_K2_SB5_O0p2 S2N133_GPU7_A_OA_MSE_20X1_MSE_SUBSPACE_OA_MSE_TARGET_OLD_K2_RADIUS_E S2N133_GPU7_B_OA_MSE_7X7_OA_MSE_HEAD_OA_MSE_SEEN_NEW_K5_QUALITY_DEFER_H S2N133_GPU7_C_OA_MSE_7X14_OA_MSE_HEAD_SOURCE_BOUNDARY_K5_SB6_O0p16)
declare -a CAND_GPU=(0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7)
declare -a CAND_KIND=(oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse oa_mse)
declare -a CAND_SLOT=(GPU0/A GPU0/B GPU0/C GPU1/A GPU1/B GPU1/C GPU2/A GPU2/B GPU2/C GPU3/A GPU3/B GPU3/C GPU4/A GPU4/B GPU4/C GPU5/A GPU5/B GPU5/C GPU6/A GPU6/B GPU6/C GPU7/A GPU7/B GPU7/C)
declare -a CAND_LANE=(phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2 phase2)
declare -a ROW_STAGE=(mse_subspace oa_mse_head mse_subspace mse_subspace oa_mse_head mse_subspace mse_subspace oa_mse_head mse_subspace mse_subspace oa_mse_head mse_subspace mse_subspace oa_mse_head oa_mse_head mse_subspace oa_mse_head oa_mse_head mse_subspace oa_mse_head oa_mse_head mse_subspace oa_mse_head oa_mse_head)
declare -a ROW_PROTOCOL=(ftrc sfe ftrc ftrc sfe ftrc ftrc sfe ftrc ftrc sfe ftrc ftrc sfe sfe ftrc sfe sfe ftrc sfe sfe ftrc sfe sfe)
declare -a ROW_TARGET_RX=(7-14 20-1 20-1 8-8 7-14 7-14 3-19 7-14 3-19 20-1 8-8 8-8 7-14 8-8 7-7 8-8 3-19 20-1 20-1 3-19 3-19 20-1 7-7 7-14)
declare -a ROW_NEW_TX=(1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16 1-14,1-16)
declare -a ROW_UNKNOWN_TX=(1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12 1-10,1-12)
declare -a ROW_K_OLD=(1 2 5 2 5 2 5 2 5 1 5 2 2 2 5 5 5 2 5 2 2 2 5 5)
declare -a ROW_K_NEW=(0 2 0 0 5 0 0 2 0 0 5 0 0 2 5 0 5 2 0 2 2 0 5 5)
declare -a ROW_SEED=(97103 97106 97736 97204 97207 97135 97305 97306 97338 97403 97407 97238 97504 97506 97842 97605 97607 97142 97705 97706 97743 97804 97807 97245)
declare -a ROW_UNKNOWN_THRESHOLD=(0.9 0.96 0.94 0.92 0.98 0.9 0.94 0.96 0.94 0.9 0.98 0.92 0.92 0.96 0.98 0.94 0.98 0.96 0.94 0.96 0.96 0.92 0.98 0.98)
declare -a ROW_OPENMAX_QUANTILE=(0.995 1.0 0.985 0.99 1.0 0.995 0.985 1.0 0.985 0.995 1.0 0.99 0.99 1.0 1.0 0.985 1.0 1.0 0.985 1.0 1.0 0.99 1.0 1.0)
declare -a ROW_OPENMAX_MIN_THRESHOLD=(0.06 0.1 0.1 0.08 0.12 0.06 0.1 0.1 0.1 0.06 0.12 0.08 0.08 0.1 0.12 0.1 0.12 0.1 0.1 0.1 0.1 0.08 0.12 0.12)
declare -a ROW_ADAPTER_RANK=(2 1 2 2 1 2 2 1 2 2 1 2 2 1 1 2 1 1 2 1 1 2 1 1)
declare -a ROW_ADAPTER_STEPS=(20 20 40 30 20 40 40 20 40 20 20 40 30 20 30 40 20 30 40 20 30 30 20 30)
declare -a ROW_ADAPTER_LR=(0.03 0.02 0.02 0.03 0.02 0.025 0.02 0.02 0.02 0.03 0.02 0.025 0.03 0.02 0.02 0.02 0.02 0.025 0.02 0.02 0.025 0.03 0.02 0.02)
declare -a ROW_KAPPA=(1.0 2.0 6.0 3.0 5.0 4.0 6.0 2.0 6.0 1.0 5.0 4.0 3.0 2.0 6.0 6.0 5.0 4.0 6.0 2.0 4.0 3.0 5.0 6.0)
declare -a ROW_SOURCE_ANCHOR_WEIGHT=(0.05 0.08 0.12 0.05 0.08 0.12 0.05 0.1 0.1 0.1 0.1 0.12 0.1 0.08 0.1 0.1 0.08 0.1 0.05 0.1 0.1 0.05 0.1 0.12)
declare -a ROW_SOURCE_CE_WEIGHT=(0.2 0.18 0.26 0.2 0.18 0.24 0.2 0.15 0.26 0.15 0.15 0.22 0.15 0.18 0.22 0.15 0.18 0.22 0.2 0.15 0.24 0.12 0.15 0.24)
declare -a ROW_UNKNOWN_MOAT_WEIGHT=(0.16 0.12 0.24 0.16 0.12 0.22 0.16 0.1 0.24 0.1 0.1 0.22 0.1 0.12 0.2 0.1 0.12 0.2 0.16 0.1 0.22 0.08 0.1 0.22)
declare -a ROW_UNKNOWN_MOAT_MARGIN=(0.55 0.45 0.62 0.55 0.45 0.6 0.55 0.4 0.62 0.4 0.4 0.58 0.4 0.45 0.55 0.4 0.45 0.55 0.55 0.4 0.58 0.5 0.4 0.58)
declare -a ROW_PSEUDO_UNKNOWN_SAMPLES_PER_PAIR=(8 6 8 8 6 8 8 4 8 4 4 8 4 6 6 4 6 6 8 4 4 4 4 6)
declare -a ROW_PSEUDO_UNKNOWN_OFFSET_SCALE=(0.12 0.18 0.12 0.12 0.18 0.12 0.12 0.22 0.12 0.22 0.22 0.12 0.22 0.18 0.18 0.22 0.18 0.18 0.12 0.22 0.18 0.15 0.22 0.18)
declare -a ROW_PSEUDO_UNKNOWN_SOURCE_BOUNDARY_SAMPLES_PER_PAIR=(2 3 6 2 3 4 4 3 6 2 3 4 2 3 5 4 3 4 4 3 5 2 3 6)
declare -a ROW_PSEUDO_UNKNOWN_SOURCE_BOUNDARY_OFFSET_SCALE=(0.18 0.18 0.18 0.2 0.2 0.14 0.16 0.22 0.16 0.24 0.2 0.2 0.18 0.18 0.18 0.16 0.2 0.22 0.18 0.22 0.2 0.2 0.2 0.16)
declare -a ROW_PSEUDO_UNKNOWN_TARGET_SHIFT_SAMPLES_PER_CLASS=(2 4 6 4 6 4 6 4 6 2 6 4 4 4 6 6 6 4 6 4 4 4 6 6)
declare -a ROW_PSEUDO_UNKNOWN_TARGET_SHIFT_OFFSET_SCALE=(0.24 0.2 0.24 0.24 0.2 0.24 0.24 0.28 0.24 0.28 0.28 0.24 0.28 0.2 0.28 0.28 0.2 0.22 0.24 0.28 0.28 0.16 0.28 0.22)
declare -a ROW_PSEUDO_UNKNOWN_TARGET_HALO_SAMPLES_PER_CLASS=(2 4 6 4 6 4 6 4 6 2 6 4 4 4 6 6 6 4 6 4 4 4 6 6)
declare -a ROW_PSEUDO_UNKNOWN_TARGET_HALO_OFFSET_SCALE=(0.3 0.35 0.3 0.4 0.32 0.3 0.3 0.35 0.3 0.4 0.32 0.4 0.3 0.35 0.32 0.4 0.32 0.35 0.3 0.35 0.35 0.4 0.32 0.32)
declare -a ROW_PSEUDO_UNKNOWN_TARGET_RING_SAMPLES_PER_CLASS=(2 4 6 4 6 4 6 4 6 2 6 4 4 4 6 6 6 4 6 4 4 4 6 6)
declare -a ROW_PSEUDO_UNKNOWN_TARGET_RING_OFFSET_SCALE=(0.35 0.45 0.35 0.55 0.4 0.35 0.35 0.45 0.35 0.55 0.4 0.55 0.35 0.45 0.4 0.55 0.4 0.45 0.35 0.45 0.45 0.55 0.4 0.4)
declare -a ROW_SUPPORT_CONTRAST_WEIGHT=(0.08 0.09999999999999999 0.16 0.12 0.14 0.08 0.16 0.09999999999999999 0.16 0.08 0.14 0.12 0.12 0.09999999999999999 0.14 0.16 0.14 0.09999999999999999 0.16 0.09999999999999999 0.09999999999999999 0.12 0.14 0.14)
declare -a ROW_SUPPORT_CONTRAST_NEGATIVE_MARGIN=(0.74 0.78 0.74 0.82 0.76 0.74 0.74 0.78 0.74 0.82 0.76 0.82 0.74 0.78 0.76 0.82 0.76 0.78 0.74 0.78 0.78 0.82 0.76 0.76)
declare -a ROW_SUPPORT_CONTRAST_POSITIVE_MARGIN=(0.86 0.88 0.86 0.9 0.87 0.86 0.86 0.88 0.86 0.9 0.87 0.9 0.86 0.88 0.87 0.9 0.87 0.88 0.86 0.88 0.88 0.9 0.87 0.87)
declare -a ROW_SOFT_PROTO_WEIGHT=(0.1 0.07 0.1 0.1 0.07 0.1 0.1 0.07 0.1 0.1 0.07 0.1 0.1 0.07 0.08 0.1 0.07 0.08 0.1 0.07 0.08 0.1 0.07 0.08)
declare -a ROW_SOFT_PROTO_TOPK=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
declare -a ROW_SOFT_PROTO_TEMPERATURE=(0.1 0.12 0.1 0.1 0.12 0.1 0.1 0.12 0.1 0.1 0.12 0.1 0.1 0.12 0.12 0.1 0.12 0.12 0.1 0.12 0.12 0.1 0.12 0.12)
declare -a ROW_OLD_BRIDGE_WEIGHT=(0.1 0.12 0.14 0.12 0.14 0.1 0.14 0.12 0.14 0.1 0.14 0.12 0.12 0.12 0.14 0.14 0.14 0.12 0.14 0.12 0.12 0.12 0.14 0.14)
declare -a ROW_OLD_BRIDGE_SAMPLES_PER_CLASS=(2 3 4 3 4 2 4 3 4 2 4 3 3 3 4 4 4 3 4 3 3 3 4 4)
declare -a ROW_OLD_BRIDGE_MAX_MIX=(0.75 0.82 0.75 0.88 0.8 0.75 0.75 0.82 0.75 0.88 0.8 0.88 0.75 0.82 0.8 0.88 0.8 0.82 0.75 0.82 0.82 0.88 0.8 0.8)
declare -a ROW_OLD_NEIGHBORHOOD_WEIGHT=(0.15 0.2 0.18 0.15 0.2 0.18 0.15 0.05 0.2 0.05 0.05 0.16 0.05 0.2 0.12 0.05 0.2 0.12 0.15 0.05 0.14 0.1 0.05 0.14)
declare -a ROW_OLD_NEIGHBORHOOD_SAMPLES_PER_CLASS=(4 3 4 4 3 4 4 2 4 2 2 4 2 3 2 2 3 3 4 2 2 2 2 3)
declare -a ROW_OLD_NEIGHBORHOOD_RADIUS=(0.04 0.08 0.04 0.04 0.08 0.04 0.04 0.1 0.04 0.1 0.1 0.04 0.1 0.08 0.1 0.1 0.08 0.08 0.04 0.1 0.1 0.06 0.1 0.08)
declare -a ROW_OLD_RETENTION_QUANTILE=(0.9 0.9 0.88 0.9 0.88 0.9 0.88 0.9 0.88 0.9 0.88 0.9 0.9 0.9 0.88 0.88 0.88 0.9 0.88 0.9 0.9 0.9 0.88 0.88)
declare -a ROW_OLD_SURROGATE_EVIDENCE_MARGIN=(0.03 0.04 0.07 0.05 0.060000000000000005 0.03 0.07 0.04 0.07 0.03 0.060000000000000005 0.05 0.05 0.04 0.060000000000000005 0.07 0.060000000000000005 0.04 0.07 0.04 0.04 0.05 0.060000000000000005 0.060000000000000005)
declare -a ROW_OLD_ANCHOR_OVERRIDE_MIN_QUALITY=(0.6 0.55 0.6 0.6 0.55 0.6 0.6 0.7 0.6 0.7 0.7 0.6 0.7 0.55 0.7 0.7 0.55 0.55 0.6 0.7 0.7 0.5 0.7 0.55)
declare -a ROW_OLD_SURROGATE_MARGIN=(0.1 0.08 0.1 0.1 0.08 0.1 0.1 0.12 0.1 0.12 0.12 0.1 0.12 0.08 0.12 0.12 0.08 0.08 0.1 0.12 0.12 0.05 0.12 0.08)
declare -a ROW_OLD_SURROGATE_MARGIN_WEIGHT=(0.08 0.05 0.1 0.08 0.05 0.1 0.08 0.1 0.12 0.1 0.1 0.1 0.1 0.05 0.08 0.1 0.05 0.08 0.08 0.1 0.08 0.03 0.1 0.08)
declare -a CAND_STATUS=()
declare -a CAND_PID=()
for _ in "${CAND_ID[@]}"; do CAND_STATUS+=("queued"); CAND_PID+=(""); done
run_phase1_safe_ssdg_candidate() {
  local i="$1" cid gpu seed out_dir
  cid="${CAND_ID[$i]}"; gpu="${CAND_GPU[$i]}"; seed="${ROW_SEED[$i]}"
  out_dir="${RUNS_ROOT}/${cid}"
  mkdir -p "${out_dir}"
  echo "[S1-SAFE-SSDG-BEGIN] cid=${cid} gpu=${gpu} split=tx_rx_day_1_7_2 ratios=0.1L/0.7U/0.2Val seed=${seed}"
  env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m SSDG.train_ssdg \
    --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2 \
    --labeled_ratio 0.10 --unlabeled_ratio 0.70 --source_val_ratio 0.20 \
    --output_dir "${out_dir}" --epochs 200 --from_scratch true \
    --use_sat_consistency --sat_train_scenario mixed_orbit \
    --sat_cons_start_epoch 60 --lambda_sat_cls 1.0 --lambda_sat_cons 0.03 \
    --eval_sat_channel true --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_eval_max_batches -1 \
    --device cuda:0 --seed "${seed}"
  echo "[S1-SAFE-SSDG-END] cid=${cid}"
}

run_phase2_oa_mse_candidate() {
  local i="$1" cid gpu protocol rx new_tx unknown_tx k_old k_new seed unknown_threshold openmax_quantile openmax_min_threshold adapter_rank adapter_steps adapter_lr kappa out_dir query_per_tx max_samples
  cid="${CAND_ID[$i]}"; gpu="${CAND_GPU[$i]}"; protocol="${ROW_PROTOCOL[$i]}"; rx="${ROW_TARGET_RX[$i]}"; new_tx="${ROW_NEW_TX[$i]}"; unknown_tx="${ROW_UNKNOWN_TX[$i]}"; k_old="${ROW_K_OLD[$i]}"; k_new="${ROW_K_NEW[$i]}"; seed="${ROW_SEED[$i]}"; unknown_threshold="${ROW_UNKNOWN_THRESHOLD[$i]}"; openmax_quantile="${ROW_OPENMAX_QUANTILE[$i]}"; openmax_min_threshold="${ROW_OPENMAX_MIN_THRESHOLD[$i]}"; adapter_rank="${ROW_ADAPTER_RANK[$i]}"; adapter_steps="${ROW_ADAPTER_STEPS[$i]}"; adapter_lr="${ROW_ADAPTER_LR[$i]}"; kappa="${ROW_KAPPA[$i]}"; source_anchor_weight="${ROW_SOURCE_ANCHOR_WEIGHT[$i]}"; source_ce_weight="${ROW_SOURCE_CE_WEIGHT[$i]}"; unknown_moat_weight="${ROW_UNKNOWN_MOAT_WEIGHT[$i]}"; unknown_moat_margin="${ROW_UNKNOWN_MOAT_MARGIN[$i]}"; pseudo_unknown_samples="${ROW_PSEUDO_UNKNOWN_SAMPLES_PER_PAIR[$i]}"; pseudo_unknown_offset="${ROW_PSEUDO_UNKNOWN_OFFSET_SCALE[$i]}"; pseudo_unknown_source_boundary_samples="${ROW_PSEUDO_UNKNOWN_SOURCE_BOUNDARY_SAMPLES_PER_PAIR[$i]}"; pseudo_unknown_source_boundary_offset="${ROW_PSEUDO_UNKNOWN_SOURCE_BOUNDARY_OFFSET_SCALE[$i]}"; pseudo_unknown_target_shift_samples="${ROW_PSEUDO_UNKNOWN_TARGET_SHIFT_SAMPLES_PER_CLASS[$i]}"; pseudo_unknown_target_shift_offset="${ROW_PSEUDO_UNKNOWN_TARGET_SHIFT_OFFSET_SCALE[$i]}"; pseudo_unknown_target_halo_samples="${ROW_PSEUDO_UNKNOWN_TARGET_HALO_SAMPLES_PER_CLASS[$i]}"; pseudo_unknown_target_halo_offset="${ROW_PSEUDO_UNKNOWN_TARGET_HALO_OFFSET_SCALE[$i]}"; old_bridge_weight="${ROW_OLD_BRIDGE_WEIGHT[$i]}"; old_bridge_samples="${ROW_OLD_BRIDGE_SAMPLES_PER_CLASS[$i]}"; old_bridge_max_mix="${ROW_OLD_BRIDGE_MAX_MIX[$i]}"; pseudo_unknown_target_ring_samples="${ROW_PSEUDO_UNKNOWN_TARGET_RING_SAMPLES_PER_CLASS[$i]}"; pseudo_unknown_target_ring_offset="${ROW_PSEUDO_UNKNOWN_TARGET_RING_OFFSET_SCALE[$i]}"; support_contrast_weight="${ROW_SUPPORT_CONTRAST_WEIGHT[$i]}"; support_contrast_negative_margin="${ROW_SUPPORT_CONTRAST_NEGATIVE_MARGIN[$i]}"; support_contrast_positive_margin="${ROW_SUPPORT_CONTRAST_POSITIVE_MARGIN[$i]}"; soft_proto_weight="${ROW_SOFT_PROTO_WEIGHT[$i]}"; soft_proto_topk="${ROW_SOFT_PROTO_TOPK[$i]}"; soft_proto_temperature="${ROW_SOFT_PROTO_TEMPERATURE[$i]}"; old_neighborhood_weight="${ROW_OLD_NEIGHBORHOOD_WEIGHT[$i]}"; old_neighborhood_samples="${ROW_OLD_NEIGHBORHOOD_SAMPLES_PER_CLASS[$i]}"; old_neighborhood_radius="${ROW_OLD_NEIGHBORHOOD_RADIUS[$i]}"; old_retention_quantile="${ROW_OLD_RETENTION_QUANTILE[$i]}"; old_surrogate_margin_weight="${ROW_OLD_SURROGATE_MARGIN_WEIGHT[$i]}"; old_surrogate_margin="${ROW_OLD_SURROGATE_MARGIN[$i]}"; old_surrogate_evidence_margin="${ROW_OLD_SURROGATE_EVIDENCE_MARGIN[$i]}"; old_anchor_override_min_quality="${ROW_OLD_ANCHOR_OVERRIDE_MIN_QUALITY[$i]}"
  out_dir="${RUNS_ROOT}/${cid}"
  query_per_tx=50
  max_samples=200
  mkdir -p "${out_dir}"
  echo "[S2-OA-MSE-BEGIN] cid=${cid} protocol=${protocol} rx=${rx} new=${new_tx} unknown=${unknown_tx} k_old=${k_old} k_new=${k_new} seed=${seed} threshold=${unknown_threshold} openmax_q=${openmax_quantile} openmax_min=${openmax_min_threshold} adapter_rank=${adapter_rank} adapter_steps=${adapter_steps} adapter_lr=${adapter_lr} kappa=${kappa} old_neighbor=${old_neighborhood_weight} old_ret_q=${old_retention_quantile} old_surr_w=${old_surrogate_margin_weight} old_surr=${old_surrogate_margin} old_surr_ev=${old_surrogate_evidence_margin} old_anchor_q=${old_anchor_override_min_quality} source_boundary=${pseudo_unknown_source_boundary_samples}@${pseudo_unknown_source_boundary_offset} target_shift=${pseudo_unknown_target_shift_samples}@${pseudo_unknown_target_shift_offset} halo=${pseudo_unknown_target_halo_samples}@${pseudo_unknown_target_halo_offset} ring=${pseudo_unknown_target_ring_samples}@${pseudo_unknown_target_ring_offset} contrast=${support_contrast_weight}/${support_contrast_negative_margin}/${support_contrast_positive_margin} soft_proto=${soft_proto_weight}/${soft_proto_topk}/${soft_proto_temperature} bridge=${old_bridge_samples}@${old_bridge_weight}/${old_bridge_max_mix}"
  env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py"     --ckpt "${TEACHER_CKPT}" --wisig_pkl "${WISIG_PKL}" --new_wisig_pkl "${NEW_WISIG_PKL}"     --out_npz "${out_dir}/features.npz" --feature_name z_id     --source_tx_ids "${SOURCE_TX_IDS}" --source_rxs "${CEN51_TRAIN_RXS}"     --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --target_old_rxs "${rx}" --target_old_channel_view satellite     --target_old_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --target_old_sat_seed "$((seed + 111))"     --new_tx_ids "${new_tx}" --new_rxs "${rx}" --unknown_tx_ids "${unknown_tx}"     --target_new_channel_view satellite --target_new_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --target_new_sat_seed "$((seed + 222))"     --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256 --max_samples_per_combo 0 --max_samples_per_tx "${max_samples}" --batch_size 512 --device cuda:0 --seed "${seed}"
  local eval_cmd=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/eval_spaceborne_fewshot.py"     --protocol "${protocol}" --feature_npz "${out_dir}/features.npz" --output_json "${out_dir}/metrics.json" --manifest_json "${out_dir}/manifest.json" --score_table_csv "${out_dir}/score_table.csv"     --source_tx_ids "${SOURCE_TX_IDS}" --target_old_tx_ids "${TARGET_OLD_TX_IDS}" --new_tx_ids "${new_tx}" --unknown_tx_ids "${unknown_tx}"     --target_old_support_per_tx "${k_old}" --target_old_query_per_tx "${query_per_tx}" --shots "${k_new}" --source_proto_per_tx 20 --source_query_per_tx 20 --query_per_tx "${query_per_tx}"     --unknown_threshold "${unknown_threshold}" --gate_mode oa_mse --openmax_tail_size 20 --openmax_quantile "${openmax_quantile}" --openmax_min_threshold "${openmax_min_threshold}"     --oa_mse_adapter_rank "${adapter_rank}" --oa_mse_adapter_steps "${adapter_steps}" --oa_mse_adapter_lr "${adapter_lr}" --kappa "${kappa}" --oa_mse_source_anchor_weight "${source_anchor_weight}" --oa_mse_source_ce_weight "${source_ce_weight}" --oa_mse_unknown_moat_weight "${unknown_moat_weight}" --oa_mse_unknown_moat_margin "${unknown_moat_margin}" --pseudo_unknown_samples_per_pair "${pseudo_unknown_samples}" --pseudo_unknown_offset_scale "${pseudo_unknown_offset}" --pseudo_unknown_source_boundary_samples_per_pair "${pseudo_unknown_source_boundary_samples}" --pseudo_unknown_source_boundary_offset_scale "${pseudo_unknown_source_boundary_offset}" --pseudo_unknown_target_shift_samples_per_class "${pseudo_unknown_target_shift_samples}" --pseudo_unknown_target_shift_offset_scale "${pseudo_unknown_target_shift_offset}" --pseudo_unknown_target_halo_samples_per_class "${pseudo_unknown_target_halo_samples}" --pseudo_unknown_target_halo_offset_scale "${pseudo_unknown_target_halo_offset}" --oa_mse_old_bridge_weight "${old_bridge_weight}" --old_bridge_samples_per_class "${old_bridge_samples}" --old_bridge_max_mix "${old_bridge_max_mix}" --pseudo_unknown_target_ring_samples_per_class "${pseudo_unknown_target_ring_samples}" --pseudo_unknown_target_ring_offset_scale "${pseudo_unknown_target_ring_offset}" --oa_mse_support_contrast_weight "${support_contrast_weight}" --old_support_contrast_negative_margin "${support_contrast_negative_margin}" --old_support_contrast_positive_margin "${support_contrast_positive_margin}" --oa_mse_soft_proto_weight "${soft_proto_weight}" --soft_proto_topk "${soft_proto_topk}" --soft_proto_temperature "${soft_proto_temperature}" --oa_mse_old_neighborhood_weight "${old_neighborhood_weight}" --old_neighborhood_samples_per_class "${old_neighborhood_samples}" --old_neighborhood_radius "${old_neighborhood_radius}" --oa_mse_old_surrogate_margin_weight "${old_surrogate_margin_weight}" --old_surrogate_margin "${old_surrogate_margin}" --old_surrogate_evidence_margin "${old_surrogate_evidence_margin}" --old_anchor_override_min_quality "${old_anchor_override_min_quality}" --old_retention_quantile "${old_retention_quantile}" --old_acc_target 0.9 --seen_new_acc_target 0.75 --seed "${seed}")
  "${eval_cmd[@]}"
  echo "[S2-OA-MSE-END] cid=${cid}"
}

launch_candidate() {
  local i="$1" cid gpu kind lane log_path pid
  cid="${CAND_ID[$i]}"; gpu="${CAND_GPU[$i]}"; kind="${CAND_KIND[$i]}"; lane="${CAND_LANE[$i]:-}"
  if [[ -z "${lane}" ]]; then
    if [[ "${kind}" == "safe_ssdg" || "${kind}" == "safe_ssdg_cvs_r01" || "${kind}" == "meta_ssl" ]]; then lane="phase1"; else lane="phase2"; fi
  fi
  log_path="${LOG_ROOT}/${cid}.out"
  if [[ "${lane}" == "phase1" && "${kind}" != "safe_ssdg" && "${kind}" != "safe_ssdg_cvs_r01" ]]; then
    CAND_STATUS[$i]="failed_unsupported_phase1_kind"
    event_row "${cid}" "FAILED_LOCAL_SCHEMA" "gpu=${gpu}" "lane=phase1" "reason=unsupported_phase1_kind:${kind}"
    return 70
  fi
  if [[ "${lane}" == "phase1" && "${PHASE1_LANE_ACTIVE}" != "0" ]]; then
    CAND_STATUS[$i]="deferred_phase1_active"
    event_row "${cid}" "DEFERRED_RETRY_CAPACITY" "gpu=${gpu}" "lane=phase1" "reason=phase1_lane_active"
    return 0
  fi
  if [[ "${lane}" == "phase2" && "${PHASE2_LANE_ACTIVE}" != "0" ]]; then
    CAND_STATUS[$i]="deferred_phase2_active"
    event_row "${cid}" "DEFERRED_RETRY_CAPACITY" "gpu=${gpu}" "lane=phase2" "reason=phase2_lane_active"
    return 0
  fi
  if [[ "${lane}" == "phase2" && "${PHASE2_LOCAL_PATCH_REQUIRED}" == "1" ]]; then
    CAND_STATUS[$i]="deferred_phase2_local_verify"
    event_row "${cid}" "DEFERRED_RETRY_LOCAL_VERIFY" "gpu=${gpu}" "lane=phase2" "reason=phase2_local_patch_required"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${lane}" == "phase1" ]]; then
      echo "[S1-DRY-RUN] cid=${cid} slot=${CAND_SLOT[$i]} gpu=${gpu} kind=${kind} lane=${lane} entrypoint=run_phase1_safe_ssdg_candidate module=SSDG.train_ssdg split=tx_rx_day_1_7_2 ratios=0.1L/0.7U/0.2Val seed=${ROW_SEED[$i]} log=${log_path}"
    else
      echo "[S2-DRY-RUN] cid=${cid} slot=${CAND_SLOT[$i]} gpu=${gpu} kind=${kind} lane=${lane} protocol=${ROW_PROTOCOL[$i]} rx=${ROW_TARGET_RX[$i]} new=${ROW_NEW_TX[$i]} unknown=${ROW_UNKNOWN_TX[$i]} threshold=${ROW_UNKNOWN_THRESHOLD[$i]} openmax_q=${ROW_OPENMAX_QUANTILE[$i]} openmax_min=${ROW_OPENMAX_MIN_THRESHOLD[$i]} adapter_rank=${ROW_ADAPTER_RANK[$i]} adapter_steps=${ROW_ADAPTER_STEPS[$i]} kappa=${ROW_KAPPA[$i]} source_ce=${ROW_SOURCE_CE_WEIGHT[$i]} unknown_moat=${ROW_UNKNOWN_MOAT_WEIGHT[$i]} unknown_margin=${ROW_UNKNOWN_MOAT_MARGIN[$i]} old_neighbor=${ROW_OLD_NEIGHBORHOOD_WEIGHT[$i]} old_ret_q=${ROW_OLD_RETENTION_QUANTILE[$i]} old_surr_w=${ROW_OLD_SURROGATE_MARGIN_WEIGHT[$i]} old_surr_ev=${ROW_OLD_SURROGATE_EVIDENCE_MARGIN[$i]} old_anchor_q=${ROW_OLD_ANCHOR_OVERRIDE_MIN_QUALITY[$i]} source_boundary=${ROW_PSEUDO_UNKNOWN_SOURCE_BOUNDARY_SAMPLES_PER_PAIR[$i]}@${ROW_PSEUDO_UNKNOWN_SOURCE_BOUNDARY_OFFSET_SCALE[$i]} target_shift=${ROW_PSEUDO_UNKNOWN_TARGET_SHIFT_SAMPLES_PER_CLASS[$i]}@${ROW_PSEUDO_UNKNOWN_TARGET_SHIFT_OFFSET_SCALE[$i]} halo=${ROW_PSEUDO_UNKNOWN_TARGET_HALO_SAMPLES_PER_CLASS[$i]}@${ROW_PSEUDO_UNKNOWN_TARGET_HALO_OFFSET_SCALE[$i]} ring=${ROW_PSEUDO_UNKNOWN_TARGET_RING_SAMPLES_PER_CLASS[$i]}@${ROW_PSEUDO_UNKNOWN_TARGET_RING_OFFSET_SCALE[$i]} contrast=${ROW_SUPPORT_CONTRAST_WEIGHT[$i]}/${ROW_SUPPORT_CONTRAST_NEGATIVE_MARGIN[$i]}/${ROW_SUPPORT_CONTRAST_POSITIVE_MARGIN[$i]} soft_proto=${ROW_SOFT_PROTO_WEIGHT[$i]}/${ROW_SOFT_PROTO_TOPK[$i]}/${ROW_SOFT_PROTO_TEMPERATURE[$i]} bridge=${ROW_OLD_BRIDGE_SAMPLES_PER_CLASS[$i]}@${ROW_OLD_BRIDGE_WEIGHT[$i]}/${ROW_OLD_BRIDGE_MAX_MIX[$i]} log=${log_path}"
    fi
    CAND_STATUS[$i]="dry_run"
    return 0
  fi
  if [[ "${kind}" == "safe_ssdg" || "${kind}" == "safe_ssdg_cvs_r01" ]]; then
    (run_phase1_safe_ssdg_candidate "${i}" > "${log_path}" 2>&1) &
  else
    (run_phase2_oa_mse_candidate "${i}" > "${log_path}" 2>&1) &
  fi
  pid="$!"
  CAND_PID[$i]="${pid}"
  CAND_STATUS[$i]="running"
  event_row "${cid}" "LAUNCHED" "gpu=${gpu}" "lane=${lane}" "pid=${pid};log=${log_path}"
}

log_msg "[S2-SCHEDULER] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CAND_ID[@]} phase1=disabled_by_user_request phase2=resolved_manytx_oa_mse_source_distribution_boundary_v1_next24_phase2_only"
for i in "${!CAND_ID[@]}"; do
  log_msg "[S2-CANDIDATE] idx=${i} id=${CAND_ID[$i]} slot=${CAND_SLOT[$i]} gpu=${CAND_GPU[$i]} kind=${CAND_KIND[$i]} lane=${CAND_LANE[$i]} rx=${ROW_TARGET_RX[$i]} new=${ROW_NEW_TX[$i]} unknown=${ROW_UNKNOWN_TX[$i]}"
done

source "${ROOT}/tools/stage2_queue_runner_template.sh"
stage2_run_queue_scheduler
