#!/usr/bin/env bash
set -euo pipefail

# First-batch union matrix for FSDG49 + CEN_A31 + VMB, created 2026-06-01.
# Hard contract, kept in plain text for audits:
# --wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver
#
# Mechanism boundary:
# - GPU0-GPU4 preserve FSDG49's confirmed receiver-client FedProx + receiver_agnostic_bex02 rx_adv core.
#   GPU0-GPU1 keep the historical CVS satellite path; GPU2-GPU4 deliberately replace that path with
#   CEN_A31-style SAT CE-only controls, so they are not claimed as full historical FSDG49 replicas.
# - GPU5-GPU7 are explicitly named VMB audit arms; they are not FSDG49 core replacements.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-20260601_163012_fsdg49_cen_a31_vmb_union}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_INDEX="${ONLY_INDEX:-}"
ONLY_RUN="${ONLY_RUN:-}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-2}"
ALLOW_DUPLICATE_RUN="${ALLOW_DUPLICATE_RUN:-0}"
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

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_val_ratio 0.9
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --epochs 200
  --fl_rounds 200
  --fl_client_key receiver
  --fl_clients_per_round 1.0
  --fl_local_epochs 2
  --fl_test_eval_interval 10
  --fl_test_eval_last_n 5
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches 0
  --num_workers 0
  --fl_num_workers 0
  --batch_size 128
  --eval_batch_size 256
  --seed 1337
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --primary_udu_weight 0.70
  --use_aug
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
)

FSDG_CORE_ARGS=(
  --train_mode fedprox
  --fedprox_mu 0.01
  --fl_local_objective receiver_agnostic_bex02
  --lambda_rx_adv 1.0
  --grl_lambda 1.0
  --lambda_fishr 0.02
  --fishr_min_domains 4
)

FSDG_CVS_SAT_ARGS=(
  --use_sat_consistency
  --fl_sat_aug_mode cvs_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
)

CEN_SAT_CE_ARGS=(
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 1.28
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.0
  --sat_cons_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

CEN_A31_LIGHT_ARGS=(
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --lambda_group_ce 0.06
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --group_ce_min_domains 2
  --groupdro_tau 0.50
  --groupdro_cap 0.65
  --use_fed_proto_stats
  --lambda_fed_proto 0.02
  --fed_proto_min_count 2
  --fed_proto_momentum 0.20
  --lambda_supcon_id 0.02
  --supcon_temp 0.12
  --generalization_feature z_id
)

ZID_CORAL_DIAG_ARGS=(
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.003
  --fl_coral_stage all
  --fl_coral_start_round 20
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

VMB_BASE_ARGS=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_batches_per_client 1
  --fl_vmb_server_lr 0.006
  --fl_vmb_server_momentum 0.9
  --fl_conflict_agg cosine_clip
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_freeze_rx_stage2
  --lambda_rx_adv 1.0
  --grl_lambda 1.0
  --lambda_tx_adv_r 0.10
  --use_tx_adv_on_zdom
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --fl_baseline_view_ce_only
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.0
  --sat_cons_start_epoch 1
  --fl_style_zdom_probe_every 10
  --fl_style_zdom_probe_real_samples 4
  --fl_style_zdom_probe_max_examples 4
)

VMB_STYLE_AUDIT_ARGS=(
  --use_fed_style_bank
  --fl_style_sampling_policy target_balanced
  --fl_style_replay_start_round 40
  --fl_style_phys_start_round 40
  --fl_style_dg_start_round 70
  --fl_style_replay_prob 0.20
  --fl_style_max_views 1
  --fl_style_transform_mix_alpha 0.35
  --fl_style_min_remote_centroids 1
)

RUN_NAMES=(
  FSDG49A_anchor_fedprox_ra_cvs_r010_r200
  FSDG49B_rxadv_zidcoral_diag_r010
  FSDG49C_cenA31_sat_dsq_groupce_r010
  FSDG49D_rxadv_cenA31_full_r010
  FSDG49E_rxadv_satonly_no_groupce_r010
  VMB_AUDIT_auto_stage20_receiver_r010
  VMB_RXADV_stage40_zdom_guard_r010
  VMB_FULL_rxadv_cenA31_audit_r010
)

if [[ "${DRY_RUN}" != "1" ]]; then
  cd "${ROOT}"
  if [[ ! -f "${ROOT}/train.py" ]]; then
    echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
    exit 2
  fi
  mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
  PID_FILE="${LOG_ROOT}/launch_pids.tsv"
  if [[ ! -f "${PID_FILE}" ]]; then
    printf "run_name\tgpu\tpid\tlog_path\toutput_dir\n" > "${PID_FILE}"
  fi
else
  PID_FILE="/dev/null"
fi

for idx in "${!RUN_NAMES[@]}"; do
  GPU="${idx}"
  RUN_NAME="${RUN_NAMES[$idx]}"
  if [[ -n "${ONLY_INDEX}" && "${ONLY_INDEX}" != "${idx}" ]]; then
    continue
  fi
  if [[ -n "${ONLY_RUN}" && "${ONLY_RUN}" != "${RUN_NAME}" ]]; then
    continue
  fi

  EXTRA_ARGS=()
  case "${RUN_NAME}" in
    FSDG49A_anchor_fedprox_ra_cvs_r010_r200)
      EXTRA_ARGS=("${FSDG_CORE_ARGS[@]}" "${FSDG_CVS_SAT_ARGS[@]}")
      ;;
    FSDG49B_rxadv_zidcoral_diag_r010)
      EXTRA_ARGS=("${FSDG_CORE_ARGS[@]}" "${FSDG_CVS_SAT_ARGS[@]}" "${ZID_CORAL_DIAG_ARGS[@]}")
      ;;
    FSDG49C_cenA31_sat_dsq_groupce_r010)
      EXTRA_ARGS=("${FSDG_CORE_ARGS[@]}" "${CEN_SAT_CE_ARGS[@]}" "${CEN_A31_LIGHT_ARGS[@]}")
      ;;
    FSDG49D_rxadv_cenA31_full_r010)
      EXTRA_ARGS=("${FSDG_CORE_ARGS[@]}" "${CEN_SAT_CE_ARGS[@]}" "${CEN_A31_LIGHT_ARGS[@]}" "${ZID_CORAL_DIAG_ARGS[@]}")
      ;;
    FSDG49E_rxadv_satonly_no_groupce_r010)
      EXTRA_ARGS=("${FSDG_CORE_ARGS[@]}" "${CEN_SAT_CE_ARGS[@]}")
      ;;
    VMB_AUDIT_auto_stage20_receiver_r010)
      EXTRA_ARGS=("${VMB_BASE_ARGS[@]}" --fl_baseline_view_ce_weight 0.80 --lambda_vmb_tx_proto 0.10 --lambda_vmb_rx_proto 0.10 --fl_vmb_pretrain_rounds 20 --fl_vmb_stage1_local_steps 2 --fl_vmb_stage1_objective ce --no_fl_vmb_stage1_use_aux_losses)
      ;;
    VMB_RXADV_stage40_zdom_guard_r010)
      EXTRA_ARGS=("${VMB_BASE_ARGS[@]}" "${VMB_STYLE_AUDIT_ARGS[@]}" --fl_baseline_view_ce_weight 0.80 --lambda_vmb_tx_proto 0.10 --lambda_vmb_rx_proto 0.10 --fl_vmb_pretrain_rounds 40 --fl_vmb_stage1_local_steps 4 --fl_vmb_stage1_objective receiver_style_pretrain --fl_vmb_stage1_use_aux_losses)
      ;;
    VMB_FULL_rxadv_cenA31_audit_r010)
      EXTRA_ARGS=("${VMB_BASE_ARGS[@]}" "${VMB_STYLE_AUDIT_ARGS[@]}" "${CEN_A31_LIGHT_ARGS[@]}" --fl_vmb_cen_a31_profile --fl_baseline_view_ce_weight 1.14 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.14 --fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_local_steps 4 --fl_vmb_stage1_objective receiver_style_pretrain --fl_vmb_stage1_use_aux_losses)
      ;;
    *)
      echo "[ERROR] no case for run: ${RUN_NAME}" >&2
      exit 3
      ;;
  esac

  if [[ "${DRY_RUN}" != "1" ]]; then
    LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
    if [[ "${ALLOW_DUPLICATE_RUN}" != "1" ]]; then
      if awk -F '\t' -v run="${RUN_NAME}" 'NR > 1 && $1 == run { found = 1 } END { exit(found ? 0 : 1) }' "${PID_FILE}"; then
        echo "[ERROR] run=${RUN_NAME} is already recorded in ${PID_FILE}; refusing duplicate launch for RUN_ID=${RUN_ID}." >&2
        echo "[ERROR] Use a new RUN_ID, or set ALLOW_DUPLICATE_RUN=1 only for an intentional recovery." >&2
        exit 5
      fi
      if [[ -e "${LOG_PATH}" || -e "${LOG_ROOT}/${RUN_NAME}" || -e "${RUNS_ROOT}/${RUN_NAME}" ]]; then
        echo "[ERROR] existing log/output path for run=${RUN_NAME}; refusing duplicate launch for RUN_ID=${RUN_ID}." >&2
        echo "[ERROR] log=${LOG_PATH}" >&2
        echo "[ERROR] log_dir=${LOG_ROOT}/${RUN_NAME}" >&2
        echo "[ERROR] output_dir=${RUNS_ROOT}/${RUN_NAME}" >&2
        echo "[ERROR] Use a new RUN_ID, or set ALLOW_DUPLICATE_RUN=1 only for an intentional recovery." >&2
        exit 6
      fi
    fi
    if ! active_pids="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
      echo "[ERROR] nvidia-smi query failed for gpu=${GPU}; refusing to launch without occupancy proof." >&2
      printf '%s\n' "${active_pids}" >&2
      exit 4
    fi
    active_count="$(printf '%s\n' "${active_pids}" | sed '/^$/d' | wc -l | tr -d ' ')"
    active_count="${active_count:-0}"
    if [[ "${active_count}" -ge "${MAX_PROCS_PER_GPU}" ]]; then
      echo "[SKIP] gpu=${GPU} active_compute=${active_count} max=${MAX_PROCS_PER_GPU} run=${RUN_NAME}" >&2
      continue
    fi
  else
    active_count=0
  fi

  CMD=(
    env "CUDA_VISIBLE_DEVICES=${GPU}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${RUN_NAME}"
    --output_dir "${RUNS_ROOT}/${RUN_NAME}"
    --log_dir "${LOG_ROOT}/${RUN_NAME}"
    "${EXTRA_ARGS[@]}"
  )

  echo "[UNION] idx=${idx} gpu=${GPU} active_compute=${active_count} run=${RUN_NAME} dry_run=${DRY_RUN}"
  printf '[UNION-CMD]'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  mkdir -p "${LOG_ROOT}/${RUN_NAME}" "${RUNS_ROOT}/${RUN_NAME}"
  nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
  PID="$!"
  printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU}" "${PID}" "${LOG_PATH}" "${RUNS_ROOT}/${RUN_NAME}" | tee -a "${PID_FILE}"
done
