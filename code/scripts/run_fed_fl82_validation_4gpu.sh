#!/usr/bin/env bash
set -uo pipefail

# FL82 federated validation launcher for CV-SincNet/CVS-RFFI on N607.
#
# Goals:
#   1. Validate that federated training is effective under the WiSig receiver/day split.
#   2. Push clean strict UDU accuracy (test_unseen_day_unseen_rx) to >= 82%.
#   3. Run named clean tests and satellite-channel split tests every communication round.
#
# Example:
#   GPU_IDS=3,4,5,7 PLAN=CORE bash scripts/run_fed_fl82_validation_4gpu.sh --dry-run
#   GPU_IDS=0,1,2,3,4,5,6 PLAN=BACKBONE_ABL bash scripts/run_fed_fl82_validation_4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -d "${CODE_ROOT}/Dataset_WigSig" ] || [ -d "${CODE_ROOT}/Dataset_ORALCE" ]; then
  WORKSPACE_ROOT="${CODE_ROOT}"
fi
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-3,4,5,7}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${WORKSPACE_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/fl82_fed_validation}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/fl82_fed_validation}"
FEWSHOT_RATIO="${FEWSHOT_RATIO:-0.1}"
EPOCHS="${EPOCHS:-200}"
FL_ROUNDS="${FL_ROUNDS:-200}"
FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
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
  sed -n '1,15p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV        GPUs to use, default 3,4,5,7
  --plan NAME          SMOKE, CORE, SAT_BASELINE, ABLATION, BACKBONE_ABL, PAIC, or FULL
  --wisig-pkl PATH     Dataset_WigSig/ManySig.pkl path
  --python PATH        Python executable, default N607 CVS-RFFI env
  --run-root PATH      Output checkpoint root
  --log-root PATH      Log root
  --ratio FLOAT        WiSig train ratio, default 0.1
  --rounds N           Federated communication rounds, default 200
  --local-epochs N     Federated local epochs, default 2
  --epochs N           Parser compatibility epoch value, default 200
  --no-skip-done       Re-run even when a launcher completion marker exists
  --stop-on-fail       Stop queue after a failed batch
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
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "${DRY_RUN}" != "1" ] && { [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; }; then
  echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
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
FL82_00_smoke_fedavg_rxday|SMOKE|Two-round FedAvg smoke test with full clean/satellite eval path enabled.|--train_mode fedavg --fl_client_key receiver_day --fl_local_objective ce --fl_rounds 2 --fl_local_epochs 1 --eval_max_batches 2 --sat_eval_max_batches 2
EOF
      ;;
    CORE)
      cat <<'EOF' >> "${QUEUE_FILE}"
FL82_01_fedavg_rxday_ce_r010|FL_BASE|FedAvg receiver_day CE baseline at 10% labels; validates plain FL under the strict UDU split.|--train_mode fedavg --fl_client_key receiver_day --fl_local_objective ce
FL82_02_fedprox_rxday_ce_r010_mu01|FL_BASE|FedProx receiver_day CE baseline at 10% labels; tests whether proximal control improves FL stability over FedAvg.|--train_mode fedprox --fl_client_key receiver_day --fedprox_mu 0.01 --fl_local_objective ce
FL82_03_fedprox_rx_ra_bex02_cvs_r010|FL_PERF|Performance anchor at 10% labels: receiver-client FedProx + receiver-agnostic BEX02 + CVS satellite consistency.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_04_fedprox_rx_ra_bex02_cvs_stylebank_r010_l3|FL_PERF|Performance-plus-diagnostics run at 10% labels: same anchor with conservative opt-in StyleBank statistics enabled.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_epochs 1 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --use_aug --use_mixstyle --use_sat_consistency --use_fl_style_bank_stats --fl_style_replay_start_round 20 --fl_style_phys_start_round 20 --fl_style_dg_start_round 40 --fl_style_max_views 1 --fl_style_replay_prob 0.25 --fl_style_phys_jitter_scale 0.25 --fl_style_phys_max_gain_delta 0.05 --fl_style_phys_max_noise_std 0.01 --fl_style_phys_max_cfo_hz 5000 --fl_style_phys_max_sro_ppm 25 --fl_style_phys_max_iq_gain_db 0.5 --fl_style_phys_max_iq_phase_deg 0.5 --fl_style_phys_max_phase_noise_std 0.0005 --fl_style_phys_min_awgn_snr_db 20 --fl_style_phys_p_lowpass 0.2 --fl_style_phys_p_multipath 0.2 --fl_style_phys_max_multipath_taps 3 --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 2
EOF
      ;;
    FULL)
      append_rows_for_plan CORE
      append_rows_for_plan SAT_BASELINE
      append_rows_for_plan ABLATION
      append_rows_for_plan BACKBONE_ABL
      append_rows_for_plan PAIC
      ;;
    SAT_BASELINE)
      cat <<'EOF' >> "${QUEUE_FILE}"
FL82_07_fedprox_rx_ra_bex02_baselineview_clearleo_r010|FL_SAT_TARGET|Baseline-origin clean+sat supervised LEO view expansion at 10% labels inside federated receiver-agnostic BEX02; trains and evaluates only clear_leo.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo --sat_view_prob 1.0 --sat_cons_start_epoch 1 --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_08_fedprox_rx_ra_bex02_baselineview_clearleo_stylebank_l3_r010|FL_SAT_TARGET|Baseline-origin clean+sat supervised LEO view expansion plus conservative opt-in StyleBank diagnostics; trains and evaluates only clear_leo.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_epochs 1 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --use_aug --use_mixstyle --use_sat_consistency --use_fl_style_bank_stats --fl_style_replay_start_round 20 --fl_style_phys_start_round 20 --fl_style_dg_start_round 40 --fl_style_max_views 1 --fl_style_replay_prob 0.25 --fl_style_phys_jitter_scale 0.25 --fl_style_phys_max_gain_delta 0.05 --fl_style_phys_max_noise_std 0.01 --fl_style_phys_max_cfo_hz 5000 --fl_style_phys_max_sro_ppm 25 --fl_style_phys_max_iq_gain_db 0.5 --fl_style_phys_max_iq_phase_deg 0.5 --fl_style_phys_max_phase_noise_std 0.0005 --fl_style_phys_min_awgn_snr_db 20 --fl_style_phys_p_lowpass 0.2 --fl_style_phys_p_multipath 0.2 --fl_style_phys_max_multipath_taps 3 --sat_train_scenarios clear_leo --sat_view_prob 1.0 --sat_cons_start_epoch 1 --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 2
FL82_10_fedprox_rx_ra_bex02_stylebank_collab_clearleo_r010|FL_SAT_TARGET|Paper-inspired virtual heterogeneous receiver run with LEO-only baseline-view SAT training/eval; StyleBank views activate GRL/Fishr and collaborative inference.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_epochs 1 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --use_aug --use_mixstyle --use_sat_consistency --use_fl_style_bank_stats --use_style_collab_eval --style_collab_views 2 --style_collab_fusion adaptive --style_collab_base_weight 1.0 --style_collab_max_aux_weight 0.75 --fl_style_replay_start_round 20 --fl_style_phys_start_round 20 --fl_style_dg_start_round 40 --fl_style_max_views 1 --fl_style_replay_prob 0.25 --fl_style_phys_jitter_scale 0.25 --fl_style_phys_max_gain_delta 0.05 --fl_style_phys_max_noise_std 0.01 --fl_style_phys_max_cfo_hz 5000 --fl_style_phys_max_sro_ppm 25 --fl_style_phys_max_iq_gain_db 0.5 --fl_style_phys_max_iq_phase_deg 0.5 --fl_style_phys_max_phase_noise_std 0.0005 --fl_style_phys_min_awgn_snr_db 20 --fl_style_phys_p_lowpass 0.2 --fl_style_phys_p_multipath 0.2 --fl_style_phys_max_multipath_taps 3 --sat_train_scenarios clear_leo --sat_view_prob 1.0 --sat_cons_start_epoch 1 --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 2
FL82_09_fedprox_rx_ra_bex02_baselineview_clearleo_l3_r010|FL_SAT_TARGET|Clear-LEO-focused baseline-origin clean+sat supervised view expansion at 10% labels; probes the requested clear_leo floors directly.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_epochs 3 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo --sat_view_prob 1.0 --sat_cons_start_epoch 1 --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    ABLATION)
      cat <<'EOF' >> "${QUEUE_FILE}"
FL82_05_fedprox_rx_ra_proto_stylebank_r010|FL_ABL|Ablation at 10% labels: add existing FedProto stats and conservative StyleBank to the receiver-agnostic CVS anchor without changing the classifier.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --use_aug --use_mixstyle --use_sat_consistency --use_fl_style_bank_stats --fl_style_replay_start_round 20 --fl_style_phys_start_round 20 --fl_style_dg_start_round 40 --fl_style_max_views 1 --fl_style_replay_prob 0.25 --fl_style_phys_jitter_scale 0.25 --fl_style_phys_max_gain_delta 0.05 --fl_style_phys_max_noise_std 0.01 --fl_style_phys_max_cfo_hz 5000 --fl_style_phys_max_sro_ppm 25 --fl_style_phys_max_iq_gain_db 0.5 --fl_style_phys_max_iq_phase_deg 0.5 --fl_style_phys_max_phase_noise_std 0.0005 --fl_style_phys_min_awgn_snr_db 20 --fl_style_phys_p_lowpass 0.2 --fl_style_phys_p_multipath 0.2 --fl_style_phys_max_multipath_taps 3 --use_fed_proto_stats --lambda_fed_proto 0.10 --fed_proto_min_count 2 --fed_proto_momentum 0.20 --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 2
FL82_06_fedprox_rx_ra_bex02_cvs_r010_l4_mu001|FL_ABL|Ablation at 10% labels: lower FedProx mu with 4 local epochs; probes whether more local learning can cross 82% without excessive drift.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.001 --fl_local_epochs 4 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    BACKBONE_ABL)
      cat <<'EOF' >> "${QUEUE_FILE}"
FL82_11_fedprox_rx_ra_bex02_baselineview_ceonly_backbone_anchor_r010|FL_BACKBONE|Anchor for optional backbone study: Lite-D no-DAC baseline_view with satellite samples contributing TX CE only.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_12_fedprox_rx_ra_bex02_baselineview_ceonly_id_phase_r010|FL_BACKBONE|Optional direction 1: ID backbone complex phase-delta time stability cues on top of the CE-only baseline-view anchor.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --id_time_stability_mode phase_delta --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_13_fedprox_rx_ra_bex02_baselineview_ceonly_id_dsq_r010|FL_BACKBONE|Optional direction 2: ID backbone differential spectral-quotient frequency stability cues on top of the CE-only baseline-view anchor.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --id_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_14_fedprox_rx_ra_bex02_baselineview_ceonly_id_phase_dsq_r010|FL_BACKBONE|Combined ID-backbone phase-delta time stability plus DSQ frequency stability cues.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --id_time_stability_mode phase_delta --id_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_15_fedprox_rx_ra_bex02_baselineview_ceonly_domain_phase_r010|FL_BACKBONE|Domain-backbone-only phase-delta time stability probe; keeps the ID backbone on the mature anchor.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_time_stability_mode phase_delta --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_16_fedprox_rx_ra_bex02_baselineview_ceonly_domain_dsq_r010|FL_BACKBONE|Domain-backbone-only DSQ frequency stability probe; keeps the ID backbone on the mature anchor.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
FL82_17_fedprox_rx_ra_bex02_baselineview_ceonly_all_phase_dsq_r010|FL_BACKBONE|Full optional backbone probe: ID phase+DSQ and domain backbone mirrors those stability cues.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --id_time_stability_mode phase_delta --id_freq_stability_mode dsq --domain_time_stability_mode same --domain_freq_stability_mode same --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    PAIC)
      cat <<'EOF' >> "${QUEUE_FILE}"
F0_FSDG49_ANCHOR|FL_PAIC|CVS-SAT-PAIC F0 anchor: historical FedProx receiver-client receiver-agnostic BEX02 route for comparison, not a new PAIC claim by itself.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode cvs_consistency --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
F1_FL82_16_CE_ONLY_DSQ|FL_PAIC|CVS-SAT-PAIC F1: replicate the SA16 semantics in FL with baseline-view CE-only, DSQ domain branch, and all-five satellite eval.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
F2_FL_PAIC_CURRICULUM|FL_PAIC|CVS-SAT-PAIC F2: add the three-stage physics-aligned curriculum to the CE-only FL baseline-view route.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_view_schedule 1@0.30:mixed_orbit;41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp --sat_view_prob 1.0 --sat_cons_start_epoch 1 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
F3_FL_PAIC_LATE_ALIGN|FL_PAIC|CVS-SAT-PAIC F3: exploratory late weak z_id alignment after curriculum; keep source-only DG and no target receiver data.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --sat_view_schedule 1@0.30:mixed_orbit;41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp --sat_view_prob 1.0 --sat_cons_start_epoch 90 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.01 --lambda_fishr 0.02 --fishr_min_domains 4
F4_STYLEBANK_DIAGNOSTIC_ONLY|FL_PAIC_DIAGNOSTIC|CVS-SAT-PAIC F4 diagnostic only: StyleBank must beat random physical stress without leakage before it can become a mainline route.|--train_mode fedprox --wisig_domain rx --fl_client_key receiver --fedprox_mu 0.01 --fl_local_objective receiver_agnostic_bex02 --fl_sat_aug_mode baseline_view --fl_baseline_view_ce_only --fl_baseline_view_ce_weight 1.0 --use_aug --use_mixstyle --use_sat_consistency --use_fl_style_bank_stats --sat_view_schedule 1@0.30:mixed_orbit;41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp --sat_view_prob 1.0 --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --domain_freq_stability_mode dsq --lambda_rx_adv 1.0 --grl_lambda 1.0 --lambda_sat_cls 0.00 --lambda_sat_cons 0.00 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    *)
      echo "ERROR: unknown plan '${plan_name}'. Use SMOKE, CORE, SAT_BASELINE, ABLATION, BACKBONE_ABL, PAIC, or FULL." >&2
      exit 2
      ;;
  esac
}

generate_queue() {
  : > "${QUEUE_FILE}"
  IFS=',' read -r -a plans <<< "${PLAN}"
  local p
  for p in "${plans[@]}"; do
    append_rows_for_plan "${p}"
  done
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
  --primary_udu_weight 0.80
  --test_eval_policy every_epoch
  --eval_max_batches 0
  --sat_eval_max_batches -1
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
  --eval_sat_on test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo
  --seed "${SEED}"
)

RUNNING_PIDS=()
RUNNING_NAMES=()
RUNNING_GPUS=()
FAIL_COUNT=0
JOB_INDEX=0

launch_one() {
  local gpu="$1"
  local run_name="$2"
  local group="$3"
  local desc="$4"
  local extra_args="$5"
  local out_dir="${RUN_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local done_file="${out_dir}/_launcher_done.txt"
  local row_args=()
  if [ -n "${extra_args}" ]; then
    read -r -a row_args <<< "${extra_args}"
  fi

  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" "${PYTHON_BIN}" -u train.py
    "${BASE_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${out_dir}"
    "${row_args[@]}"
  )

  local cmdline
  cmdline="$(printf '%q ' "${cmd[@]}")"
  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN][GPU ${gpu}] ${run_name}: ${cmdline}"
    return 0
  fi

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${done_file}" ]; then
    log_msg "[SKIP][GPU ${gpu}] ${run_name}: done marker exists at ${done_file}"
    return 0
  fi

  mkdir -p "${out_dir}"
  {
    echo "[START] $(date -Is) run=${run_name} group=${group} gpu=${gpu}"
    echo "[DESC] ${desc}"
    echo "[CMD] ${cmdline}"
    echo "[TARGET] train_ratio=0.1 required; default_epochs=200; clean strict test_unseen_day_unseen_rx >= 82.0%; LEO-only SAT evaluation uses clear_leo on every round."
  } >> "${log_file}"

  (
    "${cmd[@]}"
    status="$?"
    echo "[END] $(date -Is) run=${run_name} status=${status}" >> "${log_file}"
    if [ "${status}" -eq 0 ]; then
      date -Is > "${done_file}"
    fi
    exit "${status}"
  ) >> "${log_file}" 2>&1 &
  LAST_PID="$!"
  log_msg "[LAUNCH][GPU ${gpu}] ${run_name} pid=${LAST_PID} log=${log_file}"
}

wait_current_batch() {
  local i pid name gpu status batch_failed=0
  for i in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$i]}"
    name="${RUNNING_NAMES[$i]}"
    gpu="${RUNNING_GPUS[$i]}"
    if wait "${pid}"; then
      status=0
    else
      status="$?"
      batch_failed=1
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    log_msg "[DONE][GPU ${gpu}] ${name} pid=${pid} status=${status}"
  done
  RUNNING_PIDS=()
  RUNNING_NAMES=()
  RUNNING_GPUS=()
  if [ "${STOP_ON_FAIL}" = "1" ] && [ "${batch_failed}" -ne 0 ]; then
    log_msg "[STOP] stop-on-fail set and current batch failed."
    exit 1
  fi
}

log_msg "[SCHED] plan=${PLAN} jobs=${TOTAL_JOBS} gpus=${GPU_IDS_CSV} ratio=${FEWSHOT_RATIO} rounds=${FL_ROUNDS} local_epochs=${FL_LOCAL_EPOCHS}"
log_msg "[SCHED] run_root=${RUN_ROOT}"
log_msg "[SCHED] log_root=${LOG_ROOT}"
log_msg "[SCHED] queue=${QUEUE_FILE}"

while IFS='|' read -r run_name group desc extra_args; do
  [ -z "${run_name}" ] && continue
  gpu="${GPU_LIST[$((JOB_INDEX % ${#GPU_LIST[@]}))]}"
  launch_one "${gpu}" "${run_name}" "${group}" "${desc}" "${extra_args}"
  if [ "${DRY_RUN}" != "1" ] && [ -n "${LAST_PID:-}" ]; then
    RUNNING_PIDS+=("${LAST_PID}")
    RUNNING_NAMES+=("${run_name}")
    RUNNING_GPUS+=("${gpu}")
  fi
  JOB_INDEX=$((JOB_INDEX + 1))
  if [ "${DRY_RUN}" != "1" ] && [ "${#RUNNING_PIDS[@]}" -ge "${#GPU_LIST[@]}" ]; then
    wait_current_batch
  fi
done < "${QUEUE_FILE}"

if [ "${DRY_RUN}" != "1" ] && [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
  wait_current_batch
fi

if [ "${DRY_RUN}" = "1" ]; then
  log_msg "[DRY-RUN] complete."
elif [ "${FAIL_COUNT}" -eq 0 ]; then
  log_msg "[SCHED] complete with all jobs successful."
else
  log_msg "[SCHED] complete with ${FAIL_COUNT} failed job(s)."
  exit 1
fi
