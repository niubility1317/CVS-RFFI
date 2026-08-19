#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID_INPUT="${RUN_ID:-}"
RUNS_ROOT_INPUT="${RUNS_ROOT:-}"
LOG_ROOT_INPUT="${LOG_ROOT:-}"
RUN_ID="${RUN_ID:-phase1_adv3_mechanism32_queue_20260701}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
LAUNCH_STABILIZE_SEC="${LAUNCH_STABILIZE_SEC:-35}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATES="${ONLY_CANDIDATES:-}"
CIPG_SCREEN="${CIPG_SCREEN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    --cipg-screen) CIPG_SCREEN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ "${CIPG_SCREEN}" == "1" && -z "${RUN_ID_INPUT}" && -z "${RUNS_ROOT_INPUT}" && -z "${LOG_ROOT_INPUT}" ]]; then
  RUN_ID="phase1_advb02_cipg_mixed_screen_20260819"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
fi

candidate_enabled() {
  local cid="$1"
  [[ -z "${ONLY_CANDIDATES}" || ",${ONLY_CANDIDATES}," == *",${cid},"* ]]
}

gpu_active_count() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="${gpu}" '$1 == gpu && $3 == "C" { c++ } END { print c + 0 }'
}

print_gpu_baseline() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    echo "[ADV3M32-GPU-BASELINE] gpu=${gpu} active=$(gpu_active_count "${gpu}") cap=${STAGE2_MAX_ACTIVE_PER_GPU}"
  done
}

wait_for_gpu_slot() {
  local gpu="$1"
  local active
  while true; do
    active="$(gpu_active_count "${gpu}")"
    if [[ "${active}" -lt "${STAGE2_MAX_ACTIVE_PER_GPU}" ]]; then
      return
    fi
    echo "[ADV3M32-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 90
  done
}

set_candidate_defaults() {
  epochs=200
  label_epochs=130
  base_candidate="adv3_mechanism32_main"
  mechanism="balanced direct-vaccept hard-unknown"
  zid_start=8
  ow_start=12
  source_start=20
  soft_start=25
  proxy_start=45
  warmup=25
  lambda_ow=0.0024
  ow_tail=0.14
  ow_vacuum=0.40
  lambda_zid=0.032
  lambda_proxy=0.0045
  proxy_vacuum=0.55
  proxy_virtual_count=48
  proxy_virtual_mode="hard"
  proxy_core_q=0.85
  proxy_accept_q=0.82
  proxy_tail_q=0.92
  proxy_overflow_q=0.97
  proxy_vaccept_w=1.00
  proxy_core_w=0.35
  proxy_gate_w=0.65
  proxy_tail_w=0.20
  proxy_source_w=0.20
  proxy_cvar_alpha=0.25
  proxy_unknown_margin=0.08
  proxy_known_margin=0.05
  proxy_tau_e=0.04
  proxy_comp_temp=3.0
  proxy_comp_margin=4.0
  proxy_comp_margin_temp=3.0
  proxy_shell_width=4.0
  lambda_soft_mix=0.0045
  soft_count=24
  soft_order=3
  soft_ce=0.60
  soft_vacuum=0.35
  lambda_source=0.0035
  source_mix=0.75
  source_radius=33
  fuse_components=6
  fuse_radius=15.0
  sat_start=80
  lambda_sat_cls=0.68
  lambda_sat_cons=0
  lambda_zid_channel=0
  zid_pair_weight=1.0
  labeled_ratio=0.10
  unlabeled_ratio=0.70
  source_val_ratio=0.20
  sat_train_scenario="leo_clear_weak"
  sat_train_scenarios="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
  sat_eval_scenarios="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
  sat_schedule="1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
}

apply_candidate_variant() {
  local cid="$1"
  case "${cid}" in
    ADV3B01_CORE80_STRICT_E200)
      mechanism="direct vaccept strict core80 accept80 alpha20"
      proxy_core_q=0.80; proxy_accept_q=0.80; proxy_cvar_alpha=0.20 ;;
    ADV3B02_CORE90_SOFT_E200)
      mechanism="direct vaccept core90 accept85 alpha30"
      proxy_core_q=0.90; proxy_accept_q=0.85; proxy_cvar_alpha=0.30; proxy_core_w=0.45 ;;
    ADV3B03_MU10_ALPHA20_E200)
      mechanism="unknown margin 0.10 with alpha20 hard CVaR"
      proxy_unknown_margin=0.10; proxy_accept_q=0.80; proxy_cvar_alpha=0.20; lambda_proxy=0.0050 ;;
    ADV3B04_TAU03_ALPHA30_E200)
      mechanism="sharp energy softplus tau0.03 alpha30"
      proxy_tau_e=0.03; proxy_cvar_alpha=0.30; proxy_unknown_margin=0.05; proxy_accept_q=0.85 ;;
    ADV3B05_GATE_HIGH_TIGHT_E200)
      mechanism="component gate high weight tight temperature"
      proxy_gate_w=1.00; proxy_comp_temp=2.5; lambda_proxy=0.0045 ;;
    ADV3B06_GATE_WIDE_SHELL_E200)
      mechanism="component shell wider boundary negatives"
      proxy_gate_w=0.75; proxy_shell_width=6.0; proxy_virtual_count=56 ;;
    ADV3B07_GATE_MARGIN6_E200)
      mechanism="component prototype margin gate 6deg"
      proxy_gate_w=0.85; proxy_comp_margin=6.0; proxy_comp_margin_temp=2.5 ;;
    ADV3B08_GATE_SOFTTEMP_E200)
      mechanism="softer component gate for closed-set retention"
      proxy_gate_w=0.55; proxy_comp_temp=4.5; proxy_comp_margin_temp=4.5; proxy_core_w=0.45 ;;
    ADV3B09_HARD48_E200)
      mechanism="hard unknown pool count48 proxy-dominant"
      proxy_virtual_count=48; lambda_proxy=0.0050; proxy_vacuum=0.65; soft_ce=0.50 ;;
    ADV3B10_HARD72_E200)
      mechanism="hard unknown pool count72 lower outer weight"
      proxy_virtual_count=72; lambda_proxy=0.0040; proxy_gate_w=0.75; soft_count=24 ;;
    ADV3B11_MIXED48_E200)
      mechanism="mixed hard plus legacy virtual outliers"
      proxy_virtual_mode="mixed"; proxy_virtual_count=48; lambda_proxy=0.0045; proxy_source_w=0.25 ;;
    ADV3B12_LEGACYHARD64_E200)
      mechanism="legacy hard virtual outlier mixture"
      proxy_virtual_mode="legacy_hard"; proxy_virtual_count=64; lambda_proxy=0.0038; proxy_core_w=0.50 ;;
    ADV3B13_TAIL_STRONG_E200)
      mechanism="tail quarantine stronger, tighter tail split"
      proxy_tail_w=0.45; proxy_tail_q=0.90; proxy_overflow_q=0.96; source_radius=32 ;;
    ADV3B14_SOURCE_SAFE_STRONG_E200)
      mechanism="source overflow safe loss strong"
      proxy_source_w=0.50; proxy_tail_w=0.25; proxy_overflow_q=0.95; lambda_source=0.0040; source_radius=31 ;;
    ADV3B15_OVERFLOW_TIGHT94_E200)
      mechanism="tight core tail overflow quantile governance"
      proxy_core_q=0.80; proxy_accept_q=0.80; proxy_tail_q=0.88; proxy_overflow_q=0.94; proxy_source_w=0.35 ;;
    ADV3B16_CORE_KEEP_STRONG_E200)
      mechanism="known core accept preservation strong"
      proxy_core_w=0.80; proxy_known_margin=0.08; lambda_proxy=0.0040; lambda_zid=0.034 ;;
    ADV3B17_EARLY_PROXY35_E200)
      mechanism="early proxy direct vaccept curriculum"
      proxy_start=35; soft_start=25; warmup=30; lambda_proxy=0.0045 ;;
    ADV3B18_MID_PROXY60_E200)
      mechanism="mid proxy delayed direct vaccept curriculum"
      proxy_start=60; soft_start=25; warmup=20; lambda_proxy=0.0045 ;;
    ADV3B19_LATE_PROXY100_E200)
      mechanism="late proxy high-intensity control"
      proxy_start=100; soft_start=30; warmup=15; lambda_proxy=0.0055; proxy_gate_w=0.80 ;;
    ADV3B20_LONG_WARMUP_E220)
      mechanism="long warmup smoother reject constraints"
      epochs=220; label_epochs=140; proxy_start=40; warmup=45; lambda_proxy=0.0045; lambda_source=0.0040 ;;
    ADV3B21_SOFTCE_LOW_E200)
      mechanism="soft unknown CE low, energy-vaccept dominant"
      soft_ce=0.35; lambda_soft_mix=0.0040; lambda_proxy=0.0050; proxy_gate_w=0.75 ;;
    ADV3B22_SOFTCE_BAL_E200)
      mechanism="soft unknown CE balanced known fidelity"
      soft_ce=0.80; lambda_soft_mix=0.0045; proxy_core_w=0.55; lambda_proxy=0.0040 ;;
    ADV3B23_SOFTMIX32_E200)
      mechanism="soft unknown mixup count32 with hard vaccept"
      soft_count=32; lambda_soft_mix=0.0050; source_mix=0.85; soft_ce=0.60 ;;
    ADV3B24_SOFTMIX48_E200)
      mechanism="soft unknown mixup count48 stress"
      soft_count=48; lambda_soft_mix=0.0045; soft_ce=0.50; proxy_virtual_count=56 ;;
    ADV3B25_SOURCE_MIX075_E200)
      mechanism="source episode mixup balanced 0.75"
      lambda_source=0.0040; source_mix=0.75; source_radius=32; source_start=15 ;;
    ADV3B26_SOURCE_MIX125_E220)
      mechanism="source episode mixup strong 1.25"
      epochs=220; label_epochs=140; lambda_source=0.0045; source_mix=1.25; source_radius=31; source_start=15 ;;
    ADV3B27_ZID_OW_HEAVY_E200)
      mechanism="zid compactness and open-world feature strong"
      lambda_zid=0.040; lambda_ow=0.0032; ow_vacuum=0.50; lambda_proxy=0.0040; proxy_core_w=0.50 ;;
    ADV3B28_ZID_OW_MODERATE_E200)
      mechanism="zid open-world moderate closed-set retention"
      lambda_zid=0.026; lambda_ow=0.0020; ow_vacuum=0.30; lambda_proxy=0.0045; proxy_vaccept_w=1.15 ;;
    ADV3B29_SAT_EARLY_E200)
      mechanism="earlier satellite stress protection"
      sat_start=50; lambda_sat_cls=0.72; lambda_proxy=0.0040; proxy_core_w=0.50 ;;
    ADV3B30_SAT_STRONG_E200)
      mechanism="strong satellite stress with reject constraints"
      sat_start=70; lambda_sat_cls=0.78; proxy_start=45; lambda_proxy=0.0040
      sat_schedule="1@0.35:leo_clear_weak;31@0.65:leo_low_elev_weak,leo_rain_weak;81@0.85:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" ;;
    ADV3B31_BAL_MAIN_E160)
      mechanism="short balanced screen 160 epochs"
      epochs=160; label_epochs=100; proxy_start=35; soft_start=20; warmup=25; lambda_proxy=0.0040; lambda_zid=0.030 ;;
    ADV3B32_BAL_LONG_E240)
      mechanism="long balanced consolidation 240 epochs"
      epochs=240; label_epochs=150; proxy_start=50; soft_start=30; warmup=30; lambda_proxy=0.0045; lambda_source=0.0040 ;;
    ADV3B02_MIXED_ORBIT_E200)
      mechanism="historical ADV3B02 mixed_orbit control"
      proxy_core_q=0.90; proxy_accept_q=0.85; proxy_cvar_alpha=0.30; proxy_core_w=0.45
      labeled_ratio=0.07; unlabeled_ratio=0.63; source_val_ratio=0.30
      sat_train_scenario="mixed_orbit"
      sat_train_scenarios="mixed_orbit"
      sat_eval_scenarios="mixed_orbit"
      sat_schedule="1@0.30:mixed_orbit;41@0.60:mixed_orbit;91@0.80:mixed_orbit" ;;
    ADV3B02_CIPG_MIXED_E200)
      mechanism="ADV3B02 mixed_orbit plus TX-conditioned clean-satellite z_id geometry"
      proxy_core_q=0.90; proxy_accept_q=0.85; proxy_cvar_alpha=0.30; proxy_core_w=0.45
      lambda_zid_channel=0.18; zid_pair_weight=1.0
      labeled_ratio=0.07; unlabeled_ratio=0.63; source_val_ratio=0.30
      sat_train_scenario="mixed_orbit"
      sat_train_scenarios="mixed_orbit"
      sat_eval_scenarios="mixed_orbit"
      sat_schedule="1@0.30:mixed_orbit;41@0.60:mixed_orbit;91@0.80:mixed_orbit" ;;
    *)
      echo "[ERROR] unknown candidate variant: ${cid}" >&2
      exit 2 ;;
  esac
}

build_command() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  local pseudo_epochs=$((epochs - label_epochs))
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio "${labeled_ratio}"
    --unlabeled_ratio "${unlabeled_ratio}"
    --source_val_ratio "${source_val_ratio}"
    --output_dir "${RUNS_ROOT}/${cid}"
    --run_id "${RUN_ID}"
    --candidate_id "${cid}"
    --base_candidate "${base_candidate}"
    --epochs "${epochs}"
    --label_epochs "${label_epochs}"
    --pseudo_epochs "${pseudo_epochs}"
    --from_scratch true
    --best_metric joint_safe
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
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch "${ow_start}"
    --ow_feat_warmup_epochs "${warmup}"
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight 0
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight "${ow_tail}"
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight "${ow_vacuum}"
    --ow_feat_vacuum_width_deg 6
    --ow_feat_vacuum_hard_k 3
    --lambda_zid_compact "${lambda_zid}"
    --zid_compact_start_epoch "${zid_start}"
    --zid_compact_warmup_epochs "${warmup}"
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.35
    --zid_compact_cvar_weight 0.35
    --zid_compact_cvar_alpha 0.95
    --zid_compact_radius_deg 40
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs "${warmup}"
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count "${proxy_virtual_count}"
    --proxy_unknown_virtual_mode "${proxy_virtual_mode}"
    --proxy_unknown_energy_margin 0.0
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.0
    --proxy_unknown_virtual_detach false
    --proxy_unknown_vacuum_weight "${proxy_vacuum}"
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 3
    --proxy_unknown_vacuum_radius_deg 40
    --proxy_unknown_core_quantile "${proxy_core_q}"
    --proxy_unknown_accept_quantile "${proxy_accept_q}"
    --proxy_unknown_tail_quantile "${proxy_tail_q}"
    --proxy_unknown_overflow_quantile "${proxy_overflow_q}"
    --proxy_unknown_vaccept_weight "${proxy_vaccept_w}"
    --proxy_unknown_core_accept_weight "${proxy_core_w}"
    --proxy_unknown_component_gate_weight "${proxy_gate_w}"
    --proxy_unknown_tail_quarantine_weight "${proxy_tail_w}"
    --proxy_unknown_source_safe_weight "${proxy_source_w}"
    --proxy_unknown_vaccept_cvar_alpha "${proxy_cvar_alpha}"
    --proxy_unknown_unknown_margin "${proxy_unknown_margin}"
    --proxy_unknown_known_margin "${proxy_known_margin}"
    --proxy_unknown_energy_softplus_temperature "${proxy_tau_e}"
    --proxy_unknown_component_temperature_deg "${proxy_comp_temp}"
    --proxy_unknown_component_margin_deg "${proxy_comp_margin}"
    --proxy_unknown_component_margin_temperature_deg "${proxy_comp_margin_temp}"
    --proxy_unknown_shell_width_deg "${proxy_shell_width}"
    --lambda_soft_unknown_mixup "${lambda_soft_mix}"
    --soft_unknown_mixup_start_epoch "${soft_start}"
    --soft_unknown_mixup_warmup_epochs "${warmup}"
    --soft_unknown_mixup_count "${soft_count}"
    --soft_unknown_mixup_order "${soft_order}"
    --soft_unknown_mixup_alpha 0.5
    --soft_unknown_mixup_energy_margin 1.0
    --soft_unknown_mixup_ce_weight "${soft_ce}"
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_vacuum_weight "${soft_vacuum}"
    --soft_unknown_mixup_vacuum_width_deg 6
    --soft_unknown_mixup_vacuum_hard_k 3
    --soft_unknown_mixup_detach false
    --lambda_source_episode "${lambda_source}"
    --source_episode_start_epoch "${source_start}"
    --source_episode_warmup_epochs "${warmup}"
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_radius}"
    --source_episode_mixup_weight "${source_mix}"
    --source_episode_mixup_hard_k 3
    --phase2_export_prototypes true
    --phase2_export_path "${RUNS_ROOT}/${cid}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components "${fuse_components}"
    --phase2_fuse_merge_angle_deg 2.5
    --phase2_fuse_radius_cap_deg "${fuse_radius}"
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
    --sat_train_scenario "${sat_train_scenario}"
    --sat_train_scenarios "${sat_train_scenarios}"
    --sat_view_schedule "${sat_schedule}"
    --sat_cons_start_epoch "${sat_start}"
    --lambda_sat_cls "${lambda_sat_cls}"
    --lambda_sat_cons "${lambda_sat_cons}"
    --lambda_zid_channel_invariance "${lambda_zid_channel}"
    --zid_channel_pair_weight "${zid_pair_weight}"
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
    --eval_sat_scenarios "${sat_eval_scenarios}"
    --sat_eval_max_batches -1
    --device cuda:0
    --seed "${seed}")
}

run_candidate() {
  local cid="$1"
  local gpu="$2"
  local slot="$3"
  local seed="$4"
  if ! candidate_enabled "${cid}"; then
    echo "[ADV3M32-SKIP] id=${cid} gpu=${gpu} slot=${slot} reason=only-filter"
    return 0
  fi
  set_candidate_defaults
  apply_candidate_variant "${cid}"
  build_command "${cid}" "${gpu}" "${seed}"
  local pseudo_epochs=$((epochs - label_epochs))
  echo "[ADV3M32-CANDIDATE] id=${cid} gpu=${gpu} slot=${slot} seed=${seed} epochs=${epochs} label_epochs=${label_epochs} pseudo_epochs=${pseudo_epochs} mechanism=${mechanism}"
  printf "[ADV3M32-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}" "${LOG_ROOT}/locks" "${LOG_ROOT}/status"
  if [[ -e "${RUNS_ROOT}/${cid}" || -e "${LOG_ROOT}/${cid}.out" ]]; then
    echo "[ERROR] refusing to overwrite existing run/log for ${cid}" >&2
    return 3
  fi
  mkdir -p "${RUNS_ROOT}/${cid}"

  local lock_file="${LOG_ROOT}/locks/gpu${gpu}.lock"
  local pid
  exec 9>"${lock_file}"
  flock 9
  wait_for_gpu_slot "${gpu}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  pid=$!
  echo "${pid}" > "${LOG_ROOT}/${cid}.pid"
  echo "running pid=${pid} gpu=${gpu} slot=${slot} started=$(date -Is)" > "${LOG_ROOT}/status/${cid}.status"
  echo "[ADV3M32-LAUNCHED] id=${cid} pid=${pid} gpu=${gpu} slot=${slot} log=${LOG_ROOT}/${cid}.out"
  sleep "${LAUNCH_STABILIZE_SEC}"
  flock -u 9
  exec 9>&-

  set +e
  wait "${pid}"
  local status=$?
  set -e
  echo "exit=${status} finished=$(date -Is)" >> "${LOG_ROOT}/status/${cid}.status"
  echo "[ADV3M32-FINISHED] id=${cid} pid=${pid} gpu=${gpu} slot=${slot} exit=${status}"
  return 0
}

slot_queue() {
  local gpu="$1"
  local slot="$2"
  local cid_a="$3"
  local seed_a="$4"
  local cid_b="$5"
  local seed_b="$6"
  echo "[ADV3M32-SLOT-START] gpu=${gpu} slot=${slot} queue=${cid_a},${cid_b}"
  run_candidate "${cid_a}" "${gpu}" "${slot}" "${seed_a}" || true
  run_candidate "${cid_b}" "${gpu}" "${slot}" "${seed_b}" || true
  echo "[ADV3M32-SLOT-DONE] gpu=${gpu} slot=${slot}"
}

echo "[ADV3M32-RUN] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=32 gpus=0-7 slot_queues=16 cap_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU}"
print_gpu_baseline

if [[ "${CIPG_SCREEN}" == "1" ]]; then
  echo "[ADV3M32-CIPG-SCREEN] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=2 channel=mixed_orbit baseline=ADV3B02_MIXED_ORBIT_E200 candidate=ADV3B02_CIPG_MIXED_E200"
  slot_queue 0 0 ADV3B02_MIXED_ORBIT_E200 392033 ADV3B02_CIPG_MIXED_E200 392033
  echo "[ADV3M32-CIPG-SCREEN-DONE] run_id=${RUN_ID}"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  slot_queue 0 0 ADV3B01_CORE80_STRICT_E200 392001 ADV3B02_CORE90_SOFT_E200 392002
  slot_queue 0 1 ADV3B03_MU10_ALPHA20_E200 392003 ADV3B04_TAU03_ALPHA30_E200 392004
  slot_queue 1 0 ADV3B05_GATE_HIGH_TIGHT_E200 392005 ADV3B06_GATE_WIDE_SHELL_E200 392006
  slot_queue 1 1 ADV3B07_GATE_MARGIN6_E200 392007 ADV3B08_GATE_SOFTTEMP_E200 392008
  slot_queue 2 0 ADV3B09_HARD48_E200 392009 ADV3B10_HARD72_E200 392010
  slot_queue 2 1 ADV3B11_MIXED48_E200 392011 ADV3B12_LEGACYHARD64_E200 392012
  slot_queue 3 0 ADV3B13_TAIL_STRONG_E200 392013 ADV3B14_SOURCE_SAFE_STRONG_E200 392014
  slot_queue 3 1 ADV3B15_OVERFLOW_TIGHT94_E200 392015 ADV3B16_CORE_KEEP_STRONG_E200 392016
  slot_queue 4 0 ADV3B17_EARLY_PROXY35_E200 392017 ADV3B18_MID_PROXY60_E200 392018
  slot_queue 4 1 ADV3B19_LATE_PROXY100_E200 392019 ADV3B20_LONG_WARMUP_E220 392020
  slot_queue 5 0 ADV3B21_SOFTCE_LOW_E200 392021 ADV3B22_SOFTCE_BAL_E200 392022
  slot_queue 5 1 ADV3B23_SOFTMIX32_E200 392023 ADV3B24_SOFTMIX48_E200 392024
  slot_queue 6 0 ADV3B25_SOURCE_MIX075_E200 392025 ADV3B26_SOURCE_MIX125_E220 392026
  slot_queue 6 1 ADV3B27_ZID_OW_HEAVY_E200 392027 ADV3B28_ZID_OW_MODERATE_E200 392028
  slot_queue 7 0 ADV3B29_SAT_EARLY_E200 392029 ADV3B30_SAT_STRONG_E200 392030
  slot_queue 7 1 ADV3B31_BAL_MAIN_E160 392031 ADV3B32_BAL_LONG_E240 392032
else
  slot_queue 0 0 ADV3B01_CORE80_STRICT_E200 392001 ADV3B02_CORE90_SOFT_E200 392002 &
  slot_queue 0 1 ADV3B03_MU10_ALPHA20_E200 392003 ADV3B04_TAU03_ALPHA30_E200 392004 &
  slot_queue 1 0 ADV3B05_GATE_HIGH_TIGHT_E200 392005 ADV3B06_GATE_WIDE_SHELL_E200 392006 &
  slot_queue 1 1 ADV3B07_GATE_MARGIN6_E200 392007 ADV3B08_GATE_SOFTTEMP_E200 392008 &
  slot_queue 2 0 ADV3B09_HARD48_E200 392009 ADV3B10_HARD72_E200 392010 &
  slot_queue 2 1 ADV3B11_MIXED48_E200 392011 ADV3B12_LEGACYHARD64_E200 392012 &
  slot_queue 3 0 ADV3B13_TAIL_STRONG_E200 392013 ADV3B14_SOURCE_SAFE_STRONG_E200 392014 &
  slot_queue 3 1 ADV3B15_OVERFLOW_TIGHT94_E200 392015 ADV3B16_CORE_KEEP_STRONG_E200 392016 &
  slot_queue 4 0 ADV3B17_EARLY_PROXY35_E200 392017 ADV3B18_MID_PROXY60_E200 392018 &
  slot_queue 4 1 ADV3B19_LATE_PROXY100_E200 392019 ADV3B20_LONG_WARMUP_E220 392020 &
  slot_queue 5 0 ADV3B21_SOFTCE_LOW_E200 392021 ADV3B22_SOFTCE_BAL_E200 392022 &
  slot_queue 5 1 ADV3B23_SOFTMIX32_E200 392023 ADV3B24_SOFTMIX48_E200 392024 &
  slot_queue 6 0 ADV3B25_SOURCE_MIX075_E200 392025 ADV3B26_SOURCE_MIX125_E220 392026 &
  slot_queue 6 1 ADV3B27_ZID_OW_HEAVY_E200 392027 ADV3B28_ZID_OW_MODERATE_E200 392028 &
  slot_queue 7 0 ADV3B29_SAT_EARLY_E200 392029 ADV3B30_SAT_STRONG_E200 392030 &
  slot_queue 7 1 ADV3B31_BAL_MAIN_E160 392031 ADV3B32_BAL_LONG_E240 392032 &
  wait
fi

echo "[ADV3M32-DONE] run_id=${RUN_ID}"
