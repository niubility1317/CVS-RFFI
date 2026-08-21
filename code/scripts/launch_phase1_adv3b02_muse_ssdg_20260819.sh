#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
RUN_ID="${RUN_ID:-phase1_adv3b02_muse_ssdg_20260819}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU="${GPU:-0}"
SEED=392002
ABLATION="${ABLATION:-NONE}"
INIT_MODE="${INIT_MODE:-scratch}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
CANDIDATE_ID_OVERRIDE="${CANDIDATE_ID_OVERRIDE:-}"
MUSE_UNLABELED_BATCH_SIZE="${MUSE_UNLABELED_BATCH_SIZE:-256}"
DRY_RUN=0
ONLY_CANDIDATES="M0,M1,M2,M3"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    *) echo "[MUSE-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[MUSE-ERROR] unsafe RUN_ID: ${RUN_ID}" >&2
  exit 2
fi

candidate_selected() {
  local level="$1"
  [[ ",${ONLY_CANDIDATES}," == *",${level},"* ]]
}

validate_only() {
  local raw="${ONLY_CANDIDATES//,/ }"
  local level
  [[ -n "${raw// /}" ]] || { echo "[MUSE-ERROR] --only must not be empty" >&2; return 2; }
  for level in ${raw}; do
    case "${level}" in
      M0|M1|M2|M3) ;;
      *) echo "[MUSE-ERROR] unknown candidate: ${level}" >&2; return 2 ;;
    esac
  done
}

validate_ablation() {
  case "${ABLATION}" in
    NONE|U_PROTO|NO_U_PROTO_UPDATE|NO_U_SATELLITE_ID|NO_PRIOR|NO_PROTO|NO_PROTO_EVIDENCE|NO_TEMPORAL|NO_SATELLITE|NO_CROSSRX|NO_NUISANCE|NUISANCE_DETACHED|NO_CLASS_CAP) ;;
    *) echo "[MUSE-ERROR] unknown ablation: ${ABLATION}" >&2; return 2 ;;
  esac
  if [[ "${ABLATION}" == "U_PROTO" && "${ONLY_CANDIDATES}" != "M2" ]]; then
    echo "[MUSE-ERROR] U_PROTO requires --only=M2" >&2
    return 2
  fi
  if [[ "${ABLATION}" != "NONE" && "${ABLATION}" != "U_PROTO" && "${ONLY_CANDIDATES}" != "M3" ]]; then
    echo "[MUSE-ERROR] named ablation requires --only=M3" >&2
    return 2
  fi
}

build_ablation_args() {
  ABLATION_ARGS=()
  case "${ABLATION}" in
    NONE) ;;
    NO_PRIOR)
      ABLATION_ARGS+=(--muse_prior_alignment_gamma 0)
      ;;
    U_PROTO)
      ABLATION_ARGS+=(--muse_enable_u_prototype_update true)
      ;;
    NO_U_PROTO_UPDATE)
      ABLATION_ARGS+=(--muse_enable_u_prototype_update false)
      ;;
    NO_U_SATELLITE_ID)
      ABLATION_ARGS+=(--muse_enable_u_satellite_identity false)
      ;;
    NO_PROTO)
      ABLATION_ARGS+=(
        --muse_fusion_global_weight 0.6666667
        --muse_fusion_local_weight 0.3333333
        --muse_fusion_prototype_weight 0
        --muse_reliability_prototype_weight 0
        --muse_unlabeled_prototype_weight 0
      )
      ;;
    NO_PROTO_EVIDENCE)
      ABLATION_ARGS+=(
        --muse_use_prototype_evidence false
        --muse_fusion_global_weight 0.6666667
        --muse_fusion_local_weight 0.3333333
        --muse_fusion_prototype_weight 0
        --muse_reliability_prototype_weight 0
      )
      ;;
    NO_TEMPORAL)
      ABLATION_ARGS+=(
        --muse_reliability_stability_weight 0
        --muse_require_temporal_stability false
      )
      ;;
    NO_SATELLITE)
      ABLATION_ARGS+=(
        --muse_p_sat_s2a_end 0
        --muse_p_sat_full 0
        --muse_lambda_satellite 0
      )
      ;;
    NO_CROSSRX)
      ABLATION_ARGS+=(--muse_lambda_cross_receiver 0)
      ;;
    NO_NUISANCE)
      ABLATION_ARGS+=(--muse_lambda_nuisance 0)
      ;;
    NUISANCE_DETACHED)
      ABLATION_ARGS+=(--muse_nuisance_detached true)
      ;;
    NO_CLASS_CAP)
      ABLATION_ARGS+=(--muse_class_balanced_cap false)
      ;;
  esac
}

capability_label() {
  case "$1" in
    M0) echo "ADV3B02_CONTROL" ;;
    M1) echo "BASE" ;;
    M2) echo "BASE_FUSION_HML" ;;
    M3) echo "BASE_FUSION_HML_SATELLITE_CROSSRX_PROTO" ;;
  esac
}

build_train_command() {
  local level="$1"
  local candidate_root="$2"
  local candidate_id="$3"
  TRAIN_CMD=(env
    "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${CODE_ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_val_ratio 0.30
    --source_cal_ratio 0.15
    --source_select_ratio 0.15
    --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --output_dir "${candidate_root}"
    --run_id "${RUN_ID}"
    --candidate_id "${candidate_id}"
    --base_candidate ADV3B02_CORE90_SOFT_E200
    --epochs 200
    --batch_size 128
    --label_epochs 130
    --pseudo_epochs 70
    --phase1_source_val_selection_only true
    --checkpoint_selection final_only
    --best_metric source_val_sat_hmean
    --paic_guard_enabled true
    --paic_guard_sat_ce_delta 0.12
    --paic_guard_grad_delta 3.0
    --paic_guard_reliable_drop 0.01
    --paic_guard_cooldown_epochs 1
    --paic_guard_sat_scale 0.75
    --use_muse_ssdg true
    --muse_level "${level}"
    --muse_external_final_eval true
    --muse_epoch_basis unlabeled_loader
    --muse_unlabeled_batch_size "${MUSE_UNLABELED_BATCH_SIZE}"
    --muse_fused_student_forward true
    --muse_lr_schedule fasttrust
    --muse_hard_max_fraction 0.25
    --muse_identity_max_fraction 0.50
    --muse_final_epoch 200
    --use_unlabeled true
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
    --phase2_export_path "${candidate_root}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --endpoint_require_artifact_on_export false
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
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_training_mode concat_masked
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
    --max_grad_norm 5
    --tau_min 0.92
    --tau_max 0.97
    --pseudo_quantile 0.86
    --use_ema_teacher true
    --eval_sat_channel true
    --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_eval_max_batches -1
    --device cuda:0
    --seed "${SEED}"
  )
  if [[ "${INIT_MODE}" == "adv3b02_core90" ]]; then
    TRAIN_CMD+=(--from_scratch false --baseline_ckpt "${BASE_CKPT}")
  elif [[ "${INIT_MODE}" == "scratch" ]]; then
    TRAIN_CMD+=(--from_scratch true)
  else
    echo "[MUSE-ERROR] unknown INIT_MODE: ${INIT_MODE}" >&2
    return 2
  fi
  TRAIN_CMD+=("${ABLATION_ARGS[@]}")
}

build_eval_command() {
  local candidate_root="$1"
  EVAL_CMD=(env
    "PYTHONPATH=${CODE_ROOT}/code:${CODE_ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${CODE_ROOT}/code/scripts/eval_ssdg_sat_per_rx.py"
    --ckpt "${candidate_root}/final_ssdg.pth"
    --output_json "${candidate_root}/metrics_joint.json"
    --eval_on unseen_rx
    --scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --device cuda:0
    --max_batches -1
    --sat_seed "${SEED}"
    --strict_reconstruction
  )
}

split_joint_metrics() {
  local candidate_root="$1"
  local joint_json="${candidate_root}/metrics_joint.json"
  local error_file="${candidate_root}/metrics_split_error.txt"
  "${CONTROL_PYTHON}" -c '
import json
import sys
from pathlib import Path

joint_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
error_path = Path(sys.argv[3])
leo_scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
all_scenarios = ("clean", *leo_scenarios)

def fail(scenario, message):
    error_path.write_text(f"EVAL_FAILED_{scenario.upper()}\n{message}\n", encoding="utf-8")
    raise SystemExit(31)

def checked_metric(row, prefix, scenario):
    required = (f"{prefix}_acc", f"{prefix}_correct", f"{prefix}_total")
    if any(key not in row for key in required):
        fail(scenario, "missing metric fields")
    correct = int(row[required[1]])
    total = int(row[required[2]])
    accuracy = float(row[required[0]])
    if total <= 0 or correct < 0 or correct > total:
        fail(scenario, "invalid metric counts")
    expected = 100.0 * correct / total
    if abs(accuracy - expected) > 1e-6:
        fail(scenario, "metric accuracy does not match counts")
    return correct, total, expected

error_path.unlink(missing_ok=True)
data = json.loads(joint_path.read_text(encoding="utf-8"))
reconstruction_audit = data.get("reconstruction_audit")
if not isinstance(reconstruction_audit, dict):
    fail("joint", "missing strict reconstruction audit")
if reconstruction_audit.get("strict_requested") is not True:
    fail("joint", "strict reconstruction was not requested")
if reconstruction_audit.get("checkpoint_load_strict") is not True:
    fail("joint", "checkpoint was not restored with strict=True")
if reconstruction_audit.get("fallback_used") is not False:
    fail("joint", "fallback reconstruction is forbidden")
for key in ("missing_keys", "unexpected_keys", "shape_mismatches"):
    if int(reconstruction_audit.get(key, -1)) != 0:
        fail("joint", f"strict reconstruction reported {key}")
rows = data.get("rows")
if not isinstance(rows, list) or not rows:
    fail("clean", "joint evaluator returned no rows")

common_keys = ("name", "rx_idx", "rx_label", "days_label")
clean_by_key = {}
for row in rows:
    correct, total, accuracy = checked_metric(row, "clean", "clean")
    identity = tuple(str(row.get(key, "")) for key in common_keys)
    normalized = {
        **{key: row.get(key) for key in common_keys},
        "scenario": "clean",
        "tx_acc": accuracy,
        "tx_correct": correct,
        "tx_total": total,
    }
    previous = clean_by_key.get(identity)
    if previous is not None and previous != normalized:
        fail("clean", "inconsistent repeated clean metrics")
    clean_by_key[identity] = normalized

normalized_rows = {"clean": list(clean_by_key.values())}
for scenario in leo_scenarios:
    selected = [row for row in rows if str(row.get("scenario")) == scenario]
    if not selected:
        fail(scenario, "joint evaluator omitted scenario rows")
    normalized_rows[scenario] = []
    for row in selected:
        correct, total, accuracy = checked_metric(row, "sat", scenario)
        normalized_rows[scenario].append(
            {
                **{key: row.get(key) for key in common_keys},
                "scenario": scenario,
                "tx_acc": accuracy,
                "tx_correct": correct,
                "tx_total": total,
            }
        )

payloads = {}
for scenario in all_scenarios:
    scenario_rows = normalized_rows[scenario]
    correct = sum(int(row["tx_correct"]) for row in scenario_rows)
    total = sum(int(row["tx_total"]) for row in scenario_rows)
    if total <= 0:
        fail(scenario, "scenario aggregate has no samples")
    aggregate = {
        "scenario": scenario,
        "tx_acc": 100.0 * correct / total,
        "tx_correct": correct,
        "tx_total": total,
    }
    payloads[scenario] = {
        "schema": "ssdg_phase1_scenario_eval_v1",
        "source_schema": data.get("schema"),
        "checkpoint": data.get("checkpoint"),
        "checkpoint_epoch": data.get("checkpoint_epoch"),
        "run_name": data.get("run_name"),
        "reconstruction": data.get("reconstruction"),
        "reconstruction_audit": reconstruction_audit,
        "eval_on": data.get("eval_on"),
        "group_loader": data.get("group_loader"),
        "group_key": data.get("group_key"),
        "selected_names": data.get("selected_names"),
        "split": data.get("split"),
        "sat_seed": data.get("sat_seed"),
        "scenario": scenario,
        "aggregate": aggregate,
        "rows": scenario_rows,
    }

for scenario in all_scenarios:
    payload = payloads[scenario]
    (output_root / f"metrics_{scenario}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = payload["aggregate"]
    (output_root / f"eval_{scenario}.log").write_text(
        ("[MUSE-EVAL-SPLIT] scenario={scenario} tx_acc={tx_acc:.6f} "
         "correct={tx_correct} total={tx_total}\n").format(**aggregate)
        + f"source={joint_path}\n",
        encoding="utf-8",
    )
' "${joint_json}" "${candidate_root}" "${error_file}"
}

write_config() {
  local level="$1"
  local candidate_id="$2"
  local capabilities="$3"
  local candidate_root="$4"
  printf '{\n  "run_id": "%s",\n  "candidate": "%s",\n  "muse_level": "%s",\n  "ablation": "%s",\n  "init_mode": "%s",\n  "base_checkpoint": "%s",\n  "base_candidate": "ADV3B02_CORE90_SOFT_E200",\n  "capabilities": "%s",\n  "seed": %d,\n  "epochs": 200,\n  "labeled_batch_size": 128,\n  "unlabeled_batch_size": %d,\n  "ratios": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},\n  "checkpoint_selection": "final_only",\n  "final_evaluation": ["clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]\n}\n' \
    "${RUN_ID}" "${candidate_id}" "${level}" "${ABLATION}" "${INIT_MODE}" "${BASE_CKPT}" "${capabilities}" "${SEED}" "${MUSE_UNLABELED_BATCH_SIZE}" > "${candidate_root}/config.json"
}

run_candidate() {
  local level="$1"
  local capabilities
  local candidate_id="${level}"
  if [[ "${ABLATION}" != "NONE" ]]; then
    candidate_id="${level}_${ABLATION}"
  fi
  if [[ -n "${CANDIDATE_ID_OVERRIDE}" ]]; then
    candidate_id="${CANDIDATE_ID_OVERRIDE}"
  fi
  local candidate_root="${RUNS_ROOT}/${candidate_id}"
  local scenario
  local status
  capabilities="$(capability_label "${level}")"
  build_train_command "${level}" "${candidate_root}" "${candidate_id}"
  build_eval_command "${candidate_root}"

  echo "[MUSE-CANDIDATE] candidate=${candidate_id} capabilities=${capabilities} muse_level=${level} ablation=${ABLATION} init=${INIT_MODE} u_batch=${MUSE_UNLABELED_BATCH_SIZE} output=${candidate_root} seed=${SEED} epochs=200"
  printf '[MUSE-TRAIN-CMD] '; printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
  printf '[MUSE-EVAL-CMD] scenarios=clean,leo_clear_weak,leo_low_elev_weak,leo_rain_weak log=%s ' "${candidate_root}/eval_joint.log"
  printf '%q ' "${EVAL_CMD[@]}"
  printf '\n'
  for scenario in clean leo_clear_weak leo_low_elev_weak leo_rain_weak; do
    printf '[MUSE-EVAL-OUTPUT] scenario=%s metrics=%s log=%s\n' \
      "${scenario}" "${candidate_root}/metrics_${scenario}.json" "${candidate_root}/eval_${scenario}.log"
  done
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ -e "${candidate_root}" ]]; then
    echo "[MUSE-ERROR] refusing to overwrite existing candidate root: ${candidate_root}" >&2
    return 3
  fi
  mkdir -p "${candidate_root}"
  write_config "${level}" "${candidate_id}" "${capabilities}" "${candidate_root}"

  if ! "${TRAIN_CMD[@]}" > "${candidate_root}/train.log" 2>&1; then
    printf 'TRAIN_FAILED\n' > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} training failed; outputs preserved at ${candidate_root}" >&2
    return 4
  fi
  if [[ ! -s "${candidate_root}/final_ssdg.pth" ]]; then
    printf 'FINAL_CHECKPOINT_MISSING\n' > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} missing non-empty final_ssdg.pth" >&2
    return 5
  fi

  if ! "${EVAL_CMD[@]}" > "${candidate_root}/eval_joint.log" 2>&1; then
    printf 'EVAL_FAILED_JOINT\n' > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} status=EVAL_FAILED_JOINT; training outputs preserved" >&2
    return 6
  fi
  if [[ ! -s "${candidate_root}/eval_joint.log" || ! -s "${candidate_root}/metrics_joint.json" ]]; then
    printf 'EVAL_FAILED_JOINT\n' > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} status=EVAL_FAILED_JOINT; empty joint artifact" >&2
    return 7
  fi
  if ! split_joint_metrics "${candidate_root}"; then
    status="$(sed -n '1p' "${candidate_root}/metrics_split_error.txt" 2>/dev/null || printf 'EVAL_FAILED_METRICS_SPLIT')"
    if [[ "${status}" != EVAL_FAILED_* ]]; then
      status="EVAL_FAILED_METRICS_SPLIT"
    fi
    printf '%s\n' "${status}" > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} status=${status}; training outputs preserved" >&2
    return 8
  fi

  for scenario in clean leo_clear_weak leo_low_elev_weak leo_rain_weak; do
    if [[ ! -s "${candidate_root}/eval_${scenario}.log" || ! -s "${candidate_root}/metrics_${scenario}.json" ]]; then
      status="EVAL_FAILED_${scenario^^}"
      printf '%s\n' "${status}" > "${candidate_root}/status.txt"
      echo "[MUSE-ERROR] candidate=${level} status=${status}; empty split artifact" >&2
      return 9
    fi
  done
  printf 'ARTIFACTS_COMPLETE\n' > "${candidate_root}/status.txt"
  echo "[MUSE-COMPLETE] candidate=${level} status=ARTIFACTS_COMPLETE root=${candidate_root}"
}

validate_only
validate_ablation
build_ablation_args
echo "[MUSE-RUN] run_id=${RUN_ID} root=${RUNS_ROOT} dry_run=${DRY_RUN} gpu=${GPU} seed=${SEED} ablation=${ABLATION} ratios=0.07/0.63/0.15/0.15 roles=L_s/U_s/V_cal/V_select checkpoint_selection=final_only"
for level in M0 M1 M2 M3; do
  if candidate_selected "${level}"; then
    run_candidate "${level}"
  fi
done
