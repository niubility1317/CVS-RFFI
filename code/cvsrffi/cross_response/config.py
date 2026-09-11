"""Resolve every variant before data/model construction; fail on unknown controls."""
from copy import deepcopy
import math
from pathlib import Path

from scripts.core90_cross_response_matrix import load_config, resolve_variant, validate_baseline, validate_input_roles, ROLE_FIELDS


KEYS = {
    "enabled", "response_enabled", "decision_enabled", "identity_interaction_enabled",
    "head_only", "permanent_detach", "P", "Q", "K", "view_policy", "target_family",
    "target_dim", "statistics_epsilon", "readout_dim", "predictor_mode", "rank",
    "lambda_resp", "lambda_dec", "lambda_cross", "decision_delta", "decision_min_records",
    "interaction_weight", "role_rotation", "normalization_policy", "mixstyle_role_policy",
    "gradient_stage", "gradient_cap", "gate_min_blocks", "gate_error_ratio", "gate_stable_checks",
    "scheduler_mode", "exploration", "update_interval", "feedback_clip", "scheduler_alpha",
    "scheduler_beta", "data_seed", "lags", "gate_interval_epochs", "source_fit_max_records",
    "source_eval_max_blocks", "k_menu", "gradient_tail_prefixes", "scheduler_candidate_limit",
    "event_windows",
}

V2_KEYS = {"implementation_version", "direction_shrinkage", "diagnostic_interval", "statistics_audit",
           "gain_strategy", "evidence_config", "decision_mode", "decision_calibration",
           "interaction_mode", "response_routing", "mechanism_gate", "response_decomposition",
           "source_audit_enabled", "source_baseline_fit_steps", "source_baseline_fit_lr"}
V2_KEYS.add("geometry_min_norm")


def validate_runtime_config(c):
    unknown = set(c) - KEYS - V2_KEYS
    if unknown:
        raise ValueError(f"unused cross-response parameters: {sorted(unknown)}")
    for key in KEYS - {"event_windows"}:
        if key not in c:
            raise ValueError(f"missing cross-response parameter: {key}")
    if c.get("implementation_version", 1) not in (1, 2):
        raise ValueError("unsupported implementation_version")
    if not math.isfinite(c.get("direction_shrinkage", 0.)) or c.get("direction_shrinkage", 0.) < 0:
        raise ValueError("direction_shrinkage must be finite and nonnegative")
    if type(c.get("diagnostic_interval", 20)) is not int or c.get("diagnostic_interval", 20) < 1:
        raise ValueError("diagnostic_interval must be a positive integer")
    if c.get("decision_mode", "vectorized") not in ("vectorized", "delta", "delta_pairs"):
        raise ValueError("unknown decision_mode")
    if c.get("decision_mode") in ("delta", "delta_pairs") and not c.get("decision_calibration"):
        raise ValueError("SOURCE_PARAMETERS_UNFROZEN: delta requires source calibration and validation evidence")
    if c.get("source_audit_enabled") and (c.get("source_baseline_fit_steps") is None or c.get("source_baseline_fit_lr") is None):
        raise ValueError("source baseline fit budget must be explicit")
    if c.get("implementation_version") == 2 and c["gradient_tail_prefixes"] != ["id_backbone.cls_head"]:
        raise ValueError("V2 starts at cls_head; one feature-module extension requires fresh source gate evidence")
    if c.get("response_routing", "full_error") not in ("full_error", "decomposed"):
        raise ValueError("unknown response_routing")
    if c.get("response_routing") == "decomposed":
        d = c.get("response_decomposition") or {}
        if d.get("source_frozen") is not True or d.get("source_role") != "L_s" or not d.get("evidence_id"):
            raise ValueError("SOURCE_PARAMETERS_UNFROZEN: decomposed routing requires source noise evidence")
        if c.get("mechanism_gate") is None:
            raise ValueError("decomposed joint routing requires an explicit source mechanism gate configuration")
    if c.get("interaction_mode", "raw") not in ("raw", "normalized"):
        raise ValueError("unknown interaction_mode")
    for key in ("gate_min_blocks", "gate_stable_checks", "gate_interval_epochs", "source_fit_max_records",
                "source_eval_max_blocks", "scheduler_candidate_limit", "decision_min_records"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("statistics_epsilon", "gate_error_ratio", "feedback_clip"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("scheduler_alpha", "scheduler_beta", "decision_delta"):
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("enabled", "response_enabled", "decision_enabled", "identity_interaction_enabled",
                "head_only", "permanent_detach", "role_rotation"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be boolean, not a truthy string")
    for key, expected in (("normalization_policy", "frozen_source"),
                          ("gradient_stage", "warmup")):
        if c[key] != expected:
            raise ValueError(f"unsupported {key}: {c[key]}")
    if c["mixstyle_role_policy"] != "donor_only":
        if not (c.get("implementation_version") == 2 and c["mixstyle_role_policy"] == "original_mask" and not c["response_enabled"]):
            raise ValueError("original mask is only an explicit V2 no-response organization control")
    if not c["k_menu"] or c["K"] not in c["k_menu"] or any(type(k) is not int or k < 1 for k in c["k_menu"]):
        raise ValueError("k_menu must contain positive physical K including configured K")
    if not c["gradient_tail_prefixes"] or any(not x.startswith("id_backbone.") for x in c["gradient_tail_prefixes"]):
        raise ValueError("gradient tail must explicitly select identity backbone modules")
    if c["target_family"] not in {"fft", "autocorr", "iq", "event"}:
        raise ValueError("unknown fixed statistic family")
    expected_dim = {"fft": c["target_dim"], "autocorr": 2 * len(c["lags"]), "iq": 5,
                    "event": 3 * len(c.get("event_windows", []))}[c["target_family"]]
    if c["target_dim"] != expected_dim or expected_dim <= 0:
        raise ValueError("target_dim disagrees with the chosen statistic family")
    if c["head_only"] or c["permanent_detach"]:
        if not c["response_enabled"]:
            raise ValueError("gradient ablations require an enabled response task")
    if c["response_enabled"] and not (c["head_only"] or c["permanent_detach"]):
        if c["gradient_cap"] <= 0:
            raise ValueError("joint response gradient_cap must be positive; use permanent_detach for that control")
        if c["gate_min_blocks"] > c["source_eval_max_blocks"]:
            raise ValueError("gate_min_blocks exceeds possible source evaluation blocks: joint gate cannot open")
    for flag, weight in (("response_enabled", "lambda_resp"), ("decision_enabled", "lambda_dec"),
                         ("identity_interaction_enabled", "lambda_cross")):
        if c[flag] and c[weight] <= 0:
            raise ValueError(f"{flag} configured but {weight}=0: mechanism cannot activate")
    if c["interaction_weight"] and not c["response_enabled"]:
        raise ValueError("interaction error reweighting requires response prediction")
    return c


def apply_configuration(args):
    path = str(getattr(args, "cross_response_config", "") or "")
    if not path:
        if getattr(args, "cross_response_variant", "") or getattr(args, "cross_response_resume", ""):
            raise ValueError("cross-response variant/resume requires an explicit configuration")
        return None
    config = load_config(path)
    if str(getattr(args, "baseline_ckpt", "") or "") or not bool(getattr(args, "from_scratch", True)):
        raise ValueError("CHECKPOINT_PROVENANCE_UNVERIFIED: external initialization is not a scratch CORE90 row")
    validate_baseline(config["baseline_args"])
    c = validate_runtime_config(resolve_variant(config, str(args.cross_response_variant)))
    validate_input_roles({key: getattr(args, key, "") for key in ROLE_FIELDS})
    # The named configuration is the method authority. External data/output,
    # model seed, device and loader resource choices stay user supplied.
    external = {"seed", "num_workers", "prefetch_factor", "device", "amp", "eval_batch_size",
                "run_id", "candidate_id"}
    for key, value in config["baseline_args"].items():
        if key in external:
            continue
        if not hasattr(args, key):
            raise ValueError(f"baseline parameter has no real training parser entry: {key}")
        setattr(args, key, deepcopy(value))
    # Later methods must not enter through unrelated current-parser flags.
    forbidden = ("use_muse_ssdg", "use_crra", "use_nmfdu", "use_daot", "formal_ablation",
                 "os_gradient_surgery", "os_budget_controller", "tail_safety_state_machine",
                 "source_val_dg_health_guard", "direct_metric_reference_bank", "freeze_backbone")
    for key in forbidden:
        if bool(getattr(args, key, False)):
            raise ValueError(f"{key} is not part of the matched historical CORE90 method")
    for key in ("baseline_ckpt", "teacher_ckpt"):
        if str(getattr(args, key, "") or ""):
            raise ValueError("cross-response uses scratch provenance, not inherited external checkpoints")
    args._cross_response_resolved = deepcopy(c)
    args.cross_response_config = str(Path(path).resolve())
    return c
