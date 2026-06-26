from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Mapping, Sequence


Experiment = Dict[str, object]


def _exp(
    exp_id: str,
    layer: str,
    hypothesis: str,
    purpose: str,
    args: Mapping[str, object] | None = None,
    *,
    expected: str = "",
    batch: str = "",
) -> Experiment:
    return {
        "id": exp_id,
        "layer": layer,
        "batch": batch,
        "hypothesis": hypothesis,
        "purpose": purpose,
        "expected": expected,
        "args": dict(args or {}),
    }


def _l0() -> List[Experiment]:
    return [
        _exp("L0-00", "L0", "sanity", "raw baseline eval", {"mode": "baseline_eval"}, expected="baseline split metrics"),
        _exp("L0-01", "L0", "H2", "FJMP wrapper base_only equals raw baseline", {"fusion_mode": "base_only", "epochs": 0}),
        _exp("L0-02", "L0", "sanity", "frozen backbone grad norm check", {"check_backbone_grad": True, "epochs": 1}),
        _exp("L0-03", "L0", "H2", "random prototype no-train base_only/fused eval", {"prototype_init": "random", "epochs": 0}),
        _exp("L0-04", "L0", "H4", "init-only eval without optimization", {"init_only_eval": True, "epochs": 0}),
        _exp("L0-05", "L0", "sanity", "deterministic repeat with same seed", {"repeat_seeds": [1337, 1337, 1337]}),
    ]


def _l1() -> List[Experiment]:
    rows = [
        ("L1-00", {"feature_input": "z_id", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "base_only", "ce_on": "none"}, "baseline wrapper control"),
        ("L1-01", {"feature_input": "z_id", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "proto_only", "ce_on": "proto"}, "proto branch standalone"),
        ("L1-02", {"feature_input": "z_id", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "calibrated_logit", "ce_on": "fused"}, "reproduce P1"),
        ("L1-03", {"feature_input": "z_id", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "residual", "eta": 0.05, "ce_on": "fused"}, "conservative residual"),
        ("L1-04", {"feature_input": "z_id", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "residual", "eta": 0.10, "ce_on": "fused"}, "eta scan"),
        ("L1-05", {"feature_input": "z_id", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "residual", "eta": 0.20, "ce_on": "fused"}, "eta upper"),
        ("L1-06", {"feature_input": "z_id", "zdom_mode": "normal", "num_prototypes": 2, "fusion_mode": "calibrated_logit", "ce_on": "fused"}, "compare P3"),
        ("L1-07", {"feature_input": "z_id", "zdom_mode": "shuffled", "num_prototypes": 2, "fusion_mode": "calibrated_logit", "ce_on": "fused"}, "zdom shortcut check"),
        ("L1-08", {"init_zdom_mode": "normal", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "calibrated_logit"}, "current init mismatch"),
        ("L1-09", {"init_zdom_mode": "zero", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "calibrated_logit"}, "init/train consistency"),
        ("L1-10", {"init_zdom_mode": "mean", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "calibrated_logit"}, "neutral init"),
        ("L1-11", {"prototype_init": "class_mean", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "calibrated_logit"}, "remove domain-center shortcut"),
        ("L1-12", {"prototype_init": "kmeans", "zdom_mode": "zero", "num_prototypes": 2, "fusion_mode": "calibrated_logit"}, "natural cluster init"),
        ("L1-13", {"epochs": 7, "lr": 1e-3, "weight_decay": 1e-4}, "early-stop reproduction"),
        ("L1-14", {"epochs": 15, "lr": 1e-3, "weight_decay": 1e-4}, "degradation timing"),
        ("L1-15", {"epochs": 30, "lr": 3e-4, "weight_decay": 1e-4}, "lower lr"),
        ("L1-16", {"epochs": 30, "lr": 1e-3, "weight_decay": 1e-3}, "strong weight decay"),
    ]
    return [_exp(exp_id, "L1", "harm source localization", purpose, args) for exp_id, args, purpose in rows]


def _l2() -> List[Experiment]:
    rows = [
        ("L2-00", "base_only", {"eta": 0.0}, "baseline control"),
        ("L2-01", "proto_only", {}, "proto independent ability"),
        ("L2-02", "simple_logit", {"alpha": 0.1, "fusion_learnable": False}, "small direct proto"),
        ("L2-03", "simple_logit", {"alpha": 0.3, "fusion_learnable": False}, "medium direct proto"),
        ("L2-04", "simple_logit", {"alpha": 1.0, "fusion_learnable": False}, "aggressive direct proto"),
        ("L2-05", "calibrated_logit", {"alpha": 1.0, "beta": 1.0, "fusion_learnable": True}, "current calibrated"),
        ("L2-06", "calibrated_logit", {"alpha": 0.1, "beta": 1.0, "fusion_learnable": True}, "limited proto alpha"),
        ("L2-07", "calibrated_logit", {"alpha": 0.03, "beta": 1.0, "fusion_learnable": True}, "very conservative alpha"),
        ("L2-08", "residual", {"eta": 0.02, "fusion_learnable": False}, "most conservative residual"),
        ("L2-09", "residual", {"eta": 0.05, "fusion_learnable": False}, "recommended residual"),
        ("L2-10", "residual", {"eta": 0.10, "fusion_learnable": False}, "medium residual"),
        ("L2-11", "residual", {"eta": 0.20, "eta_max": 0.20, "fusion_learnable": False}, "strong residual"),
        ("L2-12", "residual", {"eta": 0.05, "eta_max": 0.10, "fusion_learnable": True}, "learnable residual capped 0.10"),
        ("L2-13", "residual", {"eta": 0.05, "eta_max": 0.20, "fusion_learnable": True}, "learnable residual capped 0.20"),
        ("L2-14", "probability_ensemble", {"probability_weight": 0.05}, "probability conservative"),
        ("L2-15", "probability_ensemble", {"probability_weight": 0.10}, "probability recommended"),
        ("L2-16", "confidence_gated", {"alpha": 0.1, "enable_conf_gate": True, "gate_type": "margin"}, "margin gate"),
        ("L2-17", "residual", {"eta": 0.10, "enable_conf_gate": True, "gate_type": "ood"}, "OOD gated residual candidate"),
    ]
    return [_exp(exp_id, "L2", "H2 fusion aggressiveness", purpose, {"fusion_mode": mode, **args}) for exp_id, mode, args, purpose in rows]


def _l3() -> List[Experiment]:
    rows = []
    for idx, (ce_proto, ce_fused) in enumerate([(1.0, 0.0), (0.0, 1.0), (1.0, 0.3), (0.3, 1.0), (1.0, 1.0)]):
        ce_on = "both" if ce_proto > 0 and ce_fused > 0 else ("proto" if ce_proto > 0 else "fused")
        rows.append(_exp(f"L3-{idx:02d}", "L3", "H3 CE target", f"CE proto={ce_proto} fused={ce_fused}", {"ce_on": ce_on, "ce_proto_weight": ce_proto, "ce_fused_weight": ce_fused}))
    kd_specs = [
        ("none", 0.0, 4), ("proto", 0.1, 4), ("proto", 0.3, 4), ("fused", 0.3, 4), ("fused", 1.0, 4), ("both", 0.3, 4)
    ]
    for offset, (kd_on, lam, temp) in enumerate(kd_specs, start=5):
        rows.append(_exp(f"L3-{offset:02d}", "L3", "H3 KD target", f"KD {kd_on} lambda={lam}", {"kd_on": kd_on, "lambda_kd": lam, "kd_temperature": temp}))
    dnh_specs = [(0.1, 0.0, "train"), (0.3, 0.0, "train"), (0.5, 0.0, "train"), (0.3, 0.02, "train"), (0.3, 0.0, "train+sat_aug"), (0.3, 0.0, "hard_samples")]
    for offset, (lam, margin, split) in enumerate(dnh_specs, start=11):
        rows.append(_exp(f"L3-{offset:02d}", "L3", "H1 do-no-harm", f"DNH on {split}", {"lambda_dnh": lam, "dnh_margin": margin, "dnh_split": split}))
    margin_specs = [(0.1, 0.0), (0.3, 0.0), (0.3, 0.1)]
    for offset, (lam, delta) in enumerate(margin_specs, start=17):
        rows.append(_exp(f"L3-{offset:02d}", "L3", "H1 margin preservation", f"margin preserve lambda={lam}", {"lambda_margin_preserve": lam, "margin_preserve_delta": delta}))
    cons_specs = [
        ("weak_rf", "fused_clean_fused_aug", 0.1),
        ("sat07", "base_clean_fused_sat", 0.3),
        ("mixed_orbit", "base_clean_fused_sat", 0.3),
        ("cfo_phase_noise", "symmetric_kl", 0.1),
        ("receiver_mix", "symmetric_kl", 0.1),
    ]
    for offset, (aug, target, lam) in enumerate(cons_specs, start=20):
        rows.append(_exp(f"L3-{offset:02d}", "L3", "H1 proxy consistency", f"consistency {aug}", {"proxy_aug": aug, "consistency_target": target, "lambda_proxy_cons": lam}))
    return rows


def _l4() -> List[Experiment]:
    rows: List[Experiment] = []
    for idx, (k, init, fusion) in enumerate([(1, "class_mean", "residual"), (2, "class_domain", "residual"), (3, "class_domain", "residual"), (4, "kmeans", "residual"), (7, "class_rx_center", "residual"), (14, "class_rx_day_center", "residual"), (21, "kmeans", "residual")]):
        rows.append(_exp(f"L4-{idx:02d}", "L4", "H6 K scan", f"K={k} init={init}", {"num_prototypes": k, "prototype_init": init, "fusion_mode": fusion, "eta": 0.05}))
    init_names = ["random", "class_mean", "class_domain_center", "class_rx_center", "class_day_center", "kmeans_per_class", "farthest_point", "herding", "sat_aug_centers"]
    for offset, init in enumerate(init_names, start=7):
        rows.append(_exp(f"L4-{offset:02d}", "L4", "H4 init", f"prototype init {init}", {"prototype_init": init}))
    aggs = ["logsumexp", "max", "mean", "top2_mean", "trimmed_lse", "attention_pool"]
    for offset, agg in enumerate(aggs, start=16):
        rows.append(_exp(f"L4-{offset:02d}", "L4", "H7 aggregation", f"prototype aggregation {agg}", {"prototype_aggregation": agg}))
    scales = [(4, 10, False), (8, 30, False), (12, 30, False), (8, 30, True)]
    for offset, (init_scale, max_scale, scale_reg) in enumerate(scales, start=22):
        rows.append(_exp(f"L4-{offset:02d}", "L4", "H7 scale", f"logit scale {init_scale}/{max_scale}", {"init_scale": init_scale, "max_scale": max_scale, "scale_reg": scale_reg}))
    structs = [
        (0, 0, 0, 0, "no structure"),
        (0.01, 0.003, 0.003, 0.0005, "current"),
        (0.03, 0.003, 0.003, 0.0005, "strong sep"),
        (0.01, 0.01, 0.003, 0.0005, "strong div"),
        (0.01, 0.003, 0.01, 0.0005, "strong usage"),
        (0.01, 0.003, 0.003, 0.005, "strong delta"),
    ]
    for offset, (sep, div, usage, delta, purpose) in enumerate(structs, start=26):
        rows.append(_exp(f"L4-{offset:02d}", "L4", "H6 structure", purpose, {"lambda_sep": sep, "lambda_div": div, "lambda_usage": usage, "lambda_delta": delta}))
    return rows


def _l5() -> List[Experiment]:
    rows: List[Experiment] = []
    mode_specs = [
        ("zero", False, False), ("normal", True, False), ("mean", True, False), ("shuffled", True, False),
        ("dropout", True, False), ("dropout", True, False), ("zero", False, True), ("normal", "weak", True), ("shuffled", False, True),
    ]
    for idx, (mode, cls, gate) in enumerate(mode_specs):
        args = {"zdom_mode": mode, "zdom_to_classifier": cls, "zdom_to_gate": gate}
        if idx == 4:
            args["zdom_drop_prob"] = 0.3
        if idx == 5:
            args["zdom_drop_prob"] = 0.7
        rows.append(_exp(f"L5-{idx:02d}", "L5", "H5 zdom mode", f"zdom {mode} classifier={cls} gate={gate}", args))
    for offset, (max_res, init_res) in enumerate([(0.0, 0.0), (0.1, 0.02), (0.2, 0.05), (0.5, 0.1), (1.0, 0.1)], start=9):
        rows.append(_exp(f"L5-{offset:02d}", "L5", "H5 zdom residual strength", f"max_res_scale={max_res}", {"max_res_scale": max_res, "init_res_scale": init_res}))
    diagnostics = ["zdom_domain_probe", "zid_domain_probe", "zid_tx_probe", "zdom_tx_probe", "zdom_shuffle_sensitivity", "zdom_zero_sensitivity"]
    for offset, diag in enumerate(diagnostics, start=14):
        rows.append(_exp(f"L5-{offset:02d}", "L5", "H5 zdom diagnostics", diag, {"diagnostic": diag}))
    return rows


def _l6() -> List[Experiment]:
    rows: List[Experiment] = []
    proxy_specs = [
        ("source_val", "UDU rank corr"), ("leave_rx_proxy", "UDU rank corr"), ("leave_day_proxy", "UDU rank corr"),
        ("leave_rx_day_proxy", "UDU rank corr"), ("sat07_proxy", "SAT eval corr"), ("mixed_orbit_proxy", "mixed-orbit corr"),
    ]
    for idx, (metric, purpose) in enumerate(proxy_specs):
        rows.append(_exp(f"L6-{idx:02d}", "L6", "H8 proxy-val", purpose, {"selection_metric": metric}))
    sat_specs = [
        ("leo_compact", "low"), ("leo_compact", "medium"), ("leo_compact", "high"), ("mixed_orbit", "low"),
        ("mixed_orbit", "medium"), ("mixed_orbit", "high"), ("cfo_only", "sweep"), ("phase_noise_only", "sweep"),
        ("multipath_only", "sweep"), ("awgn_only", "sweep"),
    ]
    for offset, (scenario, strength) in enumerate(sat_specs, start=6):
        rows.append(_exp(f"L6-{offset:02d}", "L6", "H8 SAT proxy", f"{scenario} {strength}", {"sat_scenario": scenario, "sat_strength": strength}))
    return rows


def _priority_batches() -> List[Experiment]:
    rows: List[Experiment] = []
    batch_a = [
        ("A00", "baseline raw eval", {"mode": "baseline_eval"}),
        ("A01", "wrapper base_only", {"fusion_mode": "base_only"}),
        ("A02", "current P1 rerun seed1", {"num_prototypes": 2, "zdom_mode": "zero"}),
        ("A03", "current P2 rerun seed1", {"num_prototypes": 3, "zdom_mode": "zero"}),
        ("A04", "current P3 rerun seed1", {"num_prototypes": 2, "zdom_mode": "normal"}),
        ("A05", "init zero train zero K2", {"init_zdom_mode": "zero", "zdom_mode": "zero", "num_prototypes": 2}),
        ("A06", "init zero train zero K3", {"init_zdom_mode": "zero", "zdom_mode": "zero", "num_prototypes": 3}),
        ("A07", "proto_only K2", {"fusion_mode": "proto_only", "num_prototypes": 2}),
        ("A08", "residual eta=0.05 K2", {"fusion_mode": "residual", "eta": 0.05, "num_prototypes": 2}),
        ("A09", "residual eta=0.10 K2", {"fusion_mode": "residual", "eta": 0.10, "num_prototypes": 2}),
        ("A10", "residual eta=0.05 + DNH", {"fusion_mode": "residual", "eta": 0.05, "lambda_dnh": 0.3}),
        ("A11", "residual eta=0.05 + DNH + SAT consistency", {"fusion_mode": "residual", "eta": 0.05, "lambda_dnh": 0.3, "lambda_proxy_cons": 0.3, "proxy_aug": "sat07"}),
    ]
    rows.extend(_exp(exp_id, "priority", "Batch A", purpose, args, batch="A") for exp_id, purpose, args in batch_a)

    for i in range(18):
        k = 3 if i >= 16 else 2
        rows.append(_exp(f"B{i:02d}", "priority", "Batch B fusion", f"fusion expansion {i}", {"num_prototypes": k, "fusion_batch_index": i}, batch="B"))
    for i in range(20):
        rows.append(_exp(f"C{i:02d}", "priority", "Batch C loss", f"loss expansion {i}", {"loss_batch_index": i}, batch="C"))
    for i in range(24):
        rows.append(_exp(f"D{i:02d}", "priority", "Batch D prototype geometry", f"prototype geometry {i}", {"geometry_batch_index": i}, batch="D"))
    for i in range(18):
        rows.append(_exp(f"E{i:02d}", "priority", "Batch E zdom", f"zdom usage {i}", {"zdom_batch_index": i}, batch="E"))
    for i in range(20):
        rows.append(_exp(f"F{i:02d}", "priority", "Batch F proxy-val", f"proxy selection {i}", {"proxy_batch_index": i}, batch="F"))
    return rows


def _sgv_bp_args(**overrides: object) -> Dict[str, object]:
    args: Dict[str, object] = {
        "model_name": "SGV-BP-FJMP",
        "ce_on": "auto",
        "kd_on": "auto",
        "num_prototypes": 3,
        "prototype_init": "class_mean",
        "aggregation": "top2_mean",
        "zdom_usage": "detached_gate_only",
        "fusion_mode": "base_protected_residual",
        "logit_calibration": "centered_temperature",
        "rho_init": 0.03,
        "rho_max_stage1": 0.10,
        "rho_max_stage2": 0.25,
        "rho_max_stage3": 0.30,
        "max_delta_norm": 3.0,
        "use_sgv": True,
        "sgv_train_strength": "low,mid",
        "sgv_eval_strength": "low,mid,high",
        "use_sat_reliability": True,
        "lambda_ce_head_clean": 0.30,
        "lambda_ce_head_sat": 0.15,
        "lambda_pres_clean": 3.0,
        "lambda_pres_sat": 1.5,
        "lambda_harm": 2.0,
        "lambda_kd_easy": 1.5,
        "lambda_kd_mid": 0.5,
        "lambda_sgv_head": 0.5,
        "lambda_sgv_safe": 1.0,
        "lambda_proto_sgv": 0.2,
        "lambda_worst_domain_view": 0.3,
        "lambda_gate_easy": 0.08,
        "lambda_gate_view_gap": 0.03,
        "lambda_delta": 0.04,
        "selection_metric": "best_proxy_safe_score",
        "epochs": 30,
    }
    args.update(overrides)
    return args


def _sgv_bp() -> List[Experiment]:
    rows = [
        _exp("EXP-00", "SGV-BP", "baseline", "baseline only", {"mode": "baseline_eval"}, batch="CORE"),
        _exp("EXP-01", "SGV-BP", "route", "original FJMP calibrated fusion", _sgv_bp_args(fusion_mode="calibrated_logit", use_sgv=False), batch="CORE"),
        _exp("EXP-02", "SGV-BP", "route", "pure FJMP head with weak CE and SGV", _sgv_bp_args(fusion_mode="proto_only", ce_on="proto", lambda_harm=0.0), batch="CORE"),
        _exp("EXP-03", "SGV-BP", "route", "pure safe residual without independent head CE", _sgv_bp_args(lambda_ce_head_clean=0.0, lambda_ce_head_sat=0.0), batch="CORE"),
        _exp("EXP-04", "SGV-BP", "route", "SGV-BP-FJMP main head plus safe residual", _sgv_bp_args(), batch="CORE"),
        _exp("EXP-05", "SGV-BP", "ablation", "remove clean-to-sat KD", _sgv_bp_args(lambda_sgv_head=0.0, lambda_sgv_safe=0.0), batch="CORE"),
        _exp("EXP-06", "SGV-BP", "ablation", "remove harm loss", _sgv_bp_args(lambda_harm=0.0), batch="CORE"),
        _exp("EXP-07", "SGV-BP", "ablation", "remove prototype assignment consistency", _sgv_bp_args(lambda_proto_sgv=0.0), batch="CORE"),
        _exp("EXP-08", "SGV-BP", "ablation", "remove worst-domain-view", _sgv_bp_args(lambda_worst_domain_view=0.0), batch="CORE"),
        _exp("EXP-09", "SGV-BP", "ablation", "remove gate view gap", _sgv_bp_args(lambda_gate_view_gap=0.0), batch="CORE"),
        _exp("EXP-10", "SGV-BP", "ablation", "remove sat reliability mask", _sgv_bp_args(use_sat_reliability=False), batch="CORE"),
        _exp("EXP-11", "SGV-BP", "exploration", "kmeans per class prototype init", _sgv_bp_args(prototype_init="kmeans_per_class"), batch="EXP"),
        _exp("EXP-12", "SGV-BP", "exploration", "prototype count K=2", _sgv_bp_args(num_prototypes=2), batch="EXP"),
        _exp("EXP-13", "SGV-BP", "exploration", "trimmed logsumexp aggregation", _sgv_bp_args(aggregation="trimmed_lse"), batch="EXP"),
        _exp("EXP-14", "SGV-BP", "exploration", "zdom zero comparison", _sgv_bp_args(zdom_usage="zero", zdom_mode="zero"), batch="EXP"),
        _exp("EXP-15", "SGV-BP", "exploration", "rho max stage3 0.40", _sgv_bp_args(rho_max_stage3=0.40), batch="EXP"),
        _exp("EXP-16", "SGV-BP", "exploration", "stage3 prototype LR decay control", _sgv_bp_args(prototype_lr_decay_epoch=16, prototype_lr_decay=0.1), batch="EXP"),
        _exp("EXP-17", "SGV-BP", "rho sweep", "conservative rho stage3 0.20", _sgv_bp_args(rho_max_stage2=0.15, rho_max_stage3=0.20), batch="RHO"),
        _exp("EXP-18", "SGV-BP", "rho sweep", "balanced rho stage3 0.35", _sgv_bp_args(rho_max_stage3=0.35), batch="RHO"),
        _exp("EXP-19", "SGV-BP", "rho sweep", "very conservative rho all stages", _sgv_bp_args(rho_max_stage1=0.05, rho_max_stage2=0.15, rho_max_stage3=0.20), batch="RHO"),
        _exp("EXP-20", "SGV-BP", "safety sweep", "strong preservation and harm", _sgv_bp_args(lambda_pres_clean=5.0, lambda_pres_sat=2.0, lambda_harm=3.0), batch="SAFETY"),
        _exp("EXP-21", "SGV-BP", "safety sweep", "lighter preservation for rescue", _sgv_bp_args(lambda_pres_clean=2.0, lambda_pres_sat=1.0, lambda_harm=1.5), batch="SAFETY"),
        _exp("EXP-22", "SGV-BP", "safety sweep", "strong gate easy suppression", _sgv_bp_args(lambda_gate_easy=0.15, lambda_gate_view_gap=0.05), batch="SAFETY"),
        _exp("EXP-23", "SGV-BP", "ce sweep", "lower head CE anti-shortcut", _sgv_bp_args(lambda_ce_head_clean=0.15, lambda_ce_head_sat=0.05), batch="LOSS"),
        _exp("EXP-24", "SGV-BP", "ce sweep", "higher hard rescue CE", _sgv_bp_args(lambda_ce_head_clean=0.45, lambda_ce_head_sat=0.20, lambda_pres_clean=4.0), batch="LOSS"),
        _exp("EXP-25", "SGV-BP", "safe ce", "disable safe CE and rely on safety losses", _sgv_bp_args(lambda_ce_safe_clean=0.0, lambda_ce_safe_sat=0.0), batch="LOSS"),
        _exp("EXP-26", "SGV-BP", "sgv strength", "sat low only warm proxy", _sgv_bp_args(sgv_train_strength="low", sgv_eval_strength="low,mid,high"), batch="SGV"),
        _exp("EXP-27", "SGV-BP", "sgv strength", "sat mid only stronger proxy", _sgv_bp_args(sgv_train_strength="mid", sgv_eval_strength="low,mid,high"), batch="SGV"),
        _exp("EXP-28", "SGV-BP", "sgv strength", "include high stress in training", _sgv_bp_args(sgv_train_strength="low,mid,high", lambda_ce_head_sat=0.05, lambda_sgv_safe=1.2), batch="SGV"),
        _exp("EXP-29", "SGV-BP", "sgv consistency", "strong SGV safe consistency", _sgv_bp_args(lambda_sgv_head=0.5, lambda_sgv_safe=1.5, lambda_sgv_margin=0.4), batch="SGV"),
        _exp("EXP-30", "SGV-BP", "sgv consistency", "head consistency dominated", _sgv_bp_args(lambda_sgv_head=1.0, lambda_sgv_safe=0.7), batch="SGV"),
        _exp("EXP-31", "SGV-BP", "prototype", "K4 class mean top2", _sgv_bp_args(num_prototypes=4), batch="PROTO"),
        _exp("EXP-32", "SGV-BP", "prototype", "K3 logsumexp aggregation", _sgv_bp_args(aggregation="logsumexp"), batch="PROTO"),
        _exp("EXP-33", "SGV-BP", "prototype", "K3 mean aggregation conservative", _sgv_bp_args(aggregation="mean"), batch="PROTO"),
        _exp("EXP-34", "SGV-BP", "prototype", "strong prototype SGV consistency", _sgv_bp_args(lambda_proto_sgv=0.5), batch="PROTO"),
        _exp("EXP-35", "SGV-BP", "prototype", "no prototype LR decay", _sgv_bp_args(prototype_lr_decay=1.0), batch="PROTO"),
        _exp("EXP-36", "SGV-BP", "zdom", "zdom normal gate-only comparison", _sgv_bp_args(zdom_usage="detached_gate_only", zdom_mode="normal"), batch="ZDOM"),
        _exp("EXP-37", "SGV-BP", "zdom", "zdom shuffled shortcut stress", _sgv_bp_args(zdom_mode="shuffled"), batch="ZDOM"),
        _exp("EXP-38", "SGV-BP", "zdom", "zdom mean neutralization", _sgv_bp_args(zdom_mode="mean"), batch="ZDOM"),
        _exp("EXP-39", "SGV-BP", "delta", "tight delta norm 1.5", _sgv_bp_args(max_delta_norm=1.5), batch="FUSION"),
        _exp("EXP-40", "SGV-BP", "delta", "wide delta norm 5.0 with strong safety", _sgv_bp_args(max_delta_norm=5.0, lambda_pres_clean=5.0, lambda_harm=3.0), batch="FUSION"),
        _exp("EXP-41", "SGV-BP", "calibration", "fixed temperature calibration", _sgv_bp_args(logit_calibration="centered_temperature_fixed"), batch="FUSION"),
        _exp("EXP-42", "SGV-BP", "optimizer", "lower projector and proto LR", _sgv_bp_args(lr_projector=3e-4, lr_proto=1e-4, lr_rho=1e-4), batch="OPT"),
        _exp("EXP-43", "SGV-BP", "optimizer", "higher gate/rho LR", _sgv_bp_args(lr_gate=2e-3, lr_rho=5e-4), batch="OPT"),
        _exp("EXP-44", "SGV-BP", "schedule", "short anti-drift 20 epoch", _sgv_bp_args(epochs=20, prototype_lr_decay_epoch=10), batch="SCHED"),
        _exp("EXP-45", "SGV-BP", "schedule", "long refinement 40 epoch", _sgv_bp_args(epochs=40, prototype_lr_decay_epoch=16, prototype_lr_decay=0.05), batch="SCHED"),
        _exp("EXP-46", "SGV-BP", "selection", "proxy score conservative rho target", _sgv_bp_args(selection_metric="best_proxy_safe_score_rho020", rho_max_stage3=0.25), batch="SELECT"),
        _exp("EXP-47", "SGV-BP", "selection", "source safe checkpoint control", _sgv_bp_args(selection_metric="best_source_safe"), batch="SELECT"),
        _exp("EXP-48", "SGV-BP", "selection", "harm safe checkpoint control", _sgv_bp_args(selection_metric="best_harm_safe"), batch="SELECT"),
    ]
    return rows


def _sgv_bp_loss_design() -> List[Experiment]:
    """Focused post-5.16 loss-design matrix for cross-domain FJMP tuning."""

    def loss_args(**overrides):
        base = {
            "lambda_ce_head_clean": 0.30,
            "lambda_ce_head_sat": 0.10,
            "lambda_ce_safe_clean": 0.02,
            "lambda_ce_safe_sat": 0.0,
            "lambda_pres_clean": 2.0,
            "lambda_pres_sat": 0.5,
            "lambda_harm": 1.5,
            "lambda_kd_easy": 1.0,
            "lambda_kd_mid": 0.2,
            "lambda_kd_hard_low_margin": 0.0,
            "lambda_sgv_head": 0.8,
            "lambda_sgv_safe": 0.3,
            "lambda_sgv_margin": 0.0,
            "lambda_proto_sgv": 0.2,
            "lambda_worst_domain_view": 0.3,
            "lambda_gate_easy": 0.05,
            "lambda_gate_view_gap": 0.02,
            "lambda_delta": 0.04,
            "selection_metric": "best_proxy_safe_score",
            "use_sat_reliability": True,
            "epochs": 30,
        }
        base.update(overrides)
        return _sgv_bp_args(**base)

    rows = [
        _exp("LD-00", "SGV-BP", "fixed-control", "main SGV-BP rerun after worst-domain group fix", _sgv_bp_args(), batch="LOSS-DESIGN"),
        _exp("LD-01", "SGV-BP", "recommended", "conservative head-led safe residual loss", loss_args(), batch="LOSS-DESIGN"),
        _exp("LD-02", "SGV-BP", "sgv-safe", "disable safe consistency; head/proto learn view stability", loss_args(lambda_sgv_safe=0.0), batch="LOSS-DESIGN"),
        _exp("LD-03", "SGV-BP", "sgv-safe", "moderate safe consistency 0.5", loss_args(lambda_sgv_safe=0.5), batch="LOSS-DESIGN"),
        _exp("LD-04", "SGV-BP", "sgv-head", "strong head consistency with weak safe consistency", loss_args(lambda_sgv_head=1.0, lambda_sgv_safe=0.3), batch="LOSS-DESIGN"),
        _exp("LD-05", "SGV-BP", "sgv-head", "head-dominated balance with mild safe consistency", loss_args(lambda_sgv_head=1.0, lambda_sgv_safe=0.7, lambda_sgv_margin=0.0), batch="LOSS-DESIGN"),
        _exp("LD-06", "SGV-BP", "worst-domain", "light worst-domain-view pressure", loss_args(lambda_worst_domain_view=0.1), batch="LOSS-DESIGN"),
        _exp("LD-07", "SGV-BP", "worst-domain", "strong worst-domain-view pressure", loss_args(lambda_worst_domain_view=0.5), batch="LOSS-DESIGN"),
        _exp("LD-08", "SGV-BP", "safety", "lighter preservation and harm for rescue room", loss_args(lambda_pres_clean=1.5, lambda_pres_sat=0.3, lambda_harm=1.0), batch="LOSS-DESIGN"),
        _exp("LD-09", "SGV-BP", "safety", "strong clean preservation with weak sat preservation", loss_args(lambda_pres_clean=3.0, lambda_pres_sat=0.3, lambda_harm=2.0), batch="LOSS-DESIGN"),
        _exp("LD-10", "SGV-BP", "kd", "minimal clean KD to avoid baseline lock-in", loss_args(lambda_kd_easy=0.5, lambda_kd_mid=0.0), batch="LOSS-DESIGN"),
        _exp("LD-11", "SGV-BP", "ce", "more head CE on sat, no safe CE", loss_args(lambda_ce_head_clean=0.35, lambda_ce_head_sat=0.20, lambda_ce_safe_clean=0.0, lambda_ce_safe_sat=0.0), batch="LOSS-DESIGN"),
        _exp("LD-12", "SGV-BP", "rho", "conservative rho cap with recommended loss", loss_args(rho_max_stage2=0.15, rho_max_stage3=0.20), batch="LOSS-DESIGN"),
        _exp("LD-13", "SGV-BP", "opt", "lower projector/proto/rho LR with recommended loss", loss_args(lr_projector=3e-4, lr_proto=1e-4, lr_rho=1e-4), batch="LOSS-DESIGN"),
        _exp("LD-14", "SGV-BP", "sgv-strength", "low-strength SGV only to reduce proxy over-constraint", loss_args(sgv_train_strength="low", lambda_sgv_head=0.8, lambda_sgv_safe=0.3), batch="LOSS-DESIGN"),
        _exp("LD-15", "SGV-BP", "schedule", "long refinement recommended loss", loss_args(epochs=40, prototype_lr_decay_epoch=16, prototype_lr_decay=0.05), batch="LOSS-DESIGN"),
    ]
    return rows


def _a03_a06_loss_attribution() -> List[Experiment]:
    """A03/A06 anchored matrix to identify which losses improve cross-domain UDU."""

    a03 = {
        "num_prototypes": 3,
        "zdom_mode": "zero",
        "save_checkpoints": False,
    }
    a06 = {
        "init_zdom_mode": "zero",
        "zdom_mode": "zero",
        "num_prototypes": 3,
        "save_checkpoints": False,
    }

    def a03_args(**overrides):
        args = dict(a03)
        args.update(overrides)
        return args

    def a06_args(**overrides):
        args = dict(a06)
        args.update(overrides)
        return args

    rows = [
        _exp(
            "R83-00",
            "FJMP-LOSS",
            "A03 control",
            "A03 exact 30-epoch control from 5.15 log",
            a03_args(),
            expected="Track whether late epochs drift from the E007 UDU high point.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-01",
            "FJMP-LOSS",
            "A06 control",
            "A06 exact 30-epoch control from 5.15 log",
            a06_args(),
            expected="Same as A03 with explicit zero init/train zdom.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-02",
            "FJMP-LOSS",
            "A03 early high",
            "A03 stopped at epoch 7 where both source logs reached 86.83 UDU",
            a03_args(epochs=7),
            expected="Reproduce the observed 86.83% unseen-day unseen-rx high point.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-03",
            "FJMP-LOSS",
            "A06 early high",
            "A06 stopped at epoch 7 where both source logs reached 86.83 UDU",
            a06_args(epochs=7),
            expected="Check whether init_zdom_mode=zero is neutral at the high point.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-04",
            "FJMP-LOSS",
            "margin-preserve gain",
            "A03 plus margin preservation seen to help cross-domain runs",
            a03_args(lambda_margin_preserve=0.3, margin_preserve_delta=0.0, epochs=22),
            expected="If UDU improves while harm stays low, margin preservation is beneficial.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-05",
            "FJMP-LOSS",
            "margin-preserve gain",
            "A06 plus margin preservation with explicit zero zdom init",
            a06_args(lambda_margin_preserve=0.3, margin_preserve_delta=0.0, epochs=22),
            expected="Confirm whether the A03 margin-preserve effect survives the A06 init setting.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-06",
            "FJMP-LOSS",
            "KD ablation",
            "A03 high-point schedule without fused KD",
            a03_args(epochs=7, kd_on="none", lambda_kd=0.0),
            expected="If UDU rises or harm falls, KD is over-locking to the source baseline.",
            batch="A03-A06-REPRO",
        ),
        _exp(
            "R83-07",
            "FJMP-LOSS",
            "CE target ablation",
            "A03 high-point schedule with proto CE instead of fused CE",
            a03_args(epochs=7, ce_on="proto", ce_proto_weight=1.0, ce_fused_weight=0.0),
            expected="Separates whether cross-domain gain comes from prototype head learning or fused calibration.",
            batch="A03-A06-REPRO",
        ),
    ]
    return rows


def _fjmp_v2() -> List[Experiment]:
    """FJMP-v2 base-anchored safe residual ablation matrix."""

    base: Dict[str, object] = {
        "fjmp_version": "v2",
        "model_name": "FJMP_V2_K3_SAFE_RESIDUAL",
        "num_prototypes": 3,
        "proto_dim": 256,
        "zdom_mode": "zero",
        "init_zdom_mode": "zero",
        "epochs": 12,
        "rho_max": 0.15,
        "delta_clip": 3.0,
        "proto_dropout": 0.10,
        "lambda_ce_trim": 1.0,
        "lambda_kd_selective": 0.3,
        "lambda_proto_div": 0.01,
        "lambda_proto_usage": 0.005,
        "lambda_assign_entropy": 0.001,
        "lambda_delta": 0.01,
        "lambda_logit_residual": 0.01,
        "lambda_gate": 0.003,
        "lambda_margin_preserve": 0.0,
        "lambda_gate_view_gap": 0.0,
        "lambda_worst_domain_view": 0.0,
        "lambda_proto_sgv": 0.0,
        "lambda_sgv_safe": 0.0,
        "lambda_sgv_margin": 0.0,
        "lambda_pres_clean": 0.0,
        "lambda_pres_sat": 0.0,
        "lambda_harm": 0.0,
        "use_sgv": False,
    }

    def args(**overrides: object) -> Dict[str, object]:
        out = dict(base)
        out.update(overrides)
        return out

    rows = [
        _exp(
            "V2-01",
            "FJMP-V2",
            "boundary CE",
            "CE_trim with legacy-like regularization control",
            args(lambda_kd_selective=0.0, lambda_proto_usage=0.0, lambda_assign_entropy=0.0, lambda_delta=0.0005),
            expected="final stability improves versus all-source fused CE",
            batch="FJMP-V2",
        ),
        _exp(
            "V2-02",
            "FJMP-V2",
            "selective KD",
            "CE_trim plus selective KD protects high-confidence correct base samples",
            args(lambda_proto_usage=0.0, lambda_assign_entropy=0.0, lambda_delta=0.0005),
            expected="harm falls without locking base errors",
            batch="FJMP-V2",
        ),
        _exp(
            "V2-03",
            "FJMP-V2",
            "anti-collapse",
            "new angular diversity plus usage balance and assignment entropy",
            args(lambda_delta=0.0005),
            expected="prototype usage is more balanced and dead prototype rate falls",
            batch="FJMP-V2",
        ),
        _exp(
            "V2-04",
            "FJMP-V2",
            "main",
            "FJMP-v2 K3 safe residual main configuration",
            args(),
            expected="stable final UDU, bounded rho/delta, low harm",
            batch="FJMP-V2",
        ),
        _exp(
            "V2-05",
            "FJMP-V2",
            "K scan",
            "K=2 conservative baseline",
            args(num_prototypes=2),
            expected="stable but potentially lower rescue",
            batch="FJMP-V2",
        ),
        _exp(
            "V2-06",
            "FJMP-V2",
            "K scan",
            "K=4 upper-capacity stress test",
            args(num_prototypes=4),
            expected="check peak/final gap and dead prototype rate",
            batch="FJMP-V2",
        ),
    ]
    return rows


def _fjmp_v3() -> List[Experiment]:
    """Aggressive rescue residual experiments for cross-domain gains."""

    base: Dict[str, object] = {
        "fjmp_version": "v3",
        "model_name": "FJMP_V3_AGGRESSIVE_RESCUE",
        "num_prototypes": 3,
        "proto_dim": 256,
        "zdom_mode": "zero",
        "init_zdom_mode": "zero",
        "epochs": 12,
        "rho_max": 0.30,
        "delta_clip": 5.0,
        "proto_dropout": 0.10,
        "proto_warmup_epochs": 3,
        "lambda_ce_proto_warmup": 1.0,
        "lambda_ce_trim": 1.5,
        "ce_trim_margin": 5.0,
        "ce_trim_tau": 1.25,
        "ce_trim_rescue_weight": 1.5,
        "lambda_kd_selective": 0.1,
        "kd_conf_threshold": 0.95,
        "kd_margin_threshold": 5.0,
        "dynamic_rho_cap": True,
        "rho_easy_cap": 0.03,
        "rho_boundary_cap": 0.30,
        "rho_easy_conf": 0.95,
        "rho_easy_margin": 5.0,
        "rho_boundary_margin": 3.0,
        "lambda_proto_div": 0.01,
        "lambda_proto_usage": 0.01,
        "lambda_assign_entropy": 0.001,
        "lambda_delta": 0.005,
        "delta_ratio_max": 0.15,
        "lambda_logit_residual": 0.003,
        "logit_residual_target_norm": 5.0,
        "lambda_gate": 0.001,
        "lambda_margin_preserve": 0.0,
        "lambda_gate_view_gap": 0.0,
        "lambda_worst_domain_view": 0.0,
        "lambda_proto_sgv": 0.0,
        "lambda_sgv_safe": 0.0,
        "lambda_sgv_margin": 0.0,
        "lambda_pres_clean": 0.0,
        "lambda_pres_sat": 0.0,
        "lambda_harm": 0.0,
        "use_sgv": False,
    }

    def args(**overrides: object) -> Dict[str, object]:
        out = dict(base)
        out.update(overrides)
        return out

    return [
        _exp(
            "V3-01",
            "FJMP-V3",
            "proto warmup",
            "proto CE warmup added to V2 main structure",
            args(lambda_ce_trim=1.0, ce_trim_margin=3.0, ce_trim_tau=0.75, ce_trim_rescue_weight=0.5, lambda_kd_selective=0.3, kd_conf_threshold=0.0, kd_margin_threshold=-1.0, dynamic_rho_cap=False, rho_max=0.15, delta_clip=3.0, lambda_delta=0.01, lambda_logit_residual=0.01),
            expected="test whether prototype warmup alone restores rescue capacity",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-02",
            "FJMP-V3",
            "wide CE trim",
            "proto warmup plus wider boundary/rescue CE",
            args(lambda_kd_selective=0.3, kd_conf_threshold=0.0, kd_margin_threshold=-1.0, dynamic_rho_cap=False, rho_max=0.15, delta_clip=3.0, lambda_delta=0.01, lambda_logit_residual=0.01),
            expected="increase cross-domain correction while still under V2 rho cap",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-03",
            "FJMP-V3",
            "hard selective KD",
            "wide CE trim with KD only on very reliable base samples",
            args(dynamic_rho_cap=False, rho_max=0.15, delta_clip=3.0, lambda_delta=0.01, lambda_logit_residual=0.01),
            expected="free low-margin and likely wrong samples from base distillation",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-04",
            "FJMP-V3",
            "dynamic rho",
            "hard selective KD plus dynamic rho caps",
            args(rho_max=0.30, delta_clip=3.0, lambda_delta=0.01, lambda_logit_residual=0.01),
            expected="shift correction budget from easy source samples to boundary samples",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-05",
            "FJMP-V3",
            "main aggressive",
            "K3 aggressive rescue residual main configuration",
            args(),
            expected="target UDU peak >= 86.8 with harm_rate <= 0.0015",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-06",
            "FJMP-V3",
            "K4 aggressive",
            "K4 version of aggressive rescue residual",
            args(num_prototypes=4),
            expected="check whether extra capacity raises peak without dead prototype drift",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-07",
            "FJMP-V3",
            "very aggressive CE",
            "K3 with stronger rescue CE and lighter regularization",
            args(lambda_ce_trim=2.0, ce_trim_margin=6.0, ce_trim_rescue_weight=2.0, lambda_delta=0.003, lambda_logit_residual=0.001),
            expected="upper bound for rescue strength; reject if source/harm thresholds fail",
            batch="FJMP-V3",
        ),
        _exp(
            "V3-08",
            "FJMP-V3",
            "higher rho cap",
            "K3 with rho cap 0.40 and same hard safeguards",
            args(rho_max=0.40, rho_boundary_cap=0.40),
            expected="stress-test residual intervention strength",
            batch="FJMP-V3",
        ),
    ]


def build_experiment_manifest(layers: Sequence[str] | None = None) -> List[Experiment]:
    selected = {str(layer).upper() for layer in layers} if layers else None
    groups = [
        _l0(),
        _l1(),
        _l2(),
        _l3(),
        _l4(),
        _l5(),
        _l6(),
        _priority_batches(),
        _sgv_bp(),
        _sgv_bp_loss_design(),
        _a03_a06_loss_attribution(),
        _fjmp_v2(),
        _fjmp_v3(),
    ]
    manifest = [row for group in groups for row in group]
    if selected is not None:
        manifest = [
            row
            for row in manifest
            if str(row["layer"]).upper() in selected or str(row.get("batch", "")).upper() in selected
        ]
    return manifest


def render_train_command(row: Mapping[str, object], base_command: str = "python -m FJMP.train_fjmp") -> str:
    args = row.get("args", {}) or {}
    parts = [base_command, "--output_dir", f"runs/fjmp_v2/{row['id']}"]
    for key, value in args.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        parts.extend([f"--{key}", str(value)])
    return " ".join(parts)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the FJMP v2 experiment manifest.")
    parser.add_argument("--layer", action="append", default=None, help="Filter by L0-L6 or batch A-F. Can be repeated.")
    parser.add_argument("--commands", action="store_true", help="Print FJMP.train_fjmp command templates instead of JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_experiment_manifest(args.layer)
    if args.commands:
        for row in manifest:
            print(render_train_command(row))
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
