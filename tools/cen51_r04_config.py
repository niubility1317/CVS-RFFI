"""Shared CEN51_R04 ratio-path configuration.

This module is intentionally small: CEN51 few-shot planners may tune below
100 samples per combo, but K >= 100 must restore the original CEN51_R04
ratio=0.1 recipe instead of applying low-shot controller overrides.
"""

from __future__ import annotations

from typing import Dict


SAT_SCENARIOS = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
RESTORE_SHOT_THRESHOLD = 100


CEN51_R04_RATIO_PARAMS: Dict[str, object] = {
    "train_mode": "centralized",
    "batch_size": 256,
    "eval_batch_size": 256,
    "dataset": "wisig",
    "wisig_protocol": "cvs_day_rx",
    "wisig_equalized": 1,
    "wisig_domain": "rx_day",
    "wisig_out_len": 256,
    "wisig_train_ratio": 0.1,
    "wisig_val_ratio": -1.0,
    "wisig_guard_gap": 8,
    "wisig_split_strategy": "random",
    "wisig_cap_strategy": "random",
    "wisig_train_days": "0,1",
    "wisig_test_days": "2,3",
    "wisig_train_rxs": "0,1,2,3,4,5,6",
    "wisig_test_rxs": "7,8,9,10,11",
    "epochs": 200,
    "test_eval_policy": "interval_final",
    "test_eval_start_epoch": 1,
    "test_eval_interval": 10,
    "eval_sat_channel": True,
    "eval_sat_on": "test_unseen_day_unseen_rx",
    "eval_sat_scenarios": SAT_SCENARIOS,
    "sat_eval_max_batches": -1,
    "arch_family": "cvsincnet",
    "slim_group": "none",
    "branch_ablation": "no_dac",
    "domain_branch_ablation": "no_stats",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    "exp_group": "s3_rxrobust_no_dac",
    "model_variant": "lite_d",
    "use_aug": True,
    "use_concat_sat_channel_aug": True,
    "concat_sat_start_epoch": 1,
    "lambda_sat_cls": 0.0,
    "lambda_sat_cons": 0.006,
    "use_sat_consistency": True,
    "sat_cons_start_epoch": 118,
    "seed": 1337,
    "use_mixstyle": True,
    "mixstyle_layers": "time_down,t1",
    "mixstyle_mix": "same_tx_crossdomain",
    "mixstyle_fallback": "skip",
    "mixstyle_strength": 0.70,
    "mixstyle_p": 0.18,
    "mixstyle_late_start": 110,
    "mixstyle_late_ramp_epochs": 40,
    "mixstyle_late_min_p": 0.05,
    "mixstyle_late_min_strength": 0.32,
    "sat_train_scenarios": SAT_SCENARIOS,
    "sat_view_prob": 1.0,
    "sat_view_schedule": (
        "1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;"
        "115@0.82:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
    ),
    "domain_freq_stability_mode": "dsq",
    "freq_stability_channels": 2,
    "primary_udu_weight": 0.84,
    "concat_sat_ce_weight": 1.19,
    "pa_orders": "1,3,5",
    "lambda_group_ce": 0.088,
    "group_ce_mode": "smooth_dro_capped",
    "group_ce_min_domains": 4,
    "group_ce_top_frac": 0.20,
    "groupdro_tau": 0.37,
    "groupdro_cap": 0.48,
    "use_proto_memory": True,
    "lambda_proto": 0.016,
    "proto_momentum": 0.970,
    "lambda_supcon_id": 0.022,
    "supcon_temp": 0.12,
    "lambda_fishr": 0.002,
    "fishr_min_domains": 4,
    "generalization_feature": "z_id",
    "collapse_guard": True,
    "collapse_guard_min_epoch": 35,
    "collapse_guard_best_margin": 12.0,
    "collapse_guard_max_skipped_delta": 2,
    "use_ema_ckpt": True,
    "ema_decay": 0.999,
    "use_swad_ckpt": True,
    "swad_interval": 1,
    "swad_start_epoch": 70,
    "swad_tolerance": 0.34,
}


def should_restore_cen51_r04(shots: int) -> bool:
    return int(shots) >= RESTORE_SHOT_THRESHOLD


def cen51_r04_ratio_params(*, seed: int = 1337) -> Dict[str, object]:
    params = dict(CEN51_R04_RATIO_PARAMS)
    params["seed"] = int(seed)
    return params
