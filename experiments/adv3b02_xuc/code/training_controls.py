"""Pure-Python training control helpers for long RFFI runs.

These helpers intentionally avoid importing torch so they can be unit-tested on
machines that only prepare launch scripts or inspect logs.
"""

from __future__ import annotations

import math
import hashlib
import json
from typing import Dict, List


def _finite_float(value, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _ramp01(epoch: int, start: int, ramp_epochs: int) -> float:
    start = int(start)
    ramp_epochs = int(ramp_epochs)
    if start <= 0:
        return 0.0
    if int(epoch) <= start:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return max(0.0, min(1.0, (int(epoch) - start) / float(ramp_epochs)))


def compute_mixstyle_epoch_state(
    *,
    epoch: int,
    base_p: float,
    base_strength: float,
    late_start: int = 0,
    ramp_epochs: int = 1,
    min_p: float = -1.0,
    min_strength: float = -1.0,
    stop_epoch: int = 0,
) -> Dict[str, float]:
    """Return the MixStyle probability/strength for a training epoch.

    The schedule keeps MixStyle active during representation discovery, then
    anneals it late so the classifier can consolidate transmitter identity
    instead of being permanently perturbed.
    """
    base_p = max(0.0, _finite_float(base_p, 0.0))
    base_strength = max(0.0, _finite_float(base_strength, 0.0))
    stop_epoch = int(stop_epoch)
    if stop_epoch > 0 and int(epoch) > stop_epoch:
        return {
            "enabled": False,
            "p": 0.0,
            "strength": 0.0,
            "phase": "stopped",
            "anneal_t": 1.0,
        }

    if base_p <= 0.0 or base_strength <= 0.0:
        return {
            "enabled": False,
            "p": 0.0,
            "strength": 0.0,
            "phase": "disabled",
            "anneal_t": 0.0,
        }

    target_p = base_p if float(min_p) < 0.0 else max(0.0, _finite_float(min_p, base_p))
    target_strength = base_strength if float(min_strength) < 0.0 else max(0.0, _finite_float(min_strength, base_strength))
    t = _ramp01(int(epoch), int(late_start), int(ramp_epochs))
    p = base_p * (1.0 - t) + min(base_p, target_p) * t
    strength = base_strength * (1.0 - t) + min(base_strength, target_strength) * t
    enabled = p > 0.0 and strength > 0.0
    return {
        "enabled": bool(enabled),
        "p": float(p if enabled else 0.0),
        "strength": float(strength if enabled else 0.0),
        "phase": "late_anneal" if t > 0.0 else "base",
        "anneal_t": float(t),
    }


def collapse_guard_decision(
    *,
    enabled: bool,
    epoch: int,
    min_epoch: int,
    train_tx_acc: float,
    val_tx_acc: float,
    test_tx_acc: float,
    random_tx_acc: float,
    best_primary_score: float,
    current_primary_score: float,
    best_margin: float,
    skipped_backward_delta: int,
    max_skipped_delta: int,
    orth_loss: float,
    random_margin: float = 3.0,
) -> Dict[str, object]:
    """Decide whether latest checkpoint should be protected from a bad epoch."""
    if (not bool(enabled)) or int(epoch) < int(min_epoch):
        return {"skip_latest": False, "reason": ""}

    random_acc = max(0.0, _finite_float(random_tx_acc, 0.0))
    random_floor = random_acc + max(0.0, _finite_float(random_margin, 3.0))
    train_acc = _finite_float(train_tx_acc, 0.0)
    val_acc = _finite_float(val_tx_acc, 0.0)
    test_acc = _finite_float(test_tx_acc, 0.0)
    best_score = _finite_float(best_primary_score, -1.0)
    cur_score = _finite_float(current_primary_score, -1.0)
    margin = max(0.0, _finite_float(best_margin, 25.0))

    random_level = (val_acc <= random_floor and test_acc <= random_floor) or (
        train_acc <= random_floor and test_acc <= random_floor
    )
    degraded_from_best = best_score > 0.0 and cur_score >= 0.0 and cur_score <= (best_score - margin)
    non_finite_orth = not math.isfinite(_finite_float(orth_loss, math.nan))
    too_many_skips = int(skipped_backward_delta) > int(max_skipped_delta)

    reasons = []
    if random_level:
        reasons.append("random_level_acc")
    if degraded_from_best:
        reasons.append("primary_drop")
    if non_finite_orth:
        reasons.append("non_finite_orth")
    if too_many_skips:
        reasons.append("unsafe_backward_spike")

    skip_latest = bool(random_level or too_many_skips or (non_finite_orth and degraded_from_best))
    return {
        "skip_latest": skip_latest,
        "reason": ",".join(reasons) if skip_latest else "",
    }


SAT_CHANNEL_SCENARIO_CONFIGS: Dict[str, Dict[str, object]] = {
    "leo_clear_weak": {
        "channel_model": "leo_residual",
        "weather": "clear",
        "scenario": "leo_residual",
        "loo_level": "light",
        "orbit_probs": {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0},
        "theta_deg": (35.0, 90.0),
        "snr_db": (22.0, 32.0),
        "cfo_std_hz": 50.0,
        "phase_noise_inc_std": (0.0, 5e-4),
        "use_residual_doppler": True,
        "apply_path_loss_to_iq": False,
        "enable_atmospheric_fading": False,
        "enable_iq_imbalance": False,
        "fading_mode": "rician",
        "K_db_range": (16.0, 24.0),
        "enable_multipath": True,
        "multipath_profile": "weak",
        "num_taps": (2, 2),
        "max_delay_samp": 2,
        "pwr_decay": 0.08,
        "agc_resid_db": (-0.2, 0.2),
    },
    "leo_low_elev_weak": {
        "channel_model": "leo_residual",
        "weather": "clear",
        "scenario": "leo_residual",
        "loo_level": "light",
        "orbit_probs": {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0},
        "theta_deg": (10.0, 35.0),
        "snr_db": (16.0, 28.0),
        "cfo_std_hz": 90.0,
        "phase_noise_inc_std": (1e-4, 8e-4),
        "use_residual_doppler": True,
        "apply_path_loss_to_iq": False,
        "enable_atmospheric_fading": False,
        "enable_iq_imbalance": False,
        "fading_mode": "shadowed_rician",
        "K_db_range": (8.0, 18.0),
        "enable_multipath": True,
        "multipath_profile": "weak",
        "num_taps": (2, 2),
        "max_delay_samp": 3,
        "pwr_decay": 0.12,
        "agc_resid_db": (-0.3, 0.3),
    },
    "leo_rain_weak": {
        "channel_model": "leo_residual",
        "weather": "rain",
        "scenario": "leo_residual",
        "loo_level": "light",
        "orbit_probs": {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0},
        "theta_deg": (20.0, 80.0),
        "snr_db": (14.0, 26.0),
        "cfo_std_hz": 70.0,
        "phase_noise_inc_std": (1e-4, 7e-4),
        "use_residual_doppler": True,
        "apply_path_loss_to_iq": False,
        "enable_atmospheric_fading": False,
        "enable_iq_imbalance": False,
        "fading_mode": "rician",
        "K_db_range": (10.0, 20.0),
        "enable_multipath": True,
        "multipath_profile": "weak",
        "num_taps": (2, 2),
        "max_delay_samp": 3,
        "pwr_decay": 0.10,
        "agc_resid_db": (-0.3, 0.3),
    },
    "clear_leo": {
        "weather": "clear",
        "scenario": "urban",
        "loo_level": "mid",
        "orbit_probs": {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0},
        "theta_deg": (30.0, 90.0),
        "snr_db": (20.0, 30.0),
        "cfo_std_hz": 200.0,
        "phase_noise_inc_std": (0.0, 2e-3),
        "enable_multipath": False,
    },
    "low_elev_leo": {
        "weather": "clear",
        "scenario": "urban",
        "loo_level": "mid",
        "orbit_probs": {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0},
        "theta_deg": (10.0, 30.0),
        "snr_db": (15.0, 28.0),
        "cfo_std_hz": 350.0,
        "phase_noise_inc_std": (5e-4, 3e-3),
        "enable_multipath": False,
    },
    "rain_leo": {
        "weather": "rain",
        "scenario": "urban",
        "loo_level": "mid",
        "orbit_probs": {"LEO": 1.0, "MEO": 0.0, "GEO": 0.0},
        "theta_deg": (20.0, 80.0),
        "snr_db": (10.0, 25.0),
        "cfo_std_hz": 250.0,
        "phase_noise_inc_std": (5e-4, 3e-3),
        "enable_multipath": False,
    },
    "storm_mp": {
        "weather": "storm",
        "scenario": "urban",
        "loo_level": "severe",
        "orbit_probs": {"LEO": 0.8, "MEO": 0.2, "GEO": 0.0},
        "theta_deg": (10.0, 35.0),
        "snr_db": (8.0, 20.0),
        "cfo_std_hz": 400.0,
        "phase_noise_inc_std": (1e-3, 4e-3),
        "enable_multipath": True,
        "num_taps": (2, 5),
        "max_delay_samp": 6,
        "pwr_decay": 0.8,
    },
    "geo_clear": {
        "weather": "clear",
        "scenario": "urban",
        "loo_level": "light",
        "orbit_probs": {"LEO": 0.0, "MEO": 0.0, "GEO": 1.0},
        "theta_deg": (25.0, 80.0),
        "snr_db": (18.0, 30.0),
        "cfo_std_hz": 100.0,
        "phase_noise_inc_std": (0.0, 1.5e-3),
        "enable_multipath": False,
    },
    "mixed_orbit": {
        "weather": "cloudy",
        "scenario": "urban",
        "loo_level": "mid",
        "orbit_probs": {"LEO": 0.6, "MEO": 0.3, "GEO": 0.1},
        "theta_deg": (10.0, 90.0),
        "snr_db": (12.0, 30.0),
        "cfo_std_hz": 300.0,
        "phase_noise_inc_std": (0.0, 3e-3),
        "enable_multipath": True,
        "num_taps": (2, 4),
        "max_delay_samp": 5,
        "pwr_decay": 0.75,
    },
}


SAT_CHANNEL_PROTOCOL_FAMILIES: Dict[str, str] = {
    "leo_clear_weak": "simplified_leo_residual_weak_v1",
    "leo_low_elev_weak": "simplified_leo_residual_weak_v1",
    "leo_rain_weak": "simplified_leo_residual_weak_v1",
    "clear_leo": "legacy_satellite_physics_holdout_v1",
    "low_elev_leo": "legacy_satellite_physics_holdout_v1",
    "rain_leo": "legacy_satellite_physics_holdout_v1",
    "storm_mp": "legacy_satellite_physics_holdout_v1",
    "geo_clear": "legacy_satellite_physics_holdout_v1",
    "mixed_orbit": "legacy_satellite_physics_holdout_v1",
}


def parse_sat_scenarios(raw: str) -> List[str]:
    """Parse a comma/semicolon/plus separated satellite scenario list."""
    seen = set()
    out: List[str] = []
    normalized = str(raw or "").replace(";", ",").replace("+", ",")
    for item in normalized.split(","):
        name = item.strip().lower().replace("-", "_")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def satellite_protocol_manifest(
    train_scenarios: List[str],
    eval_scenarios: List[str],
    *,
    require_disjoint: bool = False,
) -> Dict[str, object]:
    """Describe and optionally enforce a held-out satellite stress protocol."""

    train = parse_sat_scenarios(",".join(str(v) for v in train_scenarios))
    evaluate = parse_sat_scenarios(",".join(str(v) for v in eval_scenarios))
    unknown = sorted(
        {
            name
            for name in [*train, *evaluate]
            if name not in SAT_CHANNEL_SCENARIO_CONFIGS or name not in SAT_CHANNEL_PROTOCOL_FAMILIES
        }
    )
    train_families = sorted({SAT_CHANNEL_PROTOCOL_FAMILIES[name] for name in train if name in SAT_CHANNEL_PROTOCOL_FAMILIES})
    eval_families = sorted({SAT_CHANNEL_PROTOCOL_FAMILIES[name] for name in evaluate if name in SAT_CHANNEL_PROTOCOL_FAMILIES})
    scenario_overlap = sorted(set(train).intersection(evaluate))
    family_overlap = sorted(set(train_families).intersection(eval_families))
    config_hashes = {
        name: hashlib.sha256(
            json.dumps(
                SAT_CHANNEL_SCENARIO_CONFIGS.get(name, {}),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for name in [*train, *evaluate]
        if name in SAT_CHANNEL_SCENARIO_CONFIGS
    }
    train_hashes = sorted({config_hashes[name] for name in train if name in config_hashes})
    eval_hashes = sorted({config_hashes[name] for name in evaluate if name in config_hashes})
    config_hash_overlap = sorted(set(train_hashes).intersection(eval_hashes))
    channel_implementations = {
        name: str(SAT_CHANNEL_SCENARIO_CONFIGS.get(name, {}).get("channel_model", "legacy_full"))
        for name in [*train, *evaluate]
        if name in SAT_CHANNEL_SCENARIO_CONFIGS
    }
    train_channel_implementations = sorted(
        {channel_implementations[name] for name in train if name in channel_implementations}
    )
    eval_channel_implementations = sorted(
        {channel_implementations[name] for name in evaluate if name in channel_implementations}
    )
    channel_implementation_overlap = sorted(
        set(train_channel_implementations).intersection(eval_channel_implementations)
    )
    registry_payload = {
        name: {
            "family": SAT_CHANNEL_PROTOCOL_FAMILIES.get(name, ""),
            "config": SAT_CHANNEL_SCENARIO_CONFIGS.get(name, {}),
        }
        for name in sorted(SAT_CHANNEL_SCENARIO_CONFIGS)
    }
    registry_sha256 = hashlib.sha256(
        json.dumps(registry_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    disjoint = bool(
        train
        and evaluate
        and not unknown
        and not scenario_overlap
        and not family_overlap
        and not config_hash_overlap
        and not channel_implementation_overlap
    )
    manifest: Dict[str, object] = {
        "schema": "phase1_satellite_train_eval_protocol_v1",
        "train_scenarios": train,
        "eval_scenarios": evaluate,
        "train_families": train_families,
        "eval_families": eval_families,
        "unknown_scenarios": unknown,
        "scenario_overlap": scenario_overlap,
        "family_overlap": family_overlap,
        "scenario_config_sha256": config_hashes,
        "train_config_sha256": train_hashes,
        "eval_config_sha256": eval_hashes,
        "config_hash_overlap": config_hash_overlap,
        "channel_implementation": channel_implementations,
        "train_channel_implementations": train_channel_implementations,
        "eval_channel_implementations": eval_channel_implementations,
        "channel_implementation_overlap": channel_implementation_overlap,
        "registry_version": "phase1_satellite_protocol_registry_v1",
        "registry_sha256": registry_sha256,
        "disjoint": disjoint,
        "require_disjoint": bool(require_disjoint),
        "evaluation_claim": "SOURCE_SYNTHETIC_HELDOUT_CHANNEL_STRESS_NOT_REAL_SATELLITE",
    }
    if require_disjoint and not disjoint:
        reasons = []
        if not train:
            reasons.append("empty_train_scenarios")
        if not evaluate:
            reasons.append("empty_eval_scenarios")
        if unknown:
            reasons.append("unknown_scenarios=" + ",".join(unknown))
        if scenario_overlap:
            reasons.append("scenario_overlap=" + ",".join(scenario_overlap))
        if family_overlap:
            reasons.append("family_overlap=" + ",".join(family_overlap))
        if config_hash_overlap:
            reasons.append("config_hash_overlap=" + ",".join(config_hash_overlap))
        if channel_implementation_overlap:
            reasons.append("channel_implementation_overlap=" + ",".join(channel_implementation_overlap))
        raise ValueError(
            "Phase1 satellite training/evaluation protocol is not held-out: " + ";".join(reasons)
        )
    return manifest


def sat_channel_config_for_scenario(name: str) -> Dict[str, object]:
    """Return dataclass kwargs for a named satellite channel scenario."""
    key = str(name or "").strip().lower().replace("-", "_")
    if key not in SAT_CHANNEL_SCENARIO_CONFIGS:
        valid = ", ".join(sorted(SAT_CHANNEL_SCENARIO_CONFIGS))
        raise ValueError(f"Unknown satellite channel scenario '{name}'. Valid scenarios: {valid}")
    return dict(SAT_CHANNEL_SCENARIO_CONFIGS[key])
