#!/usr/bin/env python
"""SLEV-CI support-only logit-energy collaborative inference for Stage2-C.

SLEV-CI keeps the ADV3B02/qknn8 evidence path frozen and adds a lightweight
support-only logit-energy verifier to the collaborative decision layer. The
energy threshold is calibrated from target-old and seen-new support rows only;
unknown query rows remain evaluation-only and are never used for threshold
selection, profile selection, or fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SCRIPTS_ROOT = CODE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from phase2_collaborative_open_set_qknn_eval import (  # noqa: E402
    KNOWN_ROLES,
    UNKNOWN_ROLE,
    _event_key,
    _scenario_of,
    _split_support_query_selected,
    _stable_score,
    load_feature_npz,
)
from phase2_orbit_enpc_ci_eval import (  # noqa: E402
    EnpcProfile,
    augment_enpc_evidence,
    evaluate_enpc_collaborative_evidence,
)
from phase2_orbit_pcet_ci_eval import (  # noqa: E402
    _count_value,
    _float,
    _positive_int,
    _str,
    _target_pass,
    _write_csv,
    parse_args as _parse_pcet_args,
    run_pcet_ci,
)


UNKNOWN_LABEL = "__unknown__"
ROLE_TO_ALIAS = {"target_old": "old", "target_new": "seen_new", UNKNOWN_ROLE: "unknown"}
ALIAS_TO_ROLE = {value: key for key, value in ROLE_TO_ALIAS.items()}


@dataclass(frozen=True)
class SlevProfile:
    name: str
    description: str
    accept_confidence: float
    accept_margin: float
    accept_max_pressure: float
    support_accept_confidence: float
    reject_pressure: float
    reject_min_high_fraction: float
    reject_min_disagreement: float
    min_accept_receivers: int
    energy_weight: float

    def as_enpc_profile(self) -> EnpcProfile:
        return EnpcProfile(
            name=self.name,
            description=self.description,
            accept_confidence=self.accept_confidence,
            accept_margin=self.accept_margin,
            accept_max_pressure=self.accept_max_pressure,
            support_accept_confidence=self.support_accept_confidence,
            reject_pressure=self.reject_pressure,
            reject_min_high_fraction=self.reject_min_high_fraction,
            reject_min_disagreement=self.reject_min_disagreement,
            min_accept_receivers=self.min_accept_receivers,
        )


PROFILES: tuple[SlevProfile, ...] = (
    SlevProfile(
        name="slev_known_anchor",
        description="known-retention anchor with energy telemetry but loose rejection",
        accept_confidence=0.18,
        accept_margin=0.00,
        accept_max_pressure=1.00,
        support_accept_confidence=0.35,
        reject_pressure=1.10,
        reject_min_high_fraction=1.10,
        reject_min_disagreement=1.10,
        min_accept_receivers=1,
        energy_weight=0.10,
    ),
    SlevProfile(
        name="slev_balanced",
        description="support-only energy pressure with ENPC known protection",
        accept_confidence=0.30,
        accept_margin=0.02,
        accept_max_pressure=0.72,
        support_accept_confidence=0.56,
        reject_pressure=0.62,
        reject_min_high_fraction=0.45,
        reject_min_disagreement=0.50,
        min_accept_receivers=1,
        energy_weight=0.35,
    ),
    SlevProfile(
        name="slev_old80_energy_probe",
        description="OLD80-constrained energy probe for unknown rejection recovery",
        accept_confidence=0.40,
        accept_margin=0.00,
        accept_max_pressure=0.52,
        support_accept_confidence=0.54,
        reject_pressure=0.50,
        reject_min_high_fraction=0.20,
        reject_min_disagreement=1.20,
        min_accept_receivers=1,
        energy_weight=0.45,
    ),
    SlevProfile(
        name="slev_energy_strict",
        description="strict multi-receiver energy rejection diagnostic",
        accept_confidence=0.44,
        accept_margin=0.04,
        accept_max_pressure=0.58,
        support_accept_confidence=0.66,
        reject_pressure=0.48,
        reject_min_high_fraction=0.40,
        reject_min_disagreement=0.45,
        min_accept_receivers=2,
        energy_weight=0.55,
    ),
)


@dataclass
class SlevEnergyBundle:
    query_energy_by_row: dict[tuple[str, str], float]
    threshold_by_receiver: dict[str, float]
    global_threshold: float
    support_count: int
    support_min: float
    support_median: float
    support_max: float
    support_quantile: float
    temperature: float
    risk_temperature: float
    margin: float
    alignment_policy: str
    threshold_scope: str = "target_old_and_seen_new_support_only"


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 60.0:
        return 1.0
    if value <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-float(value)))


def _profile_names(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"", "all", "*"}:
        return [profile.name for profile in PROFILES]
    names = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    known = {profile.name for profile in PROFILES}
    unknown = sorted(set(names) - known)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown SLEV-CI profile(s): {', '.join(unknown)}")
    return names


def _logit_energy(logits: np.ndarray, temperature: float) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    scaled = arr / temp
    maxv = np.max(scaled, axis=1, keepdims=True)
    return -temp * (np.squeeze(maxv, axis=1) + np.log(np.sum(np.exp(scaled - maxv), axis=1)))


def _quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, max(0.0, min(1.0, float(q)))))


def _labels_for_role(payload: Mapping[str, Any], role: str) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    return sorted({str(tx_ids[i]) for i in np.where(roles == role)[0].tolist()})


def _target_receivers(payload: Mapping[str, Any]) -> list[str]:
    roles = np.asarray(payload["dataset_role"]).astype(str)
    rx_ids = np.asarray(payload["rx_ids"]).astype(str)
    mask = np.isin(roles, ["target_old", "target_new", UNKNOWN_ROLE])
    return sorted({str(rx_ids[i]) for i in np.where(mask)[0].tolist()})


def _split_by_receiver_role(
    payload: Mapping[str, Any],
    *,
    features: np.ndarray,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, dict[str, list[int]]]]:
    support_by_rx: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    query_by_rx: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rx in _target_receivers(payload):
        for role in ["target_old", "target_new", UNKNOWN_ROLE]:
            for label in _labels_for_role(payload, role):
                support, query = _split_support_query_selected(
                    payload,
                    features=features,
                    role=role,
                    tx_id=label,
                    rx_id=rx,
                    k_shot=int(k_shot),
                    query_per_class=int(query_per_class),
                    seed=int(seed),
                    support_selection_policy=str(support_selection_policy),
                )
                alias = ROLE_TO_ALIAS[role]
                if role in KNOWN_ROLES:
                    support_by_rx[rx][alias].extend(int(i) for i in support)
                query_by_rx[rx][alias].extend(int(i) for i in query)
    return support_by_rx, query_by_rx


def _query_energy_map(
    payload: Mapping[str, Any],
    query_by_rx: Mapping[str, Mapping[str, Sequence[int]]],
    energies: np.ndarray,
    *,
    event_alignment_policy: str,
    seed: int,
) -> dict[tuple[str, str], float]:
    tx_ids = np.asarray(payload["tx_ids"]).astype(str)
    mapping: dict[tuple[str, str], float] = {}
    target_receivers = sorted(query_by_rx)
    alignment = str(event_alignment_policy or "receiver_domain_ranked").strip().lower()
    for role_name in ["old", "seen_new", "unknown"]:
        role = ALIAS_TO_ROLE[role_name]
        for label in _labels_for_role(payload, role):
            if alignment == "strict_event_key":
                for rx in target_receivers:
                    for idx in query_by_rx[rx].get(role_name, []):
                        if str(tx_ids[int(idx)]) == label:
                            mapping[(_event_key(payload, int(idx), role_name, label), rx)] = float(energies[int(idx)])
                continue
            by_rx_scenario: dict[str, dict[str, list[int]]] = {}
            for rx in target_receivers:
                by_rx_scenario[rx] = defaultdict(list)
                for idx in query_by_rx[rx].get(role_name, []):
                    if str(tx_ids[int(idx)]) == label:
                        by_rx_scenario[rx][_scenario_of(payload, int(idx))].append(int(idx))
                for scenario in by_rx_scenario[rx]:
                    by_rx_scenario[rx][scenario] = sorted(
                        by_rx_scenario[rx][scenario],
                        key=lambda i: (
                            str(payload["day_ids"][i]),
                            str(payload["sig_ids"][i]),
                            _stable_score((rx, role_name, label, i), int(seed)),
                        ),
                    )
            common_scenarios = sorted(set().union(*(set(by_rx_scenario[rx]) for rx in target_receivers)))
            for scenario in common_scenarios:
                n = max(len(by_rx_scenario[rx].get(scenario, [])) for rx in target_receivers)
                for event_i in range(n):
                    event_id = f"{role_name}|{label}|{scenario}|rank{event_i:05d}"
                    for rx in target_receivers:
                        rows = by_rx_scenario[rx].get(scenario, [])
                        if event_i < len(rows):
                            mapping[(event_id, rx)] = float(energies[int(rows[event_i])])
    return mapping


def build_slev_energy_bundle(
    feature_npz: Path,
    *,
    k_shot: int,
    query_per_class: int,
    seed: int,
    support_selection_policy: str,
    event_alignment_policy: str,
    support_quantile: float,
    logit_temperature: float,
    risk_temperature: float,
    risk_margin: float,
) -> SlevEnergyBundle:
    payload = load_feature_npz(Path(feature_npz))
    with np.load(Path(feature_npz), allow_pickle=True) as raw_npz:
        if "tx_logits" not in raw_npz.files:
            raise ValueError("SLEV-CI requires tx_logits in the feature NPZ")
        logits = np.asarray(raw_npz["tx_logits"], dtype=np.float64)
    features = np.asarray(payload["features"], dtype=np.float32)
    if int(logits.shape[0]) != int(features.shape[0]):
        raise ValueError(f"tx_logits length mismatch: expected {features.shape[0]}, got {logits.shape[0]}")
    energies = _logit_energy(logits, float(logit_temperature))
    support_by_rx, query_by_rx = _split_by_receiver_role(
        payload,
        features=features,
        k_shot=int(k_shot),
        query_per_class=int(query_per_class),
        seed=int(seed),
        support_selection_policy=str(support_selection_policy),
    )
    support_indices: list[int] = []
    threshold_by_receiver: dict[str, float] = {}
    for rx, by_role in support_by_rx.items():
        rx_indices = sorted({int(i) for role_indices in by_role.values() for i in role_indices})
        if rx_indices:
            threshold_by_receiver[str(rx)] = _quantile([float(energies[i]) for i in rx_indices], support_quantile)
            support_indices.extend(rx_indices)
    support_indices = sorted(set(support_indices))
    support_values = [float(energies[i]) for i in support_indices]
    if not support_values:
        raise ValueError("SLEV-CI could not build target-old/seen-new support energy calibration")
    return SlevEnergyBundle(
        query_energy_by_row=_query_energy_map(
            payload,
            query_by_rx,
            energies,
            event_alignment_policy=str(event_alignment_policy),
            seed=int(seed),
        ),
        threshold_by_receiver=threshold_by_receiver,
        global_threshold=_quantile(support_values, support_quantile),
        support_count=len(support_values),
        support_min=float(np.min(support_values)),
        support_median=float(np.median(support_values)),
        support_max=float(np.max(support_values)),
        support_quantile=float(support_quantile),
        temperature=float(logit_temperature),
        risk_temperature=float(risk_temperature),
        margin=float(risk_margin),
        alignment_policy=str(event_alignment_policy),
    )


def _energy_risk(energy: float, threshold: float, *, risk_temperature: float, margin: float) -> float:
    denom = max(float(risk_temperature), 1e-6)
    return _clip01(_sigmoid((float(energy) - float(threshold) + float(margin)) / denom))


def augment_slev_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    energy_bundle: SlevEnergyBundle,
    energy_weight: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    weight = _clip01(float(energy_weight))
    for source in evidence:
        row = dict(source)
        event_id = _str(row, "event_id")
        receiver_id = _str(row, "receiver_id")
        energy = energy_bundle.query_energy_by_row.get((event_id, receiver_id))
        threshold = energy_bundle.threshold_by_receiver.get(receiver_id, energy_bundle.global_threshold)
        if energy is None:
            risk = 0.0
            row["slev_energy_map_status"] = "missing"
        else:
            risk = _energy_risk(
                float(energy),
                float(threshold),
                risk_temperature=energy_bundle.risk_temperature,
                margin=energy_bundle.margin,
            )
            row["slev_energy_map_status"] = "matched"
        enpc_pressure = _clip01(_float(row, "enpc_episode_negative_pressure", 0.0))
        combined_pressure = max(enpc_pressure, _clip01((1.0 - weight) * enpc_pressure + weight * risk))
        row["slev_logit_energy"] = "" if energy is None else float(energy)
        row["slev_support_energy_threshold"] = float(threshold)
        row["slev_global_support_energy_threshold"] = float(energy_bundle.global_threshold)
        row["slev_energy_risk"] = float(risk)
        row["slev_energy_weight"] = float(weight)
        row["slev_enpc_pressure"] = float(enpc_pressure)
        row["slev_combined_pressure"] = float(combined_pressure)
        row["slev_threshold_scope"] = energy_bundle.threshold_scope
        row["enpc_episode_negative_pressure"] = float(combined_pressure)
        out.append(row)
    return out


def _energy_summary(bundle: SlevEnergyBundle) -> dict[str, Any]:
    return {
        "threshold_scope": bundle.threshold_scope,
        "support_count": int(bundle.support_count),
        "support_quantile": float(bundle.support_quantile),
        "global_threshold": float(bundle.global_threshold),
        "support_min": float(bundle.support_min),
        "support_median": float(bundle.support_median),
        "support_max": float(bundle.support_max),
        "logit_temperature": float(bundle.temperature),
        "risk_temperature": float(bundle.risk_temperature),
        "risk_margin": float(bundle.margin),
        "alignment_policy": str(bundle.alignment_policy),
        "receiver_thresholds": dict(sorted(bundle.threshold_by_receiver.items())),
    }


def run_slev_ci(args: argparse.Namespace) -> dict[str, Any]:
    base_args = argparse.Namespace(**vars(args))
    base_args.profiles = "pcet_known_preserving"
    base = run_pcet_ci(base_args)
    base_evidence = augment_enpc_evidence(
        base.pop("_evidence_rows"),
        gap_scale=float(args.enpc_gap_scale),
        pressure_floor=float(args.enpc_pressure_floor),
    )
    energy_bundle = build_slev_energy_bundle(
        args.feature_npz,
        k_shot=int(args.k_shot),
        query_per_class=int(args.query_per_class),
        seed=int(args.seed),
        support_selection_policy=str(args.support_selection_policy),
        event_alignment_policy=str(args.event_alignment_policy),
        support_quantile=float(args.slev_energy_support_quantile),
        logit_temperature=float(args.slev_logit_temperature),
        risk_temperature=float(args.slev_energy_risk_temperature),
        risk_margin=float(args.slev_energy_risk_margin),
    )
    metadata = dict(base["qknn_metadata"])
    metadata["algorithm_wrapper"] = "SLEV-CI"
    metadata["unknown_query_eval_only"] = True
    metadata["labeled_unknown_support_used_for_boundary_fit"] = False
    metadata["threshold_scope"] = energy_bundle.threshold_scope
    metadata["in_orbit_method"] = "qknn8"
    metadata["slev_components"] = [
        "qknn8_candidate_evidence",
        "support_only_logit_energy",
        "episode_negative_pressure",
        "multi_receiver_selective_fusion",
    ]
    requested = set(_profile_names(args.slev_profiles))
    profile_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    final_evidence: list[dict[str, Any]] = []
    for profile in [profile for profile in PROFILES if profile.name in requested]:
        evidence = augment_slev_evidence(base_evidence, energy_bundle=energy_bundle, energy_weight=profile.energy_weight)
        if not final_evidence:
            final_evidence = evidence
        result = evaluate_enpc_collaborative_evidence(
            evidence,
            profile=profile.as_enpc_profile(),
            collab_counts=args.collab_counts,
            collab_group_policy=str(args.collab_group_policy),
            partial_collab_min_receivers=int(args.partial_collab_min_receivers),
            max_event_bytes=float(args.max_event_bytes),
            max_event_latency_ms=float(args.max_event_latency_ms),
            metadata=metadata,
            include_event_results=bool(args.include_event_results),
        )
        result["protocol"] = "slev_collaborative_open_set_qknn_evidence"
        result["fusion_policy"] = "slev_ci"
        result["energy_calibration"] = _energy_summary(energy_bundle)
        profile_results[profile.name] = result
        for collab_count, counts in sorted(result["counts"].items(), key=lambda item: int(item[0])):
            row = {
                "profile": profile.name,
                "profile_description": profile.description,
                "collab_count": int(collab_count),
                "old_acc": _count_value(counts, "old_acc"),
                "min_old": _count_value(counts, "min_old_class_acc", "min_old_acc"),
                "seen_new_acc": _count_value(counts, "seen_new_acc"),
                "min_seen": _count_value(counts, "min_seen_new_class_acc", "min_seen_new_acc"),
                "unknown_reject": _count_value(counts, "unknown_reject_rate", "unknown_reject_acc"),
                "unknown_FAR": _count_value(counts, "unknown_FAR", "unknown_far"),
                "known_defer": _count_value(counts, "known_defer_rate"),
                "unknown_defer": _count_value(counts, "unknown_defer_rate"),
                "bytes_per_event": _count_value(counts, "bytes_per_event", "mean_evidence_bytes"),
                "latency_ms": _count_value(counts, "latency_ms_pessimistic", "mean_latency_ms"),
                "energy_weight": float(profile.energy_weight),
                "support_energy_threshold": float(energy_bundle.global_threshold),
                "support_energy_count": int(energy_bundle.support_count),
                "target_old_acc": float(args.target_old_acc),
                "target_min_old": float(args.target_min_old),
                "target_seen_new_acc": float(args.target_seen_new_acc),
                "target_min_seen": float(args.target_min_seen),
                "target_unknown_reject": float(args.target_unknown_reject),
            }
            row["target_pass"] = _target_pass(row)
            row["resource_pass"] = (
                (float(args.max_event_bytes) <= 0.0 or row["bytes_per_event"] <= float(args.max_event_bytes))
                and (
                    float(args.max_event_latency_ms) <= 0.0
                    or row["latency_ms"] <= float(args.max_event_latency_ms)
                )
            )
            summary_rows.append(row)
    best_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["target_pass"],
            row["old_acc"] >= 0.80,
            row["unknown_reject"],
            row["old_acc"],
            row["seen_new_acc"],
            -row["known_defer"],
        ),
        reverse=True,
    )
    return {
        "algorithm": "SLEV-CI",
        "feature_npz": str(args.feature_npz),
        "profiles": [profile.__dict__ for profile in PROFILES if profile.name in requested],
        "base_pcet_known_preserving": base,
        "profile_results": profile_results,
        "summary_rows": summary_rows,
        "best_joint_row": best_rows[0] if best_rows else None,
        "qknn_metadata": metadata,
        "energy_calibration": _energy_summary(energy_bundle),
        "evidence_row_count": len(final_evidence),
        "target_gates": {
            "old_acc": float(args.target_old_acc),
            "min_old": float(args.target_min_old),
            "seen_new_acc": float(args.target_seen_new_acc),
            "min_seen": float(args.target_min_seen),
            "unknown_reject": float(args.target_unknown_reject),
        },
        "resource_constraints": {
            "evidence_packet_bytes": float(args.evidence_packet_bytes),
            "max_event_bytes": float(args.max_event_bytes),
            "max_event_latency_ms": float(args.max_event_latency_ms),
            "latency_budget_ms": float(args.latency_budget_ms),
        },
        "_evidence_rows": final_evidence,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    slev_parser = argparse.ArgumentParser(add_help=False)
    slev_parser.add_argument("--profiles", dest="slev_profiles", default="all")
    slev_parser.add_argument("--enpc_gap_scale", type=float, default=0.20)
    slev_parser.add_argument("--enpc_pressure_floor", type=float, default=0.0)
    slev_parser.add_argument("--slev_energy_support_quantile", type=float, default=0.90)
    slev_parser.add_argument("--slev_logit_temperature", type=float, default=1.0)
    slev_parser.add_argument("--slev_energy_risk_temperature", type=float, default=0.75)
    slev_parser.add_argument("--slev_energy_risk_margin", type=float, default=0.0)
    slev_args, remaining = slev_parser.parse_known_args(raw)
    args = _parse_pcet_args(remaining)
    for key, value in vars(slev_args).items():
        setattr(args, key, value)
    _profile_names(args.slev_profiles)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_slev_ci(args)
    evidence_rows = result.pop("_evidence_rows")
    result["run_command_argv"] = [str(item) for item in sys.argv]
    result["run_cwd"] = str(Path.cwd())
    result["python_executable"] = str(sys.executable)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_summary_csv:
        _write_csv(args.output_summary_csv, result["summary_rows"])
    if args.output_evidence_csv:
        _write_csv(args.output_evidence_csv, evidence_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
