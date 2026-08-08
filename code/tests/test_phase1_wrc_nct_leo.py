from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "scripts"))

from cvsrffi.phase1_gi_epior import (  # noqa: E402
    GI_EPIOR_THRESHOLD,
    GIEpiORError,
    GIEpiORHead,
    GIEpiORRuntime,
    canonical_physical_ids,
    deterministic_reference_query_split,
    fit_class_geometry,
)
from cvsrffi.phase1_wrc_nct import WRCNCTError  # noqa: E402
from eval_phase1_wrc_nct import _run as _run_clean, build_parser as build_clean_parser  # noqa: E402
from eval_phase1_wrc_nct_leo import (  # noqa: E402
    EXPECTED_SCENARIOS,
    _run,
    build_parser,
)


SOURCE = ("1-1", "1-2", "1-3", "1-4", "1-5")


def _source_arrays(rows_per_class: int = 160, dim: int = 8):
    generator = np.random.default_rng(511)
    features = []
    tx_ids = []
    rx_ids = []
    day_ids = []
    eq_ids = []
    sig_ids = []
    for class_index, tx in enumerate(SOURCE):
        center = np.zeros(dim, dtype=np.float32)
        center[class_index] = 1.0
        for row in range(rows_per_class):
            features.append(center + generator.normal(0.0, 0.012, size=dim).astype(np.float32))
            tx_ids.append(tx)
            rx_ids.append(f"rx-{row % 2}")
            day_ids.append(f"day-{row % 3}")
            eq_ids.append("1")
            sig_ids.append(f"{class_index}-{row}")
    physical = canonical_physical_ids(tx_ids, rx_ids, day_ids, eq_ids, sig_ids)
    return (
        torch.as_tensor(np.asarray(features), dtype=torch.float32),
        np.asarray(tx_ids, dtype=object),
        np.asarray(rx_ids, dtype=object),
        np.asarray(day_ids, dtype=object),
        np.asarray(eq_ids, dtype=object),
        np.asarray(sig_ids, dtype=object),
        physical,
    )


def _write_feature_npz(
    path: Path,
    features: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    day: np.ndarray,
    eq: np.ndarray,
    sig: np.ndarray,
    scenarios: np.ndarray,
) -> None:
    logits = np.full((features.shape[0], len(SOURCE)), -4.0, dtype=np.float32)
    for index, value in enumerate(tx):
        logits[index, SOURCE.index(str(value))] = 4.0
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        tx_logits=logits,
        tx_ids=tx,
        rx_ids=rx,
        day_ids=day,
        eq_ids=eq,
        sig_ids=sig,
        dataset_role=np.asarray(["source"] * len(tx)),
        channel_views=np.asarray(["clean"] * len(tx)),
        sat_scenarios=scenarios,
    )


def _write_gi_bundle(path: Path, features: torch.Tensor, tx: np.ndarray, physical: np.ndarray) -> None:
    reference, _, _ = deterministic_reference_query_split(tx, physical, SOURCE)
    prototypes, scales = fit_class_geometry(features, tx, reference, SOURCE)
    torch.manual_seed(211)
    head = GIEpiORHead().eval()
    runtime = GIEpiORRuntime(prototypes, scales, head).eval()
    state = head.state_dict()
    manifest = {
        "schema": "cvs.phase1.gi_epior_bundle.v1",
        "class_ids": list(SOURCE),
        "threshold": GI_EPIOR_THRESHOLD,
        "fit_receipt": {"schema": "synthetic"},
        "runtime_state_bytes": 0,
    }
    np.savez_compressed(
        path,
        prototypes=runtime.prototypes.detach().cpu().numpy(),
        scales=runtime.scales.detach().cpu().numpy(),
        head_0_weight=state["net.0.weight"].detach().cpu().numpy(),
        head_0_bias=state["net.0.bias"].detach().cpu().numpy(),
        head_2_weight=state["net.2.weight"].detach().cpu().numpy(),
        head_2_bias=state["net.2.bias"].detach().cpu().numpy(),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def _clean_args(tmp_path: Path, clean_npz: Path, gi_bundle: Path):
    return build_clean_parser().parse_args(
        [
            "--feature-npz",
            str(clean_npz),
            "--gi-bundle",
            str(gi_bundle),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--held-tx-ids",
            "9-1",
            "--view-name",
            "clean",
            "--output-readout-json",
            str(tmp_path / "wrc.readout.json"),
            "--output-torchscript",
            str(tmp_path / "wrc.ts"),
            "--output-metrics-json",
            str(tmp_path / "wrc.metrics.json"),
            "--output-scores-csv",
            str(tmp_path / "wrc.scores.csv"),
        ]
    )


def _leo_args(tmp_path: Path, leo_npz: Path, gi_bundle: Path):
    return build_parser().parse_args(
        [
            "--feature-npz",
            str(leo_npz),
            "--gi-bundle",
            str(gi_bundle),
            "--wrc-readout-json",
            str(tmp_path / "wrc.readout.json"),
            "--clean-scores-csv",
            str(tmp_path / "wrc.scores.csv"),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--expected-scenarios",
            ",".join(EXPECTED_SCENARIOS),
            "--output-metrics-json",
            str(tmp_path / "leo.metrics.json"),
            "--output-scores-csv",
            str(tmp_path / "leo.scores.csv"),
        ]
    )


def _context(tmp_path: Path) -> dict[str, object]:
    features, tx, rx, day, eq, sig, physical = _source_arrays()
    clean_npz = tmp_path / "clean.npz"
    _write_feature_npz(clean_npz, features.numpy(), tx, rx, day, eq, sig, np.asarray([""] * len(tx)))
    gi_bundle = tmp_path / "gi.npz"
    _write_gi_bundle(gi_bundle, features, tx, physical)
    clean_args = _clean_args(tmp_path, clean_npz, gi_bundle)
    _run_clean(clean_args)
    with Path(clean_args.output_scores_csv).open("r", encoding="utf-8", newline="") as handle:
        clean_rows = list(csv.DictReader(handle))
    eval_ids = {row["physical_id"] for row in clean_rows if row["known_evaluation"] == "1"}
    scenarios = np.empty(len(tx), dtype=object)
    for class_id in SOURCE:
        for receiver_id in sorted(set(rx.tolist())):
            rows = np.flatnonzero((tx == class_id) & (rx == receiver_id))
            evaluation_rows = [row for row in rows.tolist() if physical[row] in eval_ids]
            assert len(evaluation_rows) >= len(EXPECTED_SCENARIOS)
            for index, row in enumerate(evaluation_rows):
                scenarios[row] = EXPECTED_SCENARIOS[index % len(EXPECTED_SCENARIOS)]
            other_rows = [row for row in rows.tolist() if physical[row] not in eval_ids]
            for index, row in enumerate(other_rows):
                scenarios[row] = EXPECTED_SCENARIOS[index % len(EXPECTED_SCENARIOS)]
    assert not any(value is None for value in scenarios.tolist())
    leo_npz = tmp_path / "leo.npz"
    leo_features = features.numpy().copy()
    for index, scenario in enumerate(scenarios):
        if scenario == "leo_low_elev_weak":
            leo_features[index] += 0.01
        elif scenario == "leo_rain_weak":
            leo_features[index] -= 0.01
    _write_feature_npz(leo_npz, leo_features, tx, rx, day, eq, sig, scenarios)
    return {
        "features": features.numpy(),
        "tx": tx,
        "rx": rx,
        "day": day,
        "eq": eq,
        "sig": sig,
        "physical": physical,
        "scenarios": scenarios,
        "clean_rows": clean_rows,
        "gi_bundle": gi_bundle,
        "leo_npz": leo_npz,
    }


def test_complete_frozen_leo_closure_and_paired_physical_baseline(tmp_path: Path):
    context = _context(tmp_path)
    args = _leo_args(tmp_path, context["leo_npz"], context["gi_bundle"])
    metrics = _run(args)
    saved = json.loads(Path(args.output_metrics_json).read_text(encoding="utf-8"))
    score_rows = list(csv.DictReader(Path(args.output_scores_csv).open("r", encoding="utf-8", newline="")))
    assert metrics["calibration_performed"] is False and metrics["outer_used"] is False
    assert metrics["new_runtime_exported"] is False
    assert set(metrics["scenario_metrics"]) == set(EXPECTED_SCENARIOS)
    assert metrics["parity"]["gi_ratio_max_abs"] <= 1.0e-5
    assert len(score_rows) == len(context["tx"])
    assert saved["bindings"]["wrc_readout_sha256"]
    clean_by_id = {row["physical_id"]: row for row in context["clean_rows"]}
    for scenario in EXPECTED_SCENARIOS:
        ids = [
            str(context["physical"][index])
            for index, value in enumerate(context["scenarios"])
            if value == scenario and str(context["physical"][index]) in clean_by_id and clean_by_id[str(context["physical"][index])]["known_evaluation"] == "1"
        ]
        expected_clean_full = float(np.mean([int(clean_by_id[item]["full_correct_known"]) for item in ids]))
        observed = metrics["scenario_metrics"][scenario]["paired_clean_baseline"]["overall_full_accuracy"]
        assert observed == pytest.approx(expected_clean_full)
        assert metrics["scenario_metrics"][scenario]["known_evaluation"]["count"] == len(ids)
        assert metrics["scenario_metrics"][scenario]["paired_clean_full_drop_pp"]["overall"] == pytest.approx(
            100.0 * (expected_clean_full - metrics["scenario_metrics"][scenario]["known_evaluation"]["overall_full_accuracy"])
        )


def test_missing_scenario_and_changed_physical_set_fail_closed(tmp_path: Path):
    context = _context(tmp_path)
    missing = np.asarray(context["scenarios"], dtype=object).copy()
    missing[missing == "leo_rain_weak"] = "leo_clear_weak"
    missing_npz = tmp_path / "missing.npz"
    _write_feature_npz(
        missing_npz,
        context["features"],
        context["tx"],
        context["rx"],
        context["day"],
        context["eq"],
        context["sig"],
        missing,
    )
    with pytest.raises(WRCNCTError, match="scenario set"):
        _run(_leo_args(tmp_path, missing_npz, context["gi_bundle"]))
    eval_ids = {row["physical_id"] for row in context["clean_rows"] if row["known_evaluation"] == "1"}
    missing_rx = np.asarray(context["scenarios"], dtype=object).copy()
    for index, physical_id in enumerate(context["physical"]):
        if str(physical_id) in eval_ids and context["rx"][index] == "rx-0" and missing_rx[index] == "leo_rain_weak":
            missing_rx[index] = "leo_clear_weak"
    coverage_npz = tmp_path / "missing_rx_coverage.npz"
    _write_feature_npz(
        coverage_npz,
        context["features"],
        context["tx"],
        context["rx"],
        context["day"],
        context["eq"],
        context["sig"],
        missing_rx,
    )
    with pytest.raises(WRCNCTError, match="source RX coverage"):
        _run(_leo_args(tmp_path, coverage_npz, context["gi_bundle"]))
    changed_sig = np.asarray(context["sig"], dtype=object).copy()
    changed_sig[0] = "changed-physical-id"
    changed_npz = tmp_path / "changed.npz"
    _write_feature_npz(
        changed_npz,
        context["features"],
        context["tx"],
        context["rx"],
        context["day"],
        context["eq"],
        changed_sig,
        context["scenarios"],
    )
    with pytest.raises(WRCNCTError, match="physical-ID sets differ"):
        _run(_leo_args(tmp_path, changed_npz, context["gi_bundle"]))


def test_duplicate_physical_and_leo_value_changes_cannot_recalibrate_tau(tmp_path: Path):
    context = _context(tmp_path)
    duplicate_tx = np.asarray(context["tx"], dtype=object).copy()
    duplicate_rx = np.asarray(context["rx"], dtype=object).copy()
    duplicate_day = np.asarray(context["day"], dtype=object).copy()
    duplicate_eq = np.asarray(context["eq"], dtype=object).copy()
    duplicate_sig = np.asarray(context["sig"], dtype=object).copy()
    for values in (duplicate_tx, duplicate_rx, duplicate_day, duplicate_eq, duplicate_sig):
        values[-1] = values[0]
    duplicate_npz = tmp_path / "duplicate.npz"
    _write_feature_npz(
        duplicate_npz,
        context["features"],
        duplicate_tx,
        duplicate_rx,
        duplicate_day,
        duplicate_eq,
        duplicate_sig,
        context["scenarios"],
    )
    with pytest.raises(GIEpiORError, match="unique"):
        _run(_leo_args(tmp_path, duplicate_npz, context["gi_bundle"]))
    first_args = _leo_args(tmp_path, context["leo_npz"], context["gi_bundle"])
    first = _run(first_args)
    shifted_npz = tmp_path / "shifted.npz"
    _write_feature_npz(
        shifted_npz,
        np.asarray(context["features"], dtype=np.float32) + 4.0,
        context["tx"],
        context["rx"],
        context["day"],
        context["eq"],
        context["sig"],
        context["scenarios"],
    )
    second_args = _leo_args(tmp_path, shifted_npz, context["gi_bundle"])
    second = _run(second_args)
    assert first["tau"] == second["tau"]
    assert first["bindings"]["wrc_readout_sha256"] == second["bindings"]["wrc_readout_sha256"]
    parser = build_parser()
    assert "--alpha" not in parser.format_help()
    assert "--tau" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--alpha", "0.01"])
