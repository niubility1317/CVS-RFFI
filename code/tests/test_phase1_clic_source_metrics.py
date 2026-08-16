from __future__ import annotations

"""TDD contracts for held source-V CLIC source-only metrics.

These tests deliberately import the new file-only exporter and scorer at
module load time.  Before the production APIs exist, collection must fail at
that boundary instead of accidentally proving a synthetic helper.
"""

import copy
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import build_phase1_clic_source_v_leo_iq as TASK1_CACHE
import evaluate_phase1_clic_source_metrics as METRICS
import export_phase1_clic_source_v_leo_features as EXPORTER


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SOURCE_TX = ("tx-0", "tx-1", "tx-2", "tx-3")
V4_RUN_ID = "phase1_clic_source_metrics_20260816_v4"
V4_SMOKE_ROOT_NAME = ".smoke_phase1_clic_source_metrics_20260816_v4_F1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v4_technical_smoke_paths(
    tmp_path: Path, *, mirror_training: bool = False
) -> dict[str, Path]:
    """Create only path-shaped inputs for the root-contract boundary."""

    runs = tmp_path / "runs"
    smoke_root = runs / V4_SMOKE_ROOT_NAME
    training_root = (
        smoke_root / "training_mirror" / "phase1_clic12_20260812_v5"
        if mirror_training
        else runs / "phase1_clic12_20260812_v5"
    )
    clean_root = runs / "phase1_clic_postfreeze_20260812_v4"
    source_root = smoke_root / V4_RUN_ID
    candidate = "F1C_CLIC12"
    paths = {
        "project_root": tmp_path,
        "runs": runs,
        "smoke_root": smoke_root,
        "training_root": training_root,
        "clean_root": clean_root,
        "source_root": source_root,
        "checkpoint": training_root / candidate / "final_ssdg.pth",
        "terminal": training_root / candidate / "phase1_clic_terminal_receipt.json",
        "clean": clean_root / candidate / "source_clean_proxy.npz",
        "cache": source_root / "F1_SHARED" / "source_validation_known_leo_weak.npz",
        "cache_receipt": source_root / "F1_SHARED" / "source_validation_known_leo_weak.receipt.json",
        "pair": runs / "phase1_clic_source_pair_20260812_v3" / "F1_C_vs_G_pair.json",
        "output": source_root / candidate / "source_validation_known_leo_weak_features.npz",
        "binding": source_root / candidate / "source_validation_known_leo_weak.binding.json",
    }
    for key in ("checkpoint", "terminal", "clean", "cache", "cache_receipt", "pair"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_bytes(key.encode("ascii"))
    return paths


def _v4_technical_smoke_args(paths: dict[str, Path], *, technical_smoke: bool) -> Namespace:
    return Namespace(
        ckpt=str(paths["checkpoint"]),
        terminal_receipt_json=str(paths["terminal"]),
        clean_npz=str(paths["clean"]),
        source_v_received_iq_npz=str(paths["cache"]),
        source_v_received_iq_receipt_json=str(paths["cache_receipt"]),
        pair_json=str(paths["pair"]),
        formal_project_root=str(paths["project_root"]),
        training_run_root=str(paths["training_root"]),
        cache_run_root=str(paths["source_root"]),
        output_root=str(paths["source_root"]),
        out_npz=str(paths["output"]),
        binding_json=str(paths["binding"]),
        candidate_id="F1C_CLIC12",
        fold_index=1,
        arm="C",
        source_tx_ids=",".join(SOURCE_TX),
        batch_size=4,
        device="cpu",
        technical_smoke=technical_smoke,
    )


def test_technical_smoke_reaches_original_f1_terminal_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the formal-parent gate blocks a legal independent F1 smoke."""

    paths = _v4_technical_smoke_paths(tmp_path)
    args = _v4_technical_smoke_args(paths, technical_smoke=True)
    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        EXPORTER,
        "EXPECTED_TECHNICAL_SMOKE_PROJECT_ROOT",
        str(paths["project_root"]),
    )

    monkeypatch.setattr(
        EXPORTER.torch,
        "load",
        lambda *_args, **_kwargs: {
            "args": {
                "phase1_source_known_validation_tx_ids": "held-tx",
                "phase1_source_proxy_unknown_tx_ids": "proxy-tx",
                "wisig_out_len": 256,
            }
        },
    )

    def reaches_original_terminal(
        _checkpoint: Mapping[str, Any], *, checkpoint_path: Path, terminal_receipt_path: Path, **_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        observed.append((checkpoint_path, terminal_receipt_path))
        raise RuntimeError("reached original F1 terminal envelope")

    monkeypatch.setattr(EXPORTER._clean, "validate_clic_training_checkpoint", reaches_original_terminal)
    with pytest.raises(RuntimeError, match="reached original F1 terminal envelope"):
        EXPORTER.export_source_v_leo_features(args)

    assert observed == [(paths["checkpoint"].resolve(), paths["terminal"].resolve())]
    assert not paths["output"].exists()
    assert not paths["binding"].exists()


def test_technical_smoke_root_contract_allows_only_the_exact_f1_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a broad smoke escape hatch can admit another fold or root."""

    paths = _v4_technical_smoke_paths(tmp_path)
    monkeypatch.setattr(
        EXPORTER,
        "EXPECTED_TECHNICAL_SMOKE_PROJECT_ROOT",
        str(paths["project_root"]),
    )
    assert EXPORTER.validate_source_v_execution_roots(
        training_root=paths["training_root"],
        clean_path=paths["clean"],
        cache_root=paths["source_root"],
        output_root=paths["source_root"],
        checkpoint_path=paths["checkpoint"],
        terminal_path=paths["terminal"],
        pair_path=paths["pair"],
        formal_project_root=str(paths["project_root"]),
        fold_index=1,
        candidate_id="F1C_CLIC12",
        technical_smoke=True,
    ) is None

    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="technical smoke|F1|root"):
        EXPORTER.validate_source_v_execution_roots(
            training_root=paths["training_root"],
            clean_path=paths["clean"],
            cache_root=paths["source_root"],
            output_root=paths["source_root"],
            checkpoint_path=paths["checkpoint"],
            terminal_path=paths["terminal"],
            pair_path=paths["pair"],
            formal_project_root=str(paths["project_root"]),
            fold_index=2,
            candidate_id="F2C_CLIC12",
            technical_smoke=True,
        )

    mirrored = _v4_technical_smoke_paths(tmp_path / "mirrored", mirror_training=True)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="technical smoke|canonical|root"):
        EXPORTER.validate_source_v_execution_roots(
            training_root=mirrored["training_root"],
            clean_path=mirrored["clean"],
            cache_root=mirrored["source_root"],
            output_root=mirrored["source_root"],
            checkpoint_path=mirrored["checkpoint"],
            terminal_path=mirrored["terminal"],
            pair_path=mirrored["pair"],
            formal_project_root=str(paths["project_root"]),
            fold_index=1,
            candidate_id="F1C_CLIC12",
            technical_smoke=True,
        )


def test_formal_v4_root_contract_has_no_independent_parent_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: independent roots become legal without the narrow smoke control."""

    paths = _v4_technical_smoke_paths(tmp_path)
    monkeypatch.setattr(
        EXPORTER,
        "EXPECTED_TECHNICAL_SMOKE_PROJECT_ROOT",
        str(paths["project_root"]),
    )
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="root binding|formal|parent"):
        EXPORTER.validate_source_v_execution_roots(
            training_root=paths["training_root"],
            clean_path=paths["clean"],
            cache_root=paths["source_root"],
            output_root=paths["source_root"],
            checkpoint_path=paths["checkpoint"],
            terminal_path=paths["terminal"],
            pair_path=paths["pair"],
            formal_project_root=str(paths["project_root"]),
            fold_index=1,
            candidate_id="F1C_CLIC12",
            technical_smoke=False,
        )

    formal_root = paths["runs"] / V4_RUN_ID
    EXPORTER.validate_source_v_execution_roots(
        training_root=paths["training_root"],
        clean_path=paths["clean"],
        cache_root=formal_root,
        output_root=formal_root,
        checkpoint_path=paths["checkpoint"],
        terminal_path=paths["terminal"],
        pair_path=paths["pair"],
        formal_project_root=str(paths["project_root"]),
        fold_index=1,
        candidate_id="F1C_CLIC12",
        technical_smoke=False,
    )
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="technical smoke|root"):
        EXPORTER.validate_source_v_execution_roots(
            training_root=paths["training_root"],
            clean_path=paths["clean"],
            cache_root=formal_root,
            output_root=formal_root,
            checkpoint_path=paths["checkpoint"],
            terminal_path=paths["terminal"],
            pair_path=paths["pair"],
            formal_project_root=str(paths["project_root"]),
            fold_index=1,
            candidate_id="F1C_CLIC12",
            technical_smoke=True,
        )


def test_technical_smoke_rejects_same_shape_mirror_before_checkpoint_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a self-consistent mirror replaces the original formal root."""

    canonical = _v4_technical_smoke_paths(tmp_path / "canonical")
    mirror = _v4_technical_smoke_paths(tmp_path / "mirror_only")
    args = _v4_technical_smoke_args(mirror, technical_smoke=True)
    args.formal_project_root = str(mirror["project_root"])
    checkpoint_opened: list[Path] = []

    monkeypatch.setattr(
        EXPORTER,
        "EXPECTED_TECHNICAL_SMOKE_PROJECT_ROOT",
        str(canonical["project_root"]),
    )

    def must_not_open_checkpoint(path: str | Path, **_kwargs: Any) -> dict[str, Any]:
        checkpoint_opened.append(Path(path))
        raise AssertionError("same-shape mirror reached checkpoint opening")

    monkeypatch.setattr(EXPORTER.torch, "load", must_not_open_checkpoint)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="formal project|canonical|root"):
        EXPORTER.export_source_v_leo_features(args)

    assert checkpoint_opened == []
    assert not mirror["output"].exists()
    assert not mirror["binding"].exists()


def test_technical_smoke_rejects_relative_formal_project_root(tmp_path: Path) -> None:
    """Break caught: a relative root lets callers escape the frozen formal parent."""

    paths = _v4_technical_smoke_paths(tmp_path)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="formal project|absolute|root"):
        EXPORTER.validate_source_v_execution_roots(
            training_root=paths["training_root"],
            clean_path=paths["clean"],
            cache_root=paths["source_root"],
            output_root=paths["source_root"],
            checkpoint_path=paths["checkpoint"],
            terminal_path=paths["terminal"],
            pair_path=paths["pair"],
            formal_project_root="relative-project-root",
            fold_index=1,
            candidate_id="F1C_CLIC12",
            technical_smoke=True,
        )


def _cache_arrays(row_count: int = 12) -> dict[str, np.ndarray]:
    rows = np.arange(row_count, dtype=np.float32)
    return {
        "received_iq": np.stack(
            (np.stack((rows, rows + 1.0), axis=1), np.stack((rows + 2.0, rows + 3.0), axis=1)),
            axis=1,
        ).astype(np.float32),
        "tx_ids": np.asarray([SOURCE_TX[index % 4] for index in range(row_count)], dtype=str),
        "rx_ids": np.asarray([f"rx-{index % 2}" for index in range(row_count)], dtype=str),
        "day_ids": np.asarray([f"day-{index % 2}" for index in range(row_count)], dtype=str),
        "physical_sample_id": np.asarray([f"physical-{index:04d}" for index in range(row_count)], dtype=str),
        "sat_scenarios": np.asarray([SCENES[index % 3] for index in range(row_count)], dtype=str),
    }


def _task1_physical_ids(
    *,
    dataset_sha256: str,
    tx_ids: np.ndarray,
    rx_ids: np.ndarray,
    day_ids: np.ndarray,
    eq_ids: np.ndarray,
    sig_ids: np.ndarray,
) -> np.ndarray:
    """Construct test IDs with Task1 itself, never a copied hash formula."""

    return np.asarray(
        [
            TASK1_CACHE._physical_sample_id(
                dataset_sha256=dataset_sha256,
                tx_id=str(tx_id),
                rx_id=str(rx_id),
                day_id=str(day_id),
                eq_id=str(eq_id),
                sig_id=str(sig_id),
            )
            for tx_id, rx_id, day_id, eq_id, sig_id in zip(
                tx_ids, rx_ids, day_ids, eq_ids, sig_ids, strict=True
            )
        ],
        dtype=str,
    )


def _cache_receipt(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    physical = arrays["physical_sample_id"].astype(str).tolist()
    return {
        "schema": "cvs.phase1.clic_source_v_leo_received_iq.v1",
        "method": "P1_CLIC",
        "role": "source_validation_known_leo_weak",
        "source_v_only": True,
        "post_target_completion_audit_non_selection": True,
        "fold_index": 1,
        "training_run_id": "phase1_clic12_20260812_v5",
        "clean_evidence_run_id": "phase1_clic_postfreeze_20260812_v4",
        "source_tx_ids": list(SOURCE_TX),
        "source_validation_row_count": int(arrays["received_iq"].shape[0]),
        "source_validation_indices_sha256": "a" * 64,
        "source_validation_physical_order_sha256": hashlib.sha256(
            json.dumps(physical, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "physical_order_sha256": hashlib.sha256(
            json.dumps(physical, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "formal_scenarios": list(SCENES),
        "received_iq_npz_path": str(path),
        "received_iq_npz_sha256": _sha256(path),
        "same_received_iq_bytes_for_c_and_g": True,
        "single_leo_observation_per_physical_sample": True,
        "cross_scene_physical_sample_reuse": False,
        "clean_source_runtime_access": False,
        "target_access": False,
        "query_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "proxy_forward_rows": 0,
        "source_l_forward_rows": 0,
        "source_v_forward_rows": 0,
        "selection_access": False,
        "retry_access": False,
    }


def _write_cache(tmp_path: Path, row_count: int = 12) -> tuple[Path, Path, dict[str, np.ndarray]]:
    arrays = _cache_arrays(row_count)
    cache_path = tmp_path / "source_validation_known_leo_weak.npz"
    receipt_path = tmp_path / "source_validation_known_leo_weak.receipt.json"
    np.savez(cache_path, **arrays)
    receipt_path.write_text(json.dumps(_cache_receipt(cache_path, arrays), sort_keys=True) + "\n", encoding="utf-8")
    return cache_path, receipt_path, arrays


def _metric_rows(*, correct: int, denominator: int = 8) -> dict[str, Any]:
    if correct < 0 or correct > denominator:
        raise ValueError("test fixture metric is invalid")
    accuracy = correct / denominator
    raw = {"correct": correct, "denominator": denominator, "accuracy": accuracy}

    def partition(total: int, parts: int) -> list[int]:
        base, remainder = divmod(total, parts)
        return [base + int(index < remainder) for index in range(parts)]

    def cells(labels: list[str]) -> dict[str, dict[str, Any]]:
        denominators = partition(denominator, len(labels))
        remaining = correct
        result: dict[str, dict[str, Any]] = {}
        for label, cell_denominator in zip(labels, denominators, strict=True):
            cell_correct = min(cell_denominator, remaining)
            result[label] = {
                "correct": cell_correct,
                "denominator": cell_denominator,
                "accuracy": cell_correct / cell_denominator,
            }
            remaining -= cell_correct
        assert remaining == 0
        return result

    by_class = cells([f"tx-{index}" for index in range(4)])
    by_rx = cells([f"rx-{index}" for index in range(2)])
    by_day = cells([f"day-{index}" for index in range(2)])
    return {
        "overall": dict(raw),
        "macro_accuracy": accuracy,
        "by_class": by_class,
        "by_rx": by_rx,
        "by_day": by_day,
        "floors": {
            "overall_accuracy": accuracy,
            "min_class_accuracy": min(cell["accuracy"] for cell in by_class.values()),
            "min_rx_accuracy": min(cell["accuracy"] for cell in by_rx.values()),
            "min_day_accuracy": min(cell["accuracy"] for cell in by_day.values()),
        },
    }


def _pair_receipt(fold: int, *, g_correct: int = 8, proxy_delta: float = 0.1) -> dict[str, Any]:
    c = _metric_rows(correct=8)
    g = _metric_rows(correct=g_correct)
    return {
        "schema": METRICS.SOURCE_METRICS_PAIR_SCHEMA,
        "fold_index": fold,
        "post_target_completion_audit_non_selection": True,
        "completion_audit": "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION",
        "source_only": True,
        "source_tx_ids": list(SOURCE_TX),
        "formal_scenarios": list(SCENES),
        "source_validation_row_count": 8,
        "source_l_rows_read": 0,
        "proxy_rows_read": 0,
        "target_access": False,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "selection_access": False,
        "retry_access": False,
        "arms": {
            "C": {"clean": c, "scenes": {scene: c for scene in SCENES}},
            "G": {"clean": g, "scenes": {scene: g for scene in SCENES}},
        },
        "proxy": {
            "C": {"AUROC_unknown": 0.50, "u_gap": 0.10, "fit_rows": 0, "threshold_fit_rows": 0},
            "G": {
                "AUROC_unknown": 0.50 + proxy_delta,
                "u_gap": 0.10 + proxy_delta,
                "fit_rows": 0,
                "threshold_fit_rows": 0,
            },
        },
    }


def test_source_v_cache_snapshot_is_v_only_and_rejects_hash_or_role_drift(tmp_path: Path) -> None:
    """Break caught: reopening an unsealed, target-facing, or changed cache."""

    cache_path, receipt_path, arrays = _write_cache(tmp_path)
    snapshot = EXPORTER.read_source_v_cache_snapshot(
        cache_path=cache_path,
        cache_receipt_path=receipt_path,
        fold_index=1,
        source_tx_ids=SOURCE_TX,
        expected_row_count=12,
    )
    assert snapshot["row_count"] == 12
    assert snapshot["source_l_rows_read"] == 0
    assert snapshot["proxy_rows_read"] == 0
    assert snapshot["target_access"] is False
    assert tuple(snapshot["sat_scenarios"]) == tuple(arrays["sat_scenarios"].astype(str))

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["target_access"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="target|source-only|access"):
        EXPORTER.read_source_v_cache_snapshot(
            cache_path=cache_path,
            cache_receipt_path=receipt_path,
            fold_index=1,
            source_tx_ids=SOURCE_TX,
            expected_row_count=12,
        )


def test_source_v_cache_snapshot_rejects_post_open_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Break caught: a cache path changed after reading cannot enter a V forward."""

    cache_path, receipt_path, _arrays = _write_cache(tmp_path)
    native_sha = EXPORTER._sha256_file
    cache_calls = 0

    def drifting_sha(path: str | Path) -> str:
        nonlocal cache_calls
        if Path(path).resolve() == cache_path.resolve():
            cache_calls += 1
            if cache_calls >= 2:
                return "d" * 64
        return native_sha(path)

    monkeypatch.setattr(EXPORTER, "_sha256_file", drifting_sha)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="changed|SHA|drift"):
        EXPORTER.read_source_v_cache_snapshot(
            cache_path=cache_path,
            cache_receipt_path=receipt_path,
            fold_index=1,
            source_tx_ids=SOURCE_TX,
            expected_row_count=12,
        )
    assert cache_calls >= 2


def test_source_v_cache_snapshot_requires_sealed_validation_identity_hashes(tmp_path: Path) -> None:
    """Break caught: a cache receipt cannot omit or fake the clean-V index identity."""

    cache_path, receipt_path, _arrays = _write_cache(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_validation_indices_sha256"] = "not-a-sha256"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="validation.*SHA|SHA256"):
        EXPORTER.read_source_v_cache_snapshot(
            cache_path=cache_path,
            cache_receipt_path=receipt_path,
            fold_index=1,
            source_tx_ids=SOURCE_TX,
            expected_row_count=12,
        )


def test_pair_single_leo_common_binding_requires_sealed_source_l_identity() -> None:
    """Break caught: a V forward may not consume a PAIR policy without its sealed L binding."""

    binding = {
        "received_iq_sha256": "a" * 64,
        "physical_order_sha256": "b" * 64,
        "source_only": True,
        "single_leo_observation": True,
    }
    assert EXPORTER.validate_pair_single_leo_common_binding(binding) == binding

    malformed = dict(binding)
    malformed["physical_order_sha256"] = "not-a-sha256"
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="single-LEO|SHA256"):
        EXPORTER.validate_pair_single_leo_common_binding(malformed)

    policies = {
        scene: {
            "received_iq_sha256": binding["received_iq_sha256"],
            "physical_order_sha256": binding["physical_order_sha256"],
        }
        for scene in SCENES
    }
    EXPORTER.validate_pair_source_l_policy_binding(binding, policies)
    policies["leo_rain_weak"] = dict(policies["leo_rain_weak"], physical_order_sha256="c" * 64)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="policy|single-LEO|binding"):
        EXPORTER.validate_pair_source_l_policy_binding(binding, policies)


def _production_shape_proxy_diagnostic(
    *, auroc_unknown: float = 0.75, u_gap: float = 0.20
) -> dict[str, Any]:
    """Return the PAIR-v3 diagnostic contract emitted by the production scorer.

    The producer records its one permitted source-L geometry fit nested under
    ``fit`` and ``geometry``.  It deliberately has no top-level fit counters:
    source-validation and proxy scoring are the only zero-fit roles here.
    """

    geometry_fit_rows = 24
    geometry = {
        "fit_rows": geometry_fit_rows,
        "state_sha256": "e" * 64,
        "feature_dim": 3,
        "class_counts": {tx_id: 6 for tx_id in SOURCE_TX},
    }
    return {
        "schema": "cvs.phase1.clic_proxy_diagnostic.v1",
        "geometry": geometry,
        "geometry_state_sha256": geometry["state_sha256"],
        "fit": {
            "role": "source_L_only",
            "fit_rows": geometry_fit_rows,
            "threshold_fit_rows": 0,
            "class_counts": dict(geometry["class_counts"]),
            "feature_dim": geometry["feature_dim"],
            "normalization": "float64_totalized_l2_zero_preserved",
            "variance_ddof": 1,
            "variance_shrink_class": 0.25,
            "variance_shrink_pooled": 0.10,
            "variance_floor": 1e-6,
        },
        "source_validation_known": {
            "role": "source_validation_known",
            "count": 8,
            "mean_e_unknown": 0.10,
            "min_e_unknown": 0.05,
            "max_e_unknown": 0.20,
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        },
        "proxy_unknown": {
            "role": "proxy_unknown",
            "tx_id": "heldout-proxy-tx",
            "count": 400,
            "mean_e_unknown": 0.30,
            "min_e_unknown": 0.15,
            "max_e_unknown": 0.50,
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        },
        "AUROC_unknown": auroc_unknown,
        "u_gap": u_gap,
        "proxy_minus_known_heldout_mean_e_unknown": u_gap,
        "score_rule": "log4_minus_logsumexp_negative_full_diagonal_gaussian_nll",
        "threshold_used": False,
        "tail_policy_used": False,
        "source_validation_fit_rows": 0,
        "proxy_fit_rows": 0,
        "source_validation_threshold_rows": 0,
        "proxy_threshold_rows": 0,
    }


def _open_pair_policy_with_proxy_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Open a real PAIR receipt file while leaving its policy validator real."""

    binding = {
        "received_iq_sha256": "a" * 64,
        "physical_order_sha256": "b" * 64,
        "source_only": True,
        "single_leo_observation": True,
    }
    policies = {
        scene: {
            "received_iq_sha256": binding["received_iq_sha256"],
            "physical_order_sha256": binding["physical_order_sha256"],
        }
        for scene in SCENES
    }
    pair_path = tmp_path / "F1_C_vs_G_pair.json"
    payload = {
        "schema": EXPORTER.EXPECTED_PAIR_SCHEMA,
        "fold_index": 1,
        "source_only": True,
        "target_artifacts_present": False,
        "source_tx_ids": list(SOURCE_TX),
        "single_leo_common_binding": binding,
        "clic_source_policy_state": {"C": {"placeholder": "C"}, "G": {"placeholder": "G"}},
        "proxy_diagnostic": {"C": diagnostic, "G": copy.deepcopy(diagnostic)},
    }
    pair_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        EXPORTER._pair,
        "_validated_clic_source_policy_state",
        lambda _state, **_kwargs: {"policies": policies, "state_sha256": "c" * 64},
    )
    return EXPORTER._load_pair_policy_state(
        pair_json_path=pair_path,
        fold_index=1,
        arm="C",
        checkpoint_sha256="d" * 64,
        terminal_receipt_sha256="f" * 64,
        source_tx_ids=SOURCE_TX,
    )


def test_pair_proxy_diagnostic_accepts_production_source_l_fit_without_top_level_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a legal PAIR-v3 source-L geometry fit is rejected as V/proxy fitting."""

    diagnostic = _production_shape_proxy_diagnostic()
    assert "fit_rows" not in diagnostic
    assert "threshold_fit_rows" not in diagnostic
    state, _pair_sha, observed = _open_pair_policy_with_proxy_diagnostic(
        tmp_path, monkeypatch, diagnostic
    )
    assert state["state_sha256"] == "c" * 64
    assert observed == diagnostic


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("source_validation_known", "fit_rows"), 1),
        (("proxy_unknown", "fit_rows"), 1),
        (("source_validation_fit_rows",), 1),
        (("proxy_fit_rows",), 1),
        (("fit", "threshold_fit_rows"), 1),
        (("source_validation_known", "threshold_fit_rows"), 1),
        (("proxy_unknown", "threshold_fit_rows"), 1),
        (("source_validation_threshold_rows",), 1),
        (("proxy_threshold_rows",), 1),
    ),
)
def test_pair_proxy_diagnostic_rejects_v_proxy_or_threshold_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: int,
) -> None:
    """Break caught: V/proxy scoring or any threshold fit enters the read-only PAIR reuse path."""

    diagnostic = _production_shape_proxy_diagnostic()
    target: dict[str, Any] = diagnostic
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="proxy diagnostic.*(fit|threshold)"):
        _open_pair_policy_with_proxy_diagnostic(tmp_path, monkeypatch, diagnostic)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("fit", "fit_rows"), 0),
        (("geometry", "fit_rows"), 0),
        (("geometry", "fit_rows"), 25),
    ),
)
def test_pair_proxy_diagnostic_requires_positive_bound_source_l_geometry_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: int,
) -> None:
    """Break caught: source-L geometry fit is zero, absent, or detached from its geometry state."""

    diagnostic = _production_shape_proxy_diagnostic()
    target: dict[str, Any] = diagnostic
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="proxy diagnostic.*source-L.*fit_rows"):
        _open_pair_policy_with_proxy_diagnostic(tmp_path, monkeypatch, diagnostic)


def test_clean_v_cache_identity_requires_the_same_index_and_order_hashes() -> None:
    """Break caught: matching V rows alone cannot substitute for sealed clean/cache identities."""

    cache_receipt = {
        "source_validation_indices_sha256": "c" * 64,
        "source_validation_physical_order_sha256": "d" * 64,
    }
    c_manifest = dict(cache_receipt)
    g_manifest = dict(cache_receipt)
    METRICS.validate_clean_v_cache_identity(
        cache_receipt=cache_receipt,
        clean_manifests={"C": c_manifest, "G": g_manifest},
    )

    g_manifest["source_validation_indices_sha256"] = "e" * 64
    with pytest.raises(METRICS.CLICSourceMetricsError, match="index|identity|binding"):
        METRICS.validate_clean_v_cache_identity(
            cache_receipt=cache_receipt,
            clean_manifests={"C": c_manifest, "G": g_manifest},
        )


def test_source_v_feature_metadata_reopens_every_cache_axis() -> None:
    """Break caught: a sealed V feature file cannot reassign RX/day metric cells."""

    cache_axes = {
        "tx_ids": np.asarray(SOURCE_TX, dtype=str),
        "rx_ids": np.asarray(["rx-0", "rx-0", "rx-1", "rx-1"], dtype=str),
        "day_ids": np.asarray(["day-0", "day-1", "day-0", "day-1"], dtype=str),
        "physical_ids": np.asarray(["p-0", "p-1", "p-2", "p-3"], dtype=str),
        "scenes": np.asarray(SCENES + ("leo_clear_weak",), dtype=str),
    }
    feature_axes = {key: np.array(value, copy=True) for key, value in cache_axes.items()}
    METRICS.validate_source_v_feature_cache_metadata(
        feature_axes=feature_axes,
        cache_axes=cache_axes,
    )

    feature_axes["rx_ids"] = np.asarray(["rx-1", "rx-1", "rx-0", "rx-0"], dtype=str)
    with pytest.raises(METRICS.CLICSourceMetricsError, match="RX|cache|metadata|binding"):
        METRICS.validate_source_v_feature_cache_metadata(
            feature_axes=feature_axes,
            cache_axes=cache_axes,
        )


def test_source_v_forward_reopens_clean_v4_identity_before_forward() -> None:
    """Break caught: raw clean sig IDs cannot substitute Task1 physical IDs."""

    dataset_sha256 = "a" * 64
    tx_ids = np.asarray(SOURCE_TX, dtype=str)
    rx_ids = np.asarray(["rx-0", "rx-0", "rx-1", "rx-1"], dtype=str)
    day_ids = np.asarray(["day-0", "day-1", "day-0", "day-1"], dtype=str)
    eq_ids = np.asarray(["eq-0", "eq-1", "eq-0", "eq-1"], dtype=str)
    sig_ids = np.asarray(["sig-0", "sig-1", "sig-2", "sig-3"], dtype=str)
    physical_ids = _task1_physical_ids(
        dataset_sha256=dataset_sha256,
        tx_ids=tx_ids,
        rx_ids=rx_ids,
        day_ids=day_ids,
        eq_ids=eq_ids,
        sig_ids=sig_ids,
    )

    snapshot = {
        "receipt": {
            "source_validation_indices_sha256": "a" * 64,
            "source_validation_physical_order_sha256": "b" * 64,
        },
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "physical_ids": physical_ids,
    }
    clean_binding = {
        "validation_indices_sha256": "a" * 64,
        "validation_metadata_order_sha256": "b" * 64,
        "validation_tx_ids": tuple(tx_ids.tolist()),
        "validation_rx_ids": tuple(rx_ids.tolist()),
        "validation_day_ids": tuple(day_ids.tolist()),
        "validation_eq_ids": tuple(eq_ids.tolist()),
        "validation_sig_ids": tuple(sig_ids.tolist()),
        "manifest": {"wisig_pkl_sha256": dataset_sha256},
    }
    EXPORTER.validate_source_v_clean_v4_binding(snapshot=snapshot, clean_binding=clean_binding)

    clean_binding["validation_indices_sha256"] = "c" * 64
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="clean-v4|index|binding"):
        EXPORTER.validate_source_v_clean_v4_binding(snapshot=snapshot, clean_binding=clean_binding)


@pytest.mark.parametrize(
    "component",
    ("dataset_sha256", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"),
)
def test_source_v_physical_component_drift_stops_before_forward_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    """Break caught: any Task1 physical-ID input drift reaches neither model nor output."""

    runs = tmp_path / "runs"
    training_root = runs / "phase1_clic12_20260812_v5"
    clean_root = runs / "phase1_clic_postfreeze_20260812_v4"
    source_root = runs / V4_RUN_ID
    candidate = "F1C_CLIC12"
    checkpoint_path = training_root / candidate / "final_ssdg.pth"
    terminal_path = training_root / candidate / "terminal_receipt.json"
    clean_path = clean_root / candidate / "source_clean_proxy.npz"
    cache_path = source_root / "F1_SHARED" / "source_validation_known_leo_weak.npz"
    cache_receipt_path = source_root / "F1_SHARED" / "source_validation_known_leo_weak.receipt.json"
    pair_path = runs / "pair_v3.json"
    output_path = source_root / candidate / "source_v_features.npz"
    binding_path = source_root / candidate / "source_v_features.binding.json"
    for path in (
        checkpoint_path,
        terminal_path,
        clean_path,
        cache_path,
        cache_receipt_path,
        pair_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))

    dataset_sha256 = "a" * 64
    tx_ids = np.asarray(SOURCE_TX, dtype=str)
    rx_ids = np.asarray(["rx-0", "rx-0", "rx-1", "rx-1"], dtype=str)
    day_ids = np.asarray(["day-0", "day-1", "day-0", "day-1"], dtype=str)
    eq_ids = np.asarray(["eq-0", "eq-1", "eq-0", "eq-1"], dtype=str)
    sig_ids = np.asarray(["sig-0", "sig-1", "sig-2", "sig-3"], dtype=str)
    physical_ids = _task1_physical_ids(
        dataset_sha256=dataset_sha256,
        tx_ids=tx_ids,
        rx_ids=rx_ids,
        day_ids=day_ids,
        eq_ids=eq_ids,
        sig_ids=sig_ids,
    )
    snapshot = {
        "cache_sha256": _sha256(cache_path),
        "cache_receipt_sha256": _sha256(cache_receipt_path),
        "receipt": {
            "source_validation_indices_sha256": "a" * 64,
            "source_validation_physical_order_sha256": "b" * 64,
            "checkpoint_sha256_by_arm": {"C": _sha256(checkpoint_path)},
            "terminal_receipt_sha256_by_arm": {"C": _sha256(terminal_path)},
        },
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "physical_ids": physical_ids,
        "row_count": int(physical_ids.size),
    }
    clean_binding = {
        "sha256": _sha256(clean_path),
        "validation_indices_sha256": "a" * 64,
        "validation_metadata_order_sha256": "b" * 64,
        "validation_tx_ids": tuple(tx_ids.tolist()),
        "validation_rx_ids": tuple(rx_ids.tolist()),
        "validation_day_ids": tuple(day_ids.tolist()),
        "validation_eq_ids": tuple(eq_ids.tolist()),
        "validation_sig_ids": tuple(sig_ids.tolist()),
        "manifest": {"wisig_pkl_sha256": dataset_sha256},
    }
    drifted_snapshot = dict(snapshot)
    drifted_binding = dict(clean_binding)
    drifted_binding["manifest"] = dict(clean_binding["manifest"])
    if component == "dataset_sha256":
        drifted_binding["manifest"]["wisig_pkl_sha256"] = "b" * 64
    else:
        binding_field = f"validation_{component}"
        values = list(drifted_binding[binding_field])
        values[0] = f"drifted-{component}"
        drifted_binding[binding_field] = tuple(values)
        if component in {"tx_ids", "rx_ids", "day_ids"}:
            drifted_snapshot[component] = np.asarray(values, dtype=str)

    monkeypatch.setattr(
        EXPORTER.torch,
        "load",
        lambda *_args, **_kwargs: {
            "args": {
                "phase1_source_known_validation_tx_ids": "held-tx",
                "phase1_source_proxy_unknown_tx_ids": "proxy-tx",
                "wisig_out_len": 256,
            }
        },
    )
    monkeypatch.setattr(
        EXPORTER._clean,
        "validate_clic_training_checkpoint",
        lambda *_args, **_kwargs: ({}, {}, "C"),
    )
    monkeypatch.setattr(
        EXPORTER,
        "read_source_v_cache_snapshot",
        lambda **_kwargs: drifted_snapshot,
    )
    monkeypatch.setattr(
        EXPORTER._cache,
        "_read_clean_validation_binding",
        lambda **_kwargs: drifted_binding,
    )
    pair_calls: list[bool] = []

    def must_not_reach_pair(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
        pair_calls.append(True)
        raise AssertionError("physical-ID drift reached the post-binding forward path")

    monkeypatch.setattr(EXPORTER, "_load_pair_policy_state", must_not_reach_pair)
    args = Namespace(
        ckpt=str(checkpoint_path),
        terminal_receipt_json=str(terminal_path),
        clean_npz=str(clean_path),
        source_v_received_iq_npz=str(cache_path),
        source_v_received_iq_receipt_json=str(cache_receipt_path),
        pair_json=str(pair_path),
        training_run_root=str(training_root),
        cache_run_root=str(source_root),
        output_root=str(source_root),
        out_npz=str(output_path),
        binding_json=str(binding_path),
        candidate_id=candidate,
        fold_index=1,
        arm="C",
        source_tx_ids=",".join(SOURCE_TX),
        batch_size=4,
        device="cpu",
    )
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="physical|binding"):
        EXPORTER.export_source_v_leo_features(args)
    assert pair_calls == []
    assert not output_path.exists()
    assert not binding_path.exists()


def test_source_v_forward_payload_is_one_row_once_and_safe_bridge_avoids_legacy_api() -> None:
    """Break caught: drop/duplicate V forwards or Torch/NumPy legacy bridge use."""

    rows = 12
    payload = {
        "features": np.ones((rows, 3), dtype=np.float32),
        "tx_logits": np.eye(4, dtype=np.float32)[np.arange(rows) % 4],
        "raw_labels": np.arange(rows, dtype=np.int64) % 4,
        "domain_labels": np.zeros(rows, dtype=np.int64),
        "tx_ids": np.asarray([SOURCE_TX[index % 4] for index in range(rows)], dtype=str),
        "rx_ids": np.asarray([f"rx-{index % 2}" for index in range(rows)], dtype=str),
        "day_ids": np.asarray([f"day-{index % 2}" for index in range(rows)], dtype=str),
        "eq_ids": np.asarray(["existing_received_iq"] * rows, dtype=str),
        "sig_ids": np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
        "dataset_role": np.asarray(["source_validation_known_leo_weak"] * rows, dtype=str),
        "channel_views": np.asarray(["received_existing"] * rows, dtype=str),
        "sat_scenarios": np.asarray([SCENES[index % 3] for index in range(rows)], dtype=str),
    }
    validated = EXPORTER.validate_source_v_forward_payload(
        payload=payload,
        physical_ids=np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
        source_tx_ids=SOURCE_TX,
        expected_row_count=rows,
    )
    assert validated["single_leo_forward_count"] == rows
    assert validated["source_l_forward_rows"] == 0
    assert validated["proxy_forward_rows"] == 0

    def forbidden_numpy(_tensor: torch.Tensor) -> np.ndarray:
        raise AssertionError("Tensor.numpy() must not be used")

    def forbidden_from_numpy(_array: np.ndarray) -> torch.Tensor:
        raise AssertionError("torch.from_numpy() must not be used")

    original_numpy = torch.Tensor.numpy
    original_from_numpy = torch.from_numpy
    try:
        torch.Tensor.numpy = forbidden_numpy
        torch.from_numpy = forbidden_from_numpy
        tensor = EXPORTER.numpy_float32_to_tensor(np.ones((2, 2, 4), dtype=np.float32))
        assert tuple(tensor.shape) == (2, 2, 4)
    finally:
        torch.Tensor.numpy = original_numpy
        torch.from_numpy = original_from_numpy

    payload["sig_ids"] = np.asarray(["duplicate"] * rows, dtype=str)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="once|unique|physical"):
        EXPORTER.validate_source_v_forward_payload(
            payload=payload,
            physical_ids=np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
            source_tx_ids=SOURCE_TX,
            expected_row_count=rows,
        )

    bad_payload = dict(payload)
    bad_payload["features"] = np.full((rows, 3), np.nan, dtype=np.float32)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="non-finite"):
        EXPORTER.validate_source_v_forward_payload(
            payload=bad_payload,
            physical_ids=np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
            source_tx_ids=SOURCE_TX,
            expected_row_count=rows,
        )

    misbound_payload = dict(payload)
    misbound_payload["sig_ids"] = np.asarray([f"physical-{index}" for index in range(rows)], dtype=str)
    misbound_payload["tx_ids"] = np.asarray([SOURCE_TX[(index + 1) % 4] for index in range(rows)], dtype=str)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="label|TX|class"):
        EXPORTER.validate_source_v_forward_payload(
            payload=misbound_payload,
            physical_ids=np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
            source_tx_ids=SOURCE_TX,
            expected_row_count=rows,
        )

    noninteger_payload = dict(payload)
    noninteger_payload["sig_ids"] = np.asarray([f"physical-{index}" for index in range(rows)], dtype=str)
    noninteger_payload["raw_labels"] = (np.arange(rows, dtype=np.float32) % 4).astype(np.float32)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="label|integer|dtype"):
        EXPORTER.validate_source_v_forward_payload(
            payload=noninteger_payload,
            physical_ids=np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
            source_tx_ids=SOURCE_TX,
            expected_row_count=rows,
        )


def test_source_v_forward_payload_binds_tx_rx_day_and_scene_to_cache_rows() -> None:
    """Break caught: a physically unique forward cannot relabel cache axes or scene rows."""

    rows = 4
    payload = {
        "features": np.ones((rows, 3), dtype=np.float32),
        "tx_logits": np.eye(4, dtype=np.float32),
        "raw_labels": np.arange(rows, dtype=np.int64),
        "domain_labels": np.zeros(rows, dtype=np.int64),
        "tx_ids": np.asarray(SOURCE_TX, dtype=str),
        "rx_ids": np.asarray(["rx-0", "rx-0", "rx-1", "rx-1"], dtype=str),
        "day_ids": np.asarray(["day-0", "day-1", "day-0", "day-1"], dtype=str),
        "eq_ids": np.asarray(["existing_received_iq"] * rows, dtype=str),
        "sig_ids": np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
        "dataset_role": np.asarray(["source_validation_known_leo_weak"] * rows, dtype=str),
        "channel_views": np.asarray(["received_existing"] * rows, dtype=str),
        "sat_scenarios": np.asarray(SCENES + ("leo_clear_weak",), dtype=str),
    }
    common = {
        "physical_ids": np.asarray([f"physical-{index}" for index in range(rows)], dtype=str),
        "source_tx_ids": SOURCE_TX,
        "expected_row_count": rows,
        "expected_tx_ids": np.asarray(SOURCE_TX, dtype=str),
        "expected_rx_ids": np.asarray(["rx-0", "rx-0", "rx-1", "rx-1"], dtype=str),
        "expected_day_ids": np.asarray(["day-0", "day-1", "day-0", "day-1"], dtype=str),
        "expected_scenarios": np.asarray(SCENES + ("leo_clear_weak",), dtype=str),
    }
    EXPORTER.validate_source_v_forward_payload(payload=payload, **common)

    payload["day_ids"] = np.asarray(["day-1", "day-0", "day-1", "day-0"], dtype=str)
    with pytest.raises(EXPORTER.CLICSourceVFeatureExportError, match="metadata|day|cache|binding"):
        EXPORTER.validate_source_v_forward_payload(payload=payload, **common)


def test_known_metrics_keep_raw_cells_and_count_unknown_or_defer_as_errors() -> None:
    """Break caught: turning known unknown/defer outputs into omissions or successes."""

    truth = np.asarray(["tx-0", "tx-0", "tx-1", "tx-1", "tx-2", "tx-2", "tx-3", "tx-3"], dtype=str)
    predicted = np.asarray(["tx-0", "tx-1", "tx-1", "", "tx-2", "tx-2", "tx-3", ""], dtype=str)
    decisions = np.asarray(["registered", "registered", "registered", "unknown", "registered", "registered", "registered", "defer"], dtype=str)
    metrics = METRICS.score_known_source_rows(
        truth_tx_ids=truth,
        predicted_tx_ids=predicted,
        decisions=decisions,
        rx_ids=np.asarray(["rx-0", "rx-1"] * 4, dtype=str),
        day_ids=np.asarray(["day-0", "day-1"] * 4, dtype=str),
        physical_ids=np.asarray([f"p-{index}" for index in range(8)], dtype=str),
        role="source_validation_known_leo_weak",
        scene="leo_clear_weak",
        source_tx_ids=SOURCE_TX,
    )
    assert metrics["overall"] == {"correct": 5, "denominator": 8, "accuracy": 0.625}
    assert metrics["known_unknown_errors"] == 1
    assert metrics["known_defer_errors"] == 1
    assert set(metrics["by_class"]) == set(SOURCE_TX)
    assert set(metrics["by_rx"]) == {"rx-0", "rx-1"}
    assert set(metrics["by_day"]) == {"day-0", "day-1"}
    assert metrics["floors"]["overall_accuracy"] == 0.625


@pytest.mark.parametrize(
    ("truth", "predicted", "role", "scene", "physical"),
    (
        (np.asarray([], dtype=str), np.asarray([], dtype=str), "source_validation_known_clean", None, np.asarray([], dtype=str)),
        (np.asarray(["tx-0"], dtype=str), np.asarray(["tx-0"], dtype=str), "proxy_unknown", "leo_clear_weak", np.asarray(["p"], dtype=str)),
        (np.asarray(["tx-0", "tx-1"], dtype=str), np.asarray(["tx-0", "tx-1"], dtype=str), "source_validation_known_leo_weak", "leo_clear_weak", np.asarray(["same", "same"], dtype=str)),
    ),
)
def test_known_metric_rejects_zero_denominator_bad_role_or_physical_reuse(
    truth: np.ndarray, predicted: np.ndarray, role: str, scene: str | None, physical: np.ndarray
) -> None:
    with pytest.raises(METRICS.CLICSourceMetricsError, match="denominator|role|physical|reuse|nonempty"):
        METRICS.score_known_source_rows(
            truth_tx_ids=truth,
            predicted_tx_ids=predicted,
            decisions=np.asarray(["registered"] * len(truth), dtype=str),
            rx_ids=np.asarray(["rx-0"] * len(truth), dtype=str),
            day_ids=np.asarray(["day-0"] * len(truth), dtype=str),
            physical_ids=physical,
            role=role,
            scene=scene,
            source_tx_ids=SOURCE_TX,
        )


def test_pair_and_sixfold_gates_require_every_floor_scene_and_strict_proxy_improvement() -> None:
    """Break caught: compensating one weak cell with another or accepting zero proxy delta."""

    pair = _pair_receipt(1)
    one_fold = METRICS.evaluate_pair_noncompensating_gates(pair)
    assert one_fold["passed"] is True
    assert one_fold["fold_scene_equal_overall_delta_pp"] == 0.0

    all_folds = [_pair_receipt(fold) for fold in range(1, 7)]
    aggregate = METRICS.aggregate_source_metric_receipts(all_folds)
    assert aggregate["passed"] is True
    assert aggregate["global_18_scene_equal_overall_delta_pp"] == 0.0
    assert set(aggregate["folds"]) == {"F1", "F2", "F3", "F4", "F5", "F6"}

    weak = _pair_receipt(1, g_correct=7)
    weak["arms"]["G"]["scenes"]["leo_rain_weak"] = _metric_rows(correct=6)
    assert METRICS.evaluate_pair_noncompensating_gates(weak)["passed"] is False

    no_proxy_gain = [_pair_receipt(fold, proxy_delta=0.0) for fold in range(1, 7)]
    assert METRICS.aggregate_source_metric_receipts(no_proxy_gain)["passed"] is False


def test_pair_gate_rejects_inconsistent_raw_axes_or_target_access() -> None:
    """Break caught: altered axis totals or a target flag cannot survive gate aggregation."""

    malformed_axes = _pair_receipt(1)
    malformed_axes["arms"]["C"]["clean"]["by_class"]["tx-0"] = {
        "correct": 1,
        "denominator": 1,
        "accuracy": 1.0,
    }
    with pytest.raises(METRICS.CLICSourceMetricsError, match="axis|raw|overall|consistent"):
        METRICS.evaluate_pair_noncompensating_gates(malformed_axes)

    target_facing = _pair_receipt(1)
    target_facing["target_access"] = True
    with pytest.raises(METRICS.CLICSourceMetricsError, match="target|access|source-only"):
        METRICS.evaluate_pair_noncompensating_gates(target_facing)


def test_source_metrics_clis_are_file_invocable() -> None:
    """Break caught: a launcher cannot invoke either file-only stage as a CLI."""

    code_root = Path(__file__).resolve().parents[1]
    for name in ("export_phase1_clic_source_v_leo_features.py", "evaluate_phase1_clic_source_metrics.py"):
        completed = subprocess.run(
            [sys.executable, str(code_root / name), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_pair_scorer_accepts_production_shape_pair_diagnostic_and_binds_each_v_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: score entry rejects legal nested PAIR fits, drops clean binding, or overwrites output."""

    runs = tmp_path / "runs"
    training_root = runs / "phase1_clic12_20260812_v5"
    clean_root = runs / "phase1_clic_postfreeze_20260812_v4"
    source_root = runs / V4_RUN_ID
    paths: dict[str, Path] = {}

    def write_file(key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode("ascii"))
        paths[key] = path

    for arm in ("C", "G"):
        candidate = f"F1{arm}_CLIC12"
        write_file(f"{arm}_checkpoint", training_root / candidate / "final_ssdg.pth")
        write_file(f"{arm}_terminal", training_root / candidate / "terminal_receipt.json")
        write_file(f"{arm}_clean", clean_root / candidate / "source_clean_proxy.npz")
        write_file(f"{arm}_feature", source_root / candidate / "source_v_features.npz")
        write_file(f"{arm}_binding", source_root / candidate / "source_v_features.binding.json")
    write_file("cache", source_root / "F1_SHARED" / "source_validation_known_leo_weak.npz")
    write_file("cache_receipt", source_root / "F1_SHARED" / "source_validation_known_leo_weak.receipt.json")
    write_file("pair", runs / "pair_v3.json")

    physical = np.asarray([f"p-{index}" for index in range(12)], dtype=str)
    scenes = np.asarray([scene for scene in SCENES for _ in SOURCE_TX], dtype=str)
    tx = np.asarray(list(SOURCE_TX) * 3, dtype=str)
    rx = np.asarray(["rx-0", "rx-1", "rx-0", "rx-1"] * 3, dtype=str)
    day = np.asarray(["day-0", "day-0", "day-1", "day-1"] * 3, dtype=str)
    logits = np.eye(4, dtype=np.float64)[np.arange(12) % 4]
    cache_sha = _sha256(paths["cache"])
    cache_receipt_sha = _sha256(paths["cache_receipt"])
    policy_hashes = {"received_iq_sha256": "a" * 64, "physical_order_sha256": "b" * 64}
    policies = {scene: dict(policy_hashes) for scene in SCENES}

    def fake_open_checkpoint_arm(**kwargs: Any) -> dict[str, Any]:
        arm = kwargs["expected_arm"]
        return {
            "checkpoint_sha256": _sha256(paths[f"{arm}_checkpoint"]),
            "terminal_sha256": _sha256(paths[f"{arm}_terminal"]),
            "terminal": {},
        }

    def fake_cache_snapshot(**_kwargs: Any) -> dict[str, Any]:
        return {
            "cache_sha256": cache_sha,
            "cache_receipt_sha256": cache_receipt_sha,
            "receipt": {
                "source_validation_indices_sha256": "c" * 64,
                "source_validation_physical_order_sha256": "d" * 64,
                "checkpoint_sha256_by_arm": {arm: _sha256(paths[f"{arm}_checkpoint"]) for arm in ("C", "G")},
                "terminal_receipt_sha256_by_arm": {arm: _sha256(paths[f"{arm}_terminal"]) for arm in ("C", "G")},
            },
            "physical_ids": physical,
            "sat_scenarios": scenes,
            "tx_ids": tx,
            "rx_ids": rx,
            "day_ids": day,
            "row_count": 12,
        }

    pair_payload = {
        "schema": METRICS._pair.EXPECTED_PAIR_SCHEMA,
        "fold_index": 1,
        "source_only": True,
        "target_artifacts_present": False,
        "source_tx_ids": list(SOURCE_TX),
        "single_leo_common_binding": {**policy_hashes, "source_only": True, "single_leo_observation": True},
        "clic_source_policy_state": {"C": {"placeholder": "C"}, "G": {"placeholder": "G"}},
        "proxy_diagnostic": {
            "C": _production_shape_proxy_diagnostic(auroc_unknown=0.50, u_gap=0.10),
            "G": _production_shape_proxy_diagnostic(auroc_unknown=0.60, u_gap=0.20),
        },
    }
    assert "fit_rows" not in pair_payload["proxy_diagnostic"]["C"]
    assert "threshold_fit_rows" not in pair_payload["proxy_diagnostic"]["C"]
    feature_clean_hashes: dict[str, str] = {}

    monkeypatch.setattr(METRICS, "_open_checkpoint_arm", fake_open_checkpoint_arm)
    monkeypatch.setattr(METRICS._source_v, "read_source_v_cache_snapshot", fake_cache_snapshot)
    monkeypatch.setattr(METRICS, "_load_json", lambda path, **_kwargs: pair_payload if Path(path) == paths["pair"] else {})
    monkeypatch.setattr(
        METRICS._pair,
        "_validated_clic_source_policy_state",
        lambda _state, *, arm, **_kwargs: {"geometry": {"arm": arm}, "policies": policies, "state_sha256": f"{arm.lower()}" * 64},
    )
    monkeypatch.setattr(
        METRICS,
        "_load_clean_v_evidence",
        lambda *, expected_arm, **_kwargs: {
            "manifest": {"source_validation_indices_sha256": "c" * 64, "source_validation_physical_order_sha256": "d" * 64},
            "v_physical_keys": physical,
            "v_count": 12,
            "v_truth": tx,
            "v_rx": rx,
            "v_day": day,
            "v_logits": logits,
            "sha256": _sha256(paths[f"{expected_arm}_clean"]),
        },
    )

    def fake_load_v_feature_export(*, expected_arm: str, clean_sha256: str, **kwargs: Any) -> dict[str, Any]:
        feature_clean_hashes[expected_arm] = clean_sha256
        assert kwargs["clean_validation_indices_sha256"] == "c" * 64
        assert kwargs["clean_validation_order_sha256"] == "d" * 64
        return {
            "binding_sha256": "e" * 64,
            "manifest": {"source_v_cache_sha256": cache_sha},
            "z_id": np.ones((12, 3), dtype=np.float64),
            "tx_logits": logits,
            "tx_ids": tx,
            "rx_ids": rx,
            "day_ids": day,
            "physical_ids": physical,
            "scenes": scenes,
            "row_count": 12,
        }

    monkeypatch.setattr(METRICS, "_load_v_feature_export", fake_load_v_feature_export)
    monkeypatch.setattr(
        METRICS._pair,
        "score_clic_open_set",
        lambda _geometry, _policy, _z_id, scene_logits, _scene: {
            "decision": np.asarray(["registered"] * scene_logits.shape[0], dtype=str),
            "predicted_class": np.asarray([SOURCE_TX[int(index)] for index in np.argmax(scene_logits, axis=1)], dtype=str),
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        },
    )

    output_json = source_root / "F1_PAIR" / "source_metrics_pair.json"
    args = Namespace(
        fold_index=1,
        training_run_root=str(training_root),
        clean_run_root=str(clean_root),
        cache_run_root=str(source_root),
        output_root=str(source_root),
        output_metrics_json=str(output_json),
        source_tx_ids=",".join(SOURCE_TX),
        c_ckpt=str(paths["C_checkpoint"]),
        c_terminal_receipt_json=str(paths["C_terminal"]),
        g_ckpt=str(paths["G_checkpoint"]),
        g_terminal_receipt_json=str(paths["G_terminal"]),
        c_clean_npz=str(paths["C_clean"]),
        g_clean_npz=str(paths["G_clean"]),
        source_v_received_iq_npz=str(paths["cache"]),
        source_v_received_iq_receipt_json=str(paths["cache_receipt"]),
        pair_json=str(paths["pair"]),
        c_source_v_feature_npz=str(paths["C_feature"]),
        g_source_v_feature_npz=str(paths["G_feature"]),
        c_source_v_binding_json=str(paths["C_binding"]),
        g_source_v_binding_json=str(paths["G_binding"]),
    )
    receipt = METRICS.score_source_metrics_pair(args)
    assert output_json.is_file()
    assert receipt["target_access"] is False
    assert receipt["completion_audit"] == "POST_TARGET_COMPLETION_AUDIT_NON_SELECTION"
    assert receipt["proxy"] == {
        "C": {"AUROC_unknown": 0.50, "u_gap": 0.10, "fit_rows": 0, "threshold_fit_rows": 0},
        "G": {"AUROC_unknown": 0.60, "u_gap": 0.20, "fit_rows": 0, "threshold_fit_rows": 0},
    }
    assert feature_clean_hashes == {"C": _sha256(paths["C_clean"]), "G": _sha256(paths["G_clean"])}
    with pytest.raises(METRICS.CLICSourceMetricsError, match="overwrite"):
        METRICS.score_source_metrics_pair(args)
