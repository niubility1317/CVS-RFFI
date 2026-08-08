from __future__ import annotations

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
    canonical_physical_ids,
    deterministic_reference_query_split,
    fit_gi_epior,
)
from eval_phase1_gi_epior import _fit, _load_bundle, _score, build_parser  # noqa: E402


SOURCE = ("1-1", "1-2", "1-3", "1-4", "1-5")


def _source_arrays(rows_per_class: int = 12, dim: int = 8):
    generator = np.random.default_rng(718)
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
            value = center + generator.normal(0.0, 0.025, size=dim).astype(np.float32)
            features.append(value)
            tx_ids.append(tx)
            rx_ids.append(f"rx-{row % 3}")
            day_ids.append(f"day-{row % 2}")
            eq_ids.append("1")
            sig_ids.append(f"{class_index}-{row}")
    physical = canonical_physical_ids(tx_ids, rx_ids, day_ids, eq_ids, sig_ids)
    return (
        torch.tensor(np.asarray(features), dtype=torch.float32),
        np.asarray(tx_ids, dtype=object),
        np.asarray(rx_ids, dtype=object),
        np.asarray(day_ids, dtype=object),
        np.asarray(eq_ids, dtype=object),
        np.asarray(sig_ids, dtype=object),
        physical,
    )


def _write_npz(path: Path, *, outer_shift: float = 0.0) -> None:
    features, tx, rx, day, eq, sig, _ = _source_arrays()
    generator = np.random.default_rng(911)
    held = generator.normal(-0.4 + outer_shift, 0.02, size=(12, features.shape[1])).astype(np.float32)
    proxy = generator.normal(0.35 + outer_shift, 0.03, size=(12, features.shape[1])).astype(np.float32)
    all_features = np.concatenate([features.numpy(), held, proxy], axis=0)
    logits = np.full((all_features.shape[0], len(SOURCE)), -4.0, dtype=np.float32)
    for index, value in enumerate(tx):
        logits[index, SOURCE.index(str(value))] = 4.0
    logits[len(tx) :, 0] = 4.0
    tx_all = np.concatenate([tx, np.asarray(["9-1"] * 12), np.asarray(["9-2"] * 12)])
    rx_all = np.concatenate([rx, np.asarray(["rx-held"] * 12), np.asarray(["rx-proxy"] * 12)])
    day_all = np.concatenate([day, np.asarray(["day-0"] * 24)])
    eq_all = np.concatenate([eq, np.asarray(["1"] * 24)])
    sig_all = np.concatenate([sig, np.asarray([f"h-{i}" for i in range(12)]), np.asarray([f"p-{i}" for i in range(12)])])
    role = np.concatenate(
        [np.asarray(["source"] * len(tx)), np.asarray(["target_old"] * 12), np.asarray(["proxy_unknown"] * 12)]
    )
    np.savez_compressed(
        path,
        features=all_features,
        tx_logits=logits,
        tx_ids=tx_all,
        rx_ids=rx_all,
        day_ids=day_all,
        eq_ids=eq_all,
        sig_ids=sig_all,
        dataset_role=role,
        channel_views=np.asarray(["clean"] * len(role)),
        sat_scenarios=np.asarray([""] * len(role)),
    )


def test_physical_reference_query_split_is_disjoint_and_closed():
    _, tx, _, _, _, _, physical = _source_arrays()
    reference, query, receipt = deterministic_reference_query_split(tx, physical, SOURCE)
    assert not np.any(reference & query)
    assert set(physical[reference]).isdisjoint(set(physical[query]))
    assert np.all(reference | query)
    assert receipt["physical_overlap"] == 0
    assert all(value == {"reference": 6, "query": 6} for value in receipt["per_tx"].values())


def test_duplicate_or_nonfinite_input_fails_closed():
    features, tx, rx, day, eq, sig, physical = _source_arrays()
    tx_dup, rx_dup, day_dup, eq_dup, sig_dup = (value.copy() for value in (tx, rx, day, eq, sig))
    for value in (tx_dup, rx_dup, day_dup, eq_dup, sig_dup):
        value[-1] = value[0]
    with pytest.raises(GIEpiORError, match="unique"):
        canonical_physical_ids(tx_dup, rx_dup, day_dup, eq_dup, sig_dup)
    features[0, 0] = float("nan")
    with pytest.raises(GIEpiORError, match="finite"):
        fit_gi_epior(features, tx, physical, SOURCE)


def test_entire_tx_is_excluded_and_identity_features_receive_no_gradient():
    features, tx, _, _, _, _, physical = _source_arrays()
    features.requires_grad_(True)
    result = fit_gi_epior(features, tx, physical, SOURCE)
    assert features.grad is None
    assert result.receipt["identity_gradient_norm"] == 0.0
    assert result.receipt["head_gradient_norm"] > 0.0
    assert set(result.receipt["episodes"]) == set(SOURCE)
    assert all(value["held_reference_rows"] == 0 for value in result.receipt["episodes"].values())


def test_numpy_int64_episode_indices_do_not_require_dtype_inference(monkeypatch):
    features, tx, _, _, _, _, physical = _source_arrays()
    original = torch.as_tensor

    def guarded_as_tensor(value, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer) and dtype is None:
            raise RuntimeError("Could not infer dtype of numpy.int64")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", guarded_as_tensor)
    result = fit_gi_epior(features, tx, physical, SOURCE)
    assert result.receipt["train_rows"] > 0


def test_class_permutation_keeps_class_symmetric_scores():
    features, tx, _, _, _, _, physical = _source_arrays()
    direct = fit_gi_epior(features, tx, physical, SOURCE)
    reverse = fit_gi_epior(features, tx, physical, tuple(reversed(SOURCE)))
    probe = features[:17]
    with torch.no_grad():
        direct_score = direct.runtime(probe)[0]
        reverse_score = reverse.runtime(probe)[0]
    assert torch.allclose(direct_score, reverse_score, atol=2.0e-5, rtol=2.0e-5)


def test_runtime_torchscript_parity_and_synthetic_separation():
    features, tx, _, _, _, _, physical = _source_arrays()
    result = fit_gi_epior(features, tx, physical, SOURCE)
    script = torch.jit.script(result.runtime.eval())
    known = features[:20]
    unknown = torch.full((20, features.shape[1]), -0.5)
    with torch.no_grad():
        eager = result.runtime(torch.cat([known, unknown]))
        scripted = script(torch.cat([known, unknown]))
    for left, right in zip(eager, scripted):
        assert torch.allclose(left, right, atol=1.0e-6, rtol=1.0e-6)
    assert float(eager[0][20:].mean()) > float(eager[0][:20].mean())


def test_cli_has_fixed_threshold_and_no_quantile_controls():
    parser = build_parser()
    help_text = parser.format_help().lower()
    assert "quantile" not in help_text
    assert GI_EPIOR_THRESHOLD == 0.5
    with pytest.raises(SystemExit):
        parser.parse_args(["fit", "--quantile", "0.95"])


def test_outer_rows_do_not_change_fit_bundle(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_npz(first, outer_shift=0.0)
    _write_npz(second, outer_shift=8.0)

    def run_fit(source: Path, prefix: str):
        args = build_parser().parse_args(
            [
                "fit",
                "--feature-npz",
                str(source),
                "--source-tx-ids",
                ",".join(SOURCE),
                "--output-bundle",
                str(tmp_path / f"{prefix}.npz"),
                "--output-torchscript",
                str(tmp_path / f"{prefix}.ts"),
                "--output-receipt",
                str(tmp_path / f"{prefix}.json"),
            ]
        )
        receipt = _fit(args)
        runtime, manifest = _load_bundle(tmp_path / f"{prefix}.npz")
        return receipt, runtime, manifest

    receipt_a, runtime_a, manifest_a = run_fit(first, "a")
    receipt_b, runtime_b, manifest_b = run_fit(second, "b")
    assert receipt_a["outer_zero_fit"] and receipt_b["outer_zero_fit"]
    assert receipt_a["non_source_rows_excluded_from_fit"] == 24
    assert manifest_a["fit_receipt"]["train_rows"] == manifest_b["fit_receipt"]["train_rows"]
    probe = torch.randn(9, 8)
    with torch.no_grad():
        assert torch.allclose(runtime_a(probe)[0], runtime_b(probe)[0], atol=1.0e-7, rtol=0.0)


def test_fit_then_score_outputs_nonconfirmatory_metrics(tmp_path: Path):
    feature_npz = tmp_path / "features.npz"
    _write_npz(feature_npz)
    fit_args = build_parser().parse_args(
        [
            "fit",
            "--feature-npz",
            str(feature_npz),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--output-bundle",
            str(tmp_path / "bundle.npz"),
            "--output-torchscript",
            str(tmp_path / "runtime.ts"),
            "--output-receipt",
            str(tmp_path / "fit.json"),
        ]
    )
    _fit(fit_args)
    score_args = build_parser().parse_args(
        [
            "score",
            "--feature-npz",
            str(feature_npz),
            "--bundle",
            str(tmp_path / "bundle.npz"),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--held-tx-ids",
            "9-1",
            "--view-name",
            "clean",
            "--output-json",
            str(tmp_path / "metrics.json"),
            "--output-csv",
            str(tmp_path / "scores.csv"),
        ]
    )
    metrics = _score(score_args)
    assert metrics["outer_used_for_fit_or_calibration"] is False
    assert metrics["threshold"] == 0.5
    assert metrics["held_count"] == 12 and metrics["proxy_count"] == 12
    assert metrics["nct_ratio_continuous_only"]["thresholded"] is False
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["method"] == "GI-EpiOR"
    assert len((tmp_path / "scores.csv").read_text(encoding="utf-8").splitlines()) == 85


def test_score_does_not_use_tensor_numpy_bridge(tmp_path: Path, monkeypatch):
    feature_npz = tmp_path / "features.npz"
    _write_npz(feature_npz)
    fit_args = build_parser().parse_args(
        [
            "fit",
            "--feature-npz",
            str(feature_npz),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--output-bundle",
            str(tmp_path / "bundle.npz"),
            "--output-torchscript",
            str(tmp_path / "runtime.ts"),
            "--output-receipt",
            str(tmp_path / "fit.json"),
        ]
    )
    _fit(fit_args)

    def forbidden_numpy(_tensor):
        raise TypeError("tensor numpy bridge is unavailable")

    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_numpy)
    score_args = build_parser().parse_args(
        [
            "score",
            "--feature-npz",
            str(feature_npz),
            "--bundle",
            str(tmp_path / "bundle.npz"),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--held-tx-ids",
            "9-1",
            "--view-name",
            "clean",
            "--output-json",
            str(tmp_path / "metrics.json"),
            "--output-csv",
            str(tmp_path / "scores.csv"),
        ]
    )
    metrics = _score(score_args)
    assert metrics["known_query_count"] > 0
