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
    GIEpiORHead,
    GIEpiORRuntime,
    canonical_physical_ids,
    deterministic_reference_query_split,
    fit_class_geometry,
)
from cvsrffi.phase1_wrc_nct import (  # noqa: E402
    WRC_NCT_ALPHA,
    WRCNCTError,
    WRCNCTRuntime,
    deterministic_reference_calibration_eval_split,
    finite_upper_quantile,
)
from eval_phase1_wrc_nct import _load_npz, _run, build_parser  # noqa: E402


SOURCE = ("1-1", "1-2", "1-3", "1-4", "1-5")


def _source_arrays(rows_per_class: int = 160, dim: int = 8):
    generator = np.random.default_rng(451)
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
            features.append(center + generator.normal(0.0, 0.015, size=dim).astype(np.float32))
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


def _write_gi_bundle(path: Path, features: torch.Tensor, tx: np.ndarray, physical: np.ndarray) -> GIEpiORRuntime:
    reference, _, _ = deterministic_reference_query_split(tx, physical, SOURCE)
    prototypes, scales = fit_class_geometry(features, tx, reference, SOURCE)
    torch.manual_seed(917)
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
    return runtime


def _write_npz(path: Path, *, outer_shift: float = 0.0) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    features, tx, rx, day, eq, sig, physical = _source_arrays()
    generator = np.random.default_rng(902)
    held = generator.normal(-0.45 + outer_shift, 0.02, size=(18, features.shape[1])).astype(np.float32)
    proxy = generator.normal(0.40 + outer_shift, 0.02, size=(18, features.shape[1])).astype(np.float32)
    all_features = np.concatenate([features.numpy(), held, proxy], axis=0)
    logits = np.full((all_features.shape[0], len(SOURCE)), -4.0, dtype=np.float32)
    for index, value in enumerate(tx):
        logits[index, SOURCE.index(str(value))] = 4.0
    logits[len(tx) :, 0] = 4.0
    tx_all = np.concatenate([tx, np.asarray(["9-1"] * 18), np.asarray(["9-2"] * 18)])
    rx_all = np.concatenate([rx, np.asarray(["rx-held"] * 18), np.asarray(["rx-proxy"] * 18)])
    day_all = np.concatenate([day, np.asarray(["day-outer"] * 36)])
    eq_all = np.concatenate([eq, np.asarray(["1"] * 36)])
    sig_all = np.concatenate([sig, np.asarray([f"h-{row}" for row in range(18)]), np.asarray([f"p-{row}" for row in range(18)])])
    roles = np.concatenate(
        [np.asarray(["source"] * len(tx)), np.asarray(["target_old"] * 18), np.asarray(["proxy_unknown"] * 18)]
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
        dataset_role=roles,
        channel_views=np.asarray(["clean"] * len(roles)),
        sat_scenarios=np.asarray([""] * len(roles)),
    )
    return features, tx, physical


def _args(tmp_path: Path, feature_npz: Path, gi_bundle: Path, prefix: str):
    return build_parser().parse_args(
        [
            "--feature-npz",
            str(feature_npz),
            "--gi-bundle",
            str(gi_bundle),
            "--source-tx-ids",
            ",".join(SOURCE),
            "--held-tx-ids",
            "9-1",
            "--view-name",
            "clean",
            "--output-readout-json",
            str(tmp_path / f"{prefix}.readout.json"),
            "--output-torchscript",
            str(tmp_path / f"{prefix}.ts"),
            "--output-metrics-json",
            str(tmp_path / f"{prefix}.metrics.json"),
            "--output-scores-csv",
            str(tmp_path / f"{prefix}.scores.csv"),
        ]
    )


def test_exact_finite_upper_quantile_rank():
    tau_50, k_50, n_50 = finite_upper_quantile(np.arange(1.0, 51.0))
    tau_100, k_100, n_100 = finite_upper_quantile(np.arange(1.0, 101.0))
    assert (tau_50, k_50, n_50) == (50.0, 50, 50)
    assert (tau_100, k_100, n_100) == (99.0, 99, 100)


def test_rce_split_is_physical_disjoint_and_closed():
    _, tx, rx, _, _, _, physical = _source_arrays()
    reference, calibration, evaluation, receipt = deterministic_reference_calibration_eval_split(tx, rx, physical, SOURCE)
    assert not np.any(reference & calibration)
    assert not np.any(reference & evaluation)
    assert not np.any(calibration & evaluation)
    assert np.all(reference | calibration | evaluation)
    assert set(physical[reference]).isdisjoint(set(physical[calibration]))
    assert set(physical[reference]).isdisjoint(set(physical[evaluation]))
    assert set(physical[calibration]).isdisjoint(set(physical[evaluation]))
    assert all(value >= 50 for value in receipt["calibration_rows_per_rx"].values())
    assert all(value == {"reference": 80, "calibration": 40, "evaluation": 40} for value in receipt["per_tx"].values())


def test_source_receiver_with_under_50_calibration_rows_fails_closed():
    _, tx, _, _, _, _, physical = _source_arrays()
    rx = np.asarray(["rx-main"] * len(tx), dtype=object)
    _, calibration, _, _ = deterministic_reference_calibration_eval_split(tx, rx, physical, SOURCE)
    rx_rare = rx.copy()
    rx_rare[np.flatnonzero(calibration)[0]] = "rx-rare"
    with pytest.raises(WRCNCTError, match="requires at least 50"):
        deterministic_reference_calibration_eval_split(tx, rx_rare, physical, SOURCE)


def test_fixed_alpha_and_no_tunable_cli_surface():
    assert WRC_NCT_ALPHA == 0.02
    with pytest.raises(WRCNCTError, match="frozen"):
        finite_upper_quantile(np.arange(1.0, 51.0), alpha=0.01)
    parser = build_parser()
    assert "--alpha" not in parser.format_help()
    assert "--quantile" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--alpha", "0.01"])


def test_runtime_preserves_upstream_gi_geometry_without_renormalizing():
    raw = torch.tensor(
        [[0.1234567, 0.7654321, 0.3333333], [0.4444444, 0.5555555, 0.6666666]], dtype=torch.float32
    )
    gi = GIEpiORRuntime(raw, torch.tensor([1.0e-4, 1.2e-4]), GIEpiORHead().eval()).eval()
    runtime = WRCNCTRuntime(gi.prototypes, gi.scales, 0.95, eps=gi.eps).eval()
    assert torch.equal(runtime.prototypes, gi.prototypes)
    probe = torch.tensor([[0.3, 0.4, 0.5], [0.7, 0.1, 0.2]], dtype=torch.float32)
    with torch.no_grad():
        _, gi_d_class, gi_ratio = gi(probe)
        d1, d2, ratio, _ = runtime(probe)
    ordered = torch.sort(gi_d_class, dim=1).values
    assert torch.equal(d1, ordered[:, 0])
    assert torch.equal(d2, ordered[:, 1])
    assert torch.equal(ratio, gi_ratio)


def test_runtime_parity_and_output_closure(tmp_path: Path):
    feature_npz = tmp_path / "features.npz"
    features, tx, physical = _write_npz(feature_npz)
    gi_bundle = tmp_path / "gi_bundle.npz"
    _write_gi_bundle(gi_bundle, features, tx, physical)
    args = _args(tmp_path, feature_npz, gi_bundle, "one")
    metrics = _run(args)
    readout = json.loads(Path(args.output_readout_json).read_text(encoding="utf-8"))
    saved_metrics = json.loads(Path(args.output_metrics_json).read_text(encoding="utf-8"))
    score_lines = Path(args.output_scores_csv).read_text(encoding="utf-8").splitlines()
    runtime = torch.jit.load(str(args.output_torchscript))
    output = runtime(torch.as_tensor(features[:9], dtype=torch.float32))
    assert len(output) == 4 and output[3].dtype == torch.bool
    assert readout["alpha"] == WRC_NCT_ALPHA
    assert readout["upstream_gi_bundle_sha256"]
    assert readout["parity"]["gi_ratio_max_abs"] <= 1.0e-5
    assert readout["parity"]["eager_torchscript_accept_equal"] is True
    assert metrics["known_evaluation_count"] == 200
    for group in ("class", "rx", "day"):
        closed = metrics[f"known_min_{group}_closed_accuracy_no_reject"]
        full = metrics[f"known_min_{group}_full_accuracy"]
        assert metrics[f"known_min_{group}_drop_pp"] == pytest.approx(100.0 * (closed - full))
    assert metrics["outer_used_for_fit_or_calibration"] is False
    assert saved_metrics["method"] == "WRC-NCT"
    assert len(score_lines) == 837


def test_non_tx_metadata_is_preserved_verbatim(tmp_path: Path):
    feature_npz = tmp_path / "features.npz"
    _write_npz(feature_npz)
    payload = _load_npz(feature_npz)
    assert "day-0" in set(payload["day_ids"].tolist())
    assert "proxy_unknown" in set(payload["dataset_role"].tolist())

    with np.load(feature_npz, allow_pickle=True) as original:
        rewritten = {name: np.asarray(original[name]) for name in original.files}
    rewritten["day_ids"] = np.asarray(
        ["2021_03_01" if value == "day-0" else value for value in rewritten["day_ids"]], dtype=object
    )
    np.savez_compressed(feature_npz, **rewritten)
    payload = _load_npz(feature_npz)
    assert "2021_03_01" in set(payload["day_ids"].tolist())
    assert "20210301" not in set(payload["day_ids"].tolist())


def test_outer_numeric_perturbation_does_not_change_readout_or_threshold(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    features, tx, physical = _write_npz(first, outer_shift=0.0)
    _write_npz(second, outer_shift=8.0)
    gi_bundle = tmp_path / "gi_bundle.npz"
    _write_gi_bundle(gi_bundle, features, tx, physical)
    args_a = _args(tmp_path, first, gi_bundle, "a")
    args_b = _args(tmp_path, second, gi_bundle, "b")
    _run(args_a)
    _run(args_b)
    readout_a = json.loads(Path(args_a.output_readout_json).read_text(encoding="utf-8"))
    readout_b = json.loads(Path(args_b.output_readout_json).read_text(encoding="utf-8"))
    assert readout_a["tau"] == readout_b["tau"]
    assert readout_a["tau_r"] == readout_b["tau_r"]
    assert readout_a["finite_quantile_rank"] == readout_b["finite_quantile_rank"]
    assert readout_a["split"] == readout_b["split"]
    runtime_a = torch.jit.load(str(args_a.output_torchscript))
    runtime_b = torch.jit.load(str(args_b.output_torchscript))
    probe = torch.as_tensor(features[:17], dtype=torch.float32)
    with torch.no_grad():
        left = runtime_a(probe)
        right = runtime_b(probe)
    for first_out, second_out in zip(left[:3], right[:3]):
        assert torch.allclose(first_out, second_out, atol=1.0e-7, rtol=0.0)
    assert torch.equal(left[3], right[3])
