from __future__ import annotations

"""Behavior contracts for the source-V one-observation LEO cache.

The production module is deliberately imported directly.  Before it exists,
collection must fail at that boundary rather than silently exercising a test
double.  The real cache builder is then driven with small external-boundary
doubles while its validation, assignment, hashing and sealing behavior stays
real.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import build_phase1_clic_source_v_leo_iq as BUILDER
from build_phase1_clic_source_v_leo_iq import assign_source_v_scenarios


FORMAL_SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
TX_IDS = tuple(f"tx-{index}" for index in range(4))
RX_IDS = tuple(f"rx-{index}" for index in range(7))
SOURCE_V_DAY_IDS = ("2021_03_01", "2021_03_08")
NONFROZEN_SOURCE_V_DAY_IDS = ("2021_03_15", "2021_03_23")
ROWS_PER_TX_RX_DAY = 300
SOURCE_V_COUNT = 16_800


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _physical_key(
    tx_id: str,
    rx_id: str,
    day_id: str,
    eq_id: str,
    sig_id: str,
) -> str:
    return "\x1f".join((tx_id, rx_id, day_id, eq_id, sig_id))


def _source_v_rows(
    *,
    day_ids_for_rows: tuple[str, ...] = SOURCE_V_DAY_IDS,
    rows_per_tx_rx_day: int = ROWS_PER_TX_RX_DAY,
) -> dict[str, Any]:
    """Hand-build the sealed 4x7x2x300 held-V physical table."""

    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    eq_ids: list[str] = []
    sig_ids: list[str] = []
    physical_ids: list[str] = []
    clean_iq = np.empty((SOURCE_V_COUNT, 2, 4), dtype=np.float32)
    row = 0
    for tx_id in TX_IDS:
        for rx_id in RX_IDS:
            for day_id in day_ids_for_rows:
                for repeat in range(rows_per_tx_rx_day):
                    tx_ids.append(tx_id)
                    rx_ids.append(rx_id)
                    day_ids.append(day_id)
                    eq_ids.append("eq-1")
                    sig_ids.append(f"v-{repeat:03d}")
                    physical_ids.append(
                        f"opaque-v-{tx_id}-{rx_id}-{day_id}-{repeat:03d}"
                    )
                    clean_iq[row, :, :] = float(row)
                    row += 1
    assert row == SOURCE_V_COUNT
    return {
        "clean_iq": clean_iq,
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "eq_ids": eq_ids,
        "sig_ids": sig_ids,
        "physical_sample_ids": physical_ids,
    }


def _copy_source_v_rows(rows: dict[str, Any]) -> dict[str, Any]:
    """Keep clean-v4 drift fixtures independent from the collected V rows."""

    return {name: value.copy() for name, value in rows.items()}


def _unbalanced_source_v_rows() -> dict[str, Any]:
    """Keep 600 rows in one cell while moving 299 rows across its frozen days."""

    rows = _copy_source_v_rows(_source_v_rows())
    moved = 0
    for index, (tx_id, rx_id, day_id) in enumerate(
        zip(rows["tx_ids"], rows["rx_ids"], rows["day_ids"], strict=True)
    ):
        if (
            tx_id == TX_IDS[0]
            and rx_id == RX_IDS[0]
            and day_id == SOURCE_V_DAY_IDS[1]
            and moved < ROWS_PER_TX_RX_DAY - 1
        ):
            rows["day_ids"][index] = SOURCE_V_DAY_IDS[0]
            rows["sig_ids"][index] = f"unbalanced-day-axis-{index:05d}"
            moved += 1
    assert moved == ROWS_PER_TX_RX_DAY - 1
    return rows


def test_assign_source_v_scenarios_is_permutation_invariant_and_covers_every_axis() -> None:
    """Break caught: changing assignment from sorted opaque IDs to input order."""

    rows = _source_v_rows()
    baseline = assign_source_v_scenarios(
        rows["tx_ids"],
        rows["rx_ids"],
        rows["day_ids"],
        rows["physical_sample_ids"],
    )
    order = list(range(SOURCE_V_COUNT))[::-1]
    shuffled = assign_source_v_scenarios(
        [rows["tx_ids"][index] for index in order],
        [rows["rx_ids"][index] for index in order],
        [rows["day_ids"][index] for index in order],
        [rows["physical_sample_ids"][index] for index in order],
    )

    assert baseline == shuffled
    assert set(baseline) == set(rows["physical_sample_ids"])
    assert set(baseline.values()) == set(FORMAL_SCENES)
    for scene in FORMAL_SCENES:
        for tx_id in TX_IDS:
            assert sum(
                baseline[physical_id] == scene
                for physical_id, observed_tx in zip(
                    rows["physical_sample_ids"], rows["tx_ids"], strict=True
                )
                if observed_tx == tx_id
            ) > 0
        for rx_id in RX_IDS:
            assert sum(
                baseline[physical_id] == scene
                for physical_id, observed_rx in zip(
                    rows["physical_sample_ids"], rows["rx_ids"], strict=True
                )
                if observed_rx == rx_id
            ) > 0
        for day_id in SOURCE_V_DAY_IDS:
            assert sum(
                baseline[physical_id] == scene
                for physical_id, observed_day in zip(
                    rows["physical_sample_ids"], rows["day_ids"], strict=True
                )
                if observed_day == day_id
            ) > 0


@pytest.mark.parametrize("mutation", ("duplicate", "empty", "short", "unknown_day"))
def test_assign_source_v_scenarios_rejects_invalid_physical_or_axis_input(
    mutation: str,
) -> None:
    """Break caught: accepting ambiguous, incomplete or noncanonical V membership."""

    rows = _source_v_rows()
    tx_ids = list(rows["tx_ids"])
    rx_ids = list(rows["rx_ids"])
    day_ids = list(rows["day_ids"])
    physical_ids = list(rows["physical_sample_ids"])
    if mutation == "duplicate":
        physical_ids[-1] = physical_ids[0]
    elif mutation == "empty":
        physical_ids[0] = ""
    elif mutation == "short":
        tx_ids.pop()
        rx_ids.pop()
        day_ids.pop()
        physical_ids.pop()
    else:
        day_ids[0] = ""

    with pytest.raises(
        Exception, match="unique|duplicate|empty|length|16|physical|day|axis|row"
    ):
        assign_source_v_scenarios(tx_ids, rx_ids, day_ids, physical_ids)


def test_assign_source_v_scenarios_rejects_legacy_three_day_axis_even_at_16800_rows() -> None:
    """Break caught: restoring the old three-day count admits a wrong clean-v4 V axis."""

    legacy_rows = _source_v_rows(
        day_ids_for_rows=(*SOURCE_V_DAY_IDS, "2021_03_15"),
        rows_per_tx_rx_day=200,
    )

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError, match="frozen day axis drifted"
    ):
        assign_source_v_scenarios(
            legacy_rows["tx_ids"],
            legacy_rows["rx_ids"],
            legacy_rows["day_ids"],
            legacy_rows["physical_sample_ids"],
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_assign_source_v_scenarios_rejects_missing_or_extra_source_v_day_axis(
    mutation: str,
) -> None:
    """Break caught: a nonsealed held-V day axis is accepted despite 16800 rows."""

    rows = _source_v_rows()
    day_ids = list(rows["day_ids"])
    if mutation == "missing":
        day_ids = [SOURCE_V_DAY_IDS[0] for _ in day_ids]
    else:
        day_ids[0] = "2021_03_15"

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError, match="frozen day axis drifted"
    ):
        assign_source_v_scenarios(
            rows["tx_ids"], rows["rx_ids"], day_ids, rows["physical_sample_ids"]
        )


def test_assign_source_v_scenarios_rejects_self_consistent_nonfrozen_two_day_axis() -> None:
    """Break caught: any two substitute dates cannot self-certify the held-V axis."""

    rows = _source_v_rows(day_ids_for_rows=NONFROZEN_SOURCE_V_DAY_IDS)

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError, match="frozen day axis drifted"
    ):
        assign_source_v_scenarios(
            rows["tx_ids"],
            rows["rx_ids"],
            rows["day_ids"],
            rows["physical_sample_ids"],
        )


def test_assign_source_v_scenarios_rejects_599_1_day_split_inside_600_row_cell() -> None:
    """Break caught: a 600-row TX/RX total cannot hide 599/1 physical-day drift."""

    rows = _unbalanced_source_v_rows()
    cell_days = [
        day_id
        for tx_id, rx_id, day_id in zip(
            rows["tx_ids"], rows["rx_ids"], rows["day_ids"], strict=True
        )
        if tx_id == TX_IDS[0] and rx_id == RX_IDS[0]
    ]
    assert cell_days.count(SOURCE_V_DAY_IDS[0]) == 599
    assert cell_days.count(SOURCE_V_DAY_IDS[1]) == 1

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError, match="TX/RX/day coverage drifted"
    ):
        assign_source_v_scenarios(
            rows["tx_ids"],
            rows["rx_ids"],
            rows["day_ids"],
            rows["physical_sample_ids"],
        )


def _source_l_rows() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    eq_ids: list[str] = []
    sig_ids: list[str] = []
    for tx_id in TX_IDS:
        for rx_id in RX_IDS:
            for repeat in range(140):
                tx_ids.append(tx_id)
                rx_ids.append(rx_id)
                day_ids.append(SOURCE_V_DAY_IDS[repeat % len(SOURCE_V_DAY_IDS)])
                eq_ids.append("eq-1")
                sig_ids.append(f"l-{repeat:03d}")
    assert len(tx_ids) == 3920
    return tx_ids, rx_ids, day_ids, eq_ids, sig_ids


def _proxy_rows() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    eq_ids: list[str] = []
    sig_ids: list[str] = []
    for repeat in range(400):
        tx_ids.append("proxy-tx")
        rx_ids.append(RX_IDS[repeat % len(RX_IDS)])
        day_ids.append(SOURCE_V_DAY_IDS[repeat % len(SOURCE_V_DAY_IDS)])
        eq_ids.append("eq-1")
        sig_ids.append(f"p-{repeat:03d}")
    return tx_ids, rx_ids, day_ids, eq_ids, sig_ids


def _write_clean_npz(
    path: Path,
    *,
    candidate_id: str,
    checkpoint_sha: str,
    terminal_sha: str,
    rows: dict[str, Any],
    manifest_drift: bool = False,
    role_overlap: bool = False,
    manifest_day_ids: tuple[str, ...] | None = None,
) -> None:
    """Write a complete clean-v4-shaped metadata archive without features."""

    l_tx, l_rx, l_day, l_eq, l_sig = _source_l_rows()
    p_tx, p_rx, p_day, p_eq, p_sig = _proxy_rows()
    v_keys = [
        _physical_key(tx, rx, day, eq, sig)
        for tx, rx, day, eq, sig in zip(
            rows["tx_ids"],
            rows["rx_ids"],
            rows["day_ids"],
            rows["eq_ids"],
            rows["sig_ids"],
            strict=True,
        )
    ]
    if role_overlap:
        l_tx[0], l_rx[0], l_day[0], l_eq[0], l_sig[0] = (
            rows["tx_ids"][0],
            rows["rx_ids"][0],
            rows["day_ids"][0],
            rows["eq_ids"][0],
            rows["sig_ids"][0],
        )
    tx_ids = l_tx + list(rows["tx_ids"]) + p_tx
    rx_ids = l_rx + list(rows["rx_ids"]) + p_rx
    day_ids = l_day + list(rows["day_ids"]) + p_day
    eq_ids = l_eq + list(rows["eq_ids"]) + p_eq
    sig_ids = l_sig + list(rows["sig_ids"]) + p_sig
    roles = (
        ["labeled_fit"] * len(l_tx)
        + ["source_validation_known"] * SOURCE_V_COUNT
        + ["proxy_unknown"] * len(p_tx)
    )
    assert len(roles) == 21_120
    manifest = {
        "schema": "cvs.phase1.clic_lv_export.v1",
        "method": "P1_CLIC",
        "source_only": True,
        "candidate_id": candidate_id,
        "run_id": "phase1_clic12_20260812_v5",
        "source_tx_ids": list(TX_IDS),
        "source_receiver_ids": list(RX_IDS),
        "source_day_ids": list(
            SOURCE_V_DAY_IDS if manifest_day_ids is None else manifest_day_ids
        ),
        "source_validation_indices_sha256": _canonical_sha(
            list(range(3920, 3920 + SOURCE_V_COUNT))
        ),
        "source_validation_physical_order_sha256": _canonical_sha(v_keys),
        "labeled_validation_physical_disjoint": True,
        "labeled_validation_proxy_physical_disjoint": True,
        "labeled_row_count": len(l_tx),
        "source_validation_row_count": SOURCE_V_COUNT,
        "proxy_row_count": len(p_tx),
        "source_checkpoint_sha256": checkpoint_sha,
        "terminal_receipt_sha256": terminal_sha,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
    }
    if manifest_drift:
        manifest["source_validation_physical_order_sha256"] = "0" * 64
    np.savez(
        path,
        dataset_role=np.asarray(roles, dtype=str),
        tx_ids=np.asarray(tx_ids, dtype=str),
        rx_ids=np.asarray(rx_ids, dtype=str),
        day_ids=np.asarray(day_ids, dtype=str),
        eq_ids=np.asarray(eq_ids, dtype=str),
        sig_ids=np.asarray(sig_ids, dtype=str),
        manifest_json=np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True)),
    )


def _builder_args(
    tmp_path: Path,
    *,
    manifest_drift: bool = False,
    role_overlap: bool = False,
    c_clean_rows: dict[str, Any] | None = None,
    g_clean_rows: dict[str, Any] | None = None,
    c_manifest_day_ids: tuple[str, ...] | None = None,
    g_manifest_day_ids: tuple[str, ...] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    rows = _source_v_rows()
    c_rows = rows if c_clean_rows is None else c_clean_rows
    g_rows = rows if g_clean_rows is None else g_clean_rows
    cache_root = tmp_path / "runs" / "phase1_clic_source_metrics_20260816_v3"
    shared_dir = cache_root / "F1_SHARED"
    c_dir = tmp_path / "runs" / "phase1_clic12_20260812_v5" / "F1C_CLIC12"
    g_dir = tmp_path / "runs" / "phase1_clic12_20260812_v5" / "F1G_CLIC12"
    c_dir.mkdir(parents=True)
    g_dir.mkdir(parents=True)
    c_checkpoint = c_dir / "final_ssdg.pth"
    g_checkpoint = g_dir / "final_ssdg.pth"
    c_terminal = c_dir / "phase1_clic_terminal_receipt.json"
    g_terminal = g_dir / "phase1_clic_terminal_receipt.json"
    c_checkpoint.write_bytes(b"synthetic-C-checkpoint")
    g_checkpoint.write_bytes(b"synthetic-G-checkpoint")
    c_terminal.write_text("{}\n", encoding="utf-8")
    g_terminal.write_text("{}\n", encoding="utf-8")
    c_clean_dir = tmp_path / "runs" / "phase1_clic_postfreeze_20260812_v4" / "F1C_CLIC12"
    g_clean_dir = tmp_path / "runs" / "phase1_clic_postfreeze_20260812_v4" / "F1G_CLIC12"
    c_clean_dir.mkdir(parents=True)
    g_clean_dir.mkdir(parents=True)
    c_clean = c_clean_dir / "source_clean_proxy.npz"
    g_clean = g_clean_dir / "source_clean_proxy.npz"
    _write_clean_npz(
        c_clean,
        candidate_id="F1C_CLIC12",
        checkpoint_sha=hashlib.sha256(c_checkpoint.read_bytes()).hexdigest(),
        terminal_sha=hashlib.sha256(c_terminal.read_bytes()).hexdigest(),
        rows=c_rows,
        role_overlap=role_overlap,
        manifest_day_ids=c_manifest_day_ids,
    )
    _write_clean_npz(
        g_clean,
        candidate_id="F1G_CLIC12",
        checkpoint_sha=hashlib.sha256(g_checkpoint.read_bytes()).hexdigest(),
        terminal_sha=hashlib.sha256(g_terminal.read_bytes()).hexdigest(),
        rows=g_rows,
        manifest_drift=manifest_drift,
        manifest_day_ids=g_manifest_day_ids,
    )
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"synthetic-data-free-wisig")
    args = argparse.Namespace(
        fold_index=1,
        c_ckpt=str(c_checkpoint),
        c_terminal_receipt_json=str(c_terminal),
        c_clean_npz=str(c_clean),
        g_ckpt=str(g_checkpoint),
        g_terminal_receipt_json=str(g_terminal),
        g_clean_npz=str(g_clean),
        wisig_pkl=str(dataset_path),
        expected_wisig_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        source_tx_ids=",".join(TX_IDS),
        known_validation_tx_ids="known-tx-a",
        proxy_unknown_tx_ids="proxy-unknown-tx",
        cache_run_root=str(cache_root),
        out_npz=str(shared_dir / "source_validation_known_leo_weak.npz"),
        receipt_json=str(shared_dir / "source_validation_known_leo_weak.receipt.json"),
        batch_size=257,
        device="cpu",
    )
    return args, rows


def _install_external_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: argparse.Namespace,
    rows: dict[str, Any],
    channel_mode: str = "finite",
) -> dict[str, list[Any]]:
    """Replace only checkpoint/dataset/channel boundaries; retain builder logic."""

    import cvsrffi.eval as eval_module
    import dataset_wisig
    import export_phase1_clic_features as clean_export

    dataset_path = Path(args.wisig_pkl).resolve()
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    monkeypatch.setattr(BUILDER, "FROZEN_WISIG_SHA256", dataset_sha)
    monkeypatch.setattr(dataset_wisig, "load_wisig_compact_pkl", lambda _path: {})
    monkeypatch.setattr(dataset_wisig, "WiSigSubsetDataset", lambda *_a, **_k: object())
    monkeypatch.setattr(
        clean_export,
        "_reconstruct_source_l_v",
        lambda **_kwargs: {
            "source_base": object(),
            "labeled_indices": tuple(range(3920)),
            "unlabeled_indices": (),
            "validation_indices": tuple(range(3920, 3920 + SOURCE_V_COUNT)),
            "source_split_receipt": {},
            "tx_partition_receipt": {},
        },
    )
    monkeypatch.setattr(clean_export, "_assert_current_source_split", lambda **_kwargs: None)
    monkeypatch.setattr(BUILDER, "_collect_source_v_rows", lambda *_a, **_k: rows)
    base_config = {
        "seed": 17,
        "split_mode": "tx_rx_day_1_6_3",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_val_ratio": 0.30,
        "wisig_pkl": str(dataset_path),
        "wisig_equalized": "1",
        "wisig_out_len": 4,
        "wisig_domain": "rx_day",
        "wisig_train_days": "0,1,2",
        "wisig_test_days": "3",
        "wisig_train_rxs": "0,1,2,3,4,5,6",
        "wisig_test_rxs": "",
        "wisig_max_day123_per_combo": 0,
        "phase1_source_train_tx_ids": ",".join(TX_IDS),
        "phase1_source_known_validation_tx_ids": "known-tx-a",
        "phase1_source_proxy_unknown_tx_ids": "proxy-unknown-tx",
        "sat_fs_hz": 25e6,
        "sat_fc_hz": 2.462e9,
    }
    calls: dict[str, list[Any]] = {"channel": []}

    def fake_validated_arm(*, checkpoint_path: Path, **_kwargs):
        arm = "C" if checkpoint_path.parent.name.startswith("F1C") else "G"
        return (
            {"args": dict(base_config)},
            dict(base_config),
            {
                "source_split_sha256": "synthetic-source-split-sha",
                "source_split_count": 3920,
                "class_order_count": 4,
                "physical_order_count": 3920,
            },
            arm,
        )

    monkeypatch.setattr(BUILDER, "_load_validated_arm", fake_validated_arm)

    def fake_channel(source, scenario, _channel_args, *, gen, return_meta):
        assert return_meta is True
        calls["channel"].append((str(scenario), source.detach().cpu().tolist()))
        if channel_mode == "nonfinite":
            result = torch.full_like(source, float("nan"))
        else:
            result = source + float(FORMAL_SCENES.index(str(scenario)) + 1)
        return result, {"channel_model": "leo_residual"}

    monkeypatch.setattr(eval_module, "apply_sat_channel_for_scenario", fake_channel)
    return calls


def test_builder_seals_exact_v_cache_shared_by_c_and_g_without_legacy_bridges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: fitting V, creating per-arm bytes, or changing V metadata/order."""

    args, rows = _builder_args(tmp_path)
    calls = _install_external_doubles(monkeypatch, args=args, rows=rows)

    def forbidden_numpy(_tensor: torch.Tensor):
        raise AssertionError("Tensor.numpy() must not be used by source-V cache builder")

    def forbidden_from_numpy(_array: np.ndarray):
        raise AssertionError("torch.from_numpy() must not be used by source-V cache builder")

    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_numpy)
    monkeypatch.setattr(torch, "from_numpy", forbidden_from_numpy)
    result = BUILDER.build_source_v_received_iq(args)
    output = Path(result["out_npz"])
    receipt_path = Path(result["receipt_json"])

    assert output.is_file() and receipt_path.is_file()
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "received_iq",
            "tx_ids",
            "rx_ids",
            "day_ids",
            "physical_sample_id",
            "sat_scenarios",
        }
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    assert arrays["received_iq"].shape == (SOURCE_V_COUNT, 2, 4)
    assert np.isfinite(arrays["received_iq"]).all()
    assert len(set(arrays["physical_sample_id"].astype(str).tolist())) == SOURCE_V_COUNT
    assert len(calls["channel"]) >= 3
    assert sum(len(batch) for _scene, batch in calls["channel"]) == SOURCE_V_COUNT

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "cvs.phase1.clic_source_v_leo_received_iq.v1"
    assert receipt["role"] == "source_validation_known_leo_weak"
    assert receipt["source_v_only"] is True
    assert receipt["source_validation_row_count"] == SOURCE_V_COUNT
    assert receipt["source_day_ids"] == list(SOURCE_V_DAY_IDS)
    assert receipt["source_validation_tx_rx_day_coverage"] == {
        f"{tx_id}|{rx_id}|{day_id}": ROWS_PER_TX_RX_DAY
        for tx_id in TX_IDS
        for rx_id in RX_IDS
        for day_id in SOURCE_V_DAY_IDS
    }
    assert receipt["same_received_iq_bytes_for_c_and_g"] is True
    assert receipt["single_leo_observation_per_physical_sample"] is True
    assert receipt["cross_scene_physical_sample_reuse"] is False
    assert receipt["fit_rows"] == 0
    assert receipt["threshold_fit_rows"] == 0
    assert receipt["proxy_forward_rows"] == 0
    assert receipt["target_access"] is False
    assert receipt["query_access"] is False
    assert receipt["post_target_completion_audit_non_selection"] is True
    assert receipt["checkpoint_sha256_by_arm"]["C"] != receipt["checkpoint_sha256_by_arm"]["G"]
    assert (
        receipt["clean_validation_metadata_order_sha256_by_arm"]["C"]
        == receipt["clean_validation_metadata_order_sha256_by_arm"]["G"]
    )
    assert receipt["scene_seeds"] == {
        "leo_clear_weak": 1008,
        "leo_low_elev_weak": 1_001_011,
        "leo_rain_weak": 2_001_014,
    }
    for coverage in (
        receipt["scenario_coverage"],
        receipt["scenario_class_coverage"],
        receipt["scenario_rx_coverage"],
        receipt["scenario_day_coverage"],
    ):
        assert coverage and min(int(value) for value in coverage.values()) > 0


@pytest.mark.parametrize("failure", ("manifest_drift", "role_overlap", "nonfinite_channel"))
def test_builder_rejects_clean_or_channel_contract_drift_without_partial_output(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: sealing V when clean split evidence or received IQ is invalid."""

    args, rows = _builder_args(
        tmp_path,
        manifest_drift=failure == "manifest_drift",
        role_overlap=failure == "role_overlap",
    )
    _install_external_doubles(
        monkeypatch,
        args=args,
        rows=rows,
        channel_mode="nonfinite" if failure == "nonfinite_channel" else "finite",
    )

    with pytest.raises(Exception, match="manifest|order|overlap|finite|malformed|physical|drift"):
        BUILDER.build_source_v_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()
    assert not list(Path(args.cache_run_root).rglob("*.tmp"))


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_builder_rejects_clean_v4_missing_or_extra_day_axis_before_output(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: clean-v4 V metadata has an incomplete or surplus physical day axis."""

    c_clean_rows = _copy_source_v_rows(_source_v_rows())
    if mutation == "missing":
        for index, day_id in enumerate(c_clean_rows["day_ids"]):
            if day_id == SOURCE_V_DAY_IDS[1]:
                c_clean_rows["day_ids"][index] = SOURCE_V_DAY_IDS[0]
                c_clean_rows["sig_ids"][index] = f"missing-day-axis-{index:05d}"
    else:
        c_clean_rows["day_ids"][0] = "2021_03_15"
    args, rows = _builder_args(tmp_path, c_clean_rows=c_clean_rows)
    _install_external_doubles(monkeypatch, args=args, rows=rows)

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError,
        match="C clean-v4 V frozen day axis drifted",
    ):
        BUILDER.build_source_v_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()


def test_builder_rejects_clean_v4_manifest_day_axis_drift_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a manifest can no longer self-declare a different held-V day axis."""

    args, rows = _builder_args(
        tmp_path,
        g_manifest_day_ids=(*SOURCE_V_DAY_IDS, "2021_03_15"),
    )
    _install_external_doubles(monkeypatch, args=args, rows=rows)

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError,
        match="G clean-v4 manifest source day axis drifted",
    ):
        BUILDER.build_source_v_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()


@pytest.mark.parametrize(
    ("mutation", "manifest_day_ids", "match"),
    (
        (
            "nonfrozen_axis",
            NONFROZEN_SOURCE_V_DAY_IDS,
            "C clean-v4 V frozen day axis drifted",
        ),
        (
            "unbalanced_cell_day",
            None,
            "C clean-v4 V TX/RX/day coverage drifted",
        ),
    ),
)
def test_builder_rejects_clean_v4_frozen_day_or_cell_coverage_drift_before_output(
    mutation: str,
    manifest_day_ids: tuple[str, ...] | None,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: clean-v4 must use the physical dates and 300 rows per cell-day."""

    if mutation == "nonfrozen_axis":
        c_clean_rows = _source_v_rows(day_ids_for_rows=NONFROZEN_SOURCE_V_DAY_IDS)
    else:
        c_clean_rows = _unbalanced_source_v_rows()
    args, rows = _builder_args(
        tmp_path,
        c_clean_rows=c_clean_rows,
        c_manifest_day_ids=manifest_day_ids,
    )
    _install_external_doubles(monkeypatch, args=args, rows=rows)

    with pytest.raises(BUILDER.CLICSourceVLeoCacheError, match=match):
        BUILDER.build_source_v_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()


def test_builder_rejects_c_g_clean_v4_physical_binding_drift_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: C/G clean-v4 can no longer bind different held-V physical rows."""

    g_clean_rows = _copy_source_v_rows(_source_v_rows())
    g_clean_rows["sig_ids"][0] = "g-physical-binding-drift"
    args, rows = _builder_args(tmp_path, g_clean_rows=g_clean_rows)
    _install_external_doubles(monkeypatch, args=args, rows=rows)

    with pytest.raises(
        BUILDER.CLICSourceVLeoCacheError,
        match="source-V cache C/G clean-v4 validation_keys does not share exact V binding",
    ):
        BUILDER.build_source_v_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()


def test_builder_rejects_input_toctou_before_sealing_and_leaves_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: not rehashing an input after V materialization."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    dataset_path = Path(args.wisig_pkl).resolve()
    actual_dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    real_hash = BUILDER._sha256_file
    dataset_calls = 0

    def drifting_hash(path: str | Path) -> str:
        nonlocal dataset_calls
        if Path(path).resolve() == dataset_path:
            dataset_calls += 1
            if dataset_calls >= 3:
                return "d" * 64
            return actual_dataset_sha
        return real_hash(path)

    monkeypatch.setattr(BUILDER, "_sha256_file", drifting_hash)
    with pytest.raises(Exception, match="changed|drift|bytes|TOCTOU"):
        BUILDER.build_source_v_received_iq(args)
    assert dataset_calls >= 3
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()


def test_builder_refuses_immutable_or_noncanonical_output_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: writing a second cache or moving a fold cache outside F1_SHARED."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    BUILDER.build_source_v_received_iq(args)
    before_npz = Path(args.out_npz).read_bytes()
    before_receipt = Path(args.receipt_json).read_bytes()
    with pytest.raises(Exception, match="overwrite|immutable"):
        BUILDER.build_source_v_received_iq(args)
    assert Path(args.out_npz).read_bytes() == before_npz
    assert Path(args.receipt_json).read_bytes() == before_receipt

    bad_args, bad_rows = _builder_args(tmp_path / "noncanonical")
    bad_args.out_npz = str(Path(bad_args.cache_run_root) / "wrong.npz")
    bad_args.receipt_json = str(Path(bad_args.cache_run_root) / "wrong.json")
    _install_external_doubles(monkeypatch, args=bad_args, rows=bad_rows)
    with pytest.raises(Exception, match="canonical|binding|shared|output"):
        BUILDER.build_source_v_received_iq(bad_args)
    assert not Path(bad_args.out_npz).exists()
    assert not Path(bad_args.receipt_json).exists()


def test_builder_fails_closed_when_npz_destination_appears_during_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a concurrent immutable NPZ must never be replaced.

    The collision is injected after the builder has written its temporary
    payload but before its exclusive destination publish.  ``os.link`` has
    the required no-replace semantics on both NTFS and POSIX filesystems.
    """

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    output = Path(args.out_npz)
    receipt = Path(args.receipt_json)
    sentinel = b"concurrent-npz-sentinel\n"
    native_link = BUILDER.os.link

    def racing_link(source: str | Path, destination: str | Path, *args_: object, **kwargs: object) -> None:
        target = Path(destination)
        if target == output:
            target.write_bytes(sentinel)
        native_link(source, destination, *args_, **kwargs)

    monkeypatch.setattr(BUILDER.os, "link", racing_link)
    with pytest.raises(Exception, match="overwrite|immutable|publish|exists|concurrent"):
        BUILDER.build_source_v_received_iq(args)

    assert output.read_bytes() == sentinel
    assert not receipt.exists()
    assert not list(Path(args.cache_run_root).rglob("*.tmp"))


def test_builder_cleans_its_npz_when_receipt_destination_races_without_deleting_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a receipt collision must retain its owner and remove our half artifact."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    output = Path(args.out_npz)
    receipt = Path(args.receipt_json)
    sentinel = b"concurrent-receipt-sentinel\n"
    native_link = BUILDER.os.link

    def racing_link(source: str | Path, destination: str | Path, *args_: object, **kwargs: object) -> None:
        target = Path(destination)
        if target == receipt:
            target.write_bytes(sentinel)
        native_link(source, destination, *args_, **kwargs)

    monkeypatch.setattr(BUILDER.os, "link", racing_link)
    with pytest.raises(Exception, match="overwrite|immutable|publish|exists|concurrent"):
        BUILDER.build_source_v_received_iq(args)

    assert not output.exists()
    assert receipt.read_bytes() == sentinel
    assert not list(Path(args.cache_run_root).rglob("*.tmp"))


def test_builder_rejects_post_publish_valid_npz_replacement_without_deleting_external_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a valid-looking replacement cannot pass a post-publish TOCTOU window."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    output = Path(args.out_npz)
    receipt = Path(args.receipt_json)
    native_assert = BUILDER._assert_publication_current
    replacement_done = False
    replacement_bytes = b""

    def replace_before_identity_check(
        publication: Any, *, expected_sha256: str, label: str
    ) -> None:
        nonlocal replacement_done, replacement_bytes
        if not replacement_done and Path(publication.path) == output:
            replacement = output.with_name("external-valid-replacement.npz")
            replacement_bytes = output.read_bytes()
            replacement.write_bytes(replacement_bytes)
            os.replace(replacement, output)
            replacement_done = True
        native_assert(publication, expected_sha256=expected_sha256, label=label)

    monkeypatch.setattr(BUILDER, "_assert_publication_current", replace_before_identity_check)
    with pytest.raises(Exception, match="publication|changed|identity|immutable|TOCTOU"):
        BUILDER.build_source_v_received_iq(args)

    assert replacement_done
    assert output.read_bytes() == replacement_bytes
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "received_iq",
            "tx_ids",
            "rx_ids",
            "day_ids",
            "physical_sample_id",
            "sat_scenarios",
        }
    assert not receipt.exists()
    assert not list(Path(args.cache_run_root).rglob("*.tmp"))


def test_builder_preserves_racing_npz_temporary_file_when_exclusive_create_loses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: losing a temporary-name race must not unlink the other owner."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    temporary = Path(args.out_npz).with_name(Path(args.out_npz).name + ".tmp")
    sentinel = b"foreign-npz-temporary\n"
    native_open = Path.open
    injected = False

    def racing_open(self: Path, mode: str = "r", *args_: object, **kwargs: object):
        nonlocal injected
        if self == temporary and mode == "xb" and not injected:
            with native_open(self, "xb") as handle:
                handle.write(sentinel)
            injected = True
        return native_open(self, mode, *args_, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(Exception, match="temporary|exists|File exists"):
        BUILDER.build_source_v_received_iq(args)

    assert injected
    assert temporary.read_bytes() == sentinel
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()


def test_builder_preserves_racing_receipt_temporary_file_when_exclusive_create_loses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: receipt temporary cleanup is owned-only on Windows and POSIX."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    output = Path(args.out_npz)
    temporary = Path(args.receipt_json).with_name(Path(args.receipt_json).name + ".tmp")
    sentinel = b"foreign-receipt-temporary\n"
    native_open = Path.open
    injected = False

    def racing_open(self: Path, mode: str = "r", *args_: object, **kwargs: object):
        nonlocal injected
        if self == temporary and mode == "x" and not injected:
            with native_open(self, "xb") as handle:
                handle.write(sentinel)
            injected = True
        return native_open(self, mode, *args_, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(Exception, match="temporary|exists|File exists"):
        BUILDER.build_source_v_received_iq(args)

    assert injected
    assert temporary.read_bytes() == sentinel
    assert not output.exists()
    assert not Path(args.receipt_json).exists()


def test_builder_rejects_same_inode_valid_npz_mutation_after_publish_before_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a valid NPZ changed in place cannot become the SHA baseline."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    output = Path(args.out_npz)
    native_publish = BUILDER._atomic_save_npz
    mutation_done = False
    external_bytes = b""

    def publish_then_mutate(path: Path, payload: dict[str, Any]):
        nonlocal mutation_done, external_bytes
        publication = native_publish(path, payload)
        original_identity = output.stat().st_ino
        replacement_payload = dict(payload)
        replacement_iq = np.asarray(payload["received_iq"], dtype=np.float32).copy()
        replacement_iq[0, 0, 0] += np.float32(0.25)
        replacement_payload["received_iq"] = replacement_iq
        with output.open("r+b") as handle:
            handle.seek(0)
            handle.truncate()
            np.savez(handle, **replacement_payload)
            handle.flush()
        external_bytes = output.read_bytes()
        assert output.stat().st_ino == original_identity
        mutation_done = True
        return publication

    monkeypatch.setattr(BUILDER, "_atomic_save_npz", publish_then_mutate)
    with pytest.raises(Exception, match="publication|bytes|changed|immutable|TOCTOU"):
        BUILDER.build_source_v_received_iq(args)

    assert mutation_done
    assert output.read_bytes() == external_bytes
    with np.load(output, allow_pickle=False) as archive:
        assert np.isfinite(np.asarray(archive["received_iq"], dtype=np.float32)).all()
        assert set(archive.files) == {
            "received_iq",
            "tx_ids",
            "rx_ids",
            "day_ids",
            "physical_sample_id",
            "sat_scenarios",
        }
    assert not Path(args.receipt_json).exists()


def test_builder_rejects_same_inode_receipt_mutation_before_return_without_deleting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: receipt JSON cannot be changed in place after its immutable publish."""

    args, rows = _builder_args(tmp_path)
    _install_external_doubles(monkeypatch, args=args, rows=rows)
    output = Path(args.out_npz)
    receipt = Path(args.receipt_json)
    native_publish = BUILDER._atomic_write_json
    mutation_done = False
    external_bytes = b""

    def publish_then_mutate(path: Path, payload: dict[str, Any]):
        nonlocal mutation_done, external_bytes
        publication = native_publish(path, payload)
        original_identity = receipt.stat().st_ino
        changed = dict(payload)
        changed["target_access"] = True
        with receipt.open("r+", encoding="utf-8") as handle:
            handle.seek(0)
            handle.truncate()
            json.dump(changed, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
        external_bytes = receipt.read_bytes()
        assert receipt.stat().st_ino == original_identity
        mutation_done = True
        return publication

    monkeypatch.setattr(BUILDER, "_atomic_write_json", publish_then_mutate)
    with pytest.raises(Exception, match="publication|bytes|changed|immutable|TOCTOU"):
        BUILDER.build_source_v_received_iq(args)

    assert mutation_done
    assert json.loads(receipt.read_text(encoding="utf-8"))["target_access"] is True
    assert receipt.read_bytes() == external_bytes
    assert not output.exists()


def test_npz_write_failure_after_our_exclusive_temporary_create_cleans_only_our_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: fail-closed publication must not leave its own partial lockout file."""

    output = tmp_path / "source_validation_known_leo_weak.npz"
    temporary = output.with_name(output.name + ".tmp")

    def partial_then_fail(handle: Any, **_payload: Any) -> None:
        handle.write(b"incomplete-own-npz")
        handle.flush()
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(BUILDER.np, "savez", partial_then_fail)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        BUILDER._atomic_save_npz(output, {"received_iq": np.ones((1, 2, 2), dtype=np.float32)})

    assert not output.exists()
    assert not temporary.exists()
