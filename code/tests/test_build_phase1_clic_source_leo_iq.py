from __future__ import annotations

"""RED contracts for the source-L single-observation LEO allocator.

The allocator is intentionally imported directly.  Until the production
module/API exists, collection must fail with the missing module/API rather
than silently using a test double.
"""

from collections.abc import Mapping
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import build_phase1_clic_source_leo_iq as BUILDER
from build_phase1_clic_source_leo_iq import assign_source_l_scenarios


FORMAL_SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
TX_IDS = tuple(f"tx-{index}" for index in range(4))
RX_IDS = tuple(f"rx-{index}" for index in range(7))


def _source_l_rows(*, rows_per_cell: int = 140) -> tuple[list[str], list[str], list[str]]:
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    physical_ids: list[str] = []
    for tx_id in TX_IDS:
        for rx_id in RX_IDS:
            for repeat in range(rows_per_cell):
                tx_ids.append(tx_id)
                rx_ids.append(rx_id)
                physical_ids.append(f"physical-{tx_id}-{rx_id}-{repeat:03d}")
    return tx_ids, rx_ids, physical_ids


def _assignment_map(result: object, physical_ids: list[str]) -> dict[str, str]:
    assert isinstance(result, Mapping)
    assignment = {str(key): str(value) for key, value in result.items()}
    assert set(assignment) == set(physical_ids)
    return assignment


def test_assign_source_l_scenarios_covers_exact_cells_and_three_formal_scenes() -> None:
    tx_ids, rx_ids, physical_ids = _source_l_rows()

    assignment = _assignment_map(
        assign_source_l_scenarios(tx_ids, rx_ids, physical_ids),
        physical_ids,
    )

    assert set(assignment.values()) == set(FORMAL_SCENES)
    for tx_id in TX_IDS:
        for rx_id in RX_IDS:
            cell = [
                physical_id
                for physical_id in physical_ids
                if physical_id.startswith(f"physical-{tx_id}-{rx_id}-")
            ]
            assert len(cell) == 140
            assert {
                scene: sum(assignment[physical_id] == scene for physical_id in cell)
                for scene in FORMAL_SCENES
            } == {
                "leo_clear_weak": 47,
                "leo_low_elev_weak": 47,
                "leo_rain_weak": 46,
            }


def test_assign_source_l_scenarios_is_invariant_to_input_row_permutation() -> None:
    tx_ids, rx_ids, physical_ids = _source_l_rows()
    baseline = _assignment_map(
        assign_source_l_scenarios(tx_ids, rx_ids, physical_ids),
        physical_ids,
    )
    order = list(range(len(physical_ids)))[::-1]
    shuffled = _assignment_map(
        assign_source_l_scenarios(
            [tx_ids[index] for index in order],
            [rx_ids[index] for index in order],
            [physical_ids[index] for index in order],
        ),
        physical_ids,
    )
    assert shuffled == baseline


@pytest.mark.parametrize(
    "mutation",
    ("missing_cell", "short_cell", "extra_cell", "duplicate", "length_drift", "empty_id"),
)
def test_assign_source_l_scenarios_rejects_malformed_source_l_inputs(mutation: str) -> None:
    tx_ids, rx_ids, physical_ids = _source_l_rows()
    if mutation == "missing_cell":
        remove = next(
            index
            for index, (tx_id, rx_id) in enumerate(zip(tx_ids, rx_ids, strict=True))
            if tx_id == TX_IDS[-1] and rx_id == RX_IDS[-1]
        )
        tx_ids = tx_ids[:remove] + tx_ids[remove + 140 :]
        rx_ids = rx_ids[:remove] + rx_ids[remove + 140 :]
        physical_ids = physical_ids[:remove] + physical_ids[remove + 140 :]
    elif mutation == "short_cell":
        remove = next(
            index
            for index, (tx_id, rx_id) in enumerate(zip(tx_ids, rx_ids, strict=True))
            if tx_id == TX_IDS[0] and rx_id == RX_IDS[0]
        )
        del tx_ids[remove]
        del rx_ids[remove]
        del physical_ids[remove]
    elif mutation == "extra_cell":
        tx_ids.append(TX_IDS[0])
        rx_ids.append(RX_IDS[0])
        physical_ids.append(f"physical-{TX_IDS[0]}-{RX_IDS[0]}-140")
    elif mutation == "duplicate":
        physical_ids[-1] = physical_ids[0]
    elif mutation == "length_drift":
        rx_ids.pop()
    else:
        physical_ids[0] = ""

    with pytest.raises(Exception, match="cell|60|140|exact|duplicate|unique|length|empty|physical|TX|RX|scene"):
        assign_source_l_scenarios(tx_ids, rx_ids, physical_ids)


def _builder_rows() -> dict[str, object]:
    """Build a data-free, deterministic 4x7x140 source-L row table.

    The integration contract patches checkpoint/WiSig reconstruction, so these
    rows are deliberately synthetic and must not be interpreted as real WiSig
    evidence.
    """

    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    physical_ids: list[str] = []
    clean_iq = np.empty((len(TX_IDS) * len(RX_IDS) * 140, 2, 8), dtype=np.float32)
    row = 0
    for tx_id in TX_IDS:
        for rx_id in RX_IDS:
            for repeat in range(140):
                tx_ids.append(tx_id)
                rx_ids.append(rx_id)
                day_ids.append(f"day-{repeat % 4}")
                physical_ids.append(f"physical-{tx_id}-{rx_id}-{repeat:03d}")
                clean_iq[row, :, :] = float(row)
                row += 1
    return {
        "clean_iq": clean_iq,
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "physical_sample_ids": physical_ids,
    }


def _builder_args(tmp_path: Path, dataset_path: Path) -> argparse.Namespace:
    c_ckpt = tmp_path / "F1C_CLIC12" / "final_ssdg.pth"
    g_ckpt = tmp_path / "F1G_CLIC12" / "final_ssdg.pth"
    c_terminal = tmp_path / "c_terminal.json"
    g_terminal = tmp_path / "g_terminal.json"
    c_ckpt.parent.mkdir(parents=True)
    g_ckpt.parent.mkdir(parents=True)
    c_ckpt.write_bytes(b"synthetic-C-checkpoint")
    g_ckpt.write_bytes(b"synthetic-G-checkpoint")
    c_terminal.write_text("{}\n", encoding="utf-8")
    g_terminal.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "source_l_received_iq.npz"
    receipt = tmp_path / "source_l_received_iq.receipt.json"
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return argparse.Namespace(
        fold_index=1,
        c_ckpt=str(c_ckpt),
        c_terminal_receipt_json=str(c_terminal),
        g_ckpt=str(g_ckpt),
        g_terminal_receipt_json=str(g_terminal),
        wisig_pkl=str(dataset_path),
        expected_wisig_sha256=dataset_sha,
        source_tx_ids=",".join(TX_IDS),
        known_validation_tx_ids="known-tx-a",
        proxy_unknown_tx_ids="proxy-unknown-tx",
        out_npz=str(output),
        receipt_json=str(receipt),
        batch_size=256,
        device="cpu",
    )


def _install_builder_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    args: argparse.Namespace,
    rows: dict[str, object],
    channel_mode: str = "finite",
    config_drift: bool = False,
) -> dict[str, list[object]]:
    """Patch only external checkpoint/data/channel boundaries for the RED test."""

    import cvsrffi.eval as eval_module
    import dataset_wisig
    import export_phase1_clic_features as clean_export

    dataset_path = Path(str(args.wisig_pkl)).resolve()
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    monkeypatch.setattr(BUILDER, "FROZEN_WISIG_SHA256", dataset_sha)
    monkeypatch.setattr(dataset_wisig, "load_wisig_compact_pkl", lambda _path: {})
    monkeypatch.setattr(dataset_wisig, "WiSigSubsetDataset", lambda *_a, **_k: object())
    monkeypatch.setattr(
        clean_export,
        "_reconstruct_source_l_v",
        lambda **_kwargs: {
            "source_base": object(),
            "labeled_indices": tuple(range(len(rows["physical_sample_ids"]))),
            "unlabeled_indices": (),
            "validation_indices": (),
            "source_split_receipt": {},
            "tx_partition_receipt": {},
        },
    )
    monkeypatch.setattr(clean_export, "_assert_current_source_split", lambda **_kwargs: None)
    monkeypatch.setattr(
        BUILDER,
        "_collect_source_l_rows",
        lambda *_a, **_k: rows,
    )

    base_config = {
        "seed": 17,
        "split_mode": "tx_rx_day_1_6_3",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_val_ratio": 0.30,
        "wisig_pkl": str(dataset_path),
        "wisig_equalized": "1",
        "wisig_out_len": 8,
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
    calls: dict[str, list[object]] = {"channel": [], "rows": []}

    def fake_validated_arm(*, checkpoint_path: Path, **_kwargs):
        arm = "C" if checkpoint_path.parent.name.startswith("F1C") else "G"
        config = dict(base_config)
        if config_drift and arm == "G":
            config["wisig_train_days"] = "0,1"
        return (
            {"args": config},
            config,
            {
                "source_split_sha256": "synthetic-source-split-sha",
            },
            arm,
        )

    monkeypatch.setattr(BUILDER, "_load_validated_arm", fake_validated_arm)

    def fake_channel(source, scenario, _channel_args, *, gen, return_meta):
        assert return_meta is True
        calls["channel"].append((str(scenario), source.detach().cpu().numpy().copy()))
        if channel_mode == "nonfinite":
            result = torch.full_like(source, float("nan"))
        else:
            offset = float((FORMAL_SCENES.index(str(scenario)) + 1) * 10)
            result = source + offset
        return result, {"channel_model": "leo_residual"}

    monkeypatch.setattr(eval_module, "apply_sat_channel_for_scenario", fake_channel)
    return calls


def test_source_l_builder_seals_one_common_iq_npz_and_reopens_with_exact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"synthetic-data-free-wisig")
    args = _builder_args(tmp_path, dataset_path)
    rows = _builder_rows()
    calls = _install_builder_doubles(monkeypatch, args=args, rows=rows)

    result = BUILDER.build_source_l_received_iq(args)
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
    assert arrays["received_iq"].shape == (3920, 2, 8)
    assert np.isfinite(arrays["received_iq"]).all()
    assert len(calls["channel"]) >= 3
    assert sum(int(batch.shape[0]) for _scene, batch in calls["channel"]) == 3920
    assert sorted(
        int(value)
        for _scene, batch in calls["channel"]
        for value in np.asarray(batch)[:, 0, 0]
    ) == list(range(3920))

    expected_assignment = assign_source_l_scenarios(
        rows["tx_ids"], rows["rx_ids"], rows["physical_sample_ids"]
    )
    expected_scenes = [expected_assignment[physical_id] for physical_id in rows["physical_sample_ids"]]
    assert arrays["sat_scenarios"].astype(str).tolist() == expected_scenes

    import export_phase1_clic_leo_features as leo_export

    reopened, physical_keys, coverage = leo_export._load_existing_received_iq(
        output, source_tx_ids=TX_IDS
    )
    assert reopened["received_iq"].shape == (3920, 2, 8)
    assert len(physical_keys) == 3920 and len(set(physical_keys)) == 3920
    assert all(int(coverage[scene]["count"]) >= 20 for scene in FORMAL_SCENES)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["same_received_iq_bytes_for_c_and_g"] is True
    assert receipt["single_leo_observation_per_physical_sample"] is True
    assert receipt["source_row_count"] == 3920
    assert receipt["minimum_cell_count"] >= 20
    assert receipt["held_validation_forward_rows"] == 0
    assert receipt["proxy_forward_rows"] == 0
    assert receipt["target_access"] is False
    assert receipt["query_access"] is False
    assert receipt["fit_rows"] == 0
    assert receipt["threshold_fit_rows"] == 0
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_source_l_builder_rejects_c_g_data_config_drift_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"synthetic-data-free-wisig")
    args = _builder_args(tmp_path, dataset_path)
    _install_builder_doubles(monkeypatch, args=args, rows=_builder_rows(), config_drift=True)

    with pytest.raises(Exception, match="data|channel|drift"):
        BUILDER.build_source_l_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_source_l_builder_rejects_nonfinite_channel_and_leaves_no_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"synthetic-data-free-wisig")
    args = _builder_args(tmp_path, dataset_path)
    _install_builder_doubles(
        monkeypatch, args=args, rows=_builder_rows(), channel_mode="nonfinite"
    )

    with pytest.raises(Exception, match="non-finite|malformed"):
        BUILDER.build_source_l_received_iq(args)
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_source_l_builder_rejects_dataset_sha_toctou_and_does_not_seal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"synthetic-data-free-wisig")
    args = _builder_args(tmp_path, dataset_path)
    _install_builder_doubles(monkeypatch, args=args, rows=_builder_rows())
    actual_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    real_hash = BUILDER._sha256_file
    dataset_calls = 0

    def drift_hash(path: str | Path) -> str:
        nonlocal dataset_calls
        resolved = Path(path).resolve()
        if resolved == dataset_path.resolve():
            dataset_calls += 1
            if dataset_calls >= 3:
                return "d" * 64
            return actual_sha
        return real_hash(path)

    monkeypatch.setattr(BUILDER, "_sha256_file", drift_hash)
    with pytest.raises(Exception, match="changed|drift|bytes"):
        BUILDER.build_source_l_received_iq(args)
    assert dataset_calls >= 3
    assert not Path(args.out_npz).exists()
    assert not Path(args.receipt_json).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_source_l_builder_refuses_to_overwrite_immutable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"synthetic-data-free-wisig")
    args = _builder_args(tmp_path, dataset_path)
    _install_builder_doubles(monkeypatch, args=args, rows=_builder_rows())
    BUILDER.build_source_l_received_iq(args)
    before_output = Path(args.out_npz).read_bytes()
    before_receipt = Path(args.receipt_json).read_bytes()

    with pytest.raises(Exception, match="overwrite"):
        BUILDER.build_source_l_received_iq(args)
    assert Path(args.out_npz).read_bytes() == before_output
    assert Path(args.receipt_json).read_bytes() == before_receipt
