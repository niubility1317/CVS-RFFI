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
import subprocess

import numpy as np
import pytest
import torch

import build_phase1_clic_source_leo_iq as BUILDER
import export_phase1_clic_leo_features as LEO_EXPORT
import export_spaceborne_features as FEATURE_EXPORT
from build_phase1_clic_source_leo_iq import assign_source_l_scenarios


FORMAL_SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
TX_IDS = tuple(f"tx-{index}" for index in range(4))
RX_IDS = tuple(f"rx-{index}" for index in range(7))


def test_tensor_to_numpy_float32_avoids_legacy_numpy_c_api_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N607 runs Torch 2.1 with NumPy 2.x; Tensor.numpy() is unsafe there."""

    source = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    def forbidden_numpy(_tensor: torch.Tensor):
        raise AssertionError("Tensor.numpy() must not be used by the source-LEO builder")

    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_numpy)
    observed = BUILDER._tensor_to_numpy_float32(source)

    assert observed.dtype == np.float32
    assert observed.shape == (3, 2, 4)
    assert observed.tolist() == source.tolist()
    assert observed.flags.c_contiguous


def test_numpy_float32_to_tensor_avoids_legacy_numpy_c_api_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(24, dtype=np.float32).reshape(3, 2, 4)

    def forbidden_from_numpy(_array: np.ndarray):
        raise AssertionError("torch.from_numpy() must not be used by the source-LEO builder")

    monkeypatch.setattr(torch, "from_numpy", forbidden_from_numpy)
    observed = BUILDER._numpy_float32_to_tensor(source, device=torch.device("cpu"))

    assert observed.dtype == torch.float32
    assert tuple(observed.shape) == (3, 2, 4)
    assert observed.tolist() == source.tolist()
    assert observed.is_contiguous()


def test_leo_export_numpy_float32_to_tensor_avoids_legacy_numpy_c_api_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(48, dtype=np.float64).reshape(2, 3, 8)[:, :, ::2]

    def forbidden_from_numpy(_array: np.ndarray):
        raise AssertionError("torch.from_numpy() must not be used by the source-LEO exporter")

    monkeypatch.setattr(torch, "from_numpy", forbidden_from_numpy)
    observed = LEO_EXPORT._numpy_float32_to_tensor(source)
    expected = np.asarray(source, dtype=np.float32)

    assert observed.dtype == torch.float32
    assert tuple(observed.shape) == expected.shape
    assert observed.tolist() == expected.tolist()
    assert observed.is_contiguous()
    source[...] = -999.0
    assert observed.tolist() == expected.tolist()


def test_feature_export_tensor_to_numpy_float32_avoids_legacy_numpy_c_api_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4).transpose(1, 2)

    def forbidden_numpy(_tensor: torch.Tensor):
        raise AssertionError("Tensor.numpy() must not be used by the source-LEO feature export")

    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_numpy)
    observed = FEATURE_EXPORT._tensor_to_numpy_float32(source)

    assert observed.dtype == np.float32
    assert observed.shape == (3, 4, 2)
    assert observed.tolist() == source.tolist()
    assert observed.flags.c_contiguous


def test_feature_export_tensor_to_numpy_float32_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty|shape|finite|conversion"):
        FEATURE_EXPORT._tensor_to_numpy_float32(torch.empty((2, 0), dtype=torch.float32))


@pytest.mark.parametrize("field", ("CACHE_ROOT", "TRAINING_ROOT", "RUN_ROOT", "LOG_ROOT"))
def test_source_leo_v4_launcher_rejects_root_override(field: str, tmp_path: Path) -> None:
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "launch_phase1_clic_source_leo_export12_20260812.sh"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f"{field}={(tmp_path / 'injected').as_posix()} bash {launcher.name} --dry-run",
        ],
        cwd=launcher.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "root" in completed.stderr.lower() and "drift" in completed.stderr.lower()


def test_source_leo_sealed_cache_asset_rejects_receipt_sha_drift(tmp_path: Path) -> None:
    project_root = tmp_path
    cache_root = project_root / "runs" / "phase1_clic_source_leo_20260812_v3"
    cache_dir = cache_root / "F1_SHARED"
    cache_dir.mkdir(parents=True)
    cache_npz = cache_dir / "source_l_received_iq.npz"
    cache_npz.write_bytes(b"immutable-cache")
    receipt_path = cache_dir / "source_l_received_iq.receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase1.clic_source_leo_received_iq.v1",
                "method": "P1_CLIC",
                "fold_index": 1,
                "source_only": True,
                "source_l_only": True,
                "same_received_iq_bytes_for_c_and_g": True,
                "formal_scenarios": list(FORMAL_SCENES),
                "source_tx_ids": list(TX_IDS),
                "source_row_count": 3920,
                "source_split_sha256": "1" * 64,
                "checkpoint_sha256_by_arm": {"C": "2" * 64, "G": "3" * 64},
                "terminal_receipt_sha256_by_arm": {"C": "4" * 64, "G": "5" * 64},
                "received_iq_npz_path": str(cache_npz),
                "received_iq_npz_sha256": "0" * 64,
                "target_access": False,
                "query_access": False,
                "held_validation_forward_rows": 0,
                "proxy_forward_rows": 0,
                "fit_rows": 0,
                "threshold_fit_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    output_dir = project_root / "runs" / "phase1_clic_source_leo_20260812_v4" / "F1C_CLIC12"
    checkpoint = project_root / "runs" / "phase1_clic12_20260812_v5" / "F1C_CLIC12" / "final_ssdg.pth"
    terminal = checkpoint.with_name("phase1_clic_terminal_receipt.json")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    terminal.write_bytes(b"terminal")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoint_sha256_by_arm"] = {
        "C": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "G": "3" * 64,
    }
    receipt["terminal_receipt_sha256_by_arm"] = {
        "C": hashlib.sha256(terminal.read_bytes()).hexdigest(),
        "G": "5" * 64,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    args = argparse.Namespace(
        require_sealed_source_leo_cache=True,
        cache_run_root=str(cache_root),
        existing_received_iq_receipt_json=str(receipt_path),
        postfreeze_output_root=str(project_root / "runs" / "phase1_clic_source_leo_20260812_v4"),
        training_run_root=str(project_root / "runs" / "phase1_clic12_20260812_v5"),
        out_npz=str(output_dir / "source_leo.npz"),
        binding_json=str(output_dir / "source_leo.binding.json"),
    )
    validator = getattr(LEO_EXPORT, "_validate_sealed_source_leo_cache_asset", None)
    assert callable(validator), "sealed source-LEO cache validator is absent"
    with pytest.raises(Exception, match="SHA|hash|drift|cache"):
        validator(
            args,
            received_path=cache_npz,
            source_tx_ids=TX_IDS,
            fold=1,
            arm="C",
            candidate="F1C_CLIC12",
            checkpoint=checkpoint,
            terminal=terminal,
            terminal_receipt={"source_split_sha256": "1" * 64},
        )


def test_feature_export_loop_runs_with_legacy_tensor_numpy_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyRows(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            return (
                torch.full((2, 8), float(index + 1), dtype=torch.float32),
                torch.tensor(index, dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                {"tx": f"tx-{index}", "rx": "rx-0", "day": "0", "equalized": "1", "sig_i": str(index)},
            )

    class TinyModel:
        def eval(self):
            return self

    def identity_forward(_model, rows: torch.Tensor, _feature_name: str):
        return rows.flatten(1), rows.mean(dim=-1)

    def forbidden_numpy(_tensor: torch.Tensor):
        raise AssertionError("Tensor.numpy() must not execute in the source-LEO feature loop")

    monkeypatch.setattr(FEATURE_EXPORT, "identity_only_feature_forward", identity_forward)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_numpy)
    payload = FEATURE_EXPORT.extract_features_with_metadata(
        TinyModel(),
        torch.utils.data.DataLoader(TinyRows(), batch_size=2, shuffle=False),
        device=torch.device("cpu"),
        feature_name="z_id",
        role="source_L_leo_calibration",
        channel_view="received_existing",
        satellite_tta_policy="none",
        safe_numpy_bridge=True,
    )

    assert payload["features"].shape == (2, 16)
    assert payload["tx_logits"].shape == (2, 2)
    assert payload["features"].dtype == np.float32
    assert np.isfinite(payload["features"]).all()


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
