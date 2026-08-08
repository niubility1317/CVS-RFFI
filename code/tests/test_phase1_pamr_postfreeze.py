from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = CODE_ROOT / "scripts" / "eval_phase1_pamr_pair.py"
LAUNCHER_PATH = CODE_ROOT / "scripts" / "launch_phase1_pamr_postfreeze_20260809.sh"
_SPEC = importlib.util.spec_from_file_location("pamr_postfreeze_pair", EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PAIR
_SPEC.loader.exec_module(PAIR)


TX = ("tx-a", "tx-b", "tx-c", "tx-d")
RXS = PAIR.EXPECTED_SOURCE_RXS
DAYS = PAIR.EXPECTED_SOURCE_DAYS
SCENARIOS = PAIR.EXPECTED_SCENARIOS


# Traceability record for the frozen, three-file postfreeze handoff.
FROZEN_POSTFREEZE_TEST_CONTRACT = {
    "binding": "NPZ SHA, strict export audit, final head, local4 order",
    "isolation": "outer proxy/target rows cannot affect source pair metrics",
    "physical": "C/G ordered metadata and clean/LEO physical/scenario closure",
    "leo_view": "source profile=satellite plus satellite_tta_policy=none emits row view=single",
    "matrix": "12 clean + 12 LEO + 12 proxy + 6 C/G pair diagnostics",
}


def _source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for tx_index, tx in enumerate(TX):
            for rx_index, rx in enumerate(RXS):
                rows.append(
                    {
                        "dataset_role": "source",
                        "tx_ids": tx,
                        "rx_ids": rx,
                        "day_ids": DAYS[(scenario_index + tx_index + rx_index) % len(DAYS)],
                        "eq_ids": "1",
                        "sig_ids": f"{scenario}-{tx_index}-{rx_index}",
                        "channel_views": "clean",
                        "sat_scenarios": "",
                    }
                )
    return rows


def _basis(tx: str, *, leo: bool, arm: str) -> np.ndarray:
    index = TX.index(tx)
    vector = np.zeros(len(TX), dtype=np.float32)
    vector[index] = 1.0
    if leo:
        vector[(index + 1) % len(TX)] = 0.12 if arm == "C" else 0.07
    return vector


def _logits(tx: str, *, arm: str) -> np.ndarray:
    values = np.full(len(TX), -2.0, dtype=np.float32)
    values[TX.index(tx)] = 4.0
    if arm == "G" and tx == "tx-a":
        values[0] = 3.8
    return values


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        return torch.load(path, map_location="cpu")


def _save_checkpoint(path: Path, *, arm: str, head: np.ndarray | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    head_array = np.eye(len(TX), dtype=np.float32) if head is None else np.asarray(head, dtype=np.float32)
    receipt: dict[str, object] = {
        "class_order_contract": "LOCAL_DATA_TX_ORDER_EQUALS_CHECKPOINT_TRAIN_TX_ORDER_EQUALS_LIVE_HEAD_ROW_ORDER",
        "dataset_tx_class_order": [*TX, "tx-e", "tx-f"],
        "local_tx_class_order": list(TX),
        "checkpoint_train_tx_class_order": list(TX),
        "local_to_dataset_class_ids": [0, 1, 2, 3],
        "local_to_head_class_ids": [0, 1, 2, 3],
        "expected_tx_class_ids": [0, 1, 2, 3],
        "dataset_class_count": 6,
        "local_data_class_count": 4,
        "checkpoint_head_class_count": 4,
        "live_head_class_count": 4,
        "class_count": 4,
    }
    receipt["class_order_binding_sha256"] = PAIR._expected_binding_sha256(receipt)
    checkpoint = {
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "args": {
            "phase1_pamr_frozen_mode": True,
            "phase1_pamr_enabled": arm == "G",
            "phase1_pamr_audit_only": False,
            "lambda_pamr": 0.05 if arm == "G" else 0.0,
            "phase1_source_train_tx_ids": ",".join(TX),
            "id_feature_key": "feat_joint",
            "num_classes": 4,
        },
        "model": {PAIR._PAMR_HEAD_KEY: torch.as_tensor(head_array)},
        "split_info": {
            "tx_partition_receipt": {
                "enabled": True,
                "source_known_train_tx": list(TX),
                "training_tx_count": 4,
                "training_view_contiguous_reindex": {str(index): tx for index, tx in enumerate(TX)},
                "held_tx_loaded_by_training": False,
            }
        },
        "pamr_receipt": receipt,
    }
    torch.save(checkpoint, path)
    return PAIR._sha256_file(path)


def _manifest(*, arm: str, leo: bool, checkpoint_sha256: str) -> dict[str, object]:
    return {
        "feature_name": "z_id",
        "class_id_to_tx": list(TX),
        "logit_class_order": list(range(len(TX))),
        "source_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": {
            "checkpoint_load_strict": True,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "skipped_mismatch": 0,
        },
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
        "source_only_export": bool(leo),
        "star_ground_channel_impl": "simplified_leo_residual" if leo else "legacy_satellite",
        "satellite_tta_policy": "none",
        "channel_profile": {
            "source": {
                "view": "satellite" if leo else "clean",
                "scenarios": list(SCENARIOS) if leo else [],
                "sat_seed": 7281718,
            }
        },
    }


def _payload(*, arm: str, leo: bool, checkpoint_sha256: str) -> dict[str, np.ndarray]:
    rows = _source_rows()
    if not leo:
        rows.extend(
            [
                {
                    "dataset_role": "target_old",
                    "tx_ids": "target-old",
                    "rx_ids": RXS[0],
                    "day_ids": DAYS[0],
                    "eq_ids": "1",
                    "sig_ids": "target-old-sig",
                    "channel_views": "clean",
                    "sat_scenarios": "",
                },
                {
                    "dataset_role": "proxy_unknown",
                    "tx_ids": "proxy-unknown",
                    "rx_ids": RXS[1],
                    "day_ids": DAYS[1],
                    "eq_ids": "1",
                    "sig_ids": "proxy-unknown-sig",
                    "channel_views": "clean",
                    "sat_scenarios": "",
                },
            ]
        )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for row in rows:
        if row["dataset_role"] == "source":
            features.append(_basis(row["tx_ids"], leo=leo, arm=arm))
            logits.append(_logits(row["tx_ids"], arm=arm))
            if leo:
                row["channel_views"] = "single"
                row["sat_scenarios"] = row["sig_ids"].rsplit("-", 2)[0]
        else:
            features.append(np.ones(len(TX), dtype=np.float32))
            logits.append(np.zeros(len(TX), dtype=np.float32))
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "manifest_json": np.asarray(json.dumps(_manifest(arm=arm, leo=leo, checkpoint_sha256=checkpoint_sha256))),
    }
    for field in PAIR.METADATA_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows])
    return payload


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)


def _rewrite(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    mutate(payload)
    np.savez(path, **payload)


def _set_manifest(path: Path, mutate) -> None:
    def apply(payload):
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        mutate(manifest)
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(path, apply)


def _write_pair(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "c_checkpoint": tmp_path / "c_final_ssdg.pth",
        "g_checkpoint": tmp_path / "g_final_ssdg.pth",
        "c_clean": tmp_path / "c_clean.npz",
        "g_clean": tmp_path / "g_clean.npz",
        "c_leo": tmp_path / "c_leo.npz",
        "g_leo": tmp_path / "g_leo.npz",
    }
    c_sha = _save_checkpoint(paths["c_checkpoint"], arm="C")
    g_sha = _save_checkpoint(paths["g_checkpoint"], arm="G")
    _save(paths["c_clean"], _payload(arm="C", leo=False, checkpoint_sha256=c_sha))
    _save(paths["g_clean"], _payload(arm="G", leo=False, checkpoint_sha256=g_sha))
    _save(paths["c_leo"], _payload(arm="C", leo=True, checkpoint_sha256=c_sha))
    _save(paths["g_leo"], _payload(arm="G", leo=True, checkpoint_sha256=g_sha))
    return paths


def _args(paths: dict[str, Path], out_json: Path):
    return PAIR.build_parser().parse_args(
        [
            "--c-clean-npz", str(paths["c_clean"]),
            "--g-clean-npz", str(paths["g_clean"]),
            "--c-leo-npz", str(paths["c_leo"]),
            "--g-leo-npz", str(paths["g_leo"]),
            "--c-final-checkpoint", str(paths["c_checkpoint"]),
            "--g-final-checkpoint", str(paths["g_checkpoint"]),
            "--candidate-pair", "F1_C_vs_G",
            "--source-tx-ids", ",".join(TX),
            "--expected-source-count", "72",
            "--expected-target-old-count", "1",
            "--expected-proxy-count", "1",
            "--output-metrics-json", str(out_json),
        ]
    )


def test_pair_evaluator_closes_final_head_binding_floors_and_margin(tmp_path):
    paths = _write_pair(tmp_path)
    with np.load(paths["c_leo"], allow_pickle=False) as leo_payload:
        assert set(np.asarray(leo_payload["channel_views"]).tolist()) == {"single"}
    output = tmp_path / "pair_metrics.json"
    metrics = PAIR.evaluate(_args(paths, output))

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["policy"]["fit_performed"] is False
    assert persisted["policy"]["calibration_performed"] is False
    assert persisted["policy"]["proxy_rows_used_for_head_or_margin"] == 0
    assert persisted["bindings"]["C"]["strict_head_extract"] is True
    assert persisted["bindings"]["C"]["head_state_key"] == "id_backbone.cls_head.head.weight"
    assert persisted["bindings"]["C"]["final_checkpoint_sha256"] == PAIR._sha256_file(paths["c_checkpoint"])
    for container in (metrics["clean_source"], *metrics["leo_scenarios"].values()):
        for arm in ("C", "G"):
            for metric in ("overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy"):
                assert 0.0 <= container[arm][metric] <= 1.0
        assert set(container["G_minus_C_pp"]) == {
            "overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy"
        }
        angular = container["raw_cosine_angular_margin_diagnostic"]
        assert angular["C"]["raw_cosine_correct_count"] > 0
        assert angular["C"]["correct_vs_hardest_other_angular_margin_mean_deg"] is not None
    assert set(metrics["leo_scenarios"]) == set(SCENARIOS)
    assert metrics["leo_scenarios"][SCENARIOS[0]]["paired_cosine_diagnostic_source_clean_only"]["C"][
        "paired_clean_leo_cosine_distance_mean"
    ] > 0.0
    with pytest.raises(PAIR.PAMRPostfreezePairError, match="refusing to overwrite"):
        PAIR.evaluate(_args(paths, output))


def test_pair_evaluator_fails_closed_on_npz_checkpoint_binding_drift(tmp_path):
    paths = _write_pair(tmp_path)
    for key in ("c_clean", "c_leo"):
        _set_manifest(paths[key], lambda manifest: manifest.__setitem__("source_checkpoint_sha256", "e" * 64))
    with pytest.raises(PAIR.PAMRPostfreezePairError, match="C final checkpoint SHA256 does not bind"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_head_rows_and_class_order(tmp_path):
    paths = _write_pair(tmp_path)
    checkpoint = _load_torch(paths["c_checkpoint"])
    checkpoint["model"][PAIR._PAMR_HEAD_KEY] = checkpoint["model"][PAIR._PAMR_HEAD_KEY][:-1]
    torch.save(checkpoint, paths["c_checkpoint"])
    changed_sha = PAIR._sha256_file(paths["c_checkpoint"])
    for key in ("c_clean", "c_leo"):
        _set_manifest(paths[key], lambda manifest: manifest.__setitem__("source_checkpoint_sha256", changed_sha))
    with pytest.raises(PAIR.PAMRPostfreezePairError, match="head shape mismatch"):
        PAIR.evaluate(_args(paths, tmp_path / "rows.json"))

    paths = _write_pair(tmp_path / "order")
    checkpoint = _load_torch(paths["g_checkpoint"])
    checkpoint["args"]["phase1_source_train_tx_ids"] = ",".join(reversed(TX))
    torch.save(checkpoint, paths["g_checkpoint"])
    changed_sha = PAIR._sha256_file(paths["g_checkpoint"])
    for key in ("g_clean", "g_leo"):
        _set_manifest(paths[key], lambda manifest: manifest.__setitem__("source_checkpoint_sha256", changed_sha))
    with pytest.raises(PAIR.PAMRPostfreezePairError, match="checkpoint train TX class order mismatch"):
        PAIR.evaluate(_args(paths, tmp_path / "order.json"))

    paths = _write_pair(tmp_path / "receipt")
    checkpoint = _load_torch(paths["g_checkpoint"])
    checkpoint["pamr_receipt"]["local_to_head_class_ids"] = [1, 0, 2, 3]
    torch.save(checkpoint, paths["g_checkpoint"])
    changed_sha = PAIR._sha256_file(paths["g_checkpoint"])
    for key in ("g_clean", "g_leo"):
        _set_manifest(paths[key], lambda manifest: manifest.__setitem__("source_checkpoint_sha256", changed_sha))
    with pytest.raises(PAIR.PAMRPostfreezePairError, match="PAMR local4 class-order binding mismatch"):
        PAIR.evaluate(_args(paths, tmp_path / "receipt.json"))


def test_outer_target_proxy_values_have_zero_pair_metric_influence(tmp_path):
    paths = _write_pair(tmp_path)
    baseline = PAIR.evaluate(_args(paths, tmp_path / "baseline.json"))

    def mutate_outer(payload):
        mask = payload["dataset_role"] != "source"
        payload["features"][mask] = np.asarray([9.0, 8.0, 7.0, 6.0], dtype=np.float32)
        payload["tx_logits"][mask] = np.asarray([-9.0, 3.0, 2.0, 1.0], dtype=np.float32)

    for key in ("c_clean", "g_clean"):
        _rewrite(paths[key], mutate_outer)
    changed = PAIR.evaluate(_args(paths, tmp_path / "changed.json"))
    assert changed["clean_source"] == baseline["clean_source"]
    assert changed["leo_scenarios"] == baseline["leo_scenarios"]
    assert changed["policy"]["proxy_rows_used_for_pair_metrics"] == 0
    assert changed["policy"]["target_old_rows_used_for_pair_metrics"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["sig_ids"].__setitem__(0, "changed-signal"))
                for key in ("c_leo", "g_leo")
            ],
            "clean/LEO source physical key sets differ",
        ),
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["dataset_role"].__setitem__(-1, "source"))
                for key in ("c_clean", "g_clean")
            ],
            "clean roles mismatch",
        ),
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["sat_scenarios"].__setitem__(0, "invalid_scenario"))
                for key in ("c_leo", "g_leo")
            ],
            "LEO scenario set mismatch",
        ),
        (
            lambda paths: _set_manifest(paths["c_clean"], lambda manifest: manifest.__setitem__("checkpoint_load_strict", False)),
            "checkpoint export was not strict-loaded",
        ),
        (
            lambda paths: [
                _set_manifest(paths[key], lambda manifest: manifest.__setitem__("class_id_to_tx", list(reversed(TX))))
                for key in ("g_clean", "g_leo")
            ],
            "class label/order mismatch",
        ),
        (
            lambda paths: [
                _rewrite(
                    paths[key],
                    lambda payload: payload["rx_ids"].__setitem__(
                        np.asarray(
                            [str(sig).startswith("leo_clear_weak-") and rx == RXS[0]
                            for sig, rx in zip(payload["sig_ids"], payload["rx_ids"])]
                        ),
                        RXS[1],
                    ),
                )
                for key in ("c_leo", "g_leo")
            ],
            "lacks full source RX coverage",
        ),
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["channel_views"].__setitem__(slice(None), "wrong_nonempty_view"))
                for key in ("c_leo", "g_leo")
            ],
            "must use exactly channel_view=single",
        ),
        (
            lambda paths: [
                _set_manifest(
                    paths[key],
                    lambda manifest: manifest["channel_profile"]["source"].__setitem__("view", "clean"),
                )
                for key in ("c_leo", "g_leo")
            ],
            "source channel profile is not satellite",
        ),
        (
            lambda paths: [
                _set_manifest(
                    paths[key],
                    lambda manifest: manifest.__setitem__("satellite_tta_policy", "repair_canonical1"),
                )
                for key in ("c_leo", "g_leo")
            ],
            "LEO manifest must use satellite_tta_policy=none",
        ),
    ],
)
def test_pair_evaluator_fails_closed_on_physical_role_scenario_or_strict_load(tmp_path, mutation, message):
    paths = _write_pair(tmp_path)
    mutation(paths)
    with pytest.raises(PAIR.PAMRPostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_postfreeze_launcher_dry_run_has_frozen_42_steps_and_new_roots():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    for required in (
        "phase1_pamr_postfreeze_20260809_v1",
        "phase1_pamr12_20260809_v1",
        "eval_phase1_pamr_pair.py",
        "--c-final-checkpoint",
        "--g-final-checkpoint",
        "source_sat_seed",
        "7281718",
        "simplified_leo_residual",
        "--unknown_far_target 0.05",
        "--known_query_roles source",
        "--unknown_query_roles proxy_unknown",
        "--calibration_roles source",
        "--max_samples_per_tx \"${MAX_PER_TX}\"",
        "--batch_size \"${EXPORT_BATCH}\"",
        "--source_only_export",
        "--source_channel_view satellite",
        "--source_channel_view clean",
        "--target_old_channel_view clean",
        "--proxy_unknown_channel_view clean",
        "--satellite_tta_policy none",
        "final_ssdg.pth",
        "[[ \"${POSTFREEZE_ROOT}\" != \"${TRAIN_RUN_ROOT}\" ]]",
    ):
        assert required in text
    assert re.findall(r"^launch_candidate (\d) ([CG]) (\d)$", text, flags=re.MULTILINE) == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_pamr_postfreeze_20260809.sh", "--dry-run"],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=True,
    )
    lines = completed.stdout.splitlines()
    assert sum(line.startswith("[DRY-RUN][CLEAN_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][LEO_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PROXY_SCORE]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PAIR_SCORE]") for line in lines) == 6
    assert all("phase1_pamr_postfreeze_20260809_v1" in line for line in lines)
