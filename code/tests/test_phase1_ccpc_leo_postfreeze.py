from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = CODE_ROOT / "scripts" / "eval_phase1_ccpc_leo_pair.py"
LAUNCHER_PATH = CODE_ROOT / "scripts" / "launch_phase1_ccpc_leo_postfreeze_20260809.sh"
_SPEC = importlib.util.spec_from_file_location("ccpc_postfreeze_pair", EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PAIR
_SPEC.loader.exec_module(PAIR)


TX = ("tx-a", "tx-b", "tx-c", "tx-d")
RXS = PAIR.EXPECTED_SOURCE_RXS
DAYS = PAIR.EXPECTED_SOURCE_DAYS
SCENARIOS = PAIR.EXPECTED_SCENARIOS


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
        vector[(index + 1) % len(TX)] = 0.10 if arm == "C" else 0.16
    return vector


def _logits(tx: str, *, arm: str) -> np.ndarray:
    values = np.full(len(TX), -2.0, dtype=np.float32)
    index = TX.index(tx)
    if arm == "G" and tx == "tx-a":
        index = 1
    values[index] = 4.0
    return values


def _manifest(*, arm: str, leo: bool) -> dict[str, object]:
    checkpoint_sha256 = ("c" if arm == "C" else "d") * 64
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


def _payload(*, arm: str, leo: bool) -> dict[str, np.ndarray]:
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
                row["channel_views"] = "satellite"
        else:
            features.append(np.ones(len(TX), dtype=np.float32))
            logits.append(np.zeros(len(TX), dtype=np.float32))
    if leo:
        for row in rows:
            row["channel_views"] = "satellite"
            row["sat_scenarios"] = row["sig_ids"].rsplit("-", 2)[0]
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "manifest_json": np.asarray(json.dumps(_manifest(arm=arm, leo=leo))),
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


def _write_pair(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "c_clean": tmp_path / "c_clean.npz",
        "g_clean": tmp_path / "g_clean.npz",
        "c_leo": tmp_path / "c_leo.npz",
        "g_leo": tmp_path / "g_leo.npz",
    }
    _save(paths["c_clean"], _payload(arm="C", leo=False))
    _save(paths["g_clean"], _payload(arm="G", leo=False))
    _save(paths["c_leo"], _payload(arm="C", leo=True))
    _save(paths["g_leo"], _payload(arm="G", leo=True))
    return paths


def _args(paths: dict[str, Path], out_json: Path):
    return PAIR.build_parser().parse_args(
        [
            "--c-clean-npz",
            str(paths["c_clean"]),
            "--g-clean-npz",
            str(paths["g_clean"]),
            "--c-leo-npz",
            str(paths["c_leo"]),
            "--g-leo-npz",
            str(paths["g_leo"]),
            "--candidate-pair",
            "F1_C_vs_G",
            "--source-tx-ids",
            ",".join(TX),
            "--expected-source-count",
            "72",
            "--expected-target-old-count",
            "1",
            "--expected-proxy-count",
            "1",
            "--output-metrics-json",
            str(out_json),
        ]
    )


def test_pair_evaluator_reports_four_floors_geometry_and_closed_json(tmp_path):
    paths = _write_pair(tmp_path)
    output = tmp_path / "pair_metrics.json"
    metrics = PAIR.evaluate(_args(paths, output))

    assert output.exists()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["policy"]["fit_performed"] is False
    assert persisted["policy"]["proxy_or_target_rows_used_for_geometry_bank"] == 0
    assert persisted["clean_source"]["C"]["count"] == 72
    assert persisted["bindings"]["c_source_checkpoint_sha256"] == "c" * 64
    assert persisted["bindings"]["g_source_checkpoint_sha256"] == "d" * 64
    for container in (metrics["clean_source"], *metrics["leo_scenarios"].values()):
        for arm in ("C", "G"):
            summary = container[arm]
            for metric in ("overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy"):
                assert metric in summary
                assert 0.0 <= summary[metric] <= 1.0
        assert set(container["G_minus_C_pp"]) == {
            "overall_accuracy",
            "min_class_accuracy",
            "min_rx_accuracy",
            "min_day_accuracy",
        }
    assert set(metrics["leo_scenarios"]) == set(SCENARIOS)
    assert metrics["leo_scenarios"][SCENARIOS[0]]["geometry_diagnostic_source_clean_only"]["C"][
        "paired_clean_leo_cosine_distance_mean"
    ] > 0.0
    with pytest.raises(PAIR.PostfreezePairError, match="refusing to overwrite"):
        PAIR.evaluate(_args(paths, output))


def test_pair_evaluator_fails_closed_on_cg_metadata_mismatch(tmp_path):
    paths = _write_pair(tmp_path)
    _rewrite(paths["g_leo"], lambda payload: payload["rx_ids"].__setitem__(0, "different-rx"))
    with pytest.raises(PAIR.PostfreezePairError, match="C/G LEO metadata/scenario mismatch"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_clean_leo_physical_mismatch(tmp_path):
    paths = _write_pair(tmp_path)
    for key in ("c_leo", "g_leo"):
        _rewrite(paths[key], lambda payload: payload["sig_ids"].__setitem__(0, "changed-signal"))
    with pytest.raises(PAIR.PostfreezePairError, match="clean/LEO source physical key sets differ"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_scenario_mismatch(tmp_path):
    for_key = _write_pair(tmp_path)
    for key in ("c_leo", "g_leo"):
        _rewrite(for_key[key], lambda payload: payload["sat_scenarios"].__setitem__(0, "invalid_scenario"))
    with pytest.raises(PAIR.PostfreezePairError, match="LEO scenario set mismatch"):
        PAIR.evaluate(_args(for_key, tmp_path / "out.json"))


@pytest.mark.parametrize(("field", "replacement", "message"), [("rx_ids", "1-19", "RX coverage"), ("tx_ids", "tx-b", "TX coverage")])
def test_pair_evaluator_fails_closed_on_missing_scenario_tx_or_rx(tmp_path, field, replacement, message):
    paths = _write_pair(tmp_path)
    for key in ("c_clean", "g_clean", "c_leo", "g_leo"):
        def remove_one_scenario_axis(payload):
            scenario_rows = np.asarray(
                [str(sig).startswith("leo_clear_weak-") for sig in payload["sig_ids"]],
                dtype=bool,
            )
            if field == "rx_ids":
                affected = scenario_rows & (payload["rx_ids"] == RXS[0])
            else:
                affected = scenario_rows & (payload["tx_ids"] == TX[0])
            payload[field][affected] = replacement
        _rewrite(paths[key], remove_one_scenario_axis)
    with pytest.raises(PAIR.PostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_duplicate_physical_key(tmp_path):
    paths = _write_pair(tmp_path)
    for key in ("c_leo", "g_leo"):
        def make_duplicate(payload):
            for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"):
                payload[field][1] = payload[field][0]
        _rewrite(paths[key], make_duplicate)
    with pytest.raises(PAIR.PostfreezePairError, match="duplicate physical keys"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_class_order_mismatch(tmp_path):
    paths = _write_pair(tmp_path)

    def reverse_class_order(payload):
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        manifest["class_id_to_tx"] = list(reversed(manifest["class_id_to_tx"]))
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(paths["g_clean"], reverse_class_order)
    with pytest.raises(PAIR.PostfreezePairError, match="class label/order mismatch"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_clean_role_count_mismatch(tmp_path):
    paths = _write_pair(tmp_path)
    for key in ("c_clean", "g_clean"):
        _rewrite(paths[key], lambda payload: payload["dataset_role"].__setitem__(-1, "source"))
    with pytest.raises(PAIR.PostfreezePairError, match="clean roles mismatch"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_non_strict_checkpoint_load_audit(tmp_path):
    paths = _write_pair(tmp_path)

    def make_non_strict(payload):
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        manifest["checkpoint_load_strict"] = False
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(paths["c_clean"], make_non_strict)
    with pytest.raises(PAIR.PostfreezePairError, match="checkpoint export was not strict-loaded"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_same_arm_checkpoint_binding_mismatch(tmp_path):
    paths = _write_pair(tmp_path)

    def change_checkpoint_hash(payload):
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        manifest["source_checkpoint_sha256"] = "e" * 64
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(paths["c_leo"], change_checkpoint_hash)
    with pytest.raises(PAIR.PostfreezePairError, match="C clean/LEO source checkpoint SHA256 differs"):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_postfreeze_launcher_dry_run_has_the_frozen_42_step_matrix():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    for required in (
        "phase1_ccpc_leo_postfreeze_20260809",
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
        "--source_days \"${SOURCE_DAYS}\"",
        "--source_rxs \"${SOURCE_RXS}\"",
        "final_ssdg.pth",
    ):
        assert required in text
    assert re.findall(r"^launch_candidate (\d) ([CG]) (\d)$", text, flags=re.MULTILINE) == [
        ("1", "C", "0"),
        ("5", "G", "0"),
        ("1", "G", "1"),
        ("5", "C", "1"),
        ("2", "C", "2"),
        ("6", "G", "2"),
        ("2", "G", "3"),
        ("6", "C", "3"),
        ("3", "C", "4"),
        ("3", "G", "5"),
        ("4", "C", "6"),
        ("4", "G", "7"),
    ]
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_ccpc_leo_postfreeze_20260809.sh", "--dry-run"],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    lines = completed.stdout.splitlines()
    assert sum(line.startswith("[DRY-RUN][CLEAN_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][LEO_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PROXY_SCORE]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PAIR_SCORE]") for line in lines) == 6
    assert all("phase1_ccpc_leo_postfreeze_20260809" in line for line in lines)
