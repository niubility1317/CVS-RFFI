from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import export_phase1_rcrmd_features as EXPORT
import evaluate_phase1_rcrmd_postfreeze_pair as PAIR
from cvsrffi import phase1_rcrmd as RCRMD


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_rcrmd_postfreeze_20260810.sh"
FOLD = 1
TX_IDS = PAIR._icmt.FROZEN_FOLD_SOURCE_TX[FOLD]
KNOWN_TX = (PAIR._icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[FOLD],)
PROXY_TX = (PAIR._icmt.FROZEN_FOLD_PROXY_TX[FOLD],)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _common_cells() -> tuple[dict[str, object], list[dict[str, object]]]:
    cells_by_scene: dict[str, object] = {}
    events: list[dict[str, object]] = []
    counts = {
        RCRMD._receiver_key(receiver, class_id): 1
        for receiver in RCRMD.FROZEN_RCRMD_SOURCE_RECEIVER_IDS
        for class_id in RCRMD.FROZEN_RCRMD_CLASS_IDS
    }
    weights = RCRMD._batch_cell_weights(RCRMD.FROZEN_RCRMD_SOURCE_RECEIVER_IDS, counts)
    for scenario in RCRMD.FROZEN_RCRMD_SCENARIOS:
        cells = RCRMD._cell_template(RCRMD.FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
        for cell in cells.values():
            cell.update({"rows": 1, "batches": 1, "nonempty_batches": 1})
        cells_by_scene[scenario] = cells
        events.append(
            {
                "scenario": scenario,
                "n_rc": dict(counts),
                "effective_weights": json.loads(json.dumps(weights)),
            }
        )
    return cells_by_scene, events


def _receipt(
    arm: str,
    *,
    labeled_sha: str | None = None,
    split_sha: str | None = None,
    source_tx_ids: tuple[str, ...] = TX_IDS,
    known_tx_ids: tuple[str, ...] = KNOWN_TX,
    proxy_tx_ids: tuple[str, ...] = PROXY_TX,
) -> dict[str, object]:
    enabled = arm == "G"
    receipt = RCRMD.rcrmd_config_receipt(
        RCRMD.RCRMDConfig(frozen_mode=True, enabled=enabled, loss_weight=0.02 if enabled else 0.0)
    )
    common_cells, batch_cells = _common_cells()
    receipt.update(
        {
            "baseline_sha256": _sha("baseline"),
            "initial_checkpoint_sha256": _sha("initial"),
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": _sha("class-order"),
            "source_labeled_indices_sha256": labeled_sha or _sha("labeled"),
            "source_split_manifest_sha256": split_sha or _sha("split"),
            "source_receiver_ids": list(RCRMD.FROZEN_RCRMD_SOURCE_RECEIVER_IDS),
            "source_receiver_count": 7,
            "source_receiver_ids_sha256": EXPORT._canonical_json_sha256(
                list(RCRMD.FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
            ),
            "source_receiver_provenance": EXPORT.SOURCE_RECEIVER_PROVENANCE,
            "optimizer_type": "AdamW",
            "optimizer_initial_state_sha256": _sha("optimizer"),
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "source_train_tx": list(source_tx_ids),
            "source_known_validation_tx": list(known_tx_ids),
            "source_proxy_unknown_tx": list(proxy_tx_ids),
            "local_tx_class_order": list(source_tx_ids),
            "checkpoint_train_tx_class_order": list(source_tx_ids),
            "dataset_tx_class_order": list(source_tx_ids) + list(known_tx_ids) + list(proxy_tx_ids),
            "local_to_dataset_class_ids": [0, 1, 2, 3],
            "local_to_head_class_ids": [0, 1, 2, 3],
            "expected_tx_class_ids": [0, 1, 2, 3],
            "dataset_class_count": 6,
            "local_data_class_count": 4,
            "checkpoint_head_class_count": 4,
            "live_head_class_count": 4,
            "common_batch_sequence_sha256": _sha("batch-sequence"),
            "common_batch_sequence_batches": 3,
            "common_batch_sequence_rows": 84,
            "common_scenario_batches": {scenario: 1 for scenario in RCRMD.FROZEN_RCRMD_SCENARIOS},
            "rcrmd_common_cells": common_cells,
            "rcrmd_common_batch_cells": batch_cells,
        }
    )
    if not enabled:
        return receipt
    scenes: dict[str, object] = {}
    for scenario in RCRMD.FROZEN_RCRMD_SCENARIOS:
        cells = RCRMD._cell_template(RCRMD.FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
        for cell in cells.values():
            cell.update(
                {
                    "rows": 1,
                    "active_q": 1,
                    "finite_q": 1,
                    "q_sum": 1.0,
                    "g_sum": 1.0,
                    "loss_sum": 1.0 / 28.0,
                    "batches": 1,
                    "nonempty_batches": 1,
                    "finite_batches": 1,
                }
            )
        scenes[scenario] = cells
    receipt.update(
        {
            "rcrmd_scenes": scenes,
            "rcrmd_batches": 3,
            "rcrmd_total_rows": 84,
            "rcrmd_active_q": 84,
            "rcrmd_loss_sum": 3.0,
            "rcrmd_g_batch_aux": [{"active_q": 28} for _ in RCRMD.FROZEN_RCRMD_SCENARIOS],
            "rcrmd_gradient_audit_attempted": True,
            "rcrmd_gradient_audit_completed": True,
            "rcrmd_gradient_audit": {
                "raw_unscaled": True,
                "diagnostic_only": True,
                "touches_amp_optimizer_rng": False,
                "shared_encoder": {"parameter_count": 2.0, "norm": 0.125},
                "classifier_head": {"parameter_count": 2.0, "norm": 0.25},
            },
        }
    )
    return receipt


def _checkpoint(
    path: Path,
    arm: str,
    receipt: dict[str, object] | None = None,
    *,
    fold: int = FOLD,
    source_tx_ids: tuple[str, ...] = TX_IDS,
    known_tx_ids: tuple[str, ...] = KNOWN_TX,
    proxy_tx_ids: tuple[str, ...] = PROXY_TX,
) -> Path:
    candidate = f"F{fold}{arm}_RCRMD12"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "candidate_id": candidate,
        "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
        "model": {},
        "args": {
            "split_mode": "tx_rx_day_1_6_3",
            "model_variant": "lite_d",
            "id_feature_key": "feat_joint",
            "phase1_source_train_tx_ids": ",".join(source_tx_ids),
            "phase1_source_known_validation_tx_ids": known_tx_ids[0],
            "phase1_source_proxy_unknown_tx_ids": proxy_tx_ids[0],
            "checkpoint_selection": "final_only",
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
            "seed": 7281105,
            "phase1_rcrmd_frozen_mode": True,
            "phase1_rcrmd_enabled": arm == "G",
            "lambda_rcrmd": 0.02 if arm == "G" else 0.0,
            "candidate_id": candidate,
            "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
        },
        "rcrmd_receipt": receipt
        if receipt is not None
        else _receipt(
            arm,
            source_tx_ids=source_tx_ids,
            known_tx_ids=known_tx_ids,
            proxy_tx_ids=proxy_tx_ids,
        ),
    }
    torch.save(payload, path)
    return path


def _valid_checkpoint_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    training_root = tmp_path / PAIR.EXPECTED_TRAINING_RUN_LEAF
    c_path = _checkpoint(training_root / "F1C_RCRMD12" / "final_ssdg.pth", "C")
    g_path = _checkpoint(training_root / "F1G_RCRMD12" / "final_ssdg.pth", "G")
    return training_root, c_path, g_path


def test_totalized_float64_geometry_retains_zero_and_rejects_nonfinite() -> None:
    features = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normal = PAIR.normalize_rcrmd_float64(features, label="fixture")
    assert normal.dtype == np.float64
    assert np.allclose(normal, np.asarray([[0.6, 0.8], [0.0, 0.0]], dtype=np.float64))
    tx = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    geometry = PAIR.fit_frozen_rcrmd_diagonal_gaussian(
        np.vstack((np.eye(4), np.eye(4))), tx, ("a", "b", "c", "d")
    )
    scores = PAIR.score_frozen_rcrmd_nll(np.zeros((1, 4), dtype=np.float64), geometry)
    assert scores.shape == (1,) and math.isfinite(float(scores[0]))
    with pytest.raises(Exception, match="non-finite"):
        PAIR.normalize_rcrmd_float64(np.asarray([[np.nan, 1.0]]), label="bad")


@pytest.mark.parametrize("arm", ["C", "G"])
def test_rcrmd_terminal_receipt_revalidates_raw_84_cell_contract(arm: str) -> None:
    checked = EXPORT.validate_rcrmd_terminal_receipt(
        _receipt(arm),
        arm=arm,
        source_tx_ids=TX_IDS,
        known_validation_tx_ids=KNOWN_TX,
        proxy_unknown_tx_ids=PROXY_TX,
    )
    assert checked["rcrmd_terminal_contract_passed"] is True
    assert checked["source_receiver_ids"] == list(range(7))
    assert all(len(checked["rcrmd_common_cells"][scene]) == 28 for scene in RCRMD.FROZEN_RCRMD_SCENARIOS)


@pytest.mark.parametrize(
    ("arm", "mutate", "message"),
    [
        ("C", lambda receipt: receipt.__setitem__("source_receiver_count", 6), "source_receiver_count"),
        ("G", lambda receipt: receipt.__setitem__("source_receiver_ids_sha256", _sha("bad-rx")), "receiver SHA"),
        ("C", lambda receipt: receipt.__setitem__("source_receiver_provenance", "other"), "provenance"),
        ("G", lambda receipt: receipt["rcrmd_common_cells"]["leo_rain_weak"].pop("rx0|tx0"), "common receiver/class cells"),
        ("G", lambda receipt: receipt["rcrmd_gradient_audit"]["classifier_head"].__setitem__("norm", 0.0), "VJP"),
        ("G", lambda receipt: receipt.__setitem__("lambda", 0.01), "lambda"),
    ],
)
def test_training_receipt_drift_fails_closed(arm, mutate, message) -> None:
    receipt = _receipt(arm)
    mutate(receipt)
    with pytest.raises(EXPORT.RCRMDSplitExportError, match=message):
        EXPORT.validate_rcrmd_terminal_receipt(
            receipt,
            arm=arm,
            source_tx_ids=TX_IDS,
            known_validation_tx_ids=KNOWN_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_sha256", _sha("other-baseline")),
        ("source_receiver_ids_sha256", _sha("other-rx")),
        ("source_receiver_count", 6),
        ("checkpoint_train_tx_class_order", ["wrong", *TX_IDS[1:]]),
        ("optimizer_initial_state_sha256", _sha("other-adamw")),
        ("common_batch_sequence_sha256", _sha("other-sequence")),
    ],
)
def test_common_training_binding_rejects_c_g_mismatch(field: str, value: object) -> None:
    c_receipt = _receipt("C")
    g_receipt = _receipt("G")
    g_receipt[field] = value
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="common training binding"):
        PAIR.validate_rcrmd_common_training_binding(c_receipt, g_receipt)


def test_g_only_auxiliary_fields_are_not_erroneously_compared_to_control() -> None:
    c_receipt = _receipt("C")
    g_receipt = _receipt("G")
    g_receipt["rcrmd_loss_sum"] = 4.0
    binding = PAIR.validate_rcrmd_common_training_binding(c_receipt, g_receipt)
    assert binding["passed"] is True
    assert all(field not in binding["fields"] for field in PAIR.G_ONLY_RECEIPT_FIELDS)


@pytest.mark.parametrize(
    ("arm", "mutate", "message"),
    [
        ("C", lambda payload: payload.__setitem__("run_id", "other"), "run_id"),
        ("G", lambda payload: payload["args"].__setitem__("lambda_rcrmd", 0.01), "lambda_rcrmd"),
        ("G", lambda payload: payload["rcrmd_receipt"].__setitem__("schema", "wrong"), "schema"),
        ("G", lambda payload: payload["args"].__setitem__("phase1_rcrmd_enabled", False), "phase1_rcrmd_enabled"),
    ],
)
def test_checkpoint_validator_rejects_training_receipt_and_arm_drift(tmp_path, arm, mutate, message):
    training_root, c_path, g_path = _valid_checkpoint_pair(tmp_path)
    target = c_path if arm == "C" else g_path
    payload = torch.load(target, map_location="cpu")
    mutate(payload)
    torch.save(payload, target)
    with pytest.raises(EXPORT.RCRMDSplitExportError, match=message):
        EXPORT.validate_rcrmd_training_checkpoint(
            torch.load(target, map_location="cpu"),
            checkpoint_path=target,
            source_tx_ids=TX_IDS,
            known_validation_tx_ids=KNOWN_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )
    assert training_root.name == PAIR.EXPECTED_TRAINING_RUN_LEAF


def test_fixed_proxy_selection_is_not_cli_tunable() -> None:
    args = argparse.Namespace(
        proxy_days="2021_03_01",
        proxy_rxs=",".join(EXPORT.FROZEN_PROXY_RXS),
        max_proxy_samples_per_tx=400,
    )
    with pytest.raises(EXPORT.RCRMDSplitExportError, match="proxy_days"):
        EXPORT._coerce_frozen_proxy_args(args)
    fixed = EXPORT._coerce_frozen_proxy_args(argparse.Namespace())
    assert fixed.proxy_days == ",".join(EXPORT.FROZEN_PROXY_DAYS)
    assert fixed.proxy_rxs == ",".join(EXPORT.FROZEN_PROXY_RXS)
    assert fixed.max_proxy_samples_per_tx == 400


def test_launcher_is_exactly_42_steps_and_rcrmd_only() -> None:
    relative_launcher = "scripts/launch_phase1_rcrmd_postfreeze_20260810.sh"
    syntax = subprocess.run(
        ["bash", "-n", relative_launcher],
        check=False,
        capture_output=True,
        text=True,
        cwd=CODE_ROOT,
    )
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(
        ["bash", relative_launcher, "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=CODE_ROOT,
    )
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if line.startswith("[DRY-RUN]")]
    assert len(lines) == 42
    assert sum("RCRMD_CLEAN_EXPORT" in line for line in lines) == 12
    assert sum("RCRMD_LEO_EXPORT_AND_BIND" in line for line in lines) == 12
    assert sum("FROZEN_LOGITS_PROXY_BINDING" in line for line in lines) == 12
    assert sum("RCRMD_PAIR_SCORE" in line for line in lines) == 6
    assert all("_CAGM12" not in line and "_ICMT12" not in line for line in lines)
    assert sum("phase1_rcrmd12_20260810_v1" in line for line in lines) == 30
    assert all("phase1_rcrmd_postfreeze_20260810_v1" in line for line in lines)


def _load_icmt_fixture_module():
    path = CODE_ROOT / "tests" / "test_phase1_icmt_postfreeze.py"
    spec = importlib.util.spec_from_file_location("_icmt_fixture_for_rcrmd", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_npz_manifest(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]).copy() for name in data.files}
    manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
    mutate(manifest)
    payload["manifest_json"] = np.asarray(json.dumps(manifest))
    np.savez(path, **payload)


def _write_frozen_proxy_logits(
    clean: Path, proxy: Path, scores: Path, source_tx_ids: tuple[str, ...]
) -> None:
    scorer = PAIR._logits_reject_module()
    scorer.evaluate(
        argparse.Namespace(
            feature_npz=str(clean),
            source_tx_ids=",".join(source_tx_ids),
            unknown_tx_ids="",
            known_query_roles="source_validation_known",
            unknown_query_roles="proxy_unknown",
            calibration_roles="source_validation_known",
            conf_quantile=0.05,
            margin_quantile=0.05,
            energy_quantile=0.95,
            disable_conf_gate=False,
            disable_margin_gate=False,
            disable_energy_gate=False,
            unknown_far_target=0.05,
            output_json=str(proxy),
            score_table_csv=str(scores),
        )
    )


def _build_rcrmd_pair_fixture(
    tmp_path: Path, *, fold: int = 1, root: Path | None = None
) -> dict[str, object]:
    fixture = _load_icmt_fixture_module()
    root = root or (tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    fixture._write_pair(root, fold=fold)
    training_root = root.parent / PAIR.EXPECTED_TRAINING_RUN_LEAF
    source_tx_ids = PAIR._icmt.FROZEN_FOLD_SOURCE_TX[fold]
    known_tx_ids = (PAIR._icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold],)
    proxy_tx_ids = (PAIR._icmt.FROZEN_FOLD_PROXY_TX[fold],)
    paths: dict[str, object] = {
        "root": root.resolve(),
        "training_root": training_root.resolve(),
        "source_tx_ids": source_tx_ids,
    }
    for arm in ("C", "G"):
        old_candidate = f"F{fold}{arm}_ICMT12"
        candidate = f"F{fold}{arm}_RCRMD12"
        old_dir = root / old_candidate
        new_dir = root / candidate
        shutil.move(str(old_dir), str(new_dir))
        clean = new_dir / "icmt_clean_l_v_proxy_final_only.npz"
        leo = new_dir / "source_leo_final_only.npz"
        proxy = new_dir / "proxy_logits_open_set_metrics.json"
        scores = new_dir / "proxy_logits_open_set_scores.csv"
        binding = new_dir / "source_leo_binding.json"
        with np.load(clean, allow_pickle=False) as data:
            clean_manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        receipt = _receipt(
            arm,
            labeled_sha=str(clean_manifest["labeled_indices_sha256"]),
            split_sha=str(clean_manifest["source_split_receipt"]["split_manifest_sha256"]),
            source_tx_ids=source_tx_ids,
            known_tx_ids=known_tx_ids,
            proxy_tx_ids=proxy_tx_ids,
        )
        checkpoint = _checkpoint(
            training_root / candidate / "final_ssdg.pth",
            arm,
            receipt,
            fold=fold,
            source_tx_ids=source_tx_ids,
            known_tx_ids=known_tx_ids,
            proxy_tx_ids=proxy_tx_ids,
        )
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checked = EXPORT.validate_rcrmd_terminal_receipt(
            receipt,
            arm=arm,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_tx_ids,
            proxy_unknown_tx_ids=proxy_tx_ids,
        )
        raw_receipt_sha = EXPORT._canonical_json_sha256(receipt)

        def clean_mutate(manifest: dict[str, object]) -> None:
            for field in (
                "icmt_receipt_schema",
                "icmt_enabled",
                "icmt_source_labeled_indices_sha256",
                "icmt_source_split_manifest_sha256",
            ):
                manifest.pop(field, None)
            manifest.update(
                {
                    "schema": PAIR.EXPECTED_LV_EXPORT_SCHEMA,
                    "method": "P1_RCRMD",
                    "checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": checkpoint_sha,
                    "candidate_id": candidate,
                    "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "training_run_contract": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "rcrmd_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                    "rcrmd_enabled": arm == "G",
                    "rcrmd_source_labeled_indices_sha256": receipt["source_labeled_indices_sha256"],
                    "rcrmd_source_split_manifest_sha256": receipt["source_split_manifest_sha256"],
                    "rcrmd_source_receiver_ids_sha256": receipt["source_receiver_ids_sha256"],
                    "rcrmd_source_receiver_ids": list(range(7)),
                    "rcrmd_source_receiver_count": 7,
                    "rcrmd_source_receiver_provenance": EXPORT.SOURCE_RECEIVER_PROVENANCE,
                    "rcrmd_frozen_cells_per_scene": 28,
                    "rcrmd_receipt_sha256": raw_receipt_sha,
                    "rcrmd_terminal_contract": checked["rcrmd_terminal_contract"],
                    "rcrmd_terminal_contract_passed": True,
                    "rcrmd_lambda": 0.02 if arm == "G" else 0.0,
                    "rcrmd_loss_global_denominator": "4_TIMES_FIXED_SOURCE_RECEIVER_COUNT",
                    "rcrmd_common_physical_rx_class_scene_nrc_bound": True,
                    "rcrmd_batch_order_bound": True,
                    "proxy_selection_frozen_not_cli_tunable": True,
                }
            )

        _rewrite_npz_manifest(clean, clean_mutate)
        _rewrite_npz_manifest(
            leo,
            lambda manifest: manifest.update(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": checkpoint_sha,
                }
            ),
        )
        _write_frozen_proxy_logits(clean, proxy, scores, source_tx_ids)
        fixture._write_leo_binding(
            binding,
            leo_path=leo,
            clean_path=clean,
            checkpoint=checkpoint,
            candidate=candidate,
            fold=fold,
            arm=arm,
            training_root=training_root,
            output_root=root,
        )
        raw_binding = json.loads(binding.read_text(encoding="utf-8"))
        with np.load(leo, allow_pickle=False) as data:
            leo_manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        raw_binding.update(
            {
                "schema": PAIR.EXPECTED_LEO_BINDING_SCHEMA,
                "method": "P1_RCRMD",
                "candidate_id": candidate,
                "training_run_root": str(training_root.resolve()),
                "postfreeze_output_root": str(root.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "training_run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                "leo_npz_path": str(leo.resolve()),
                "leo_npz_sha256": hashlib.sha256(leo.read_bytes()).hexdigest(),
                "leo_manifest_sha256": fixture.PAIR._icmt_leo._canonical_json_sha256(leo_manifest),
                "rcrmd_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                "rcrmd_receipt_sha256": raw_receipt_sha,
                "rcrmd_terminal_contract": checked["rcrmd_terminal_contract"],
                "rcrmd_terminal_contract_passed": True,
                "rcrmd_lambda": 0.02 if arm == "G" else 0.0,
                "rcrmd_loss_global_denominator": "4_TIMES_FIXED_SOURCE_RECEIVER_COUNT",
                "rcrmd_source_receiver_ids": list(range(7)),
                "rcrmd_source_receiver_count": 7,
                "rcrmd_source_receiver_ids_sha256": receipt["source_receiver_ids_sha256"],
                "rcrmd_source_receiver_provenance": EXPORT.SOURCE_RECEIVER_PROVENANCE,
                "rcrmd_frozen_cells_per_scene": 28,
                "rcrmd_common_physical_rx_class_scene_nrc_bound": True,
                "rcrmd_batch_order_bound": True,
            }
        )
        binding.write_text(json.dumps(raw_binding, sort_keys=True), encoding="utf-8")
        prefix = arm.lower()
        paths.update(
            {
                f"{prefix}_checkpoint": checkpoint,
                f"{prefix}_clean": clean,
                f"{prefix}_leo": leo,
                f"{prefix}_proxy": proxy,
                f"{prefix}_scores": scores,
                f"{prefix}_binding": binding,
            }
        )
    return paths


def _pair_args(
    paths: dict[str, object], output: Path, *, fold: int = 1, priors: tuple[Path, ...] = ()
):
    return PAIR.build_parser().parse_args(
        [
            "--c-clean-npz", str(paths["c_clean"]),
            "--g-clean-npz", str(paths["g_clean"]),
            "--c-leo-npz", str(paths["c_leo"]),
            "--g-leo-npz", str(paths["g_leo"]),
            "--c-leo-binding-json", str(paths["c_binding"]),
            "--g-leo-binding-json", str(paths["g_binding"]),
            "--c-final-checkpoint", str(paths["c_checkpoint"]),
            "--g-final-checkpoint", str(paths["g_checkpoint"]),
            "--c-proxy-metrics-json", str(paths["c_proxy"]),
            "--g-proxy-metrics-json", str(paths["g_proxy"]),
            "--c-proxy-scores-csv", str(paths["c_scores"]),
            "--g-proxy-scores-csv", str(paths["g_scores"]),
            "--source-tx-ids", ",".join(paths["source_tx_ids"]),
            "--candidate-pair", f"F{fold}_C_vs_G",
            "--fold-index", str(fold),
            "--postfreeze-matrix-id", PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
            "--postfreeze-output-root", str(paths["root"]),
            "--training-run-root", str(paths["training_root"]),
            "--expected-source-count", "72",
            "--expected-proxy-count", "400",
            "--output-metrics-json", str(output),
        ]
        + (
            ["--aggregate-prior-pair-metrics-json", ",".join(str(path) for path in priors)]
            if priors
            else []
        )
    )


def test_pair_closes_raw_checkpoint_clean_leo_proxy_and_rcrmd_receipt_chain(tmp_path):
    paths = _build_rcrmd_pair_fixture(tmp_path)
    metrics = PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair.json"))
    assert metrics["schema"] == PAIR.EXPECTED_PAIR_SCHEMA
    assert metrics["rcrmd_training_receipt_revalidation"]["C"]["candidate"] == "F1C_RCRMD12"
    assert metrics["rcrmd_training_receipt_revalidation"]["G"]["terminal_contract_passed"] is True
    assert metrics["policy"]["geometry_fit_role"] == "labeled_fit"
    assert metrics["policy"]["proxy_unknown_fit_rows"] == 0
    common = metrics["rcrmd_common_training_binding"]
    assert common["passed"] is True
    assert set(common["fields"]) == set(PAIR.COMMON_TRAINING_BINDING_FIELDS)
    assert common["fields"]["source_receiver_ids"] == list(range(7))
    assert metrics["verdict"].startswith(("REJECT", "PENDING_MAIN"))


def test_pair_rejects_source_leo_and_proxy_binding_attacks(tmp_path):
    paths = _build_rcrmd_pair_fixture(
        tmp_path, root=tmp_path / "source-attack" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    )
    clean = Path(paths["c_clean"])
    _rewrite_npz_manifest(clean, lambda manifest: manifest.__setitem__("rcrmd_source_receiver_count", 6))
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="source_receiver_count"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair-source.json"))

    paths = _build_rcrmd_pair_fixture(
        tmp_path, root=tmp_path / "leo-attack" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    )
    binding = Path(paths["g_binding"])
    raw = json.loads(binding.read_text(encoding="utf-8"))
    raw["rcrmd_frozen_cells_per_scene"] = 27
    binding.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="frozen_cells_per_scene"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair-leo.json"))

    paths = _build_rcrmd_pair_fixture(
        tmp_path, root=tmp_path / "proxy-attack" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    )
    for prefix in ("c", "g"):
        clean = Path(paths[f"{prefix}_clean"])
        _rewrite_npz_manifest(clean, lambda manifest: manifest.__setitem__("proxy_row_count", 399))
        proxy = Path(paths[f"{prefix}_proxy"])
        raw = json.loads(proxy.read_text(encoding="utf-8"))
        raw["manifest"]["proxy_row_count"] = 399
        proxy.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="proxy"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair-proxy.json"))


def test_one_row_proxy_attack_is_not_hidden_by_artifact_sha(tmp_path):
    paths = _build_rcrmd_pair_fixture(tmp_path)
    clean = Path(paths["c_clean"])
    with np.load(clean, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]).copy() for name in data.files}
    proxy_index = int(np.flatnonzero(np.asarray(payload["dataset_role"]).astype(str) == "proxy_unknown")[0])
    payload["tx_logits"][proxy_index] = np.asarray([100.0, -100.0, -100.0, -100.0], dtype=np.float32)
    np.savez(clean, **payload)
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="proxy.*raw logits"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair-one-row.json"))


def test_f6_reloads_all_five_prior_raw_artifact_chains(tmp_path):
    root = tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    priors: list[Path] = []
    final_metrics = None
    for fold in range(1, 7):
        paths = _build_rcrmd_pair_fixture(tmp_path, fold=fold, root=root)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        final_metrics = PAIR.evaluate(
            _pair_args(paths, output, fold=fold, priors=tuple(priors) if fold == 6 else ())
        )
        if fold < 6:
            priors.append(output)
    assert final_metrics is not None
    aggregate = final_metrics["matrix_aggregate"]
    assert aggregate["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert len(aggregate["prior_pair_metrics_bindings"]) == 5
    assert all(item["raw_artifacts_recomputed"] is True for item in aggregate["prior_pair_metrics_bindings"])
    assert aggregate["verdict"] in {"PENDING_MAIN_REVIEW", "REJECT_P1_RCRMD_PERMANENT"}


def test_f6_rejects_prior_summary_and_raw_artifact_tamper(tmp_path):
    paths = _build_rcrmd_pair_fixture(tmp_path)
    output = Path(paths["root"]) / "F1_C_vs_G_pair_metrics.json"
    PAIR.evaluate(_pair_args(paths, output))
    record = json.loads(output.read_text(encoding="utf-8"))
    record["rcrmd_common_training_binding"]["fields"]["common_batch_sequence_sha256"] = _sha("self-report")
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="common training binding"):
        PAIR._recompute_rcrmd_prior_pair_artifacts(
            record,
            output_root=Path(paths["root"]),
            matrix_id=PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
            training_root=Path(paths["training_root"]),
            expected_scenarios=PAIR._icmt.EXPECTED_SCENARIOS,
        )

    root = tmp_path / "raw-prior" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    root.parent.mkdir()
    priors: list[Path] = []
    paths_by_fold: dict[int, dict[str, object]] = {}
    for fold in range(1, 6):
        paths = _build_rcrmd_pair_fixture(tmp_path, fold=fold, root=root)
        paths_by_fold[fold] = paths
        pair_output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_pair_args(paths, pair_output, fold=fold))
        priors.append(pair_output)
    prior_clean = Path(paths_by_fold[1]["c_clean"])
    with np.load(prior_clean, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]).copy() for name in data.files}
    proxy_index = int(np.flatnonzero(np.asarray(payload["dataset_role"]).astype(str) == "proxy_unknown")[0])
    payload["features"][proxy_index] = np.zeros(4, dtype=np.float32)
    np.savez(prior_clean, **payload)
    prior_record = json.loads(priors[0].read_text(encoding="utf-8"))
    prior_record["bindings"]["c_clean_npz_sha256"] = hashlib.sha256(prior_clean.read_bytes()).hexdigest()
    priors[0].write_text(json.dumps(prior_record, sort_keys=True), encoding="utf-8")
    f6_paths = _build_rcrmd_pair_fixture(tmp_path, fold=6, root=root)
    with pytest.raises(PAIR.RCRMDPostfreezePairError, match="raw-artifact recomputation|does not match"):
        PAIR.evaluate(
            _pair_args(
                f6_paths,
                root / "F6_C_vs_G_pair_metrics.json",
                fold=6,
                priors=tuple(priors),
            )
        )
