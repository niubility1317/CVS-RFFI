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

import export_phase1_cagm_features as EXPORT
import evaluate_phase1_cagm_postfreeze_pair as PAIR
from cvsrffi import phase1_cagm as CAGM


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_cagm_postfreeze_20260810.sh"
FOLD = 1
TX_IDS = PAIR._icmt.FROZEN_FOLD_SOURCE_TX[FOLD]
KNOWN_TX = (PAIR._icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[FOLD],)
PROXY_TX = (PAIR._icmt.FROZEN_FOLD_PROXY_TX[FOLD],)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _term(value: float, batches: int) -> dict[str, float | int]:
    return {
        "batches": batches,
        "finite_batches": batches,
        "sum_delta": float(value * batches),
        "sum_sq_delta": float(value * value * batches),
    }


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
    receipt = CAGM.cagm_config_receipt(
        CAGM.CAGMConfig(frozen_mode=True, enabled=enabled, loss_weight=0.02 if enabled else 0.0)
    )
    receipt.update(
        {
            "baseline_sha256": _sha("baseline"),
            "initial_checkpoint_sha256": _sha("initial"),
            "checkpoint_role": "training_final_only",
            "class_order_binding_sha256": _sha("class-order"),
            "source_labeled_indices_sha256": labeled_sha or _sha("labeled"),
            "source_split_manifest_sha256": split_sha or _sha("split"),
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
            "common_batch_sequence_rows": 24,
            "common_scenario_batches": {scenario: 1 for scenario in CAGM.FROZEN_CAGM_SCENARIOS},
        }
    )
    if not enabled:
        return receipt
    radius = {f"tx{index}": _term(0.01 + index * 0.001, 1) for index in range(4)}
    gram = {
        f"tx{left}|tx{right}": _term(0.02 + left * 0.001 + right * 0.0001, 1)
        for left in range(4)
        for right in range(left + 1, 4)
    }
    scenes: dict[str, object] = {}
    for scenario in CAGM.FROZEN_CAGM_SCENARIOS:
        scenes[scenario] = {
            "batches": 1,
            "total_rows": 8,
            "valid_rows": 8,
            "clean_zero_rows": 0,
            "leo_zero_rows": 0,
            "union_zero_rows": 0,
            "both_zero_rows": 0,
            "per_tx_valid_rows": {str(index): 2 for index in range(4)},
            "radius_terms": radius,
            "gram_terms": gram,
        }
    receipt.update(
        {
            "cagm_scenes": scenes,
            "cagm_radius_terms": {
                key: _term(float(value["sum_delta"]) / 3.0, 3) for key, value in radius.items()
            },
            "cagm_gram_terms": {
                key: _term(float(value["sum_delta"]) / 3.0, 3) for key, value in gram.items()
            },
            "cagm_batches": 3,
            "cagm_total_rows": 24,
            "cagm_valid_rows": 24,
            "cagm_clean_zero_rows": 0,
            "cagm_leo_zero_rows": 0,
            "cagm_union_zero_rows": 0,
            "cagm_both_zero_rows": 0,
            "cagm_gradient_audit_attempted": True,
            "cagm_gradient_audit_completed": True,
            "cagm_gradient_audit": {
                "raw_unscaled": True,
                "diagnostic_only": True,
                "shared_encoder": {"parameter_count": 2.0, "norm": 0.125},
                "classifier_head": {
                    "parameter_count": 2.0,
                    "none_parameters": 1.0,
                    "zero_parameters": 1.0,
                    "nonzero_parameters": 0.0,
                    "none_or_zero_expected": True,
                },
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
    candidate = f"F{fold}{arm}_CAGM12"
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
            "phase1_cagm_frozen_mode": True,
            "phase1_cagm_enabled": arm == "G",
            "lambda_cagm": 0.02 if arm == "G" else 0.0,
            "candidate_id": candidate,
            "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
        },
        "cagm_receipt": receipt
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
    c_path = _checkpoint(training_root / "F1C_CAGM12" / "final_ssdg.pth", "C")
    g_path = _checkpoint(training_root / "F1G_CAGM12" / "final_ssdg.pth", "G")
    return training_root, c_path, g_path


def test_totalized_float64_geometry_matches_formula_retains_zero_and_rejects_nonfinite():
    features = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normal = PAIR.normalize_cagm_float64(features, label="fixture")
    assert normal.dtype == np.float64
    assert np.allclose(normal, np.asarray([[0.6, 0.8], [0.0, 0.0]], dtype=np.float64))
    tx = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    geometry = PAIR.fit_frozen_cagm_diagonal_gaussian(
        np.vstack((np.eye(4), np.eye(4))), tx, ("a", "b", "c", "d")
    )
    scores = PAIR.score_frozen_cagm_nll(np.zeros((1, 4), dtype=np.float64), geometry)
    assert scores.shape == (1,)
    assert math.isfinite(float(scores[0]))
    with pytest.raises(Exception, match="non-finite"):
        PAIR.normalize_cagm_float64(np.asarray([[np.nan, 1.0]]), label="bad")


@pytest.mark.parametrize("arm", ["C", "G"])
def test_cagm_terminal_receipt_revalidates_raw_contract(arm: str):
    checked = EXPORT.validate_cagm_terminal_receipt(
        _receipt(arm),
        arm=arm,
        source_tx_ids=TX_IDS,
        known_validation_tx_ids=KNOWN_TX,
        proxy_unknown_tx_ids=PROXY_TX,
    )
    assert checked["cagm_terminal_contract_passed"] is True
    assert checked["schema"] == EXPORT.EXPECTED_RECEIPT_SCHEMA


@pytest.mark.parametrize(
    ("arm", "mutate"),
    [
        ("G", lambda receipt: receipt.pop("joint_zero_mask_aux_only")),
        ("G", lambda receipt: receipt.__setitem__("joint_zero_mask_aux_only", False)),
        ("C", lambda receipt: receipt.pop("joint_zero_mask_aux_only")),
        ("C", lambda receipt: receipt.__setitem__("joint_zero_mask_aux_only", True)),
    ],
)
def test_raw_receipt_joint_zero_mask_is_required_and_arm_exact(arm, mutate):
    receipt = _receipt(arm)
    mutate(receipt)
    with pytest.raises(EXPORT.CAGMSplitExportError, match="joint_zero_mask_aux_only"):
        EXPORT.validate_cagm_terminal_receipt(
            receipt,
            arm=arm,
            source_tx_ids=TX_IDS,
            known_validation_tx_ids=KNOWN_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "sequence_sha",
            lambda receipt: receipt.__setitem__(
                "common_batch_sequence_sha256", _sha("G-sequence")
            ),
        ),
        (
            "sequence_rows",
            lambda receipt: receipt.__setitem__("common_batch_sequence_rows", 25),
        ),
        (
            "scenario",
            lambda receipt: receipt["common_scenario_batches"].__setitem__(
                "leo_rain_weak", 2
            ),
        ),
        (
            "baseline",
            lambda receipt: receipt.__setitem__("baseline_sha256", _sha("G-baseline")),
        ),
        (
            "initial",
            lambda receipt: receipt.__setitem__(
                "initial_checkpoint_sha256", _sha("G-initial")
            ),
        ),
        (
            "optimizer_type",
            lambda receipt: receipt.__setitem__("optimizer_type", "SGD"),
        ),
        (
            "optimizer_sha",
            lambda receipt: receipt.__setitem__(
                "optimizer_initial_state_sha256", _sha("G-optimizer")
            ),
        ),
        (
            "optimizer_empty",
            lambda receipt: receipt.__setitem__("optimizer_initial_state_empty", False),
        ),
        (
            "strict_type",
            lambda receipt: receipt.__setitem__("common_batch_sequence_rows", 24.0),
        ),
        (
            "strict_scenario_keys",
            lambda receipt: receipt["common_scenario_batches"].__setitem__("extra", 1),
        ),
        (
            "missing_field",
            lambda receipt: receipt.pop("common_batch_sequence_sha256"),
        ),
    ],
)
def test_common_training_binding_rejects_g_only_tamper(case, mutate):
    c_receipt = _receipt("C")
    g_receipt = _receipt("G")
    mutate(g_receipt)
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="common training binding"):
        PAIR.validate_cagm_common_training_binding(c_receipt, g_receipt)


@pytest.mark.parametrize(
    ("arm", "mutate", "message"),
    [
        ("C", lambda receipt: receipt.__setitem__("cagm_batches", 1), "N/A-or-zero"),
        ("G", lambda receipt: receipt.__setitem__("loss_divisor", 9), "divisor"),
        (
            "G",
            lambda receipt: receipt["cagm_gradient_audit"]["classifier_head"].__setitem__("nonzero_parameters", 1.0),
            "encoder/head",
        ),
        ("G", lambda receipt: receipt["cagm_scenes"].pop("leo_rain_weak"), "terminal"),
    ],
)
def test_cagm_receipt_rejects_control_divisor_vjp_and_term_coverage_drift(arm, mutate, message):
    receipt = _receipt(arm)
    mutate(receipt)
    with pytest.raises(EXPORT.CAGMSplitExportError, match=message):
        EXPORT.validate_cagm_terminal_receipt(
            receipt,
            arm=arm,
            source_tx_ids=TX_IDS,
            known_validation_tx_ids=KNOWN_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )


@pytest.mark.parametrize(
    ("arm", "mutate", "message"),
    [
        ("C", lambda payload: payload.__setitem__("run_id", "other"), "run_id"),
        ("G", lambda payload: payload["args"].__setitem__("lambda_cagm", 0.01), "lambda_cagm"),
        ("G", lambda payload: payload["cagm_receipt"].__setitem__("schema", "wrong"), "schema"),
        ("G", lambda payload: payload["args"].__setitem__("phase1_cagm_enabled", False), "phase1_cagm_enabled"),
    ],
)
def test_checkpoint_validator_rejects_root_weight_schema_and_arm_tamper(tmp_path, arm, mutate, message):
    training_root, c_path, g_path = _valid_checkpoint_pair(tmp_path)
    target = c_path if arm == "C" else g_path
    payload = torch.load(target, map_location="cpu")
    mutate(payload)
    torch.save(payload, target)
    with pytest.raises(EXPORT.CAGMSplitExportError, match=message):
        EXPORT.validate_cagm_training_checkpoint(
            torch.load(target, map_location="cpu"),
            checkpoint_path=target,
            source_tx_ids=TX_IDS,
            known_validation_tx_ids=KNOWN_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )
    assert training_root.name == PAIR.EXPECTED_TRAINING_RUN_LEAF


def test_pair_preflight_reloads_cagm_receipts_and_rejects_matrix_or_root_drift(tmp_path):
    training_root, c_path, g_path = _valid_checkpoint_pair(tmp_path)
    output_root = tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    output_root.mkdir()
    args = argparse.Namespace(
        postfreeze_matrix_id=PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
        postfreeze_output_root=str(output_root),
        training_run_root=str(training_root),
        source_tx_ids=",".join(TX_IDS),
        fold_index=1,
        candidate_pair="F1_C_vs_G",
        c_final_checkpoint=str(c_path),
        g_final_checkpoint=str(g_path),
    )
    root, tx_ids, fold, common_binding = PAIR._prevalidate_current_args(args)
    assert root == training_root.resolve() and tx_ids == TX_IDS and fold == 1
    assert common_binding["passed"] is True
    assert common_binding["fields"]["optimizer_type"] == "AdamW"
    args.postfreeze_matrix_id = "wrong"
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="matrix_id"):
        PAIR._prevalidate_current_args(args)


def test_fixed_proxy_selection_is_not_cli_tunable():
    args = argparse.Namespace(proxy_days="2021_03_01", proxy_rxs=",".join(EXPORT.FROZEN_PROXY_RXS), max_proxy_samples_per_tx=400)
    with pytest.raises(EXPORT.CAGMSplitExportError, match="proxy_days"):
        EXPORT._coerce_frozen_proxy_args(args)
    fixed = EXPORT._coerce_frozen_proxy_args(argparse.Namespace())
    assert fixed.proxy_days == ",".join(EXPORT.FROZEN_PROXY_DAYS)
    assert fixed.proxy_rxs == ",".join(EXPORT.FROZEN_PROXY_RXS)
    assert fixed.max_proxy_samples_per_tx == 400


def test_launcher_is_exactly_42_steps_and_cagm_only():
    relative_launcher = "scripts/launch_phase1_cagm_postfreeze_20260810.sh"
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
    assert sum("CAGM_CLEAN_EXPORT" in line for line in lines) == 12
    assert sum("CAGM_LEO_EXPORT_AND_BIND" in line for line in lines) == 12
    assert sum("FROZEN_LOGITS_PROXY_BINDING" in line for line in lines) == 12
    assert sum("CAGM_PAIR_SCORE" in line for line in lines) == 6
    assert all("_ICMT12" not in line for line in lines)
    assert sum("phase1_cagm12_20260810_v2" in line for line in lines) == 30
    assert all("phase1_cagm_postfreeze_20260810_v2" in line for line in lines)
    assert all("phase1_cagm12_20260810_v1" not in line for line in lines)


def test_pair_source_recomputes_prior_raw_artifacts_and_never_signs_allow():
    source = (CODE_ROOT / "evaluate_phase1_cagm_postfreeze_pair.py").read_text(encoding="utf-8")
    assert "_recompute_cagm_prior_pair_artifacts" in source
    assert "raw-artifact recomputation" in source
    assert "REJECT_P1_CAGM_PERMANENT" in source
    assert "ALLOW" not in source


def _load_icmt_fixture_module():
    path = CODE_ROOT / "tests" / "test_phase1_icmt_postfreeze.py"
    spec = importlib.util.spec_from_file_location("_icmt_fixture_for_cagm", path)
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


def _build_cagm_pair_fixture(
    tmp_path: Path, *, fold: int = 1, root: Path | None = None
) -> dict[str, object]:
    """Adapt signed ICMT-v2 synthetic raw artifacts to genuine CAGM identities."""

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
        candidate = f"F{fold}{arm}_CAGM12"
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
        checked = EXPORT.validate_cagm_terminal_receipt(
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
                    "method": "P1_CAGM",
                    "checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": checkpoint_sha,
                    "candidate_id": candidate,
                    "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "training_run_contract": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "cagm_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                    "cagm_enabled": arm == "G",
                    "cagm_source_labeled_indices_sha256": receipt["source_labeled_indices_sha256"],
                    "cagm_source_split_manifest_sha256": receipt["source_split_manifest_sha256"],
                    "cagm_receipt_sha256": raw_receipt_sha,
                    "cagm_terminal_contract": checked["cagm_terminal_contract"],
                    "cagm_terminal_contract_passed": True,
                    "cagm_loss_divisor": 10,
                    "cagm_clean_statistics_detached": True,
                    "cagm_joint_zero_mask_aux_only": receipt[
                        "joint_zero_mask_aux_only"
                    ],
                    "proxy_selection_frozen_not_cli_tunable": True,
                }
            )

        def leo_mutate(manifest: dict[str, object]) -> None:
            manifest["checkpoint"] = str(checkpoint.resolve())
            manifest["source_checkpoint_sha256"] = checkpoint_sha

        _rewrite_npz_manifest(clean, clean_mutate)
        _rewrite_npz_manifest(leo, leo_mutate)
        fixture._proxy_metrics(proxy, scores, clean)
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
                "method": "P1_CAGM",
                "candidate_id": candidate,
                "training_run_root": str(training_root.resolve()),
                "postfreeze_output_root": str(root.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "training_run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                "leo_npz_path": str(leo.resolve()),
                "leo_npz_sha256": hashlib.sha256(leo.read_bytes()).hexdigest(),
                "leo_manifest_sha256": fixture.PAIR._icmt_leo._canonical_json_sha256(leo_manifest),
                "cagm_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                "cagm_receipt_sha256": raw_receipt_sha,
                "cagm_terminal_contract": checked["cagm_terminal_contract"],
                "cagm_terminal_contract_passed": True,
                "cagm_loss_divisor": 10,
                "cagm_clean_statistics_detached": True,
                "cagm_joint_zero_mask_aux_only": receipt[
                    "joint_zero_mask_aux_only"
                ],
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


def test_cagm_pair_closes_raw_checkpoint_clean_leo_proxy_and_receipt_chain(tmp_path):
    paths = _build_cagm_pair_fixture(tmp_path)
    metrics = PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair.json"))
    assert metrics["schema"] == PAIR.EXPECTED_PAIR_SCHEMA
    assert metrics["cagm_training_receipt_revalidation"]["C"]["candidate"] == "F1C_CAGM12"
    assert metrics["cagm_training_receipt_revalidation"]["G"]["terminal_contract_passed"] is True
    assert metrics["policy"]["geometry_fit_role"] == "labeled_fit"
    assert metrics["policy"]["proxy_unknown_fit_rows"] == 0
    common = metrics["cagm_common_training_binding"]
    assert set(common) == {"passed", "fields", "sha256"}
    assert common["passed"] is True
    assert set(common["fields"]) == set(PAIR.COMMON_TRAINING_BINDING_FIELDS)
    assert common["fields"]["optimizer_type"] == "AdamW"
    assert metrics["cagm_training_receipt_revalidation"]["C"][
        "joint_zero_mask_aux_only"
    ] is False
    assert metrics["cagm_training_receipt_revalidation"]["G"][
        "joint_zero_mask_aux_only"
    ] is True


def test_raw_joint_mask_cannot_be_bypassed_by_manifest_true(tmp_path):
    paths = _build_cagm_pair_fixture(tmp_path)
    g_clean = Path(paths["g_clean"])
    with np.load(g_clean, allow_pickle=False) as data:
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
    assert manifest["cagm_joint_zero_mask_aux_only"] is True
    checkpoint = Path(paths["g_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    payload["cagm_receipt"]["joint_zero_mask_aux_only"] = False
    torch.save(payload, checkpoint)
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="joint_zero_mask_aux_only"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair.json"))


def test_clean_and_leo_receipts_copy_proven_arm_specific_joint_mask(tmp_path):
    paths = _build_cagm_pair_fixture(tmp_path)
    for prefix, expected in (("c", False), ("g", True)):
        clean = Path(paths[f"{prefix}_clean"])
        with np.load(clean, allow_pickle=False) as data:
            manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        binding = json.loads(Path(paths[f"{prefix}_binding"]).read_text(encoding="utf-8"))
        assert manifest["cagm_joint_zero_mask_aux_only"] is expected
        assert binding["cagm_joint_zero_mask_aux_only"] is expected


def test_cagm_pair_rejects_synchronized_proxy_count_manifest_tamper(tmp_path):
    paths = _build_cagm_pair_fixture(tmp_path)
    for prefix in ("c", "g"):
        clean = Path(paths[f"{prefix}_clean"])
        _rewrite_npz_manifest(clean, lambda manifest: manifest.__setitem__("proxy_row_count", 399))
        proxy = Path(paths[f"{prefix}_proxy"])
        raw = json.loads(proxy.read_text(encoding="utf-8"))
        raw["manifest"]["proxy_row_count"] = 399
        proxy.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="proxy"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair.json"))


def test_cagm_f6_reloads_all_five_prior_raw_artifact_chains(tmp_path):
    root = tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    priors: list[Path] = []
    final_metrics = None
    for fold in range(1, 7):
        paths = _build_cagm_pair_fixture(tmp_path, fold=fold, root=root)
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


def test_cagm_f6_prior_raw_recompute_rejects_g_sequence_binding_tamper(tmp_path):
    paths = _build_cagm_pair_fixture(tmp_path)
    output = Path(paths["root"]) / "F1_C_vs_G_pair_metrics.json"
    PAIR.evaluate(_pair_args(paths, output))
    record = json.loads(output.read_text(encoding="utf-8"))
    checkpoint = Path(paths["g_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    payload["cagm_receipt"]["common_batch_sequence_sha256"] = _sha(
        "tampered-G-sequence"
    )
    torch.save(payload, checkpoint)
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="common training binding"):
        PAIR._recompute_cagm_prior_pair_artifacts(
            record,
            output_root=Path(paths["root"]),
            matrix_id=PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
            training_root=Path(paths["training_root"]),
            expected_scenarios=PAIR._icmt.EXPECTED_SCENARIOS,
        )


def test_cagm_f6_prior_recompute_rejects_self_reported_common_binding_tamper(tmp_path):
    paths = _build_cagm_pair_fixture(tmp_path)
    output = Path(paths["root"]) / "F1_C_vs_G_pair_metrics.json"
    PAIR.evaluate(_pair_args(paths, output))
    record = json.loads(output.read_text(encoding="utf-8"))
    record["cagm_common_training_binding"]["fields"][
        "common_batch_sequence_sha256"
    ] = _sha("self-reported-sequence")
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="common training binding"):
        PAIR._recompute_cagm_prior_pair_artifacts(
            record,
            output_root=Path(paths["root"]),
            matrix_id=PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
            training_root=Path(paths["training_root"]),
            expected_scenarios=PAIR._icmt.EXPECTED_SCENARIOS,
        )


def test_cagm_f6_rejects_prior_one_row_feature_change_after_sha_sync(tmp_path):
    root = tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    priors: list[Path] = []
    paths_by_fold: dict[int, dict[str, object]] = {}
    for fold in range(1, 6):
        paths = _build_cagm_pair_fixture(tmp_path, fold=fold, root=root)
        paths_by_fold[fold] = paths
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_pair_args(paths, output, fold=fold))
        priors.append(output)
    prior_clean = Path(paths_by_fold[1]["c_clean"])
    with np.load(prior_clean, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]).copy() for name in data.files}
    proxy_index = int(np.flatnonzero(np.asarray(payload["dataset_role"]).astype(str) == "proxy_unknown")[0])
    payload["features"][proxy_index] = np.zeros(4, dtype=np.float32)
    np.savez(prior_clean, **payload)
    prior_record = json.loads(priors[0].read_text(encoding="utf-8"))
    prior_record["bindings"]["c_clean_npz_sha256"] = hashlib.sha256(prior_clean.read_bytes()).hexdigest()
    priors[0].write_text(json.dumps(prior_record, sort_keys=True), encoding="utf-8")
    f6_paths = _build_cagm_pair_fixture(tmp_path, fold=6, root=root)
    with pytest.raises(PAIR.CAGMPostfreezePairError, match="raw-artifact recomputation|does not match"):
        PAIR.evaluate(
            _pair_args(
                f6_paths,
                root / "F6_C_vs_G_pair_metrics.json",
                fold=6,
                priors=tuple(priors),
            )
        )
