from __future__ import annotations

"""Mechanical postfreeze closure tests for the frozen P1-RCMMC pair.

The tests intentionally reuse the existing ICMT synthetic rows only as a
data-free fixture.  Every persisted manifest/checkpoint/binding is rewritten
to the RCMMC identity before the pair evaluator sees it; no training or
performance data are read.
"""

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

import evaluate_phase1_rcmmc_postfreeze_pair as PAIR
import export_phase1_rcmmc_features as EXPORT
from cvsrffi import phase1_rcmmc as RCMMC


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_rcmmc_postfreeze_20260811.sh"
FOLD = 1
TX_IDS = tuple(PAIR._icmt.FROZEN_FOLD_SOURCE_TX[FOLD])
KNOWN_TX = (PAIR._icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[FOLD],)
PROXY_TX = (PAIR._icmt.FROZEN_FOLD_PROXY_TX[FOLD],)
RX_IDS = tuple(
    int(value)
    for value in getattr(
        RCMMC,
        "FROZEN_RCMMC_SOURCE_RECEIVER_IDS",
        range(int(RCMMC.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT)),
    )
)
SCENARIOS = tuple(RCMMC.FROZEN_RCMMC_SCENARIOS)
CELL_COUNT = int(RCMMC.FROZEN_RCMMC_CELL_COUNT)


def _load_core_fixture_module():
    path = CODE_ROOT / "tests" / "test_phase1_rcmmc.py"
    spec = importlib.util.spec_from_file_location("_rcmmc_core_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_icmt_fixture_module():
    path = CODE_ROOT / "tests" / "test_phase1_icmt_postfreeze.py"
    spec = importlib.util.spec_from_file_location("_icmt_fixture_for_rcmmc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CORE = _load_core_fixture_module()


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    fn = getattr(EXPORT, "_canonical_json_sha256", None)
    if fn is None:
        fn = getattr(RCMMC, "_canonical_sha256")
    return str(fn(value))


def _source_receiver_sha() -> str:
    return _canonical(list(RX_IDS))


def _receipt(
    arm: str,
    *,
    labeled_sha: str | None = None,
    split_sha: str | None = None,
    source_tx_ids: tuple[str, ...] = TX_IDS,
    known_tx_ids: tuple[str, ...] = KNOWN_TX,
    proxy_tx_ids: tuple[str, ...] = PROXY_TX,
) -> dict[str, object]:
    """Build a valid raw terminal receipt through the existing RCMMC core."""

    enabled = arm == "G"
    receipt = dict(_CORE._sealed_receipt(enabled=enabled))
    # The terminal RCMMC receipt is scalar/count/SHA-only: source receiver
    # tokens may be consumed by the split receipt but never persisted here.
    receipt.pop("source_receiver_ids", None)
    receipt.pop("frozen_source_receiver_ids", None)
    receipt.update(
        {
            "method": "P1_RCMMC",
            "baseline_sha256": _sha("rcmmc-baseline"),
            "initial_checkpoint_sha256": _sha("rcmmc-initial"),
            "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
            "baseline_path": "geosat_c_final.pth",
            "checkpoint_epoch": 40,
            "strict_model_keys": True,
            "missing_model_keys": [],
            "unexpected_model_keys": [],
            "class_order_binding_sha256": _sha("rcmmc-class-order"),
            "source_partition_sha256": _sha("rcmmc-partition"),
            "source_labeled_indices_sha256": labeled_sha or _sha("rcmmc-labeled"),
            "source_split_manifest_sha256": split_sha or _sha("rcmmc-split"),
            "source_train_tx": list(source_tx_ids),
            "source_known_validation_tx": list(known_tx_ids),
            "source_proxy_unknown_tx": list(proxy_tx_ids),
            "source_train_tx_count": len(source_tx_ids),
            "source_known_validation_tx_count": len(known_tx_ids),
            "source_proxy_unknown_tx_count": len(proxy_tx_ids),
            "local_tx_class_order": list(source_tx_ids),
            "checkpoint_train_tx_class_order": list(source_tx_ids),
            "dataset_tx_class_order": list(source_tx_ids + known_tx_ids + proxy_tx_ids),
            "local_to_dataset_class_ids": [0, 1, 2, 3],
            "local_to_head_class_ids": [0, 1, 2, 3],
            "expected_tx_class_ids": [0, 1, 2, 3],
            "dataset_class_count": 6,
            "local_data_class_count": 4,
            "checkpoint_head_class_count": 4,
            "live_head_class_count": 4,
            "source_receiver_count": len(RX_IDS),
            "frozen_source_receiver_count": len(RX_IDS),
            "frozen_cells_per_scene": CELL_COUNT,
            "source_receiver_ids_sha256": _source_receiver_sha(),
            "source_receiver_order_sha256": _source_receiver_sha(),
            "source_receiver_provenance": getattr(
                EXPORT,
                "SOURCE_RECEIVER_PROVENANCE",
                "SOURCE_SPLIT_RECEIPT_ORDERED_SOURCE_RECEIVERS_PHYSICAL_ID_BOUND_L_ONLY",
            ),
            "optimizer_type": "AdamW",
            "optimizer_initial_state_sha256": _sha("rcmmc-optimizer"),
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
            "checkpoint_role": "training_final_only",
            "common_l_base_head_input_path_verified": True,
        }
    )
    if not enabled:
        for index, scenario in enumerate(SCENARIOS, start=1):
            receipt = _CORE._common_bind(receipt, epoch=1, batch_index=index, scenario=scenario)
        return receipt

    # Build one legal 128×160 batch and reuse the scalar receipt across the
    # three frozen scene cycles.  This is a synthetic receipt fixture only.
    model = _CORE._BindingModel(dim=int(RCMMC.FROZEN_RCMMC_FEATURE_DIM)).train()
    labels, receivers = _CORE._batch_cells()
    clean_out = model.paired_output(torch.randn(labels.numel(), int(RCMMC.FROZEN_RCMMC_FEATURE_DIM)))
    leo_out = model.paired_output(torch.randn(labels.numel(), int(RCMMC.FROZEN_RCMMC_FEATURE_DIM)))
    loss, info = _CORE._loss_and_info(clean_out["z_id"], leo_out["z_id"], labels, receivers)
    audit = RCMMC.rcmmc_aux_gradient_audit(
        loss,
        clean_out["z_id"],
        leo_out["z_id"],
        RCMMC.rcmmc_shared_encoder_and_head_parameters(model),
    )
    for index, scenario in enumerate(SCENARIOS, start=1):
        receipt = _CORE._common_bind(receipt, epoch=1, batch_index=index, scenario=scenario)
        receipt = RCMMC.update_rcmmc_receipt(receipt, info, scenario=scenario, epoch=1, batch_index=index)
    receipt = RCMMC.update_rcmmc_gradient_audit_receipt(receipt, audit)
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
    candidate = f"F{fold}{arm}_RCMMC12"
    path.parent.mkdir(parents=True, exist_ok=True)
    effective_receipt = receipt if receipt is not None else _receipt(
        arm,
        source_tx_ids=source_tx_ids,
        known_tx_ids=known_tx_ids,
        proxy_tx_ids=proxy_tx_ids,
    )
    payload = {
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "candidate_id": candidate,
        "run_id": getattr(PAIR, "EXPECTED_TRAINING_RUN_LEAF", "phase1_rcmmc12_20260811_v1"),
        "model": {},
        "args": {
            "split_mode": "tx_rx_day_1_6_3",
            "model_variant": "lite_d",
            "id_feature_key": "feat_joint",
            "phase1_source_train_tx_ids": ",".join(source_tx_ids),
            "phase1_source_known_validation_tx_ids": ",".join(known_tx_ids),
            "phase1_source_proxy_unknown_tx_ids": ",".join(proxy_tx_ids),
            "checkpoint_selection": "final_only",
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
            "seed": 7281105,
            "phase1_rcmmc_frozen_mode": True,
            "phase1_rcmmc_enabled": arm == "G",
            "lambda_rcmmc": 0.02 if arm == "G" else 0.0,
            # The checkpoint validator intentionally re-runs the frozen core
            # argument contract; keep this fixture's namespace complete rather
            # than weakening that scientific validator for synthetic tests.
            "batch_size": 128,
            "from_scratch": False,
            "baseline_ckpt": "geosat_c_final.pth",
            "freeze_backbone": False,
            "amp": True,
            "epochs": 40,
            "label_epochs": 40,
            "pseudo_epochs": 0,
            "phase1_source_val_selection_only": True,
            "use_sat_consistency": True,
            "lambda_sat_cons": 0.10,
            "lambda_sat_cls": 0.0,
            "sat_cons_start_epoch": 1,
            "sat_view_prob": 1.0,
            "sat_train_scenarios": ",".join(SCENARIOS),
            "sat_view_schedule": "",
            "use_concat_sat_channel_aug": False,
            "use_unlabeled": False,
            "use_tx_rx_balanced_sampler": False,
            "use_aug": False,
            "use_mixstyle": False,
            "reject_head": False,
            "use_ema_teacher": False,
            "teacher_ckpt": "",
            "lambda_teacher_clean_kl": 0.0,
            "lambda_teacher_sat_kl": 0.0,
            "lambda_teacher_zid_mse": 0.0,
            "candidate_id": candidate,
            "run_id": getattr(PAIR, "EXPECTED_TRAINING_RUN_LEAF", "phase1_rcmmc12_20260811_v1"),
        },
        "rcmmc_receipt": effective_receipt,
        "split_info": {
            "source_split_receipt": {
                "schema": "cvs.phase1.source_split_receipt.v1",
                "source_receivers": list(RX_IDS),
                "labeled_indices_sha256": str(effective_receipt.get("source_labeled_indices_sha256", "")),
                "split_manifest_sha256": str(effective_receipt.get("source_split_manifest_sha256", "")),
            },
            "tx_partition_receipt": {
                "partition_sha256": str(effective_receipt.get("source_partition_sha256", "")),
            },
        },
    }
    torch.save(payload, path)
    return path


def _rewrite_npz_manifest(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]).copy() for name in data.files}
    manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
    mutate(manifest)
    payload["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    np.savez(path, **payload)


def _write_frozen_proxy_logits(clean: Path, proxy: Path, scores: Path, source_tx_ids: tuple[str, ...]) -> None:
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


def _build_rcmmc_pair_fixture(
    tmp_path: Path, *, fold: int = FOLD, root: Path | None = None
) -> dict[str, object]:
    """Reuse the signed synthetic rows while replacing every identity with RCMMC."""

    fixture = _load_icmt_fixture_module()
    root = root or (tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    fixture._write_pair(root, fold=fold)
    training_root = root.parent / PAIR.EXPECTED_TRAINING_RUN_LEAF
    source_tx_ids = tuple(PAIR._icmt.FROZEN_FOLD_SOURCE_TX[fold])
    known_tx_ids = (PAIR._icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold],)
    proxy_tx_ids = (PAIR._icmt.FROZEN_FOLD_PROXY_TX[fold],)
    paths: dict[str, object] = {
        "root": root.resolve(),
        "training_root": training_root.resolve(),
        "source_tx_ids": source_tx_ids,
    }
    clean_artifact = getattr(PAIR, "EXPECTED_CLEAN_ARTIFACT", "icmt_clean_l_v_proxy_final_only.npz")
    for arm in ("C", "G"):
        old_candidate = f"F{fold}{arm}_ICMT12"
        candidate = f"F{fold}{arm}_RCMMC12"
        old_dir = root / old_candidate
        new_dir = root / candidate
        shutil.move(str(old_dir), str(new_dir))
        old_train = training_root / old_candidate
        new_train = training_root / candidate
        if old_train.exists():
            shutil.move(str(old_train), str(new_train))
        clean = new_dir / clean_artifact
        leo = new_dir / "source_leo_final_only.npz"
        proxy = new_dir / "proxy_logits_open_set_metrics.json"
        scores = new_dir / "proxy_logits_open_set_scores.csv"
        binding = new_dir / "source_leo_binding.json"
        with np.load(clean, allow_pickle=False) as data:
            clean_manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        receipt = _receipt(
            arm,
            labeled_sha=str(clean_manifest.get("labeled_indices_sha256", _sha("labeled"))),
            split_sha=str(
                dict(clean_manifest.get("source_split_receipt", {})).get(
                    "split_manifest_sha256", _sha("split")
                )
            ),
            source_tx_ids=source_tx_ids,
            known_tx_ids=known_tx_ids,
            proxy_tx_ids=proxy_tx_ids,
        )
        checkpoint = _checkpoint(
            new_train / "final_ssdg.pth",
            arm,
            receipt,
            fold=fold,
            source_tx_ids=source_tx_ids,
            known_tx_ids=known_tx_ids,
            proxy_tx_ids=proxy_tx_ids,
        )
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checked = EXPORT.validate_rcmmc_terminal_receipt(
            receipt,
            arm=arm,
        )
        raw_receipt_sha = _canonical(receipt)

        def clean_mutate(manifest: dict[str, object]) -> None:
            for field in tuple(manifest):
                if str(field).lower().startswith(tuple(getattr(PAIR, "LEGACY_IDENTITY_PREFIXES", ("icmt_", "rcrmd_", "rcat_", "recte_", "hscf_")))):
                    manifest.pop(field, None)
            manifest.update(
                {
                    "schema": PAIR.EXPECTED_LV_EXPORT_SCHEMA,
                    "method": "P1_RCMMC",
                    "checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": checkpoint_sha,
                    "candidate_id": candidate,
                    "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "training_run_contract": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "rcmmc_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                    "rcmmc_enabled": arm == "G",
                    "rcmmc_receipt_sha256": raw_receipt_sha,
                    "rcmmc_terminal_contract": checked["rcmmc_terminal_contract"],
                    "rcmmc_terminal_contract_passed": True,
                    "rcmmc_lambda": RCMMC.FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0,
                    "rcmmc_frozen_batch_size": RCMMC.FROZEN_RCMMC_BATCH_SIZE,
                    "rcmmc_feature_dim": RCMMC.FROZEN_RCMMC_FEATURE_DIM,
                    "rcmmc_local_class_count": len(RCMMC.FROZEN_RCMMC_CLASS_IDS),
                    "rcmmc_loss_global_denominator": RCMMC.FROZEN_RCMMC_TERM_DIVISOR,
                    "rcmmc_fixed_batch_size": RCMMC.FROZEN_RCMMC_BATCH_SIZE,
                    "rcmmc_fixed_feature_dim": RCMMC.FROZEN_RCMMC_FEATURE_DIM,
                    "rcmmc_fixed_local_class_count": len(RCMMC.FROZEN_RCMMC_CLASS_IDS),
                    "rcmmc_fixed_cells_per_scene": RCMMC.FROZEN_RCMMC_TERM_DIVISOR,
                    "rcmmc_source_receiver_count": len(RX_IDS),
                    "rcmmc_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
                    "rcmmc_source_receiver_ids_sha256": _source_receiver_sha(),
                    "rcmmc_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
                    "rcmmc_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
                    "rcmmc_source_partition_sha256": str(receipt["source_partition_sha256"]),
                    "rcmmc_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
                    "rcmmc_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
                    "rcmmc_common_scenario_batches": {
                        str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()
                    },
                    "rcmmc_common_cells_sha256": _canonical(receipt.get("rcmmc_common_cells", {})),
                    "rcmmc_g_scenes_sha256": _canonical(receipt.get("rcmmc_scenes", {})) if arm == "G" else "",
                    "rcmmc_clean_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
                    "rcmmc_leo_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
                    "rcmmc_common_physical_order_bound": True,
                    "rcmmc_common_scene_cycle_bound": True,
                    "rcmmc_raw_vjp_required": True,
                    "rcmmc_leo_encoder_vjp_finite_nonzero": True,
                    "rcmmc_clean_head_vjp_na_none_or_zero": True,
                    "proxy_selection_frozen_not_cli_tunable": True,
                }
            )

        _rewrite_npz_manifest(clean, clean_mutate)
        _rewrite_npz_manifest(
            leo,
            lambda manifest: manifest.update(
                {
                    "schema": PAIR.EXPECTED_LV_EXPORT_SCHEMA,
                    "method": "P1_RCMMC",
                    "checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": checkpoint_sha,
                    "candidate_id": candidate,
                    "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
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
                "method": "P1_RCMMC",
                "candidate_id": candidate,
                "training_run_root": str(training_root.resolve()),
                "postfreeze_output_root": str(root.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "training_run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                "leo_npz_path": str(leo.resolve()),
                "leo_npz_sha256": hashlib.sha256(leo.read_bytes()).hexdigest(),
                "leo_manifest_sha256": fixture.PAIR._icmt_leo._canonical_json_sha256(leo_manifest),
                "rcmmc_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                "rcmmc_receipt_sha256": raw_receipt_sha,
                "rcmmc_terminal_contract": checked["rcmmc_terminal_contract"],
                "rcmmc_terminal_contract_passed": True,
                "rcmmc_enabled": arm == "G",
                "rcmmc_lambda": RCMMC.FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0,
                    "rcmmc_frozen_batch_size": RCMMC.FROZEN_RCMMC_BATCH_SIZE,
                "rcmmc_feature_dim": RCMMC.FROZEN_RCMMC_FEATURE_DIM,
                "rcmmc_local_class_count": len(RCMMC.FROZEN_RCMMC_CLASS_IDS),
                "rcmmc_loss_global_denominator": RCMMC.FROZEN_RCMMC_TERM_DIVISOR,
                "rcmmc_fixed_batch_size": RCMMC.FROZEN_RCMMC_BATCH_SIZE,
                "rcmmc_fixed_feature_dim": RCMMC.FROZEN_RCMMC_FEATURE_DIM,
                "rcmmc_fixed_local_class_count": len(RCMMC.FROZEN_RCMMC_CLASS_IDS),
                "rcmmc_fixed_cells_per_scene": RCMMC.FROZEN_RCMMC_TERM_DIVISOR,
                "rcmmc_source_receiver_count": len(RX_IDS),
                "rcmmc_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
                "rcmmc_source_receiver_ids_sha256": _source_receiver_sha(),
                "rcmmc_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
                "rcmmc_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
                "rcmmc_source_partition_sha256": str(receipt["source_partition_sha256"]),
                "rcmmc_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
                "rcmmc_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
                "rcmmc_common_scenario_batches": {
                    str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()
                },
                "rcmmc_common_cells_sha256": _canonical(receipt.get("rcmmc_common_cells", {})),
                "rcmmc_g_scenes_sha256": _canonical(receipt.get("rcmmc_scenes", {})) if arm == "G" else "",
                "rcmmc_clean_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
                "rcmmc_leo_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
                "rcmmc_single_leo_forward_bound": True,
                "rcmmc_physical_tx_rx_day_binding_required": True,
                "rcmmc_common_physical_order_bound": True,
                "rcmmc_common_scene_cycle_bound": True,
                "rcmmc_raw_vjp_required": True,
                "rcmmc_leo_encoder_vjp_finite_nonzero": True,
                "rcmmc_clean_head_vjp_na_none_or_zero": True,
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


def _pair_args(paths: dict[str, object], output: Path, *, fold: int = 1, priors: tuple[Path, ...] = ()):
    values = [
        "--c-clean-npz", str(paths["c_clean"]), "--g-clean-npz", str(paths["g_clean"]),
        "--c-leo-npz", str(paths["c_leo"]), "--g-leo-npz", str(paths["g_leo"]),
        "--c-leo-binding-json", str(paths["c_binding"]), "--g-leo-binding-json", str(paths["g_binding"]),
        "--c-final-checkpoint", str(paths["c_checkpoint"]), "--g-final-checkpoint", str(paths["g_checkpoint"]),
        "--c-proxy-metrics-json", str(paths["c_proxy"]), "--g-proxy-metrics-json", str(paths["g_proxy"]),
        "--c-proxy-scores-csv", str(paths["c_scores"]), "--g-proxy-scores-csv", str(paths["g_scores"]),
        "--source-tx-ids", ",".join(paths["source_tx_ids"]), "--candidate-pair", f"F{fold}_C_vs_G",
        "--fold-index", str(fold), "--postfreeze-matrix-id", PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
        "--postfreeze-output-root", str(paths["root"]), "--training-run-root", str(paths["training_root"]),
        "--expected-scenarios", ",".join(SCENARIOS), "--expected-source-days", ",".join(PAIR._icmt.EXPECTED_SOURCE_DAYS),
        "--expected-source-rxs", ",".join(PAIR._icmt.EXPECTED_SOURCE_RXS), "--source-sat-seed", "7281718",
        "--expected-source-count", "72", "--expected-proxy-count", "400", "--output-metrics-json", str(output),
    ]
    if priors:
        values += ["--aggregate-prior-pair-metrics-json", ",".join(str(path) for path in priors)]
    return PAIR.build_parser().parse_args(values)


def test_rcmmc_receipt_b128_d160_local4_fixed28_source_receiver_sha_count_and_c_g_scene_contract() -> None:
    for arm in ("C", "G"):
        checked = EXPORT.validate_rcmmc_terminal_receipt(
            _receipt(arm), arm=arm,
        )
        assert checked["schema"] == EXPORT.EXPECTED_RECEIPT_SCHEMA
        assert checked["frozen_batch_size"] == 128
        assert checked["frozen_feature_dim"] == 160
        assert checked["local_class_count"] == 4
        assert checked["frozen_cells_per_scene"] == 28
        assert checked["source_receiver_count"] == 7
        assert checked["source_receiver_ids_sha256"] == _source_receiver_sha()
        assert "source_receiver_ids" not in checked
        assert "frozen_source_receiver_ids" not in checked
        assert checked["rcmmc_terminal_contract_passed"] is True
        if arm == "C":
            assert int(checked["rcmmc_batches"]) == 0
            assert float(checked["rcmmc_loss_sum"]) == 0.0
            assert not checked.get("rcmmc_scenes")
            assert not checked.get("rcmmc_gradient_audit")
        else:
            assert set(checked["rcmmc_scenes"]) == set(SCENARIOS)
            assert all(len(checked["rcmmc_scenes"][scene]) == 84 // 3 for scene in SCENARIOS)
            assert all(checked["rcmmc_scene_positive_batches"][scene] > 0 for scene in SCENARIOS)
            audit = checked["rcmmc_gradient_audit"]
            for group in ("feat_joint_leo", "shared_encoder"):
                assert math.isfinite(float(audit[group]["norm"])) and float(audit[group]["norm"]) > 0.0
            assert float(audit["clean_feat_joint"]["nonzero_parameters"]) == 0.0
            assert float(audit["classifier_head"]["nonzero_parameters"]) == 0.0


@pytest.mark.parametrize("raw_field", ["source_receiver_ids", "frozen_source_receiver_ids"])
def test_rcmmc_terminal_receipt_rejects_nested_raw_receiver_tokens(raw_field: str) -> None:
    receipt = _receipt("G")
    receipt["nested_unrecognized"] = {"deep": {raw_field: ["opaque-rs0", "opaque-rs1"]}}
    with pytest.raises(EXPORT.RCMMCSplitExportError, match="raw source receiver token"):
        EXPORT.validate_rcmmc_terminal_receipt(receipt, arm="G")


def test_rcmmc_same_fold_common_binding_is_strict_but_g_only_evidence_is_not_compared() -> None:
    c_receipt, g_receipt = _receipt("C"), _receipt("G")
    binding = PAIR.validate_rcmmc_common_training_binding(c_receipt, g_receipt)
    assert binding["passed"] is True
    assert set(binding["fields"]) == set(PAIR.COMMON_TRAINING_BINDING_FIELDS)
    g_receipt["rcmmc_loss_sum"] = 99.0
    assert PAIR.validate_rcmmc_common_training_binding(c_receipt, g_receipt)["passed"] is True
    g_receipt["common_batch_sequence_sha256"] = _sha("tampered-sequence")
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="common training binding"):
        PAIR.validate_rcmmc_common_training_binding(c_receipt, g_receipt)


def test_rcmmc_safe_zero_nonfinite_and_l_only_float64_gaussian_are_fail_closed() -> None:
    features = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = PAIR.safe_totalized_l2_float64(features, label="fixture")
    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized, np.asarray([[0.6, 0.8], [0.0, 0.0]], dtype=np.float64))
    tx = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    geometry = PAIR.fit_frozen_rcmmc_diagonal_gaussian(np.vstack((np.eye(4), np.eye(4))), tx, ("a", "b", "c", "d"))
    scores = PAIR.score_frozen_rcmmc_nll(np.zeros((1, 4), dtype=np.float64), geometry)
    assert scores.shape == (1,) and np.isfinite(scores).all()
    with pytest.raises(Exception, match="non-finite"):
        PAIR.safe_totalized_l2_float64(np.asarray([[np.nan, 1.0]]), label="bad")


def test_rcmmc_launcher_is_exactly_42_steps_and_dry_run_is_frozen() -> None:
    relative = f"scripts/{LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative], cwd=CODE_ROOT, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative, "--dry-run"], cwd=CODE_ROOT, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if line.startswith("[DRY-RUN]")]
    assert len(lines) == 42
    assert sum("RCMMC_CLEAN_EXPORT" in line for line in lines) == 12
    assert sum("RCMMC_LEO_EXPORT_AND_BIND" in line for line in lines) == 12
    assert sum("FROZEN_LOGITS_PROXY_BINDING" in line for line in lines) == 12
    assert sum("RCMMC_PAIR_SCORE" in line for line in lines) == 6
    assert sum("phase1_rcmmc12_20260811_v1" in line for line in lines) == 30
    assert all("phase1_rcmmc_postfreeze_20260811_v1" in line for line in lines)
    assert all("_RCMMC12" in line for line in lines)
    assert all(not any(old in line for old in ("_ICMT12", "_RCAT12", "_RCRMD12", "_RECTE12", "_HSCF12")) for line in lines)
    expected_gpu = ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    candidate_lines = [line for line in lines if "RCMMC_CLEAN_EXPORT" in line]
    assert [next(token.split("=")[-1] for token in line.split() if token.startswith("CUDA_VISIBLE_DEVICES=")) for line in candidate_lines] == expected_gpu


def test_rcmmc_pair_closes_raw_checkpoint_clean_leo_proxy_and_common_binding(tmp_path: Path) -> None:
    paths = _build_rcmmc_pair_fixture(tmp_path)
    metrics = PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair.json"))
    assert metrics["schema"] == PAIR.EXPECTED_PAIR_SCHEMA
    assert metrics["rcmmc_training_receipt_revalidation"]["C"]["candidate"] == "F1C_RCMMC12"
    assert metrics["rcmmc_training_receipt_revalidation"]["G"]["terminal_contract_passed"] is True
    assert metrics["policy"]["geometry_fit_role"] == "labeled_fit"
    assert metrics["policy"]["proxy_unknown_fit_rows"] == 0
    assert metrics["rcmmc_common_training_binding"]["passed"] is True
    assert metrics["verdict"].startswith(("REJECT", "PENDING_MAIN"))


def test_rcmmc_leo_binding_closes_manysig_physical_tx_rx_day_and_three_scenes(tmp_path: Path) -> None:
    paths = _build_rcmmc_pair_fixture(tmp_path)
    for arm in ("c", "g"):
        binding = json.loads(Path(paths[f"{arm}_binding"]).read_text(encoding="utf-8"))
        assert binding["schema"] == PAIR.EXPECTED_LEO_BINDING_SCHEMA
        assert binding["method"] == "P1_RCMMC"
        assert binding["dataset_sha256"] == PAIR._icmt.FROZEN_WISIG_SHA256
        selection = binding["source_selection"]
        assert tuple(selection["source_tx_ids"]) == TX_IDS
        assert tuple(selection["source_rx_ids"]) == tuple(PAIR._icmt.EXPECTED_SOURCE_RXS)
        assert tuple(selection["source_day_ids"]) == tuple(PAIR._icmt.EXPECTED_SOURCE_DAYS)
        assert selection["source_sat_seed"] == 7281718
        with np.load(paths[f"{arm}_leo"], allow_pickle=False) as data:
            tx = np.asarray(data["tx_ids"]).reshape(-1).astype(str)
            rx = np.asarray(data["rx_ids"]).reshape(-1).astype(str)
            day = np.asarray(data["day_ids"]).reshape(-1).astype(str)
            sig = np.asarray(data["sig_ids"]).reshape(-1).astype(str)
            scenes = np.asarray(data["sat_scenarios"]).reshape(-1).astype(str)
        physical = {"\x1f".join((tx[i], rx[i], day[i], sig[i])) for i in range(tx.size)}
        assert len(physical) == tx.size
        assert set(tx) == set(TX_IDS)
        assert set(rx) == set(PAIR._icmt.EXPECTED_SOURCE_RXS)
        assert set(day) == set(PAIR._icmt.EXPECTED_SOURCE_DAYS)
        assert set(scenes) == set(SCENARIOS)
        for scene in SCENARIOS:
            mask = scenes == scene
            assert int(mask.sum()) == 24
            assert len({"\x1f".join((tx[i], rx[i], day[i], sig[i])) for i in np.flatnonzero(mask)}) == 24


def test_rcmmc_four_floor_overall_and_proxy_double_gate_are_noncompensating(tmp_path: Path) -> None:
    paths = _build_rcmmc_pair_fixture(tmp_path)
    metrics = PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "gates.json"))
    gates = metrics["postfreeze_gates"]
    floor_names = {"overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy"}
    assert set(gates["clean_four_floors_ge_minus2pp"]["metric_passes"]) == floor_names
    assert set(gates["leo_scenario_four_floors_ge_minus2pp"]["by_scenario"]) == set(SCENARIOS)
    proxy = gates["proxy_continuous_two_strict_improvements"]
    assert proxy["passed"] is (proxy["strict_AUROC_improvement"] and proxy["strict_proxy_known_gap_improvement"])
    assert proxy["diagnostic_only_non_compensating"] is True
    assert "non-compensating" in str(PAIR.FROZEN_POSTFREEZE_CONTRACT)


@pytest.mark.parametrize("legacy_field", ["icmt_receipt_schema", "rcat_enabled", "rcrmd_source_split_manifest_sha256"])
def test_rcmmc_rejects_legacy_method_identity_even_when_proxy_is_recomputed(tmp_path: Path, legacy_field: str) -> None:
    paths = _build_rcmmc_pair_fixture(tmp_path)
    _rewrite_npz_manifest(Path(paths["c_clean"]), lambda manifest: manifest.__setitem__(legacy_field, True))
    _write_frozen_proxy_logits(Path(paths["c_clean"]), Path(paths["c_proxy"]), Path(paths["c_scores"]), TX_IDS)
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="historical method identity"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "legacy.json"))


@pytest.mark.parametrize("raw_field", ["source_receiver_ids", "frozen_source_receiver_ids"])
@pytest.mark.parametrize("artifact", ["clean_manifest", "leo_binding"])
def test_rcmmc_rejects_raw_receiver_tokens_recursively(
    tmp_path: Path, raw_field: str, artifact: str,
) -> None:
    paths = _build_rcmmc_pair_fixture(
        tmp_path, root=tmp_path / f"{artifact}_{raw_field}" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID,
    )
    if artifact == "clean_manifest":
        _rewrite_npz_manifest(
            Path(paths["c_clean"]),
            lambda manifest: manifest.__setitem__(raw_field, ["opaque-rs0", "opaque-rs1"]),
        )
        _write_frozen_proxy_logits(
            Path(paths["c_clean"]), Path(paths["c_proxy"]), Path(paths["c_scores"]), TX_IDS,
        )
    else:
        binding_path = Path(paths["g_binding"])
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding[raw_field] = ["opaque-rs0", "opaque-rs1"]
        binding_path.write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="raw source receiver token"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "raw_receiver.json"))


def test_rcmmc_source_leo_and_proxy_binding_attacks_fail_closed(tmp_path: Path) -> None:
    paths = _build_rcmmc_pair_fixture(tmp_path, root=tmp_path / "source" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    _rewrite_npz_manifest(Path(paths["c_clean"]), lambda manifest: manifest.__setitem__("rcmmc_frozen_batch_size", 64))
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="batch_size"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "source.json"))
    paths = _build_rcmmc_pair_fixture(tmp_path, root=tmp_path / "leo" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    binding = json.loads(Path(paths["g_binding"]).read_text(encoding="utf-8"))
    binding["rcmmc_feature_dim"] = 159
    Path(paths["g_binding"]).write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="feature_dim"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "leo.json"))
    paths = _build_rcmmc_pair_fixture(tmp_path, root=tmp_path / "proxy" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    for arm in ("c", "g"):
        _rewrite_npz_manifest(Path(paths[f"{arm}_clean"]), lambda manifest: manifest.__setitem__("proxy_row_count", 399))
        raw = json.loads(Path(paths[f"{arm}_proxy"]).read_text(encoding="utf-8"))
        raw["manifest"]["proxy_row_count"] = 399
        Path(paths[f"{arm}_proxy"]).write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="proxy"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "proxy.json"))


def test_rcmmc_f6_reopens_all_prior_raw_artifacts_and_rejects_tamper(tmp_path: Path) -> None:
    root = tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    priors: list[Path] = []
    for fold in range(1, 6):
        paths = _build_rcmmc_pair_fixture(tmp_path, fold=fold, root=root)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_pair_args(paths, output, fold=fold))
        priors.append(output)
    paths = _build_rcmmc_pair_fixture(tmp_path, fold=6, root=root)
    output = root / "F6_C_vs_G_pair_metrics.json"
    metrics = PAIR.evaluate(_pair_args(paths, output, fold=6, priors=tuple(priors)))
    assert metrics["matrix_aggregate"]["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert all(item["raw_artifacts_recomputed"] is True for item in metrics["matrix_aggregate"]["prior_pair_metrics_bindings"])

    tampered = json.loads(priors[0].read_text(encoding="utf-8"))
    tampered["rcmmc_common_training_binding"]["fields"]["common_batch_sequence_sha256"] = _sha("tampered")
    priors[0].write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="common binding"):
        PAIR.evaluate(_pair_args(paths, root / "F6_tampered.json", fold=6, priors=tuple(priors)))

    # Reopen the raw prior artifacts as well: changing one feature byte must
    # invalidate the sealed prior binding even when no summary JSON is edited.
    raw_root = tmp_path / "raw_prior" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    raw_priors: list[Path] = []
    raw_paths_by_fold: dict[int, dict[str, object]] = {}
    for fold in range(1, 6):
        raw_paths = _build_rcmmc_pair_fixture(tmp_path, fold=fold, root=raw_root)
        raw_paths_by_fold[fold] = raw_paths
        raw_output = raw_root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_pair_args(raw_paths, raw_output, fold=fold))
        raw_priors.append(raw_output)
    raw_f6 = _build_rcmmc_pair_fixture(tmp_path, fold=6, root=raw_root)
    with np.load(raw_paths_by_fold[1]["c_clean"], allow_pickle=False) as data:
        raw_payload = {name: np.asarray(data[name]).copy() for name in data.files}
    raw_payload["features"][0, 0] = raw_payload["features"][0, 0] + np.float32(1.0)
    np.savez(raw_paths_by_fold[1]["c_clean"], **raw_payload)
    with pytest.raises(PAIR.RCMMCPostfreezePairError, match="current artifact|raw-artifact recomputation|proxy"):
        PAIR.evaluate(_pair_args(raw_f6, raw_root / "F6_raw_tampered.json", fold=6, priors=tuple(raw_priors)))
