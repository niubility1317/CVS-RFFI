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

import evaluate_phase1_hscf_postfreeze_pair as PAIR
import export_phase1_hscf_features as EXPORT
from cvsrffi import phase1_hscf as HSCF


CODE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_hscf_postfreeze_20260811.sh"
FOLD = 1
TX_IDS = tuple(PAIR._icmt.FROZEN_FOLD_SOURCE_TX[FOLD])
KNOWN_TX = (PAIR._icmt.FROZEN_FOLD_KNOWN_HELDOUT_TX[FOLD],)
PROXY_TX = (PAIR._icmt.FROZEN_FOLD_PROXY_TX[FOLD],)


def _load_core_fixture_module():
    path = CODE_ROOT / "tests" / "test_phase1_hscf.py"
    spec = importlib.util.spec_from_file_location("_hscf_core_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CORE = _load_core_fixture_module()


def _load_icmt_fixture_module():
    path = CODE_ROOT / "tests" / "test_phase1_icmt_postfreeze.py"
    spec = importlib.util.spec_from_file_location("_icmt_fixture_for_hscf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _receipt(
    arm: str,
    *,
    labeled_sha: str | None = None,
    split_sha: str | None = None,
    source_tx_ids: tuple[str, ...] = TX_IDS,
    known_tx_ids: tuple[str, ...] = KNOWN_TX,
    proxy_tx_ids: tuple[str, ...] = PROXY_TX,
) -> dict[str, object]:
    """Build a raw receipt from the HSCF core's own terminal fixture."""

    enabled = arm == "G"
    receipt = (
        _CORE._build_g_terminal_receipt()[0]
        if enabled
        else _CORE._sealed_receipt(enabled=False)
    )
    receipt = dict(receipt)
    if not enabled:
        for index, scenario in enumerate(HSCF.FROZEN_HSCF_SCENARIOS, start=1):
            receipt = _CORE._bind_common(receipt, index=index, scenario=scenario)
    receipt.update(
        {
            "baseline_sha256": _sha("hscf-baseline"),
            "initial_checkpoint_sha256": _sha("hscf-initial"),
            "warm_start_mode": "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP",
            "baseline_path": "geosat_c_final.pth",
            "checkpoint_epoch": 40,
            "strict_model_keys": True,
            "missing_model_keys": [],
            "unexpected_model_keys": [],
            "class_order_binding_sha256": _sha("hscf-class-order"),
            "source_partition_sha256": _sha("hscf-partition"),
            "source_labeled_indices_sha256": labeled_sha or _sha("hscf-labeled"),
            "source_split_manifest_sha256": split_sha or _sha("hscf-split"),
            "source_labeled_provenance": EXPORT.SOURCE_L_PROVENANCE,
            "optimizer_type": "AdamW",
            "optimizer_initial_state_sha256": _sha("hscf-optimizer"),
            "optimizer_initial_state_empty": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
            "checkpoint_role": "training_final_only",
            "source_train_tx": list(source_tx_ids),
            "source_known_validation_tx": list(known_tx_ids),
            "source_proxy_unknown_tx": list(proxy_tx_ids),
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
            "common_l_base_head_input_path_verified": True,
            "common_loader_drop_last": True,
            "common_order_contract": "C_G_IDENTICAL_SEED_SAMPLER_PHYSICAL_IDS_AND_CLEAR_LOW_RAIN_SEQUENCE",
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
    candidate = f"F{fold}{arm}_HSCF12"
    args = vars(_CORE._frozen_args(enabled=arm == "G", epochs=40, batch_size=128)).copy()
    _CORE._disable_peer_flags(argparse.Namespace(**args))
    args.update(
        {
            "split_mode": "tx_rx_day_1_6_3",
            "model_variant": "lite_d",
            "phase1_source_train_tx_ids": ",".join(source_tx_ids),
            "phase1_source_known_validation_tx_ids": ",".join(known_tx_ids),
            "phase1_source_proxy_unknown_tx_ids": ",".join(proxy_tx_ids),
            "checkpoint_selection": "final_only",
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
            "seed": 7281105,
            "candidate_id": candidate,
            "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "candidate_id": candidate,
        "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
        "model": {},
        "args": args,
        "hscf_receipt": receipt
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


def _build_hscf_pair_fixture(
    tmp_path: Path, *, fold: int = 1, root: Path | None = None
) -> dict[str, object]:
    """Reuse the signed ICMT rows while replacing every persisted identity with HSCF."""

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
    for arm in ("C", "G"):
        old_candidate = f"F{fold}{arm}_ICMT12"
        candidate = f"F{fold}{arm}_HSCF12"
        old_dir = root / old_candidate
        new_dir = root / candidate
        shutil.move(str(old_dir), str(new_dir))
        clean = new_dir / PAIR.EXPECTED_CLEAN_ARTIFACT
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
        checked = EXPORT.validate_hscf_terminal_receipt(
            receipt,
            arm=arm,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_tx_ids,
            proxy_unknown_tx_ids=proxy_tx_ids,
        )
        raw_receipt_sha = EXPORT._canonical_json_sha256(receipt)

        def clean_mutate(manifest: dict[str, object]) -> None:
            for field in tuple(manifest):
                if str(field).lower().startswith(PAIR.LEGACY_IDENTITY_PREFIXES):
                    manifest.pop(field, None)
            manifest.update(
                {
                    "schema": PAIR.EXPECTED_LV_EXPORT_SCHEMA,
                    "method": "P1_HSCF",
                    "checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": checkpoint_sha,
                    "candidate_id": candidate,
                    "run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "training_run_contract": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                    "hscf_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                    "hscf_enabled": arm == "G",
                    "hscf_receipt_sha256": raw_receipt_sha,
                    "hscf_terminal_contract": checked["hscf_terminal_contract"],
                    "hscf_terminal_contract_passed": True,
                    "hscf_lambda": HSCF.FROZEN_HSCF_LAMBDA if arm == "G" else 0.0,
                    "hscf_fixed_batch_size": HSCF.FROZEN_HSCF_BATCH_SIZE,
                    "hscf_fixed_local_class_count": HSCF.FROZEN_HSCF_CLASS_COUNT,
                    "hscf_loss_global_denominator": HSCF.FROZEN_HSCF_GLOBAL_DENOMINATOR,
                    "hscf_source_partition_sha256": receipt["source_partition_sha256"],
                    "hscf_source_labeled_indices_sha256": receipt["source_labeled_indices_sha256"],
                    "hscf_source_split_manifest_sha256": receipt["source_split_manifest_sha256"],
                    "hscf_source_labeled_provenance": EXPORT.SOURCE_L_PROVENANCE,
                    "hscf_common_physical_order_bound": True,
                    "hscf_common_scene_cycle_bound": True,
                    "hscf_raw_vjp_per_scene_required": True,
                    "hscf_exact_head_weight_vjp_nonzero_required": True,
                    "hscf_head_bias_aux_vjp_na_none_or_zero": True,
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
                "method": "P1_HSCF",
                "candidate_id": candidate,
                "training_run_root": str(training_root.resolve()),
                "postfreeze_output_root": str(root.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "training_run_id": PAIR.EXPECTED_TRAINING_RUN_LEAF,
                "leo_npz_path": str(leo.resolve()),
                "leo_npz_sha256": hashlib.sha256(leo.read_bytes()).hexdigest(),
                "leo_manifest_sha256": fixture.PAIR._icmt_leo._canonical_json_sha256(leo_manifest),
                "hscf_receipt_schema": EXPORT.EXPECTED_RECEIPT_SCHEMA,
                "hscf_receipt_sha256": raw_receipt_sha,
                "hscf_terminal_contract": checked["hscf_terminal_contract"],
                "hscf_terminal_contract_passed": True,
                "hscf_lambda": HSCF.FROZEN_HSCF_LAMBDA if arm == "G" else 0.0,
                "hscf_fixed_batch_size": HSCF.FROZEN_HSCF_BATCH_SIZE,
                "hscf_fixed_local_class_count": HSCF.FROZEN_HSCF_CLASS_COUNT,
                "hscf_loss_global_denominator": HSCF.FROZEN_HSCF_GLOBAL_DENOMINATOR,
                "hscf_source_partition_sha256": receipt["source_partition_sha256"],
                "hscf_source_labeled_indices_sha256": receipt["source_labeled_indices_sha256"],
                "hscf_source_split_manifest_sha256": receipt["source_split_manifest_sha256"],
                "hscf_source_labeled_provenance": EXPORT.SOURCE_L_PROVENANCE,
                "hscf_common_physical_order_bound": True,
                "hscf_common_scene_cycle_bound": True,
                "hscf_raw_vjp_per_scene_required": True,
                "hscf_exact_head_weight_vjp_nonzero_required": True,
                "hscf_head_bias_aux_vjp_na_none_or_zero": True,
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
    if priors:
        values += ["--aggregate-prior-pair-metrics-json", ",".join(str(path) for path in priors)]
    return PAIR.build_parser().parse_args(values)


def test_hscf_receipt_b128_k4_512_lambda_and_c_g_scene_vjp_contract() -> None:
    for arm in ("C", "G"):
        checked = EXPORT.validate_hscf_terminal_receipt(
            _receipt(arm),
            arm=arm,
            source_tx_ids=TX_IDS,
            known_validation_tx_ids=KNOWN_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )
        assert checked["fixed_batch_size"] == 128
        assert checked["fixed_local_class_count"] == 4
        assert checked["loss_global_denominator"] == 512
        assert checked["lambda"] == (0.02 if arm == "G" else 0.0)
        assert checked["hscf_terminal_contract_passed"] is True
        if arm == "C":
            assert checked["hscf_batches"] == 0
            assert checked["hscf_loss_sum"] == 0.0
        else:
            assert set(checked["hscf_scenes"]) == set(HSCF.FROZEN_HSCF_SCENARIOS)
            assert all(checked["hscf_scenes"][scene]["positive_batches"] > 0 for scene in HSCF.FROZEN_HSCF_SCENARIOS)
            assert set(checked["hscf_gradient_audit_scenes"]) == set(HSCF.FROZEN_HSCF_SCENARIOS)
            for audit in checked["hscf_gradient_audit_scenes"].values():
                for group in ("leo_raw_logits", "shared_encoder", "head_weight"):
                    assert audit[group]["norm"] > 0.0 and math.isfinite(audit[group]["norm"])
                assert audit["clean_raw_logits"]["nonzero_parameters"] == 0.0
                assert audit["head_bias"]["nonzero_parameters"] == 0.0


def test_hscf_same_fold_common_binding_is_strict_but_g_only_evidence_is_not_compared() -> None:
    binding = PAIR.validate_hscf_common_training_binding(_receipt("C"), _receipt("G"))
    assert binding["passed"] is True
    assert set(binding["fields"]) == set(PAIR.COMMON_TRAINING_BINDING_FIELDS)
    g = _receipt("G")
    g["hscf_loss_sum"] = 99.0
    assert PAIR.validate_hscf_common_training_binding(_receipt("C"), g)["passed"] is True
    g["common_batch_sequence_sha256"] = _sha("tampered-sequence")
    with pytest.raises(PAIR.HSCFPostfreezePairError, match="common training binding"):
        PAIR.validate_hscf_common_training_binding(_receipt("C"), g)


def test_hscf_float64_gaussian_is_l_only_zero_totalized_and_nonfinite_fails_closed() -> None:
    features = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = PAIR.normalize_hscf_float64(features, label="fixture")
    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized, np.asarray([[0.6, 0.8], [0.0, 0.0]], dtype=np.float64))
    tx = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    geometry = PAIR.fit_frozen_hscf_diagonal_gaussian(np.vstack((np.eye(4), np.eye(4))), tx, ("a", "b", "c", "d"))
    score = PAIR.score_frozen_hscf_nll(np.zeros((1, 4), dtype=np.float64), geometry)
    assert score.shape == (1,) and np.isfinite(score).all()
    with pytest.raises(Exception, match="non-finite"):
        PAIR.normalize_hscf_float64(np.asarray([[np.nan, 1.0]]), label="bad")


def test_hscf_launcher_is_exactly_42_steps_and_dry_run_is_frozen() -> None:
    relative_launcher = "scripts/launch_phase1_hscf_postfreeze_20260811.sh"
    syntax = subprocess.run(["bash", "-n", relative_launcher], cwd=CODE_ROOT, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative_launcher, "--dry-run"], cwd=CODE_ROOT, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    lines = [line for line in dry.stdout.splitlines() if line.startswith("[DRY-RUN]")]
    assert len(lines) == 42
    assert sum("HSCF_CLEAN_EXPORT" in line for line in lines) == 12
    assert sum("HSCF_LEO_EXPORT_AND_BIND" in line for line in lines) == 12
    assert sum("FROZEN_LOGITS_PROXY_BINDING" in line for line in lines) == 12
    assert sum("HSCF_PAIR_SCORE" in line for line in lines) == 6
    assert sum("phase1_hscf12_20260811_v2" in line for line in lines) == 30
    assert all("phase1_hscf_postfreeze_20260811_v1" in line for line in lines)
    assert all("_HSCF12" in line for line in lines)
    assert all("_ICMT12" not in line and "_RCAT12" not in line and "_RCRMD12" not in line for line in lines)


def test_hscf_pair_closes_receipt_clean_leo_proxy_and_common_binding(tmp_path: Path) -> None:
    paths = _build_hscf_pair_fixture(tmp_path)
    metrics = PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "pair.json"))
    assert metrics["schema"] == PAIR.EXPECTED_PAIR_SCHEMA
    assert metrics["hscf_training_receipt_revalidation"]["C"]["candidate"] == "F1C_HSCF12"
    assert metrics["hscf_training_receipt_revalidation"]["G"]["terminal_contract_passed"] is True
    assert metrics["policy"]["geometry_fit_role"] == "labeled_fit"
    assert metrics["policy"]["proxy_unknown_fit_rows"] == 0
    assert metrics["hscf_common_training_binding"]["passed"] is True
    assert metrics["verdict"].startswith(("REJECT", "PENDING_MAIN"))


def test_hscf_leo_binding_closes_manysig_physical_tx_rx_day_and_three_scenes(tmp_path: Path) -> None:
    paths = _build_hscf_pair_fixture(tmp_path)
    for arm in ("c", "g"):
        binding = json.loads(Path(paths[f"{arm}_binding"]).read_text(encoding="utf-8"))
        assert binding["schema"] == PAIR.EXPECTED_LEO_BINDING_SCHEMA
        assert binding["method"] == "P1_HSCF"
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
        for scene in HSCF.FROZEN_HSCF_SCENARIOS:
            mask = scenes == scene
            assert int(mask.sum()) == 24
            assert len({"\x1f".join((tx[i], rx[i], day[i], sig[i])) for i in np.flatnonzero(mask)}) == 24


def test_hscf_four_floor_overall_and_proxy_double_gate_are_noncompensating(tmp_path: Path) -> None:
    paths = _build_hscf_pair_fixture(tmp_path)
    metrics = PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "gates.json"))
    gates = metrics["postfreeze_gates"]
    floor_names = {"overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy"}
    assert set(gates["clean_four_floors_ge_minus2pp"]["metric_passes"]) == floor_names
    assert set(gates["leo_scenario_four_floors_ge_minus2pp"]["by_scenario"]) == set(HSCF.FROZEN_HSCF_SCENARIOS)
    proxy = gates["proxy_continuous_two_strict_improvements"]
    assert proxy["passed"] is (proxy["strict_AUROC_improvement"] and proxy["strict_proxy_known_gap_improvement"])
    assert proxy["diagnostic_only_non_compensating"] is True
    assert "non-compensating" in PAIR.FROZEN_POSTFREEZE_CONTRACT["HSCF-PF-10"]


@pytest.mark.parametrize("legacy_field", ["icmt_receipt_schema", "rcat_enabled", "rcrmd_source_split_manifest_sha256"])
def test_hscf_rejects_legacy_method_identity_even_when_proxy_is_recomputed(tmp_path: Path, legacy_field: str) -> None:
    paths = _build_hscf_pair_fixture(tmp_path)
    _rewrite_npz_manifest(Path(paths["c_clean"]), lambda manifest: manifest.__setitem__(legacy_field, True))
    _write_frozen_proxy_logits(Path(paths["c_clean"]), Path(paths["c_proxy"]), Path(paths["c_scores"]), TX_IDS)
    with pytest.raises(PAIR.HSCFPostfreezePairError, match="historical method identity"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "legacy.json"))


def test_hscf_source_leo_and_proxy_binding_attacks_fail_closed(tmp_path: Path) -> None:
    paths = _build_hscf_pair_fixture(tmp_path, root=tmp_path / "source" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    _rewrite_npz_manifest(Path(paths["c_clean"]), lambda manifest: manifest.__setitem__("hscf_fixed_batch_size", 64))
    with pytest.raises(PAIR.HSCFPostfreezePairError, match="hscf_fixed_batch_size"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "source.json"))
    paths = _build_hscf_pair_fixture(tmp_path, root=tmp_path / "leo" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    binding = json.loads(Path(paths["g_binding"]).read_text(encoding="utf-8"))
    binding["hscf_loss_global_denominator"] = 256
    Path(paths["g_binding"]).write_text(json.dumps(binding, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.HSCFPostfreezePairError, match="loss_global_denominator"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "leo.json"))
    paths = _build_hscf_pair_fixture(tmp_path, root=tmp_path / "proxy" / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID)
    for arm in ("c", "g"):
        _rewrite_npz_manifest(Path(paths[f"{arm}_clean"]), lambda manifest: manifest.__setitem__("proxy_row_count", 399))
        raw = json.loads(Path(paths[f"{arm}_proxy"]).read_text(encoding="utf-8"))
        raw["manifest"]["proxy_row_count"] = 399
        Path(paths[f"{arm}_proxy"]).write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.HSCFPostfreezePairError, match="proxy"):
        PAIR.evaluate(_pair_args(paths, Path(paths["root"]) / "proxy.json"))


def test_hscf_f6_reopens_all_prior_raw_artifacts_and_rejects_tamper(tmp_path: Path) -> None:
    root = tmp_path / PAIR.EXPECTED_POSTFREEZE_MATRIX_ID
    priors: list[Path] = []
    for fold in range(1, 6):
        paths = _build_hscf_pair_fixture(tmp_path, fold=fold, root=root)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_pair_args(paths, output, fold=fold))
        priors.append(output)
    paths = _build_hscf_pair_fixture(tmp_path, fold=6, root=root)
    output = root / "F6_C_vs_G_pair_metrics.json"
    metrics = PAIR.evaluate(_pair_args(paths, output, fold=6, priors=tuple(priors)))
    aggregate = metrics["matrix_aggregate"]
    assert aggregate["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert all(item["raw_artifacts_recomputed"] is True for item in aggregate["prior_pair_metrics_bindings"])

    tampered = json.loads(priors[0].read_text(encoding="utf-8"))
    tampered["hscf_common_training_binding"]["fields"]["common_batch_sequence_sha256"] = _sha("tampered")
    priors[0].write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(PAIR.HSCFPostfreezePairError, match="common training binding"):
        PAIR.evaluate(_pair_args(paths, root / "F6_tampered.json", fold=6, priors=tuple(priors)))
