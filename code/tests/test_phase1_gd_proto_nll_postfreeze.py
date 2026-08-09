from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = CODE_ROOT / "cvsrffi" / "eval_phase1_gd_proto_nll_pair.py"
EXPORTER_PATH = CODE_ROOT / "export_phase1_gd_proto_nll_features.py"
LAUNCHER_PATH = CODE_ROOT / "scripts" / "launch_phase1_gd_proto_nll_postfreeze_20260810.sh"
_SPEC = importlib.util.spec_from_file_location("gd_proto_nll_postfreeze_pair", EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PAIR
_SPEC.loader.exec_module(PAIR)


RXS = PAIR.EXPECTED_SOURCE_RXS
DAYS = PAIR.EXPECTED_SOURCE_DAYS
SCENARIOS = PAIR.EXPECTED_SCENARIOS
TEST_MATRIX_ID = "test_phase1_gd_proto_nll_postfreeze_matrix_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_hot(index: int) -> np.ndarray:
    value = np.zeros(4, dtype=np.float32)
    value[int(index)] = 1.0
    return value


def _logits(tx: str, tx_ids: tuple[str, ...]) -> np.ndarray:
    values = np.full(4, -2.0, dtype=np.float32)
    values[tx_ids.index(tx)] = 4.0
    return values


def _physical_keys(rows: list[dict[str, str]]) -> list[str]:
    return [
        "\x1f".join(row[field] for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"))
        for row in rows
    ]


def _strict_load_manifest(checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "feature_name": "z_id",
        "checkpoint": str(checkpoint.resolve()),
        "classification_head_contract": PAIR.EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
        "class_id_to_tx": list(tx_ids),
        "logit_class_order": list(range(4)),
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
        "satellite_tta_policy": "none",
    }


def _clean_rows(tx_ids: tuple[str, ...], proxy_tx: str) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    indices: list[int] = []
    source_index = 0
    for role, per_class, offset in (("labeled_fit", 3, 0), ("source_validation_known", 2, 100)):
        for tx_index, tx in enumerate(tx_ids):
            for sample in range(per_class):
                rows.append(
                    {
                        "dataset_role": role,
                        "tx_ids": tx,
                        "rx_ids": RXS[(tx_index + sample) % len(RXS)],
                        "day_ids": DAYS[(tx_index + sample) % len(DAYS)],
                        "eq_ids": "1",
                        "sig_ids": f"{role}-{tx_index}-{sample}",
                        "channel_views": "clean",
                        "sat_scenarios": "",
                    }
                )
                indices.append(source_index if role == "labeled_fit" else offset + source_index - 12)
                source_index += 1
    for sample in range(4):
        rows.append(
            {
                "dataset_role": "proxy_unknown",
                "tx_ids": proxy_tx,
                "rx_ids": RXS[sample],
                "day_ids": DAYS[sample % len(DAYS)],
                "eq_ids": "1",
                "sig_ids": f"proxy-{sample}",
                "channel_views": "clean",
                "sat_scenarios": "",
            }
        )
        indices.append(-1 - sample)
    return rows, np.asarray(indices, dtype=np.int64)


def _clean_manifest(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    tx_ids: tuple[str, ...],
    fold: int,
    rows: list[dict[str, str]],
    source_base_indices: np.ndarray,
) -> dict[str, object]:
    labeled = np.asarray([row["dataset_role"] == "labeled_fit" for row in rows], dtype=bool)
    validation = np.asarray(
        [row["dataset_role"] == "source_validation_known" for row in rows], dtype=bool
    )
    proxy = np.asarray([row["dataset_role"] == "proxy_unknown" for row in rows], dtype=bool)
    labeled_indices = source_base_indices[labeled].tolist()
    validation_indices = source_base_indices[validation].tolist()
    unlabeled_indices = list(range(200, 220))
    physical = np.asarray(_physical_keys(rows), dtype=object)
    receipt = {
        "labeled_indices_sha256": PAIR._index_sha256(labeled_indices),
        "unlabeled_indices_sha256": PAIR._index_sha256(unlabeled_indices),
        "source_validation_indices_sha256": PAIR._index_sha256(validation_indices),
        "wisig_pkl_sha256": "",
    }
    manifest = _strict_load_manifest(checkpoint, checkpoint_sha256, tx_ids)
    manifest.update(
        {
            "schema": "cvs.phase1.gd_proto_nll_lv_export.v1",
            "checkpoint_role": "training_final_only",
            "checkpoint_selection": "final_only",
            "source_only_export": False,
            "channel_profile": {
                "labeled_fit": {"view": "clean", "scenarios": []},
                "source_validation_known": {"view": "clean", "scenarios": []},
                "proxy_unknown": {"view": "clean", "scenarios": []},
            },
            "split_mode": "tx_rx_day_1_6_3",
            "seed": 7281105,
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
            "source_tx_ids": list(tx_ids),
            "wisig_pkl_sha256": PAIR.FROZEN_WISIG_SHA256,
            "expected_wisig_pkl_sha256": PAIR.FROZEN_WISIG_SHA256,
            "checkpoint_declared_wisig_pkl_sha256": "",
            "checkpoint_declared_wisig_pkl_sha256_empty_caveat": True,
            "dataset_path_checkpoint_equal": True,
            "known_validation_outer_tx_ids": [PAIR.FROZEN_FOLD_KNOWN_HELDOUT_TX[fold]],
            "proxy_unknown_tx_ids": [PAIR.FROZEN_FOLD_PROXY_TX[fold]],
            "source_split_receipt": receipt,
            "source_split_receipt_checkpoint_equal": True,
            "tx_partition_receipt_checkpoint_equal": True,
            "labeled_indices_sha256": receipt["labeled_indices_sha256"],
            "unlabeled_indices_sha256": receipt["unlabeled_indices_sha256"],
            "source_validation_indices_sha256": receipt["source_validation_indices_sha256"],
            "labeled_physical_keys_sha256": PAIR._canonical_json_sha256(physical[labeled].tolist()),
            "source_validation_physical_keys_sha256": PAIR._canonical_json_sha256(
                physical[validation].tolist()
            ),
            "proxy_physical_keys_sha256": PAIR._canonical_json_sha256(physical[proxy].tolist()),
            "labeled_source_validation_physical_disjoint": True,
            "labeled_validation_proxy_physical_disjoint": True,
            "labeled_row_count": int(labeled.sum()),
            "unlabeled_row_count": len(unlabeled_indices),
            "source_validation_row_count": int(validation.sum()),
            "proxy_row_count": int(proxy.sum()),
            "forwarded_roles": ["labeled_fit", "source_validation_known", "proxy_unknown"],
            "unlabeled_forward_rows": 0,
            "unlabeled_features_persisted": False,
        }
    )
    return manifest


def _clean_payload(
    *, arm: str, checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...], fold: int
) -> dict[str, np.ndarray]:
    rows, source_base_indices = _clean_rows(tx_ids, PAIR.FROZEN_FOLD_PROXY_TX[fold])
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for row in rows:
        role = row["dataset_role"]
        if role in {"labeled_fit", "source_validation_known"}:
            features.append(_one_hot(tx_ids.index(row["tx_ids"])))
            logits.append(_logits(row["tx_ids"], tx_ids))
        else:
            features.append(_one_hot(0) if arm == "C" else -np.ones(4, dtype=np.float32))
            logits.append(np.zeros(4, dtype=np.float32))
    manifest = _clean_manifest(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        tx_ids=tx_ids,
        fold=fold,
        rows=rows,
        source_base_indices=source_base_indices,
    )
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "source_base_indices": source_base_indices,
        "manifest_json": np.asarray(json.dumps(manifest)),
    }
    for field in PAIR.METADATA_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows])
    return payload


def _leo_payload(
    *, checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...]
) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for tx_index, tx in enumerate(tx_ids):
            for rx_index, rx in enumerate(RXS):
                rows.append(
                    {
                        "dataset_role": "source",
                        "tx_ids": tx,
                        "rx_ids": rx,
                        "day_ids": DAYS[(scenario_index + tx_index + rx_index) % len(DAYS)],
                        "eq_ids": "1",
                        "sig_ids": f"{scenario}-{tx_index}-{rx_index}",
                        "channel_views": PAIR.EXPECTED_LEO_RUNTIME_VIEW,
                        "sat_scenarios": scenario,
                    }
                )
                features.append(_one_hot(tx_index))
                logits.append(_logits(tx, tx_ids))
    manifest = _strict_load_manifest(checkpoint, checkpoint_sha256, tx_ids)
    manifest.update(
        {
            "source_only_export": True,
            "star_ground_channel_impl": "simplified_leo_residual",
            "channel_profile": {
                "source": {
                    "view": "satellite",
                    "scenarios": list(SCENARIOS),
                    "sat_seed": 7281718,
                }
            },
        }
    )
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "manifest_json": np.asarray(json.dumps(manifest)),
    }
    for field in PAIR.METADATA_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows])
    return payload


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _rewrite(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    mutate(payload)
    np.savez(path, **payload)


def _set_manifest(path: Path, mutate) -> None:
    def apply(payload: dict[str, np.ndarray]) -> None:
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        mutate(manifest)
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(path, apply)


def _proxy_metrics(path: Path, clean_path: Path) -> None:
    with np.load(clean_path, allow_pickle=False) as data:
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        roles = np.asarray(data["dataset_role"])
    path.write_text(
        json.dumps(
            {
                "phase": "phase1_only_logits_open_set_reject",
                "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
                "feature_npz": str(clean_path.resolve()),
                "source_tx_ids": manifest["source_tx_ids"],
                "known_query_roles": ["source_validation_known"],
                "unknown_query_roles": ["proxy_unknown"],
                "known_query_count": int(np.sum(roles == "source_validation_known")),
                "unknown_query_count": int(np.sum(roles == "proxy_unknown")),
                "AUROC_unknown": 0.5,
                "unknown_FAR": 0.5,
                "manifest": manifest,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_pair(root: Path, *, fold: int = 1) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    tx_ids = PAIR.FROZEN_FOLD_SOURCE_TX[fold]
    directory = root / f"F{fold}_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    training_root = root.parent / PAIR.EXPECTED_TRAINING_RUN_LEAF
    c_checkpoint = training_root / f"F{fold}C_GD_PROTO_NLL12" / "final_ssdg.pth"
    g_checkpoint = training_root / f"F{fold}G_GD_PROTO_NLL12" / "final_ssdg.pth"
    c_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    g_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    c_checkpoint.write_bytes(f"GD-PROTO-NLL-C-{fold}".encode("ascii"))
    g_checkpoint.write_bytes(f"GD-PROTO-NLL-G-{fold}".encode("ascii"))
    paths: dict[str, object] = {
        "root": root.resolve(),
        "training_root": training_root.resolve(),
        "source_tx_ids": tx_ids,
        "c_checkpoint": c_checkpoint,
        "g_checkpoint": g_checkpoint,
        "c_clean": directory / "c_clean.npz",
        "g_clean": directory / "g_clean.npz",
        "c_leo": directory / "c_leo.npz",
        "g_leo": directory / "g_leo.npz",
        "c_proxy": directory / "c_proxy.json",
        "g_proxy": directory / "g_proxy.json",
    }
    _save(
        paths["c_clean"],
        _clean_payload(
            arm="C", checkpoint=c_checkpoint, checkpoint_sha256=_sha(c_checkpoint), tx_ids=tx_ids, fold=fold
        ),
    )
    _save(
        paths["g_clean"],
        _clean_payload(
            arm="G", checkpoint=g_checkpoint, checkpoint_sha256=_sha(g_checkpoint), tx_ids=tx_ids, fold=fold
        ),
    )
    _save(
        paths["c_leo"],
        _leo_payload(checkpoint=c_checkpoint, checkpoint_sha256=_sha(c_checkpoint), tx_ids=tx_ids),
    )
    _save(
        paths["g_leo"],
        _leo_payload(checkpoint=g_checkpoint, checkpoint_sha256=_sha(g_checkpoint), tx_ids=tx_ids),
    )
    _proxy_metrics(paths["c_proxy"], paths["c_clean"])
    _proxy_metrics(paths["g_proxy"], paths["g_clean"])
    return paths


def _args(
    paths: dict[str, object], output: Path, *, fold: int = 1, priors: tuple[Path, ...] = ()
) -> object:
    command = [
        "--c-clean-npz", str(paths["c_clean"]),
        "--g-clean-npz", str(paths["g_clean"]),
        "--c-leo-npz", str(paths["c_leo"]),
        "--g-leo-npz", str(paths["g_leo"]),
        "--c-final-checkpoint", str(paths["c_checkpoint"]),
        "--g-final-checkpoint", str(paths["g_checkpoint"]),
        "--c-proxy-metrics-json", str(paths["c_proxy"]),
        "--g-proxy-metrics-json", str(paths["g_proxy"]),
        "--candidate-pair", f"F{fold}_C_vs_G",
        "--fold-index", str(fold),
        "--postfreeze-matrix-id", TEST_MATRIX_ID,
        "--postfreeze-output-root", str(paths["root"]),
        "--training-run-root", str(paths["training_root"]),
        "--source-tx-ids", ",".join(paths["source_tx_ids"]),
        "--expected-source-count", "72",
        "--expected-proxy-count", "4",
        "--output-metrics-json", str(output),
    ]
    if priors:
        command.extend(["--aggregate-prior-pair-metrics-json", ",".join(str(path) for path in priors)])
    return PAIR.build_parser().parse_args(command)


def test_float64_geometry_matches_formula_totalizes_zero_and_rejects_nonfinite():
    tx_ids = ("a", "b", "c", "d")
    features = np.asarray(
        [
            [1.0, 0.1], [1.0, -0.1],
            [0.1, 1.0], [-0.1, 1.0],
            [-1.0, 0.1], [-1.0, -0.1],
            [0.1, -1.0], [-0.1, -1.0],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.asarray(tx_ids, dtype=object), 2)
    geometry = PAIR.fit_frozen_diagonal_gaussian(features, labels, tx_ids)
    normalized = features.astype(np.float64) / np.linalg.norm(features.astype(np.float64), axis=1)[:, None]
    means = np.stack([normalized[labels == tx].mean(axis=0) for tx in tx_ids])
    raw_variances = np.stack([normalized[labels == tx].var(axis=0, ddof=1) for tx in tx_ids])
    pooled = raw_variances.mean(axis=0)
    variances = np.maximum(1e-6, 0.9 * raw_variances + 0.1 * pooled[None, :])
    np.testing.assert_allclose(geometry["means"], means, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(geometry["variances"], variances, rtol=0.0, atol=1e-15)
    probe = np.asarray([[0.8, 0.2], [-0.2, -0.8]], dtype=np.float32)
    probe64 = probe.astype(np.float64) / np.linalg.norm(probe.astype(np.float64), axis=1)[:, None]
    nll = 0.5 * np.sum(
        (probe64[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
        + np.log(2.0 * math.pi * variances)[None, :, :],
        axis=2,
    )
    maximum = np.max(-nll, axis=1)
    expected = math.log(4.0) - (
        maximum + np.log(np.sum(np.exp(-nll - maximum[:, None]), axis=1))
    )
    observed = PAIR.score_frozen_gd_proto_nll(probe, geometry)
    assert observed.dtype == np.float64
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)
    piecewise_input = np.concatenate((probe, np.zeros((1, 2), dtype=np.float32)), axis=0)
    normalized = PAIR._normalize_float64(piecewise_input, label="piecewise test")
    np.testing.assert_allclose(normalized[:2], probe64, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(normalized[2], np.zeros(2, dtype=np.float64))
    totalized_scores = PAIR.score_frozen_gd_proto_nll(piecewise_input, geometry)
    assert totalized_scores.shape == (3,)
    assert np.isfinite(totalized_scores).all()
    with pytest.raises(PAIR.GDProtoNLLPostfreezePairError, match="non-finite"):
        PAIR.score_frozen_gd_proto_nll(np.asarray([[np.nan, 0.0]], dtype=np.float32), geometry)


def test_pair_closes_l_only_v_known_proxy_strict_and_classifier_gates(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    output = Path(paths["root"]) / "F1_C_vs_G_pair_metrics.json"
    metrics = PAIR.evaluate(_args(paths, output))
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == metrics["schema"]
    assert metrics["schema"] == "cvs.phase1.gd_proto_nll_postfreeze_pair.v2"
    assert metrics["policy"]["geometry_fit_role"] == "labeled_fit"
    assert metrics["policy"]["source_validation_fit_rows"] == 0
    assert metrics["clean_source_validation"]["C"]["count"] == 8
    assert metrics["proxy_continuous_guardrail"]["strict_AUROC_improvement"] is True
    assert metrics["proxy_continuous_guardrail"]["strict_proxy_known_gap_improvement"] is True
    assert metrics["postfreeze_gates"]["fold_verdict"] == "PENDING_GLOBAL_18_GRID"
    assert metrics["phase3_unknown_capability_claim"] == "NOT_EVALUATED"
    with pytest.raises(PAIR.GDProtoNLLPostfreezePairError, match="refusing to overwrite"):
        PAIR.evaluate(_args(paths, output))


def test_zero_v_row_is_retained_and_counted_for_both_arms(tmp_path):
    paths = _write_pair(tmp_path / "matrix")

    def zero_first_validation(payload: dict[str, np.ndarray]) -> None:
        index = int(np.flatnonzero(payload["dataset_role"] == "source_validation_known")[0])
        payload["features"][index] = 0.0

    for key in ("c_clean", "g_clean"):
        _rewrite(paths[key], zero_first_validation)
    metrics = PAIR.evaluate(_args(paths, Path(paths["root"]) / "zero_v_retained.json"))
    assert metrics["clean_source_validation"]["C"]["count"] == 8
    assert metrics["clean_source_validation"]["G"]["count"] == 8
    for arm in ("C", "G"):
        role = metrics["feature_norm_receipt"][arm]["roles"]["source_validation_known"]
        assert role == {
            "total_rows": 8,
            "positive_norm_rows": 7,
            "zero_norm_rows": 1,
            "nonfinite_rows": 0,
            "retained_rows": 8,
            "dropped_rows": 0,
            "count_closed": True,
        }
        assert metrics["proxy_continuous_guardrail"][arm]["known_heldout"]["count"] == 8


def test_pair_permanently_rejects_without_both_strict_continuous_improvements(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    with np.load(paths["c_clean"], allow_pickle=False) as data:
        c_features = np.asarray(data["features"]).copy()
    _rewrite(paths["g_clean"], lambda payload: payload.__setitem__("features", c_features))
    metrics = PAIR.evaluate(_args(paths, Path(paths["root"]) / "no_improvement.json"))
    guardrail = metrics["proxy_continuous_guardrail"]
    assert guardrail["G_minus_C"]["AUROC_unknown"] == 0.0
    assert guardrail["G_minus_C"]["proxy_minus_known_heldout_mean_u"] == 0.0
    assert guardrail["passed"] is False
    assert metrics["postfreeze_gates"]["fold_verdict"] == "REJECT_GD_PROTO_NLL_PERMANENT"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda paths: [
                _set_manifest(paths[key], lambda manifest: manifest.__setitem__("unlabeled_forward_rows", 1))
                for key in ("c_clean", "g_clean")
            ],
            "forwarded U rows",
        ),
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["source_base_indices"].__setitem__(12, 0))
                for key in ("c_clean", "g_clean")
            ],
            "source_base_indices contain duplicates",
        ),
        (
            lambda paths: _set_manifest(
                paths["c_clean"],
                lambda manifest: manifest.__setitem__("classification_head_contract", "FORGED_HEAD"),
            ),
            "classification_head_contract",
        ),
    ],
)
def test_pair_fails_closed_on_u_index_or_head_tamper(tmp_path, mutate, message):
    paths = _write_pair(tmp_path / "matrix")
    mutate(paths)
    with pytest.raises(PAIR.GDProtoNLLPostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "tampered.json"))


def test_pair_rejects_complete_arm_swap_and_non_v3_training_root(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    swapped = _args(paths, Path(paths["root"]) / "swapped.json")
    swapped.c_clean_npz, swapped.g_clean_npz = str(paths["g_clean"]), str(paths["c_clean"])
    swapped.c_leo_npz, swapped.g_leo_npz = str(paths["g_leo"]), str(paths["c_leo"])
    swapped.c_final_checkpoint, swapped.g_final_checkpoint = str(paths["g_checkpoint"]), str(paths["c_checkpoint"])
    swapped.c_proxy_metrics_json, swapped.g_proxy_metrics_json = str(paths["g_proxy"]), str(paths["c_proxy"])
    with pytest.raises(PAIR.GDProtoNLLPostfreezePairError, match="C final checkpoint path"):
        PAIR.evaluate(swapped)

    wrong_root = Path(paths["root"]).parent / "phase1_gd_proto_nll12_20260809_v2"
    wrong_root.mkdir(parents=True, exist_ok=True)
    wrong = _args(paths, Path(paths["root"]) / "wrong_root.json")
    wrong.training_run_root = str(wrong_root)
    with pytest.raises(PAIR.GDProtoNLLPostfreezePairError, match="training run root leaf must be"):
        PAIR.evaluate(wrong)


def test_six_fold_aggregate_closes_same_matrix_root_training_root_and_prior_receipts(tmp_path):
    root = tmp_path / "matrix"
    priors: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(root, fold=fold)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        priors.append(output)
    final_paths = _write_pair(root, fold=6)
    final = PAIR.evaluate(
        _args(final_paths, root / "F6_C_vs_G_pair_metrics.json", fold=6, priors=tuple(priors))
    )
    aggregate = final["matrix_aggregate"]
    assert aggregate["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert aggregate["verdict"] == "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
    assert aggregate["global_18_cell_equal_weight_G_minus_C_pp"] == {
        metric: 0.0 for metric in PAIR.CLASSIFICATION_METRICS
    }
    receipt = json.loads(priors[0].read_text(encoding="utf-8"))
    receipt["feature_norm_receipt"]["C"]["roles"]["labeled_fit"]["zero_norm_rows"] += 1
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.GDProtoNLLPostfreezePairError, match="positive/zero/nonfinite counts do not close"):
        PAIR.evaluate(_args(final_paths, root / "F6_retry.json", fold=6, priors=tuple(priors)))


def test_launcher_is_exactly_42_steps_and_exporter_never_forwards_u():
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    exporter = EXPORTER_PATH.read_text(encoding="utf-8")
    evaluator = EVALUATOR_PATH.read_text(encoding="utf-8")
    for required in (
        "phase1_gd_proto_nll_postfreeze_20260810_v2",
        "phase1_gd_proto_nll12_20260809_v3",
        "export_phase1_gd_proto_nll_features.py",
        "F${fold}${arm}_GD_PROTO_NLL12",
        "--known_query_roles source_validation_known",
        "--calibration_roles source_validation_known",
        "--unknown_query_roles proxy_unknown",
        "--expected-wisig-sha256 \"${WISIG_SHA256}\"",
        "--source_only_export",
        "--source_channel_view satellite",
        "--satellite_tta_policy none",
        "CUDA_VISIBLE_DEVICES=\"\"",
    ):
        assert required in launcher
    assert re.findall(r"^launch_candidate (\d) ([CG]) (\d)$", launcher, flags=re.MULTILINE) == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_gd_proto_nll_postfreeze_20260810.sh", "--dry-run"],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=True,
    )
    lines = completed.stdout.splitlines()
    assert sum(line.startswith("[DRY-RUN][GD_CLEAN_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][LEO_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PROXY_SCORE]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PAIR_SCORE]") for line in lines) == 6
    assert len(lines) == 42
    assert all("phase1_gd_proto_nll_postfreeze_20260810_v2" in line for line in lines)
    assert "phase1_gd_proto_nll_postfreeze_20260810_v1" not in launcher
    invalid = subprocess.run(
        [
            "bash", "-c",
            "TRAIN_RUN_ROOT='/tmp/phase1_gd_proto_nll12_20260809_v2' "
            "bash scripts/launch_phase1_gd_proto_nll_postfreeze_20260810.sh --dry-run",
        ],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=False,
    )
    assert invalid.returncode == 3
    assert "TRAIN_RUN_ROOT leaf must be phase1_gd_proto_nll12_20260809_v3" in invalid.stderr
    assert "unlabeled_loader" not in exporter
    assert '"unlabeled_forward_rows": 0' in exporter
    assert '"unlabeled_features_persisted": False' in exporter
    assert "_require_split_receipts_match(checkpoint, reconstructed)" in exporter
    assert "import torch" not in evaluator
    export_spec = importlib.util.spec_from_file_location("gd_proto_nll_split_export", EXPORTER_PATH)
    assert export_spec is not None and export_spec.loader is not None
    export_module = importlib.util.module_from_spec(export_spec)
    sys.modules[export_spec.name] = export_module
    export_spec.loader.exec_module(export_module)
    with pytest.raises(export_module.GDProtoNLLSplitExportError, match="frozen value"):
        export_module._require_frozen_dataset_sha256(
            actual=PAIR.FROZEN_WISIG_SHA256,
            expected="0" * 64,
            checkpoint_declared="",
        )
