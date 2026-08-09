from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = CODE_ROOT / "scripts" / "eval_phase1_cp_sfce_pair.py"
LAUNCHER_PATH = CODE_ROOT / "scripts" / "launch_phase1_cp_sfce_postfreeze_20260809.sh"
_SPEC = importlib.util.spec_from_file_location("cp_sfce_postfreeze_pair", EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PAIR
_SPEC.loader.exec_module(PAIR)


TX = PAIR.FROZEN_FOLD_SOURCE_TX[1]
RXS = PAIR.EXPECTED_SOURCE_RXS
DAYS = PAIR.EXPECTED_SOURCE_DAYS
SCENARIOS = PAIR.EXPECTED_SCENARIOS
TEST_MATRIX_ID = "test_phase1_cp_sfce_postfreeze_matrix_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_rows(tx_ids: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
                        "channel_views": "clean",
                        "sat_scenarios": "",
                    }
                )
    return rows


def _logits(tx: str, tx_ids: tuple[str, ...]) -> np.ndarray:
    values = np.full(len(tx_ids), -2.0, dtype=np.float32)
    values[tx_ids.index(tx)] = 4.0
    return values


def _manifest(
    *, leo: bool, checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...]
) -> dict[str, object]:
    return {
        "feature_name": "z_id",
        "checkpoint": str(checkpoint.resolve()),
        "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
        "class_id_to_tx": list(tx_ids),
        "logit_class_order": list(range(len(tx_ids))),
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


def _payload(
    *, leo: bool, checkpoint: Path, checkpoint_sha256: str, tx_ids: tuple[str, ...]
) -> dict[str, np.ndarray]:
    rows = _source_rows(tx_ids)
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
            features.append(_logits(row["tx_ids"], tx_ids))
            logits.append(_logits(row["tx_ids"], tx_ids))
            if leo:
                row["channel_views"] = PAIR.EXPECTED_LEO_RUNTIME_VIEW
                row["sat_scenarios"] = row["sig_ids"].rsplit("-", 2)[0]
        else:
            features.append(np.ones(len(tx_ids), dtype=np.float32))
            logits.append(np.zeros(len(tx_ids), dtype=np.float32))
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "tx_logits": np.asarray(logits, dtype=np.float32),
        "manifest_json": np.asarray(
            json.dumps(
                _manifest(
                    leo=leo,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    tx_ids=tx_ids,
                )
            )
        ),
    }
    for field in PAIR.METADATA_FIELDS:
        payload[field] = np.asarray([row[field] for row in rows])
    return payload


def _save(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _rewrite(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    mutate(payload)
    np.savez(path, **payload)


def _set_manifest(path: Path, mutate) -> None:
    def apply(payload: dict[str, np.ndarray]) -> None:
        manifest = json.loads(str(np.asarray(payload["manifest_json"]).item()))
        mutate(manifest)
        payload["manifest_json"] = np.asarray(json.dumps(manifest))

    _rewrite(path, apply)


def _set_proxy_manifest(path: Path, mutate) -> None:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    manifest = receipt["manifest"]
    mutate(manifest)
    path.write_text(json.dumps(receipt), encoding="utf-8")


def _proxy_metrics(path: Path, clean_path: Path, manifest: dict[str, object], tx_ids: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {
                "phase": "phase1_only_logits_open_set_reject",
                "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
                "feature_npz": str(clean_path.resolve()),
                "source_tx_ids": list(tx_ids),
                "known_query_roles": ["source"],
                "unknown_query_roles": ["proxy_unknown"],
                "known_query_count": 72,
                "unknown_query_count": 1,
                "AUROC_unknown": 0.80,
                "unknown_FAR": 0.20,
                "manifest": manifest,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_pair(root: Path, *, fold: int = 1) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    tx_ids = PAIR.FROZEN_FOLD_SOURCE_TX[int(fold)]
    directory = root / f"F{fold}_inputs"
    directory.mkdir(parents=True, exist_ok=True)
    training_root = root.parent / PAIR.EXPECTED_TRAINING_RUN_LEAF
    c_checkpoint = training_root / f"F{fold}C_CP_SFCE12" / "final_ssdg.pth"
    g_checkpoint = training_root / f"F{fold}G_CP_SFCE12" / "final_ssdg.pth"
    c_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    g_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    c_checkpoint.write_bytes(f"CP-SFCE-C-{fold}".encode("ascii"))
    g_checkpoint.write_bytes(f"CP-SFCE-G-{fold}".encode("ascii"))
    c_sha, g_sha = _sha(c_checkpoint), _sha(g_checkpoint)
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
        _payload(leo=False, checkpoint=c_checkpoint, checkpoint_sha256=c_sha, tx_ids=tx_ids),
    )
    _save(
        paths["g_clean"],
        _payload(leo=False, checkpoint=g_checkpoint, checkpoint_sha256=g_sha, tx_ids=tx_ids),
    )
    _save(
        paths["c_leo"],
        _payload(leo=True, checkpoint=c_checkpoint, checkpoint_sha256=c_sha, tx_ids=tx_ids),
    )
    _save(
        paths["g_leo"],
        _payload(leo=True, checkpoint=g_checkpoint, checkpoint_sha256=g_sha, tx_ids=tx_ids),
    )
    _proxy_metrics(
        paths["c_proxy"],
        paths["c_clean"],
        _manifest(leo=False, checkpoint=c_checkpoint, checkpoint_sha256=c_sha, tx_ids=tx_ids),
        tx_ids,
    )
    _proxy_metrics(
        paths["g_proxy"],
        paths["g_clean"],
        _manifest(leo=False, checkpoint=g_checkpoint, checkpoint_sha256=g_sha, tx_ids=tx_ids),
        tx_ids,
    )
    return paths


def _args(paths: dict[str, object], output: Path, *, fold: int = 1, priors: tuple[Path, ...] = ()):  # noqa: PLR0913
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
        "--expected-target-old-count", "1",
        "--expected-proxy-count", "1",
        "--output-metrics-json", str(output),
    ]
    if priors:
        command.extend(["--aggregate-prior-pair-metrics-json", ",".join(str(path) for path in priors)])
    return PAIR.build_parser().parse_args(command)


def _force_wrong_source_logit(path: Path, tx_ids: tuple[str, ...], *, scenario: str | None = None) -> None:
    def apply(payload: dict[str, np.ndarray]) -> None:
        selected = payload["tx_ids"] == tx_ids[0]
        if scenario is not None:
            selected &= payload["sat_scenarios"] == scenario
        wrong = np.full(len(tx_ids), -2.0, dtype=np.float32)
        wrong[1] = 4.0
        payload["tx_logits"][selected] = wrong

    _rewrite(path, apply)


def test_cp_pair_closes_final_only_bindings_and_four_floors(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    output = Path(paths["root"]) / "F1_C_vs_G_pair_metrics.json"
    metrics = PAIR.evaluate(_args(paths, output))
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema"] == "cvs.phase1.cp_sfce_postfreeze_pair.v1"
    assert persisted["policy"]["fit_performed"] is False
    assert persisted["policy"]["checkpoint_weights_loaded"] is False
    assert persisted["bindings"]["c_final_checkpoint_sha256"] == _sha(paths["c_checkpoint"])
    assert metrics["postfreeze_gates"]["fold_verdict"] == "PENDING_GLOBAL_18_GRID"
    for container in (metrics["clean_source"], *metrics["leo_scenarios"].values()):
        assert set(container["G_minus_C_pp"]) == set(PAIR.CLASSIFICATION_METRICS)
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="refusing to overwrite"):
        PAIR.evaluate(_args(paths, output))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["channel_views"].__setitem__(slice(None), "wrong"))
                for key in ("c_leo", "g_leo")
            ],
            "must use exactly channel_view=single",
        ),
        (
            lambda paths: [
                _set_manifest(paths[key], lambda manifest: manifest["channel_profile"]["source"].__setitem__("view", "clean"))
                for key in ("c_leo", "g_leo")
            ],
            "source channel profile is not satellite",
        ),
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["sig_ids"].__setitem__(0, "changed"))
                for key in ("c_leo", "g_leo")
            ],
            "clean/LEO source physical key sets differ",
        ),
        (
            lambda paths: [
                _set_manifest(paths[key], lambda manifest: manifest.__setitem__("checkpoint_load_strict", False))
                for key in ("c_clean", "g_clean")
            ],
            "checkpoint export was not strict-loaded",
        ),
    ],
)
def test_cp_pair_fails_closed_on_view_profile_physical_or_strict_binding(tmp_path, mutation, message):
    paths = _write_pair(tmp_path / "matrix")
    mutation(paths)
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "out.json"))


def test_cp_pair_rejects_forged_head_contract_in_all_npz_and_proxy_manifests(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    for key in ("c_clean", "g_clean", "c_leo", "g_leo"):
        _set_manifest(
            paths[key],
            lambda manifest: manifest.__setitem__("classification_head_contract", "FORGED_WRONG_HEAD"),
        )
    for key in ("c_proxy", "g_proxy"):
        _set_proxy_manifest(
            paths[key],
            lambda manifest: manifest.__setitem__("classification_head_contract", "FORGED_WRONG_HEAD"),
        )
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="classification_head_contract"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "forged_head.json"))


def test_cp_pair_rejects_forged_head_contract_in_proxy_binding_manifest(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    _set_proxy_manifest(
        paths["c_proxy"],
        lambda manifest: manifest.__setitem__("classification_head_contract", "FORGED_WRONG_HEAD"),
    )
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="C proxy manifest classification_head_contract"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "forged_proxy_head.json"))


def test_cp_pair_rejects_complete_c_g_arm_swap_and_v1_training_root(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    swapped = _args(paths, Path(paths["root"]) / "swapped.json")
    # Swap every C/G input as a group; the frozen arm path still rejects it.
    swapped.c_clean_npz, swapped.g_clean_npz = str(paths["g_clean"]), str(paths["c_clean"])
    swapped.c_leo_npz, swapped.g_leo_npz = str(paths["g_leo"]), str(paths["c_leo"])
    swapped.c_final_checkpoint, swapped.g_final_checkpoint = str(paths["g_checkpoint"]), str(paths["c_checkpoint"])
    swapped.c_proxy_metrics_json, swapped.g_proxy_metrics_json = str(paths["g_proxy"]), str(paths["c_proxy"])
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="C final checkpoint path does not match frozen candidate path"):
        PAIR.evaluate(swapped)

    v1_root = Path(paths["root"]).parent / "phase1_cp_sfce12_20260809_v1"
    v1_root.mkdir(parents=True, exist_ok=True)
    v1_args = _args(paths, Path(paths["root"]) / "v1_root.json")
    v1_args.training_run_root = str(v1_root)
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="training run root leaf must be"):
        PAIR.evaluate(v1_args)


def test_cp_pair_rejects_manifest_checkpoint_path_arm_mismatch(tmp_path):
    paths = _write_pair(tmp_path / "matrix")
    _set_manifest(
        paths["c_leo"],
        lambda manifest: manifest.__setitem__("checkpoint", str(Path(paths["g_checkpoint"]).resolve())),
    )
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="C LEO manifest checkpoint path does not bind frozen F1C_CP_SFCE12"):
        PAIR.evaluate(_args(paths, Path(paths["root"]) / "manifest_arm_mismatch.json"))


def test_cp_pair_outer_values_have_zero_floor_influence_and_proxy_is_non_compensating(tmp_path):
    clean_paths = _write_pair(tmp_path / "clean")
    clean = PAIR.evaluate(_args(clean_paths, Path(clean_paths["root"]) / "clean.json"))
    mutated_paths = _write_pair(tmp_path / "mutated")
    for key in ("c_clean", "g_clean"):
        def mutate_outer(payload):
            mask = payload["dataset_role"] != "source"
            payload["features"][mask] = 999.0
            payload["tx_logits"][mask] = np.asarray([999.0, -999.0, -999.0, -999.0], dtype=np.float32)
        _rewrite(mutated_paths[key], mutate_outer)
    mutated = PAIR.evaluate(_args(mutated_paths, Path(mutated_paths["root"]) / "mutated.json"))
    assert mutated["clean_source"] == clean["clean_source"]
    assert mutated["leo_scenarios"] == clean["leo_scenarios"]
    proxy = json.loads(Path(mutated_paths["g_proxy"]).read_text(encoding="utf-8"))
    proxy["unknown_FAR"] = 0.21
    Path(mutated_paths["g_proxy"]).write_text(json.dumps(proxy), encoding="utf-8")
    rejected = PAIR.evaluate(_args(mutated_paths, Path(mutated_paths["root"]) / "proxy_reject.json"))
    assert rejected["postfreeze_gates"]["fold_verdict"] == "REJECT_CP_SFCE_PERMANENT"


def test_cp_pair_six_fold_equal_weight_aggregate_and_prior_identity_fail_closed(tmp_path):
    root = tmp_path / "matrix"
    priors: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(root, fold=fold)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        priors.append(output)
    final_paths = _write_pair(root, fold=6)
    final = PAIR.evaluate(_args(final_paths, root / "F6_C_vs_G_pair_metrics.json", fold=6, priors=tuple(priors)))
    aggregate = final["matrix_aggregate"]
    assert aggregate["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert aggregate["verdict"] == "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
    assert aggregate["global_18_cell_equal_weight_G_minus_C_pp"] == {
        metric: 0.0 for metric in PAIR.CLASSIFICATION_METRICS
    }

    receipt = json.loads(priors[0].read_text(encoding="utf-8"))
    receipt["candidate_pair"] = "F2_C_vs_G"
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match="candidate_pair does not match frozen fold 1"):
        PAIR.evaluate(_args(final_paths, root / "F6_retry.json", fold=6, priors=tuple(priors)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt, root: receipt.__setitem__("source_tx_ids", list(reversed(receipt["source_tx_ids"]))),
            "source TX order does not match frozen fold 1",
        ),
        (
            lambda receipt, root: receipt.__setitem__("postfreeze_output_root", str((root / "other").resolve())),
            "output root mismatch",
        ),
        (
            lambda receipt, root: receipt.__setitem__(
                "training_run_root", str((root.parent / "phase1_cp_sfce12_20260809_v1").resolve())
            ),
            "training root mismatch",
        ),
        (
            lambda receipt, root: receipt["bindings"].__setitem__("c_candidate", "F1G_CP_SFCE12"),
            "c_candidate does not match frozen fold 1",
        ),
        (
            lambda receipt, root: receipt["postfreeze_gates"]["technical_binding"].__setitem__("passed", "true"),
            "technical binding is not strictly true",
        ),
        (
            lambda receipt, root: receipt["proxy_guardrail"]["C"].__setitem__("AUROC_unknown", float("nan")),
            "proxy guardrail has non-finite or out-of-range value",
        ),
    ],
)
def test_cp_prior_matrix_binding_and_proxy_values_fail_closed(tmp_path, mutate, message):
    root = tmp_path / "matrix"
    priors: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(root, fold=fold)
        output = root / f"F{fold}_C_vs_G_pair_metrics.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        priors.append(output)
    receipt = json.loads(priors[0].read_text(encoding="utf-8"))
    mutate(receipt, root)
    priors[0].write_text(json.dumps(receipt), encoding="utf-8")
    final_paths = _write_pair(root, fold=6)
    with pytest.raises(PAIR.CPSFCEPostfreezePairError, match=message):
        PAIR.evaluate(_args(final_paths, root / "F6_C_vs_G_pair_metrics.json", fold=6, priors=tuple(priors)))


def test_cp_postfreeze_launcher_is_42_steps_and_rejects_non_v2_training_root():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    for required in (
        "phase1_cp_sfce_postfreeze_20260809_v1",
        "phase1_cp_sfce12_20260809_v2",
        "eval_phase1_cp_sfce_pair.py",
        "--source_only_export",
        "--source_channel_view satellite",
        "--source_channel_view clean",
        "--satellite_tta_policy none",
        "--unknown_far_target 0.05",
        "CUDA_VISIBLE_DEVICES=\"\"",
        "[[ \"${POSTFREEZE_ROOT}\" != \"${TRAIN_RUN_ROOT}\" ]]",
        "TRAIN_RUN_ROOT leaf must be phase1_cp_sfce12_20260809_v2",
        "--training-run-root",
    ):
        assert required in text
    assert re.findall(r"^launch_candidate (\d) ([CG]) (\d)$", text, flags=re.MULTILINE) == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_cp_sfce_postfreeze_20260809.sh", "--dry-run"],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=True,
    )
    lines = completed.stdout.splitlines()
    assert sum(line.startswith("[DRY-RUN][CLEAN_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][LEO_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PROXY_SCORE]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PAIR_SCORE]") for line in lines) == 6
    assert len(lines) == 42
    assert all("phase1_cp_sfce_postfreeze_20260809_v1" in line for line in lines)
    invalid = subprocess.run(
        [
            "bash",
            "-c",
            "TRAIN_RUN_ROOT='/tmp/phase1_cp_sfce12_20260809_v1' "
            "bash scripts/launch_phase1_cp_sfce_postfreeze_20260809.sh --dry-run",
        ],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode == 3
    assert "TRAIN_RUN_ROOT leaf must be phase1_cp_sfce12_20260809_v2" in invalid.stderr
    assert "import torch" not in EVALUATOR_PATH.read_text(encoding="utf-8")
