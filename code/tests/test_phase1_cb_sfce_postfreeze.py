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
EVALUATOR_PATH = CODE_ROOT / "scripts" / "eval_phase1_cb_sfce_pair.py"
LAUNCHER_PATH = CODE_ROOT / "scripts" / "launch_phase1_cb_sfce_postfreeze_20260809.sh"
_SPEC = importlib.util.spec_from_file_location("cb_sfce_postfreeze_pair", EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PAIR
_SPEC.loader.exec_module(PAIR)


TX = PAIR.FROZEN_FOLD_SOURCE_TX[1]
RXS = PAIR.EXPECTED_SOURCE_RXS
DAYS = PAIR.EXPECTED_SOURCE_DAYS
SCENARIOS = PAIR.EXPECTED_SCENARIOS
TEST_MATRIX_ID = "test_phase1_cb_sfce_postfreeze_matrix_v1"


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


def _manifest(*, leo: bool, checkpoint_sha256: str, tx_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "feature_name": "z_id",
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


def _payload(*, leo: bool, checkpoint_sha256: str, tx_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
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
        "manifest_json": np.asarray(json.dumps(_manifest(leo=leo, checkpoint_sha256=checkpoint_sha256, tx_ids=tx_ids))),
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


def _proxy_metrics(
    path: Path, clean_path: Path, manifest: dict[str, object], *, arm: str, tx_ids: tuple[str, ...]
) -> None:
    payload = {
        "phase": "phase1_only_logits_open_set_reject",
        "threshold_scope": "source_calibrated_only_no_target_support_no_unknown_query_tuning",
        "feature_npz": str(clean_path.resolve()),
        "source_tx_ids": list(tx_ids),
        "known_query_roles": ["source"],
        "unknown_query_roles": ["proxy_unknown"],
        "known_query_count": 72,
        "unknown_query_count": 1,
        "AUROC_unknown": 0.80 if arm == "C" else 0.80,
        "unknown_FAR": 0.20 if arm == "C" else 0.20,
        "manifest": manifest,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_pair(tmp_path: Path, *, fold: int = 1, postfreeze_root: Path | None = None) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tx_ids = PAIR.FROZEN_FOLD_SOURCE_TX[int(fold)]
    root = (postfreeze_root if postfreeze_root is not None else tmp_path.parent).resolve()
    root.mkdir(parents=True, exist_ok=True)
    c_checkpoint = tmp_path / "c_final_ssdg.pth"
    g_checkpoint = tmp_path / "g_final_ssdg.pth"
    c_checkpoint.write_bytes(b"CB-SFCE-C-final-checkpoint")
    g_checkpoint.write_bytes(b"CB-SFCE-G-final-checkpoint")
    c_sha = PAIR._sha256_file(c_checkpoint)
    g_sha = PAIR._sha256_file(g_checkpoint)
    paths = {
        "c_clean": tmp_path / "c_clean.npz",
        "g_clean": tmp_path / "g_clean.npz",
        "c_leo": tmp_path / "c_leo.npz",
        "g_leo": tmp_path / "g_leo.npz",
        "c_checkpoint": c_checkpoint,
        "g_checkpoint": g_checkpoint,
        "c_proxy": tmp_path / "c_proxy.json",
        "g_proxy": tmp_path / "g_proxy.json",
    }
    _save(paths["c_clean"], _payload(leo=False, checkpoint_sha256=c_sha, tx_ids=tx_ids))
    _save(paths["g_clean"], _payload(leo=False, checkpoint_sha256=g_sha, tx_ids=tx_ids))
    _save(paths["c_leo"], _payload(leo=True, checkpoint_sha256=c_sha, tx_ids=tx_ids))
    _save(paths["g_leo"], _payload(leo=True, checkpoint_sha256=g_sha, tx_ids=tx_ids))
    _proxy_metrics(paths["c_proxy"], paths["c_clean"], _manifest(leo=False, checkpoint_sha256=c_sha, tx_ids=tx_ids), arm="C", tx_ids=tx_ids)
    _proxy_metrics(paths["g_proxy"], paths["g_clean"], _manifest(leo=False, checkpoint_sha256=g_sha, tx_ids=tx_ids), arm="G", tx_ids=tx_ids)
    paths["source_tx_ids"] = tx_ids
    paths["postfreeze_root"] = root
    return paths


def _args(paths: dict[str, object], out_json: Path, *, fold: int = 1, priors: tuple[Path, ...] = ()):  # noqa: PLR0913
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
        "--postfreeze-output-root", str(paths["postfreeze_root"]),
        "--source-tx-ids", ",".join(paths["source_tx_ids"]),
        "--expected-source-count", "72",
        "--expected-target-old-count", "1",
        "--expected-proxy-count", "1",
        "--output-metrics-json", str(out_json),
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


def test_pair_evaluator_closes_four_floors_proxy_and_final_checkpoint_binding(tmp_path):
    paths = _write_pair(tmp_path)
    output = tmp_path / "pair_metrics.json"
    metrics = PAIR.evaluate(_args(paths, output))

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["policy"]["fit_performed"] is False
    assert persisted["policy"]["checkpoint_weights_loaded"] is False
    assert persisted["bindings"]["c_final_checkpoint_sha256"] == PAIR._sha256_file(paths["c_checkpoint"])
    assert persisted["bindings"]["g_final_checkpoint_sha256"] == PAIR._sha256_file(paths["g_checkpoint"])
    assert metrics["postfreeze_gates"]["fold_verdict"] == "PENDING_GLOBAL_18_GRID"
    assert metrics["proxy_guardrail"]["passed"] is True
    assert set(metrics["leo_scenarios"]) == set(SCENARIOS)
    for container in (metrics["clean_source"], *metrics["leo_scenarios"].values()):
        assert set(container["G_minus_C_pp"]) == set(PAIR.CLASSIFICATION_METRICS)
        for arm in ("C", "G"):
            for metric in PAIR.CLASSIFICATION_METRICS:
                assert 0.0 <= container[arm][metric] <= 1.0
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match="refusing to overwrite"):
        PAIR.evaluate(_args(paths, output))


def test_pair_evaluator_uses_real_single_runtime_view_and_satellite_manifest_profile(tmp_path):
    paths = _write_pair(tmp_path)
    for key in ("c_leo", "g_leo"):
        _rewrite(paths[key], lambda payload: payload["channel_views"].__setitem__(slice(None), "wrong"))
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match="must use exactly channel_view=single"):
        PAIR.evaluate(_args(paths, tmp_path / "wrong_view.json"))

    paths = _write_pair(tmp_path / "profile")
    for key in ("c_leo", "g_leo"):
        _set_manifest(paths[key], lambda manifest: manifest["channel_profile"]["source"].__setitem__("view", "clean"))
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match="source channel profile is not satellite"):
        PAIR.evaluate(_args(paths, tmp_path / "wrong_profile.json"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda paths: [
                _rewrite(paths[key], lambda payload: payload["sig_ids"].__setitem__(0, "changed"))
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
                _rewrite(paths[key], lambda payload: payload["sat_scenarios"].__setitem__(0, "invalid"))
                for key in ("c_leo", "g_leo")
            ],
            "LEO scenario set mismatch",
        ),
        (
            lambda paths: [
                _set_manifest(paths[key], lambda manifest: manifest.__setitem__("checkpoint_load_strict", False))
                for key in ("c_clean", "g_clean")
            ],
            "checkpoint export was not strict-loaded",
        ),
        (
            lambda paths: [
            _set_manifest(paths[key], lambda manifest: manifest.__setitem__("class_id_to_tx", list(reversed(TX))))
                for key in ("g_clean", "g_leo")
            ],
            "class label/order mismatch",
        ),
    ],
)
def test_pair_evaluator_fails_closed_on_physical_role_scenario_strict_or_order_binding(tmp_path, mutation, message):
    paths = _write_pair(tmp_path)
    mutation(paths)
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match=message):
        PAIR.evaluate(_args(paths, tmp_path / "out.json"))


def test_pair_evaluator_fails_closed_on_missing_scenario_rx_and_checkpoint_sha(tmp_path):
    paths = _write_pair(tmp_path)
    for key in ("c_leo", "g_leo"):
        def remove_rx(payload):
            selected = (payload["sat_scenarios"] == SCENARIOS[0]) & (payload["rx_ids"] == RXS[0])
            payload["rx_ids"][selected] = RXS[1]
        _rewrite(paths[key], remove_rx)
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match="lacks full source RX coverage"):
        PAIR.evaluate(_args(paths, tmp_path / "missing_rx.json"))

    paths = _write_pair(tmp_path / "checkpoint")
    paths["c_checkpoint"].write_bytes(b"mutated-final-checkpoint")
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match="final checkpoint SHA256 does not bind"):
        PAIR.evaluate(_args(paths, tmp_path / "wrong_checkpoint.json"))


def test_outer_target_and_proxy_values_have_zero_pair_metric_influence(tmp_path):
    clean_paths = _write_pair(tmp_path / "clean")
    clean_metrics = PAIR.evaluate(_args(clean_paths, tmp_path / "clean_metrics.json"))
    mutated_paths = _write_pair(tmp_path / "mutated")
    for key in ("c_clean", "g_clean"):
        def mutate_outer(payload):
            outer = payload["dataset_role"] != "source"
            payload["features"][outer] = 999.0
            payload["tx_logits"][outer] = np.asarray([999.0, -999.0, -999.0, -999.0], dtype=np.float32)
        _rewrite(mutated_paths[key], mutate_outer)
    mutated_metrics = PAIR.evaluate(_args(mutated_paths, tmp_path / "mutated_metrics.json"))
    assert mutated_metrics["clean_source"] == clean_metrics["clean_source"]
    assert mutated_metrics["leo_scenarios"] == clean_metrics["leo_scenarios"]
    assert mutated_metrics["postfreeze_gates"] == clean_metrics["postfreeze_gates"]


def test_proxy_guardrail_is_non_compensating_and_rejects_a_worse_far(tmp_path):
    paths = _write_pair(tmp_path)
    g_proxy = json.loads(paths["g_proxy"].read_text(encoding="utf-8"))
    g_proxy["unknown_FAR"] = 0.21
    paths["g_proxy"].write_text(json.dumps(g_proxy, sort_keys=True), encoding="utf-8")
    metrics = PAIR.evaluate(_args(paths, tmp_path / "proxy_reject.json"))
    assert metrics["proxy_guardrail"]["unknown_FAR_non_increase"] is False
    assert metrics["postfreeze_gates"]["fold_verdict"] == "REJECT_CB_SFCE_PERMANENT"


def test_fold_and_matrix_gates_are_non_compensating_and_permanent_reject(tmp_path):
    prior_outputs: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(tmp_path / f"fold{fold}", fold=fold)
        if fold == 2:
            _force_wrong_source_logit(paths["g_leo"], paths["source_tx_ids"], scenario=SCENARIOS[1])
        output = tmp_path / f"fold{fold}.json"
        record = PAIR.evaluate(_args(paths, output, fold=fold))
        prior_outputs.append(output)
        if fold == 2:
            assert record["postfreeze_gates"]["fold_verdict"] == "REJECT_CB_SFCE_PERMANENT"
    final_paths = _write_pair(tmp_path / "fold6", fold=6)
    final = PAIR.evaluate(_args(final_paths, tmp_path / "fold6.json", fold=6, priors=tuple(prior_outputs)))
    aggregate = final["matrix_aggregate"]
    assert aggregate["verdict"] == "REJECT_CB_SFCE_PERMANENT"
    assert aggregate["gates"]["leo_18of18_four_floors_ge_minus2pp"]["passed"] is False
    assert aggregate["phase3_unknown_capability_claim"] == "NOT_EVALUATED"


def test_six_fold_aggregate_is_exactly_18_cell_equal_weight_and_no_new_head_path(tmp_path):
    prior_outputs: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(tmp_path / f"pass_fold{fold}", fold=fold)
        output = tmp_path / f"pass_fold{fold}.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        prior_outputs.append(output)
    final_paths = _write_pair(tmp_path / "pass_fold6", fold=6)
    final = PAIR.evaluate(_args(final_paths, tmp_path / "pass_fold6.json", fold=6, priors=tuple(prior_outputs)))
    aggregate = final["matrix_aggregate"]
    assert aggregate["fold_indices"] == [1, 2, 3, 4, 5, 6]
    assert aggregate["verdict"] == "PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW"
    assert aggregate["global_18_cell_equal_weight_G_minus_C_pp"] == {
        metric: 0.0 for metric in PAIR.CLASSIFICATION_METRICS
    }
    assert len(aggregate["prior_pair_metrics_bindings"]) == 5
    evaluator_text = EVALUATOR_PATH.read_text(encoding="utf-8")
    assert "import torch" not in evaluator_text
    assert "torch.load" not in evaluator_text
    assert "id_backbone.cls_head.head.weight" not in evaluator_text


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt, root: receipt.__setitem__("candidate_pair", "F2_C_vs_G"),
            "candidate_pair does not match frozen fold 1",
        ),
        (
            lambda receipt, root: receipt.__setitem__("source_tx_ids", list(reversed(receipt["source_tx_ids"]))),
            "source TX order does not match frozen fold 1",
        ),
        (
            lambda receipt, root: receipt.__setitem__("postfreeze_output_root", str((root / "other_run").resolve())),
            "output root mismatch",
        ),
        (
            lambda receipt, root: receipt.__setitem__("postfreeze_matrix_id", "other_matrix"),
            "matrix_id mismatch",
        ),
        (
            lambda receipt, root: receipt["postfreeze_gates"]["technical_binding"].__setitem__("passed", "false"),
            "technical binding is not strictly true",
        ),
        (
            lambda receipt, root: receipt["proxy_guardrail"]["C"].__setitem__("AUROC_unknown", float("nan")),
            "proxy guardrail has non-finite or out-of-range value",
        ),
        (
            lambda receipt, root: receipt["proxy_guardrail"]["G"].__setitem__("unknown_FAR", 1.1),
            "proxy guardrail has non-finite or out-of-range value",
        ),
    ],
)
def test_matrix_aggregate_fails_closed_on_prior_identity_or_receipt_drift(tmp_path, mutate, message):
    prior_outputs: list[Path] = []
    for fold in range(1, 6):
        paths = _write_pair(tmp_path / f"prior_fold{fold}", fold=fold)
        output = tmp_path / f"prior_fold{fold}.json"
        PAIR.evaluate(_args(paths, output, fold=fold))
        prior_outputs.append(output)
    receipt = json.loads(prior_outputs[0].read_text(encoding="utf-8"))
    mutate(receipt, tmp_path)
    prior_outputs[0].write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    final_paths = _write_pair(tmp_path / "prior_fold6", fold=6)
    with pytest.raises(PAIR.CBSFCEPostfreezePairError, match=message):
        PAIR.evaluate(_args(final_paths, tmp_path / "prior_fold6.json", fold=6, priors=tuple(prior_outputs)))


def test_postfreeze_launcher_dry_run_has_frozen_42_steps_cpu_pairs_and_new_root():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    for required in (
        "phase1_cb_sfce_postfreeze_20260809_v1",
        "phase1_cb_sfce12_20260809_v1",
        "eval_phase1_cb_sfce_pair.py",
        "--c-final-checkpoint",
        "--g-final-checkpoint",
        "--postfreeze-matrix-id \"${POSTFREEZE_RUN_ID}\"",
        "--postfreeze-output-root \"${POSTFREEZE_ROOT}\"",
        "--aggregate-prior-pair-metrics-json",
        "source_sat_seed",
        "7281718",
        "simplified_leo_residual",
        "--unknown_far_target 0.05",
        "--known_query_roles source",
        "--unknown_query_roles proxy_unknown",
        "--calibration_roles source",
        "--source_only_export",
        "--source_channel_view satellite",
        "--source_channel_view clean",
        "--target_old_channel_view clean",
        "--proxy_unknown_channel_view clean",
        "--satellite_tta_policy none",
        "CUDA_VISIBLE_DEVICES=\"\"",
        "[[ \"${POSTFREEZE_ROOT}\" != \"${TRAIN_RUN_ROOT}\" ]]",
        "final_ssdg.pth",
    ):
        assert required in text
    assert re.findall(r"^launch_candidate (\d) ([CG]) (\d)$", text, flags=re.MULTILINE) == [
        ("1", "C", "0"), ("5", "G", "0"), ("1", "G", "1"), ("5", "C", "1"),
        ("2", "C", "2"), ("6", "G", "2"), ("2", "G", "3"), ("6", "C", "3"),
        ("3", "C", "4"), ("3", "G", "5"), ("4", "C", "6"), ("4", "G", "7"),
    ]
    assert PAIR.FROZEN_FOLD_SOURCE_TX == {
        1: ("20-15", "20-19", "6-15", "8-20"),
        2: ("14-10", "20-19", "6-15", "8-20"),
        3: ("14-10", "14-7", "6-15", "8-20"),
        4: ("14-10", "14-7", "20-15", "8-20"),
        5: ("14-10", "14-7", "20-15", "20-19"),
        6: ("14-7", "20-15", "20-19", "6-15"),
    }
    completed = subprocess.run(
        ["bash", "scripts/launch_phase1_cb_sfce_postfreeze_20260809.sh", "--dry-run"],
        cwd=str(CODE_ROOT), text=True, capture_output=True, check=True,
    )
    lines = completed.stdout.splitlines()
    assert sum(line.startswith("[DRY-RUN][CLEAN_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][LEO_EXPORT]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PROXY_SCORE]") for line in lines) == 12
    assert sum(line.startswith("[DRY-RUN][PAIR_SCORE]") for line in lines) == 6
    assert len(lines) == 42
    assert sum("--aggregate-prior-pair-metrics-json" in line for line in lines) == 1
    assert all("phase1_cb_sfce_postfreeze_20260809_v1" in line for line in lines)
