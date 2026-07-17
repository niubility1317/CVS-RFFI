from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys

import numpy as np

from cvsrffi.stage2_ciaf import Int8DomainClassComponent
from cvsrffi.stage2_diag_cosine_exploration import registered_feature


SCRIPTS = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_d25_support_only_concat as runner


OLD = ("old-a", "old-b", "old-c")
NEW = ("new-x", "new-y")


def _component() -> Int8DomainClassComponent:
    q = np.zeros((4, len(OLD), 160), dtype=np.int8)
    scale = np.full((4, len(OLD)), 1.0 / 127.0, dtype=np.float16)
    mask = np.ones((4, len(OLD)), dtype=np.uint8)
    for domain in range(4):
        for class_index in range(len(OLD)):
            q[domain, class_index, class_index] = 127
            q[domain, class_index, 40 + domain] = domain + 1
    return Int8DomainClassComponent(q, scale, mask, OLD)


def _fold_support() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    for class_index, label in enumerate(OLD + NEW):
        for rank in range(10):
            z = np.zeros(160, dtype=np.float32)
            fft = np.zeros(96, dtype=np.float32)
            rf = np.zeros(32, dtype=np.float32)
            z[class_index] = 1.0
            fft[class_index] = 1.0
            rf[class_index] = 1.0
            delta = np.float32(0.002 * (rank - 4.5))
            z[80 + class_index] = delta
            fft[60 + class_index] = delta
            rf[20 + class_index] = delta
            labels.append(label)
            ranks.append(rank)
            z_rows.append(z)
            fft_rows.append(fft)
            rf_rows.append(rf)
    return (
        {
            "labels": np.asarray(labels),
            "ranks": np.asarray(ranks, dtype=np.int64),
        },
        np.stack(z_rows),
        np.stack(fft_rows),
        np.stack(rf_rows),
    )


def test_parser_and_run_signature_expose_no_query_truth_or_scorer_surface() -> None:
    forbidden = ("query", "truth", "scorer", "role", "quota", "assignment", "source", "clean")
    destinations = {action.dest.lower() for action in runner.build_parser()._actions}
    parameters = {name.lower() for name in inspect.signature(runner.run).parameters}
    for names in (destinations, parameters):
        assert not any(token in name for name in names for token in forbidden)


def test_candidate_lock_contains_exactly_two_controls_and_three_d25_routes() -> None:
    candidates = runner.preregistered_candidates()
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D25_C1,
        runner.D25_C2,
    )
    assert len(candidates) == 5
    assert runner.D25_CANDIDATES == (runner.D25_C0, runner.D25_C1, runner.D25_C2)
    lock = runner._candidate_lock(candidates)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v1"
    assert len(lock["candidates"]) == 5
    assert len(lock["sha256"]) == 64
    assert sum(row["eligible_positive_route"] for row in lock["candidates"]) == 3
    assert [row["family"] for row in lock["candidates"]] == [
        "control",
        "control",
        "d25",
        "d25",
        "d25",
    ]


def test_precomputed_blocks_rebuild_historical_b3_feature_exactly() -> None:
    rng = np.random.default_rng(713101)
    iq = rng.normal(size=(7, 2, 256)).astype(np.float32)
    z_id160 = rng.normal(size=(7, 160)).astype(np.float32)
    fft96 = runner.spectral_logmag_sketch(iq)
    rf32 = runner.rf_statistics(iq)
    rebuilt = runner._d1_feature_from_blocks(z_id160, fft96, rf32)
    reference = registered_feature(iq, z_id160)
    assert rebuilt.shape == (7, 288)
    np.testing.assert_allclose(rebuilt, reference, atol=1.0e-7)


def test_matrix_cardinality_is_locked_to_five_by_three_by_five() -> None:
    assert len(runner.preregistered_candidates()) == 5
    assert len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) == 3
    assert len(runner.HELD_RANKS) == 5
    assert 5 * 3 * 5 == 75
    assert all(len(pair) == 2 for pair in runner.HELD_RANKS)
    assert len(set(runner.HELD_RANKS)) == len(runner.HELD_RANKS)
    source = inspect.getsource(runner.run)
    assert "expected_rows != 75" in source
    assert '"cvs.phase2.d25.support_fold.v1"' in source


def test_d25_fold_freezes_old_prefix_scores_and_one_support_row_per_iq() -> None:
    rows, z_id160, fft96, rf32 = _fold_support()
    row = runner._evaluate_d25_fold(
        _component(),
        rows,
        z_id160,
        fft96,
        rf32,
        old_classes=OLD,
        new_classes=NEW,
        held_ranks=(8, 9),
        candidate_id=runner.D25_C1,
        config=runner.preregistered_candidates()[runner.D25_C1],
    )
    assert row["fit_k_shot"] == 8
    assert row["old_prefix_sha256_before"] == row["old_prefix_sha256_after"]
    assert len(row["old_prefix_sha256_before"]) == 64
    assert row["old_score_columns_bitwise_unchanged"] is True
    assert row["resource"]["old_score_columns_bitwise_unchanged_after_registration"] is True
    assert row["resource"]["support_view_count"] == 1
    assert row["resource"]["support_row_multiplicity"] == 1
    assert row["resource"]["derived_support_rows"] == 0
    assert row["resource"]["additional_physical_sample_count"] == 0
    assert row["resource"]["additional_leo_overlay_count"] == 0
    assert row["resource"]["query_features_used_for_fit"] is False
    assert row["resource"]["query_labels_used_for_fit"] is False
    assert row["geometry_summary"]["schema"] == "cvs.phase2.d25.geometry_summary.v1"


def _written_artifact_names() -> set[str]:
    tree = ast.parse(inspect.getsource(runner.run))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        function_name = (
            function.attr if isinstance(function, ast.Attribute) else function.id
            if isinstance(function, ast.Name)
            else ""
        )
        if function_name not in {"_write_json", "_write_jsonl"}:
            continue
        path = node.args[0]
        if (
            isinstance(path, ast.BinOp)
            and isinstance(path.op, ast.Div)
            and isinstance(path.right, ast.Constant)
            and isinstance(path.right.value, str)
        ):
            names.add(path.right.value)
    return names


def test_artifact_schema_is_fixed_and_never_writes_features_or_prototypes() -> None:
    assert _written_artifact_names() == {
        "training_log.jsonl",
        "support_audit.json",
        "selection.json",
        "resource_audit.json",
        "geometry_audit.json",
        "RECEIPT.json",
    }
    source = inspect.getsource(runner.run)
    schemas = {
        "cvs.phase2.d25.support_fold.v1",
        "cvs.phase2.d25.support_audit.v1",
        "cvs.phase2.d25.selection.v1",
        "cvs.phase2.d25.resource_matrix.v1",
        "cvs.phase2.d25.geometry_matrix.v1",
        "cvs.phase2.d25.receipt.v1",
    }
    assert all(f'"{schema}"' in source for schema in schemas)
    artifact_names = _written_artifact_names()
    assert not any(
        forbidden in name.lower()
        for name in artifact_names
        for forbidden in ("feature", "prototype", "embedding", "logit", "iq")
    )
    # Feature arrays remain process-local dictionaries; only per-row one-way
    # hashes and aggregate audits are admitted to the persisted support audit.
    assert '"fft96_sha256"' in source
    assert '"rf32_sha256"' in source
    assert '"scene_z"' not in " ".join(artifact_names)
    assert '"scene_fft"' not in " ".join(artifact_names)
    assert '"scene_rf"' not in " ".join(artifact_names)


def test_operator_lineage_keeps_one_row_and_no_extra_view_or_overlay() -> None:
    rows = {
        "tokens": np.asarray(["sample-a", "sample-b"]),
        "hashes": np.asarray(["1" * 64, "2" * 64]),
    }
    lineage = runner._operator_lineage(rows)
    assert len(lineage) == 2
    for row in lineage:
        assert row["feature_operator_ids"] == [
            "adv3b02_zid160_base_v1",
            "same_received_iq_fft96_v1",
            "same_received_iq_rf32_v1",
        ]
        assert row["support_row_multiplicity"] == 1
        assert row["derived_support_rows"] == 0
        assert row["additional_physical_sample_count"] == 0
        assert row["additional_leo_overlay_count"] == 0
