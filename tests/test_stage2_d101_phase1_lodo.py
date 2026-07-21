from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pytest
import cvsrffi.stage2_d101_phase1_lodo as lock101
import cvsrffi.stage2_d99_d100_phase1_lodo as base
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer
from test_stage2_d101_shrinkage_rda import _state as _d101_state
from test_stage2_d99_d100_phase1_lodo import (
    CLASSES,
    RECEIVERS,
    _admission_row,
    _archive,
    _base_d99,
    _development_authority,
    _grid,
    _ground_bundle,
    _scorer,
)


def _grid101() -> dict[str, list[float]]:
    return {
        "block_variance_z160": [0.8],
        "block_variance_fft96": [1.1],
        "block_variance_rf32": [0.7],
        "prior_dof": [8.0],
        "target_rank_k5plus": [2.0],
        "lambda_relative": [0.08],
        "rda_temperature": [0.9],
        "d101_alpha": [0.35],
    }


def _complete_record_grid() -> dict[str, list[float]]:
    grid = _grid()
    grid["eta"] = [0.25, 0.30]
    grid["alpha"] = [0.35, 0.45]
    return grid


def _complete_record_grid101() -> dict[str, list[float]]:
    grid = _grid101()
    grid["lambda_relative"] = [0.08, 0.09]
    return grid


def _head_metrics(*, nll: float, gain: float = 0.0) -> dict:
    b_old = 0.80 + gain
    a_old = 0.75 + gain
    new = 0.72 + gain
    harmonic = 2.0 * a_old * new / (a_old + new)
    return {
        "row_count": 50,
        "balanced_accuracy": 0.74 + gain,
        "worst_class_floor": 0.65 + gain,
        "pseudo_old_accuracy": a_old,
        "pseudo_new_accuracy": new,
        "harmonic_old_new": harmonic,
        "balanced_nll": nll,
        "brier": 0.2,
        "per_class_accuracy": {name: 0.74 + gain for name in CLASSES},
        "B_old_pre_increment_accuracy": b_old,
        "A_old_post_increment_accuracy": a_old,
        "seen_new_accuracy": new,
        "H_old_new": harmonic,
        "all_registered_class_floor": 0.65 + gain,
        "forgetting_B_minus_A": b_old - a_old,
        "before_state_is_independently_rebuilt_old_only": True,
        "before_state_is_not_post_head_logit_mask": True,
    }


def _synthetic_joint(**kwargs) -> dict:
    episode = kwargs["episode"]
    fold = kwargs["fold"]
    query_indices = np.asarray(kwargs["query_indices"], dtype=np.int64)
    outer = kwargs["outer_held_receiver"]
    split_name = kwargs["split_name"]
    base_candidate = dict(kwargs["base_candidate"])
    candidate101 = dict(kwargs["d101_candidate"])
    control_mode = kwargs.get("d100_control_mode", "D100_POSITIVE_ALPHA")
    binding = {
        "outer_held_receiver": outer,
        "pseudo_target_receiver": episode.receiver,
        "k_shot": episode.k_shot,
        "fold_id": fold["fold_id"],
        "split_name": split_name,
        "d100_control_mode": control_mode,
        "d100_effective_alpha": float(base_candidate["alpha"]),
        "d100_fallback_prediction_exact_p99": (
            control_mode == "D99_FALLBACK_AFTER_D100_GUARD"
        ),
        "support_indices": episode.support.tolist(),
        "calibration_indices": episode.calibration.tolist(),
        "evaluation_indices": episode.evaluation.tolist(),
        "query_indices": query_indices.tolist(),
        "support_receipt": lock101._array_receipt(episode.support),
        "calibration_receipt": lock101._array_receipt(episode.calibration),
        "evaluation_receipt": lock101._array_receipt(episode.evaluation),
        "query_receipt": lock101._array_receipt(query_indices),
    }
    d99_metrics = _head_metrics(nll=0.45)
    d100_metrics = (
        deepcopy(d99_metrics)
        if control_mode == "D99_FALLBACK_AFTER_D100_GUARD"
        else _head_metrics(nll=0.40)
    )
    d101_metrics = _head_metrics(nll=0.30, gain=0.01)
    # Preserve a strictly better or equal forgetting drop.
    d101_metrics["forgetting_B_minus_A"] = d100_metrics["forgetting_B_minus_A"]
    row = {
        "schema": f"{lock101.SCHEMA}.joint_row",
        "task_receipt_sha256": "a" * 64,
        "outer_held_receiver": outer,
        "receiver": episode.receiver,
        "k_shot": episode.k_shot,
        "fold_id": fold["fold_id"],
        "pseudo_old": list(fold["pseudo_old"]),
        "pseudo_new": list(fold["pseudo_new"]),
        "split_name": split_name,
        "d100_control_mode": control_mode,
        "d100_effective_alpha": float(base_candidate["alpha"]),
        "d100_fallback_prediction_exact_p99": (
            control_mode == "D99_FALLBACK_AFTER_D100_GUARD"
        ),
        "base_candidate": base_candidate,
        "base_candidate_sha256": lock101.canonical_sha256(base_candidate),
        "d101_candidate": candidate101,
        "d101_candidate_sha256": lock101.canonical_sha256(candidate101),
        "episode_binding": binding,
        "typed_d81_batch_receipt_sha256": "b" * 64,
        "typed_d99_bank_receipt_sha256": "c" * 64,
        "d100_state_receipt_sha256": "d" * 64,
        "d101_state_receipt_sha256": "e" * 64,
        "four_heads_same_episode_and_query_receipt": True,
        "d100_d101_share_exact_typed_d99_bank_and_p99": True,
        "metrics": {
            "d81": _head_metrics(nll=0.50),
            "d99": d99_metrics,
            "d100": d100_metrics,
            "d101": d101_metrics,
        },
        "d99_d101_complementarity": {
            "prediction_event_count": 50,
            "disagreement_count": 6,
            "right_correct_when_left_wrong_count": 3,
            "left_correct_when_right_wrong_count": 3,
            "class_balanced_oracle_union_accuracy": 0.96,
            "same_episode_same_truth_same_class_balanced_denominator": True,
        },
        "d99_d100_complementarity": {
            "prediction_event_count": 50,
            "disagreement_count": 4,
            "right_correct_when_left_wrong_count": 2,
            "left_correct_when_right_wrong_count": 2,
            "class_balanced_oracle_union_accuracy": 0.90,
            "same_episode_same_truth_same_class_balanced_denominator": True,
        },
        "d99_d101_changed_count": 6,
        "ground_coverage_rho": 0.2,
        "ground_weight": 0.1,
        "ground_bundle_receipt_sha256": "f" * 64,
        "ground_domain_ids": list(kwargs["outer_ground_bundle"].domain_ids),
        "held_quantization_margin": {
            "scope": "held_phase1_evaluation_transient_fp64_teacher_vs_persistent_int8",
            "row_count": 50,
            "top1_agreement": 1.0,
            "teacher_winner_margin_sign_flip_count": 0,
            "teacher_winner_margin_sign_flip_rate": 0.0,
            "large_margin_threshold": kwargs["gate_lock"].large_margin_threshold,
            "large_margin_row_count": 40,
            "large_margin_flip_count": 0,
            "teacher_persisted": False,
            "checks": {
                "top1_agreement": True,
                "margin_sign_flip_rate": True,
                "large_margin_flip_count": True,
            },
            "passed": True,
        },
        "support_fit_quantization_diagnostic": {
            "scope": "support_fit_diagnostic_not_held_lodo_margin_authority"
        },
        "resource": {"complete_combined_resource_claim": False},
        "resource_defer": "D81_PERSISTENT_HEAD_AND_COMPLETE_GROUND_WIRE_UNAVAILABLE",
        "formal_phase1_eligible": False,
        "target_authority": False,
    }
    row["joint_row_sha256"] = lock101.canonical_sha256(row)
    return row


def _synthetic_joint_d101_gate_fail(**kwargs) -> dict:
    row = _synthetic_joint(**kwargs)
    row["metrics"]["d101"] = deepcopy(row["metrics"]["d100"])
    row["d99_d101_complementarity"] = {
        **row["d99_d101_complementarity"],
        "disagreement_count": 0,
        "right_correct_when_left_wrong_count": 0,
        "left_correct_when_right_wrong_count": 0,
        "class_balanced_oracle_union_accuracy": row["d99_d100_complementarity"][
            "class_balanced_oracle_union_accuracy"
        ],
    }
    row["d99_d101_changed_count"] = 0
    row["joint_row_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in row.items() if key != "joint_row_sha256"}
    )
    return row


def _synthetic_base(**kwargs) -> dict:
    episode = kwargs["episode"]
    fold = kwargs["fold"]
    row = _admission_row(changed_count=2)
    row.update(
        {
            "receiver": episode.receiver,
            "k_shot": episode.k_shot,
            "fold_id": fold["fold_id"],
            "pseudo_old": list(fold["pseudo_old"]),
            "pseudo_new": list(fold["pseudo_new"]),
            "candidate": dict(kwargs["candidate"]),
        }
    )
    outer_bundle = kwargs["outer_bundle"]
    row["d101_selection_binding"] = {
        "outer_held_receiver": kwargs["outer_receiver"],
        "outer_train_receivers": list(kwargs["train_receivers"]),
        "support_indices": episode.support.tolist(),
        "query_indices": episode.calibration.tolist(),
        "support_receipt": lock101._array_receipt(episode.support),
        "query_receipt": lock101._array_receipt(episode.calibration),
        "typed_d99_bank_receipt_sha256": "1" * 64,
        "ground_bundle_receipt_sha256": outer_bundle.bundle_sha256,
        "ground_domain_ids": list(outer_bundle.domain_ids),
    }
    return row


def _synthetic_base_d100_all_fail(**kwargs: object) -> dict[str, object]:
    row = _synthetic_base(**kwargs)
    row["fused"] = {**row["d99"], "balanced_nll": row["d99"]["balanced_nll"]}
    return row


def test_d101_grid_and_fold_order_are_permutation_invariant():
    grid = _grid101()
    permuted = {key: list(reversed(value)) for key, value in reversed(list(grid.items()))}
    assert lock101.d101_candidate_grid(grid) == lock101.d101_candidate_grid(permuted)
    first = lock101._normalize_folds(CLASSES)
    second = lock101._normalize_folds(tuple(reversed(CLASSES)))
    assert first == second


def test_incremental_metrics_use_independent_before_probability_not_post_mask():
    after = np.asarray([[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    before = np.asarray([[0.9, 0.1], [0.8, 0.2]])
    metrics = lock101._incremental_metrics(
        after,
        before,
        np.asarray([0, 1, 2]),
        np.asarray([0, 1]),
        CLASSES,
        CLASSES[:2],
        CLASSES[2:],
    )
    assert metrics["B_old_pre_increment_accuracy"] == 0.5
    assert metrics["A_old_post_increment_accuracy"] == 0.5
    assert metrics["before_state_is_independently_rebuilt_old_only"] is True
    assert metrics["before_state_is_not_post_head_logit_mask"] is True


def test_hard_gate_enforces_counts_receiver_coverage_oracle_pairs_and_k1():
    gate = lock101.D101LODOGateLock()
    episode = base.Episode(
        "rx-a", 1, np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5])
    )
    bundle = _ground_bundle()
    candidate = {key: value[0] for key, value in _grid().items()}
    candidate101 = {key: value[0] for key, value in _grid101().items()}
    rows = []
    for receiver in RECEIVERS[:2]:
        local_episode = base.Episode(
            receiver, 1, episode.support, episode.calibration, episode.evaluation
        )
        rows.append(
            _synthetic_joint(
                episode=local_episode,
                fold={"fold_id": "f", "pseudo_old": CLASSES[:2], "pseudo_new": CLASSES[2:]},
                query_indices=episode.evaluation,
                base_candidate=candidate,
                d101_candidate=candidate101,
                outer_ground_bundle=bundle,
                gate_lock=gate,
                outer_held_receiver=receiver,
                split_name="outer_held_evaluation",
            )
        )
    summary = lock101._aggregate_joint_rows(rows, gate, k_shot=1)
    assert summary["passed"] is True
    assert summary["minimum_each_direction_rescue_count"] == 5
    assert len(summary["d101_rescue_receiver_set"]) == 2
    assert summary["oracle_union_gain"] >= 0.0025
    degraded = deepcopy(rows)
    degraded[0]["metrics"]["d101"]["all_registered_class_floor"] = 0.1
    assert lock101._aggregate_joint_rows(degraded, gate, k_shot=1)["passed"] is False


def test_transient_held_teacher_margin_is_real_and_not_persisted():
    _bundle, _config, ground, _metric, bank, support, config101, state = _d101_state(1)
    query = np.ascontiguousarray(support[0][:4], dtype=np.float32)
    gate = lock101.D101LODOGateLock(
        minimum_top1_agreement=0.0,
        maximum_margin_sign_flip_rate=1.0,
        maximum_large_margin_flip_count=10_000,
    )
    audit = lock101.held_quantization_margin_audit(
        state, bank, ground, config101, query, gate
    )
    assert audit["row_count"] == 4
    assert audit["teacher_persisted"] is False
    assert audit["scope"].startswith("held_phase1_evaluation")
    assert audit["passed"] is True


def test_real_k1_d100_guard_fallback_rebuilds_alpha_zero_state_and_scores_d101():
    validated = base.validate_feature_archive(_archive())
    arrays = validated["arrays"]
    episode = base.build_receiver_lodo_episodes(validated, seed=991)[RECEIVERS[0]][1]
    fold = lock101._normalize_folds(CLASSES)[0]
    full_bundle = _ground_bundle()
    authority = _development_authority(full_bundle)
    outer_receiver = episode.receiver
    outer_domain = authority.receiver_domain_map[outer_receiver]
    outer_bundle = base._subset_ground_bundle(
        full_bundle, held_domain=outer_domain, pseudo_old=CLASSES
    )
    requested = {key: values[0] for key, values in _grid().items()}
    selected_d99 = {
        field: requested[field] for field in lock101.SHARED_D99_FIELDS
    }
    requested_id = lock101.canonical_sha256(requested)
    control = lock101._select_d100_control(
        [{"candidate_id": requested_id, "eligible": False}], selected_d99
    )
    assert control is not None
    assert control["control_mode"] == "D99_FALLBACK_AFTER_D100_GUARD"
    assert control["source_requested_candidate_id"] == requested_id
    effective = control["effective_parameters"]
    assert effective["alpha"] == 0.0
    assert effective["lambda0"] == 1.0
    assert effective["ridge_temperature"] == 1.0
    assert control["effective_control_id"] == lock101.canonical_sha256(effective)

    query_indices = episode.evaluation
    pseudo_old = tuple(fold["pseudo_old"])
    old_mask = np.isin(arrays["labels"][query_indices].astype(str), pseudo_old)
    scorer_contract = {
        "schema": "cvs.phase1.d81.episode_scorer.v1",
        "receipt_sha256": "7" * 64,
    }
    prepared_cache = {}
    gate = lock101.D101LODOGateLock()
    row = lock101._evaluate_joint_candidate(
        arrays=arrays,
        episode=episode,
        query_indices=query_indices,
        fold=fold,
        base_candidate=effective,
        d101_candidate={key: values[0] for key, values in _grid101().items()},
        base_d99_config=_base_d99(full_bundle),
        outer_ground_bundle=outer_bundle,
        authority=authority,
        d81_logits=np.zeros((len(query_indices), len(CLASSES)), dtype=np.float32),
        old_d81_logits=np.zeros(
            (int(np.sum(old_mask)), len(pseudo_old)), dtype=np.float32
        ),
        scorer_contract=scorer_contract,
        gate_lock=gate,
        prepared_cache=prepared_cache,
        outer_held_receiver=outer_receiver,
        split_name="outer_held_evaluation",
        d100_control_mode=control["control_mode"],
    )
    assert row["base_candidate_sha256"] == control["effective_control_id"]
    assert row["d100_effective_alpha"] == 0.0
    assert row["d100_fallback_prediction_exact_p99"] is True
    assert row["metrics"]["d100"] == row["metrics"]["d99"]
    assert row["d101_state_receipt_sha256"] != row["d100_state_receipt_sha256"]
    assert row["held_quantization_margin"]["teacher_persisted"] is False
    assert lock101._verify_joint_row(row, outer_receiver, outer_domain, gate)


def test_end_to_end_synthetic_nested_receipt_is_semantic_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    arrays = _archive()
    archive_path = tmp_path / "phase1.npz"
    np.savez(archive_path, **arrays)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        base,
        "_validate_feature_archive_manifest",
        lambda *args, **kwargs: {
            "full_phase1_lock": False,
            "development_lock_frozen": True,
            "manifest_sha256": "3" * 64,
        },
    )

    def fast_scorer(self, support, labels, query, classes):
        del self
        labels = np.asarray(labels).astype(str)
        prototypes = np.stack(
            [np.mean(support[labels == name], axis=0) for name in classes]
        )
        return np.asarray(query, np.float64) @ np.asarray(prototypes, np.float64).T

    monkeypatch.setattr(D81Phase1EpisodeScorer, "__call__", fast_scorer)
    monkeypatch.setattr(
        lock101, "_evaluate_base_calibration", _synthetic_base_d100_all_fail
    )
    monkeypatch.setattr(lock101, "_evaluate_joint_candidate", _synthetic_joint)
    semantic_verify = lock101.verify_receipt
    bundle = _ground_bundle()
    scorer = _scorer()
    receipt = lock101.run_phase1_d101_nested_lodo(
        archive_path,
        manifest_path,
        "4" * 64,
        ground_bundle=bundle,
        ground_authority=_development_authority(bundle),
        base_d99_config=_base_d99(bundle),
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        base_scorer_receipt_sha256=scorer.scorer_id,
        d99_d100_grid=_complete_record_grid(),
        d101_grid=_complete_record_grid101(),
        gate_lock=lock101.D101LODOGateLock(),
        code_sha256=lock101.current_code_sha256(),
        seed=91,
    )
    assert receipt["status"] == lock101.STATUS_ADMITTED
    assert receipt["formal_phase1_lock"] is False
    assert receipt["target_authority"] is False
    assert receipt["resource_audit"]["complete_combined_resource_claim"] is False
    assert receipt["protocol_audit"]["whole_method_unseen_receiver_generalization_claim"] is False
    assert receipt["protocol_audit"]["r7_final_metrics_used_for_d101_selection"] is False
    probe = deepcopy(receipt)
    probe_sha = probe.pop("receipt_sha256")
    assert probe_sha == lock101.canonical_sha256(probe)
    assert receipt["folds_canonical"] == list(
        lock101._normalize_folds(receipt["classes_canonical_set"])
    )
    first_selection = receipt["k_results"]["1"]["outer_selections"][0]
    outer = first_selection["outer_held_receiver"]
    outer_domain = receipt["ground"]["receiver_domain_map"][outer]
    assert all(
        lock101._verify_base_selection_row(row, outer, outer_domain)
        for record in first_selection["d99_candidates"]
        for row in record["rows"]
    )
    assert all(
        lock101._verify_joint_row(row, outer, outer_domain)
        for record in first_selection["d101_candidates"]
        for row in record["rows"]
    )
    assert all(
        lock101._verify_joint_row(row, row["outer_held_receiver"], receipt["ground"]["receiver_domain_map"][row["outer_held_receiver"]])
        for row in receipt["k_results"]["1"]["outer_held_evaluation_rows"]
    )
    rebuilt99 = [
        lock101._record_d99_candidate(
            record["candidate_id"], record["parameters"], record["rows"], 1
        )
        for record in first_selection["d99_candidates"]
    ]
    assert lock101._jsonable(rebuilt99) == lock101._jsonable(first_selection["d99_candidates"])
    rebuilt100 = [
        lock101._record_d100_candidate(
            record["candidate_id"], record["parameters"], record["rows"]
        )
        for record in first_selection["d100_candidates"]
    ]
    assert lock101._jsonable(rebuilt100) == lock101._jsonable(first_selection["d100_candidates"])
    gate = lock101.D101LODOGateLock(**receipt["gate_lock"])
    rebuilt101 = [
        lock101._record_d101_candidate(
            record["candidate_id"], record["parameters"], record["rows"], gate, 1
        )
        for record in first_selection["d101_candidates"]
    ]
    assert lock101._jsonable(rebuilt101) == lock101._jsonable(first_selection["d101_candidates"])
    result1 = receipt["k_results"]["1"]
    assert lock101._jsonable(
        lock101._aggregate_joint_rows(result1["outer_held_evaluation_rows"], gate, k_shot=1)
    ) == lock101._jsonable(result1["outer_hard_gate"])
    assert lock101._rebuild_grid_from_candidates(
        receipt["base_candidates"], base._GRID_FIELDS, base.candidate_grid
    ) == receipt["base_candidates"]
    assert lock101._rebuild_grid_from_candidates(
        receipt["d101_candidates"], lock101.D101_GRID_FIELDS, lock101.d101_candidate_grid
    ) == receipt["d101_candidates"]
    for k_text, result in receipt["k_results"].items():
        k_value = int(k_text)
        assert result["all_outer_nested_winners_available"] is True
        assert result["passed"] is True
        for selection in result["outer_selections"]:
            outer_value = selection["outer_held_receiver"]
            domain_value = receipt["ground"]["receiver_domain_map"][outer_value]
            assert lock101._winner(
                [
                    lock101._record_d99_candidate(
                        record["candidate_id"], record["parameters"], record["rows"], k_value
                    )
                    for record in selection["d99_candidates"]
                ]
            )["candidate_id"] == selection["selected_d99_candidate_id"]
            selected99 = lock101._winner(
                [
                    lock101._record_d99_candidate(
                        record["candidate_id"], record["parameters"], record["rows"], k_value
                    )
                    for record in selection["d99_candidates"]
                ]
            )
            assert selected99 is not None
            records100 = [
                lock101._record_d100_candidate(
                    record["candidate_id"], record["parameters"], record["rows"]
                )
                for record in selection["d100_candidates"]
            ]
            control = lock101._select_d100_control(records100, selected99["parameters"])
            assert control == selection["selected_d100_control"]
            assert control["effective_control_id"] == selection[
                "selected_d100_candidate_id"
            ]
            assert control["control_mode"] == "D99_FALLBACK_AFTER_D100_GUARD"
            assert control["effective_parameters"]["alpha"] == 0.0
            assert selection["selected_d100_source_requested_candidate_id"] == control[
                "source_requested_candidate_id"
            ]
            assert lock101._winner(
                [
                    lock101._record_d101_candidate(
                        record["candidate_id"], record["parameters"], record["rows"], gate, k_value
                    )
                    for record in selection["d101_candidates"]
                ]
            )["candidate_id"] == selection["selected_d101_candidate_id"]
            assert all(
                lock101._verify_joint_row(row, outer_value, domain_value)
                for row in result["outer_held_evaluation_rows"]
                if row["outer_held_receiver"] == outer_value
            )
            selected_outer_rows = [
                row
                for row in result["outer_held_evaluation_rows"]
                if row["outer_held_receiver"] == outer_value
            ]
            assert selected_outer_rows
            assert all(
                row["d100_control_mode"] == "D99_FALLBACK_AFTER_D100_GUARD"
                and row["d100_effective_alpha"] == 0.0
                and row["d100_fallback_prediction_exact_p99"] is True
                for row in selected_outer_rows
            )
    assert semantic_verify(receipt)
    for result in receipt["k_results"].values():
        expected_outer_keys = {
            (receiver, fold["fold_id"])
            for receiver in receipt["receivers_canonical"]
            for fold in receipt["folds_canonical"]
        }
        actual_outer_keys = [
            (row["outer_held_receiver"], row["fold_id"])
            for row in result["outer_held_evaluation_rows"]
        ]
        assert len(actual_outer_keys) == len(set(actual_outer_keys))
        assert set(actual_outer_keys) == expected_outer_keys
        for selection in result["outer_selections"]:
            assert selection["outer_held_receiver"] not in selection["outer_train_receivers"]
            assert selection["outer_held_ground_domain"] not in selection["outer_ground_domain_ids"]
            expected_inner_keys = {
                (receiver, fold["fold_id"])
                for receiver in selection["outer_train_receivers"]
                for fold in receipt["folds_canonical"]
            }
            for candidate_key in ("d99_candidates", "d100_candidates", "d101_candidates"):
                for candidate in selection[candidate_key]:
                    actual_inner_keys = [
                        (row["receiver"], row["fold_id"]) for row in candidate["rows"]
                    ]
                    assert len(actual_inner_keys) == len(set(actual_inner_keys))
                    assert set(actual_inner_keys) == expected_inner_keys
        for row in result["outer_held_evaluation_rows"]:
            binding = row["episode_binding"]
            assert not set(binding["support_indices"]) & set(binding["query_indices"])

    tampered = deepcopy(receipt)
    tampered["k_results"]["1"]["outer_selections"][0][
        "selected_d100_candidate_id"
    ] = "0" * 64
    tampered["receipt_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    assert lock101.verify_receipt(tampered) is False

    mask_forgery = deepcopy(receipt)
    row = mask_forgery["k_results"]["1"]["outer_held_evaluation_rows"][0]
    row["metrics"]["d101"]["before_state_is_not_post_head_logit_mask"] = False
    row["joint_row_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in row.items() if key != "joint_row_sha256"}
    )
    mask_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in mask_forgery.items() if key != "receipt_sha256"}
    )
    assert lock101.verify_receipt(mask_forgery) is False

    margin_forgery = deepcopy(receipt)
    margin_row = margin_forgery["k_results"]["1"]["outer_held_evaluation_rows"][0]
    margin_row["held_quantization_margin"]["top1_agreement"] = 0.0
    margin_row["joint_row_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in margin_row.items() if key != "joint_row_sha256"}
    )
    margin_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in margin_forgery.items() if key != "receipt_sha256"}
    )
    assert lock101.verify_receipt(margin_forgery) is False

    authority_forgery = deepcopy(receipt)
    authority_forgery["formal_phase1_lock"] = True
    authority_forgery["target_authority"] = True
    authority_forgery["resource_audit"]["complete_combined_resource_claim"] = True
    authority_forgery["protocol_audit"]["r7_final_metrics_used_for_d101_selection"] = True
    authority_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {key: value for key, value in authority_forgery.items() if key != "receipt_sha256"}
    )
    assert lock101.verify_receipt(authority_forgery) is False

    effective_parameter_forgery = deepcopy(receipt)
    forged_selection = effective_parameter_forgery["k_results"]["1"][
        "outer_selections"
    ][0]
    source_requested_id = forged_selection["selected_d100_control"][
        "source_requested_candidate_id"
    ]
    source_requested = next(
        record
        for record in forged_selection["d100_candidates"]
        if record["candidate_id"] == source_requested_id
    )
    forged_selection["selected_d100_control"]["effective_parameters"]["alpha"] = (
        source_requested["requested_parameters"]["alpha"]
    )
    effective_parameter_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in effective_parameter_forgery.items()
            if key != "receipt_sha256"
        }
    )
    assert lock101.verify_receipt(effective_parameter_forgery) is False

    for candidate_key, protected_id_key in (
        ("d99_candidates", "selected_d99_candidate_id"),
        ("d100_candidates", "selected_d100_source_requested_candidate_id"),
        ("d101_candidates", "selected_d101_candidate_id"),
    ):
        missing_candidate_forgery = deepcopy(receipt)
        forged_selection = missing_candidate_forgery["k_results"]["1"][
            "outer_selections"
        ][0]
        records = forged_selection[candidate_key]
        protected_id = forged_selection[protected_id_key]
        victim_index = next(
            index
            for index, record in enumerate(records)
            if record["candidate_id"] != protected_id
        )
        records.pop(victim_index)
        missing_candidate_forgery["receipt_sha256"] = lock101.canonical_sha256(
            {
                key: value
                for key, value in missing_candidate_forgery.items()
                if key != "receipt_sha256"
            }
        )
        assert lock101.verify_receipt(missing_candidate_forgery) is False

    reordered_candidate_forgery = deepcopy(receipt)
    reordered_candidate_forgery["k_results"]["1"]["outer_selections"][0][
        "d101_candidates"
    ].reverse()
    reordered_candidate_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in reordered_candidate_forgery.items()
            if key != "receipt_sha256"
        }
    )
    assert lock101.verify_receipt(reordered_candidate_forgery) is False

    duplicate_selection_forgery = deepcopy(receipt)
    forged_selections = duplicate_selection_forgery["k_results"]["1"][
        "outer_selections"
    ]
    forged_selections[1] = deepcopy(forged_selections[0])
    duplicate_selection_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in duplicate_selection_forgery.items()
            if key != "receipt_sha256"
        }
    )
    assert lock101.verify_receipt(duplicate_selection_forgery) is False

    duplicate_outer_row_forgery = deepcopy(receipt)
    forged_result = duplicate_outer_row_forgery["k_results"]["1"]
    forged_outer_rows = forged_result["outer_held_evaluation_rows"]
    forged_outer_rows[1] = deepcopy(forged_outer_rows[0])
    forged_result["outer_hard_gate"] = lock101._aggregate_joint_rows(
        forged_outer_rows, gate, k_shot=1
    )
    forged_result["passed"] = bool(forged_result["outer_hard_gate"]["passed"])
    forged_scientific = all(
        value["passed"] for value in duplicate_outer_row_forgery["k_results"].values()
    )
    duplicate_outer_row_forgery["scientific_phase1_hard_gate_passed"] = forged_scientific
    duplicate_outer_row_forgery["status"] = (
        lock101.STATUS_ADMITTED if forged_scientific else lock101.STATUS_REJECTED
    )
    duplicate_outer_row_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in duplicate_outer_row_forgery.items()
            if key != "receipt_sha256"
        }
    )
    assert lock101.verify_receipt(duplicate_outer_row_forgery) is False

    duplicate_inner_row_forgery = deepcopy(receipt)
    forged_inner_selection = duplicate_inner_row_forgery["k_results"]["1"][
        "outer_selections"
    ][0]
    forged_inner_record = forged_inner_selection["d101_candidates"][0]
    forged_inner_rows = deepcopy(forged_inner_record["rows"])
    forged_inner_rows[1] = deepcopy(forged_inner_rows[0])
    forged_inner_selection["d101_candidates"][0] = lock101._record_d101_candidate(
        forged_inner_record["candidate_id"],
        forged_inner_record["parameters"],
        forged_inner_rows,
        gate,
        1,
    )
    duplicate_inner_row_forgery["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in duplicate_inner_row_forgery.items()
            if key != "receipt_sha256"
        }
    )
    assert lock101.verify_receipt(duplicate_inner_row_forgery) is False


    monkeypatch.setattr(
        lock101, "_evaluate_joint_candidate", _synthetic_joint_d101_gate_fail
    )
    rejected = lock101.run_phase1_d101_nested_lodo(
        archive_path,
        manifest_path,
        "4" * 64,
        ground_bundle=bundle,
        ground_authority=_development_authority(bundle),
        base_d99_config=_base_d99(bundle),
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        base_scorer_receipt_sha256=scorer.scorer_id,
        d99_d100_grid=_complete_record_grid(),
        d101_grid=_complete_record_grid101(),
        gate_lock=lock101.D101LODOGateLock(),
        code_sha256=lock101.current_code_sha256(),
        seed=91,
    )
    assert rejected["status"] == lock101.STATUS_REJECTED
    assert rejected["scientific_phase1_hard_gate_passed"] is False
    assert semantic_verify(rejected)
    assert all(
        result["all_outer_nested_winners_available"] is False
        and result["outer_held_evaluation_rows"] == []
        and result["outer_hard_gate"] is None
        and result["passed"] is False
        for result in rejected["k_results"].values()
    )
    assert all(
        selection["selection_status"] == "REJECTED_D101_INNER_GATE"
        and selection["selected_d101_candidate_id"] is None
        for result in rejected["k_results"].values()
        for selection in result["outer_selections"]
    )

    failed_outer = receipt["receivers_canonical"][0]

    def _synthetic_joint_mixed_outer(**kwargs):
        if kwargs["outer_held_receiver"] == failed_outer:
            return _synthetic_joint_d101_gate_fail(**kwargs)
        return _synthetic_joint(**kwargs)

    monkeypatch.setattr(lock101, "_evaluate_joint_candidate", _synthetic_joint_mixed_outer)
    partial = lock101.run_phase1_d101_nested_lodo(
        archive_path,
        manifest_path,
        "4" * 64,
        ground_bundle=bundle,
        ground_authority=_development_authority(bundle),
        base_d99_config=_base_d99(bundle),
        base_scorer=scorer,
        base_scorer_id=scorer.scorer_id,
        base_scorer_receipt_sha256=scorer.scorer_id,
        d99_d100_grid=_complete_record_grid(),
        d101_grid=_complete_record_grid101(),
        gate_lock=lock101.D101LODOGateLock(),
        code_sha256=lock101.current_code_sha256(),
        seed=91,
    )
    assert partial["status"] == lock101.STATUS_REJECTED
    assert semantic_verify(partial)
    fold_ids = {fold["fold_id"] for fold in partial["folds_canonical"]}
    successful_receivers = set(partial["receivers_canonical"]) - {failed_outer}
    expected_partial_keys = {
        (receiver, fold_id)
        for receiver in successful_receivers
        for fold_id in fold_ids
    }
    for result in partial["k_results"].values():
        actual_partial_keys = {
            (row["outer_held_receiver"], row["fold_id"])
            for row in result["outer_held_evaluation_rows"]
        }
        assert actual_partial_keys == expected_partial_keys
        assert result["all_outer_nested_winners_available"] is False
        assert result["outer_hard_gate"] is None
        assert result["passed"] is False

    missing_partial_row = deepcopy(partial)
    missing_partial_row["k_results"]["1"]["outer_held_evaluation_rows"].pop()
    missing_partial_row["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in missing_partial_row.items()
            if key != "receipt_sha256"
        }
    )
    assert semantic_verify(missing_partial_row) is False

    duplicate_partial_row = deepcopy(partial)
    partial_rows = duplicate_partial_row["k_results"]["1"][
        "outer_held_evaluation_rows"
    ]
    partial_rows[1] = deepcopy(partial_rows[0])
    duplicate_partial_row["receipt_sha256"] = lock101.canonical_sha256(
        {
            key: value
            for key, value in duplicate_partial_row.items()
            if key != "receipt_sha256"
        }
    )
    assert semantic_verify(duplicate_partial_row) is False


def test_formal_and_target_paths_are_unconditionally_blocked():
    signature = inspect.signature(lock101.run_phase1_d101_nested_lodo)
    assert not any(
        name in signature.parameters
        for name in ("evaluator", "rows", "r7_metrics", "target_metrics", "n607_result")
    )
    with pytest.raises(lock101.D101Phase1LODOError, match="cannot authorize"):
        lock101.predict_formal()
