from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

import pytest

from cvsrffi.stage2_d105_target25_runner import (
    ARMS,
    D105Target25GPUSchedule,
    D105Target25PredictionOutput,
    D105Target25PredictionRequest,
    D105Target25RunnerError,
    D105Target25ScenarioPlan,
    D105Target25StatePlan,
    D105Target25TruthLabels,
    DEVELOPMENT_CLAIM_SCOPE,
    FORMAL_CLAIM_SCOPE,
    OUTER_ROW_COUNT,
    SCENARIO_ARM_PAIR_COUNT,
    SCENARIO_ROW_COUNT,
    STATE_PREDICTION_SURFACE_COUNT,
    TARGET25_SEED,
    TARGET25_SLICES,
    build_d105_target25_truth_side_manifest,
    canonical_sha256,
    execute_d105_target25_predictions,
    freeze_d105_target25_plan,
    load_d105_target25_truth_catalog_manifest,
    prepare_d105_target25_run,
    score_d105_target25_from_catalog_file,
    score_d105_target25_truth_side,
    seal_d105_target25_truth_catalog_manifest,
    summarize_d105_target25_outputs,
    verify_d105_target25_score_manifest,
    verify_d105_target25_prediction_manifest,
    write_d105_target25_truth_side_manifest,
)
from cvsrffi.stage2_d105_phase1_bundle import (
    D105_CANDIDATE_RUNTIME_ENTRYPOINTS,
    D105_CANDIDATE_RUNTIME_FILES,
)


RECEIVERS = ("rx-a", "rx-b", "rx-c", "rx-d", "rx-e")
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_CANDIDATE_CHECKPOINT_SHA = _sha("candidate-checkpoint")
_CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
_CANDIDATE_CORE_FILES = D105_CANDIDATE_RUNTIME_FILES
_CANDIDATE_RUNTIME_DOCUMENT = {
    "schema": "cvs.stage2.d105.candidate_runtime_manifest.v1",
    "candidate_id": "D105-CBRC+LPO-RC",
    "protocol_schema": "p2_min_v1",
    "checkpoint_sha256": _CANDIDATE_CHECKPOINT_SHA,
    "entrypoints": dict(D105_CANDIDATE_RUNTIME_ENTRYPOINTS),
    "core_file_sha256": {
        relative: hashlib.sha256((_CODE_ROOT / relative).read_bytes()).hexdigest()
        for relative in _CANDIDATE_CORE_FILES
    },
}
_CANDIDATE_RUNTIME_SHA = canonical_sha256(_CANDIDATE_RUNTIME_DOCUMENT)
_CANDIDATE_LOCK_DOCUMENT = {
    "schema": "cvs.stage2.d105.candidate_method_lock.v1",
    "candidate_id": "D105-CBRC+LPO-RC",
    "protocol_schema": "p2_min_v1",
    "checkpoint_sha256": _CANDIDATE_CHECKPOINT_SHA,
    "d105_candidate_runtime_manifest_sha256": _CANDIDATE_RUNTIME_SHA,
    "d105_cbrc": {
        "semantic_revision": "cbrc_mb4_task_balanced_huber_irls4_fp16_v1",
        "code_dim": 4,
        "domain_dim": 32,
        "allowed_k": [1, 5, 10],
        "irls_steps": 4,
        "old_new_task_mass": [0.5, 0.5],
        "k1_zero_coefficient": True,
        "ground_old_multiprototype_enabled": False,
        "deployment_coefficient_dtype": "float16",
        "query_transform": "relu_l2norm_pre_relu_plus_mb4",
        "query_state_updates": 0,
    },
    "student_t_qknn": {
        "student_nu": 3.0,
        "kernel_effective_dim": 12,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.35,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "temperature": 0.85,
        "support_storage": "int8_fp16_scale",
    },
    "four_arm": {
        "arms": ["M0", "M_DA", "M_HEAD", "M_JOINT"],
        "same_da_state_for_da_and_joint": True,
        "same_head_code_config_for_head_and_joint": True,
        "query_truth_surface": False,
    },
    "source_held": {
        "receiver_held_k": [1, 5, 10],
        "class_loco_k": 1,
        "tx_probe_algorithm": "receiver_held_ridge_l2_0.01",
        "tx_probe_max_balanced_accuracy": 0.25,
        "int8_min_top1_agreement": 0.995,
        "large_margin_minimum": 0.10,
        "large_margin_flip_max": 0,
        "truth_open_after_prediction": True,
    },
    "target25": {
        "seed": 713102,
        "claim_scope": "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE",
        "formal_launch_authority": False,
        "slices": [[10, 5], [10, 10], [10, 20], [5, 20], [1, 20]],
        "leo_scenarios": list(SCENARIOS),
        "outer_row_count": 25,
        "scenario_arm_pair_count": 300,
        "state_prediction_surface_count": 600,
    },
}
_CANDIDATE_LOCK_SHA = canonical_sha256(_CANDIDATE_LOCK_DOCUMENT)
_CANDIDATE_FILES = TemporaryDirectory()
_CANDIDATE_ROOT = Path(_CANDIDATE_FILES.name)
_CANDIDATE_RUNTIME_PATH = _CANDIDATE_ROOT / "candidate_runtime.json"
_CANDIDATE_LOCK_PATH = _CANDIDATE_ROOT / "candidate_lock.json"
_CANDIDATE_RUNTIME_PATH.write_bytes(
    json.dumps(
        _CANDIDATE_RUNTIME_DOCUMENT,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
)
_CANDIDATE_LOCK_PATH.write_bytes(
    json.dumps(
        _CANDIDATE_LOCK_DOCUMENT,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
)
for _candidate_path in (_CANDIDATE_RUNTIME_PATH, _CANDIDATE_LOCK_PATH):
    _candidate_path.chmod(stat.S_IREAD)


def _state(
    *,
    receiver: str,
    k_shot: int,
    new_count: int,
    scenario: str,
    stage: str,
    old: tuple[str, ...],
    new: tuple[str, ...],
) -> D105Target25StatePlan:
    classes = old if stage == "S_B" else old + new
    support = tuple(
        f"{stage}-support-{receiver}-{scenario}-n{new_count}-{class_id}-s{shot}"
        for class_id in classes
        for shot in range(k_shot)
    )
    old_query = tuple(
        f"oldq-{receiver}-{scenario}-n{new_count}-{index}" for index in range(3)
    )
    query = old_query if stage == "S_B" else old_query + tuple(
        f"newq-{receiver}-{scenario}-n{new_count}-{index}" for index in range(3)
    )
    return D105Target25StatePlan(
        stage=stage,
        capsule_id=_sha(f"capsule:{stage}:{receiver}:{scenario}:n{new_count}"),
        split_id=_sha(f"split:{stage}:{receiver}:{scenario}:k{k_shot}:n{new_count}"),
        authority_receipt_sha256=_sha(
            f"authority:{stage}:{receiver}:{scenario}:k{k_shot}:n{new_count}"
        ),
        authority_envelope_sha256=_sha(f"authority-envelope:{receiver}"),
        data_feature_runtime_sha256=_sha("d92-feature-runtime"),
        data_materialization_lock_sha256=_sha("d92-materialization-lock"),
        d105_candidate_runtime_manifest_sha256=_CANDIDATE_RUNTIME_SHA,
        d105_candidate_method_lock_sha256=_CANDIDATE_LOCK_SHA,
        support_physical_ids=support,
        query_physical_ids=query,
        registered_classes=classes,
        old_classes=old,
        new_classes=() if stage == "S_B" else new,
        prediction_context_sha256=_sha(
            f"d105-feature-tap:{stage}:{receiver}:{scenario}:k{k_shot}:n{new_count}"
        ),
    )


def _scenario(
    receiver: str, k_shot: int, new_count: int, scenario: str
) -> D105Target25ScenarioPlan:
    old = ("old-a", "old-b")
    new = tuple(f"new-{index:02d}" for index in range(new_count))
    return D105Target25ScenarioPlan(
        scenario=scenario,
        before=_state(
            receiver=receiver,
            k_shot=k_shot,
            new_count=new_count,
            scenario=scenario,
            stage="S_B",
            old=old,
            new=new,
        ),
        after=_state(
            receiver=receiver,
            k_shot=k_shot,
            new_count=new_count,
            scenario=scenario,
            stage="S_C",
            old=old,
            new=new,
        ),
    )


def _plan_inputs():
    return {
        (receiver, k_shot, new_count): tuple(
            _scenario(receiver, k_shot, new_count, scenario)
            for scenario in SCENARIOS
        )
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    }


def _plan():
    return freeze_d105_target25_plan(
        candidate_runtime_manifest_path=_CANDIDATE_RUNTIME_PATH,
        candidate_method_lock_path=_CANDIDATE_LOCK_PATH,
        receivers=RECEIVERS,
        scenario_plans=_plan_inputs(),
        seed=TARGET25_SEED,
    )


def _make_writable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _predictor(requests: list[D105Target25PredictionRequest]):
    def predict(request: D105Target25PredictionRequest) -> D105Target25PredictionOutput:
        requests.append(request)
        base = tuple(
            request.registered_classes[index % len(request.registered_classes)]
            for index in range(len(request.query_physical_ids))
        )
        da = tuple(reversed(base))
        head = base if request.k_shot == 1 else tuple(reversed(base))
        joint = da if request.k_shot == 1 else base
        return D105Target25PredictionOutput(
            stage=request.stage,
            registration_state=request.registration_state,
            arm_predictions={
                "M0": base,
                "M_DA": da,
                "M_HEAD": head,
                "M_JOINT": joint,
            },
            state_receipt_sha256=_sha(
                f"state:{request.row_id}:{request.scenario}:{request.stage}"
            ),
            predictor_receipt_sha256=_sha(
                f"predictor:{request.row_id}:{request.scenario}:{request.stage}"
            ),
            feature_receipt_sha256=_sha(
                f"feature:{request.prediction_context_sha256}:{request.stage}"
            ),
            resource_receipt_sha256=_sha(
                f"resource:{request.gpu_id}:{request.worker_slot}:{request.stage}"
            ),
            logit_sha256_by_arm={
                arm: _sha(
                    f"logit:{request.row_id}:{request.scenario}:{request.stage}:{arm}"
                )
                for arm in ARMS
            },
            arm_prediction_sha256_by_arm={
                arm: _sha(
                    f"top1:{request.row_id}:{request.scenario}:{request.stage}:{arm}"
                )
                for arm in ARMS
            },
        )

    return predict


def _truth_catalog_document(run):
    states = []
    for row in run.plan.rows:
        for scenario in row.scenarios:
            before_labels = {
                physical_id: scenario.before.old_classes[index % len(scenario.before.old_classes)]
                for index, physical_id in enumerate(
                    scenario.before.query_physical_ids
                )
            }
            for state in (scenario.before, scenario.after):
                labels = []
                next_new_index = 0
                for physical_id in state.query_physical_ids:
                    if physical_id in before_labels:
                        labels.append(before_labels[physical_id])
                    else:
                        labels.append(
                            state.new_classes[next_new_index % len(state.new_classes)]
                        )
                        next_new_index += 1
                states.append(
                    {
                        "row_id": row.row_id,
                        "scenario": scenario.scenario,
                        "stage": state.stage,
                        "registration_state": state.registration_state,
                        "query_physical_ids": list(state.query_physical_ids),
                        "query_physical_root_sha256": state.query_physical_root_sha256,
                        "registered_classes": list(state.registered_classes),
                        "old_classes": list(state.old_classes),
                        "new_classes": list(state.new_classes),
                        "labels": labels,
                    }
                )
    return {
        "schema": "cvs.phase2.d105.target25_runner.v1.truth_catalog",
        "plan_receipt_sha256": run.plan.plan_receipt_sha256,
        "states": states,
    }


def test_freeze_target25_exact_pair_surface_coverage_and_k5_pairing() -> None:
    plan = _plan()
    assert plan.claim_scope == DEVELOPMENT_CLAIM_SCOPE
    assert plan.formal_launch_authority is False
    assert plan.seed == TARGET25_SEED
    assert len(plan.rows) == OUTER_ROW_COUNT
    assert sum(len(row.scenarios) for row in plan.rows) == SCENARIO_ROW_COUNT
    assert sum(len(row.scenarios) * len(ARMS) for row in plan.rows) == SCENARIO_ARM_PAIR_COUNT
    assert (
        sum(len(row.scenarios) * len(ARMS) * 2 for row in plan.rows)
        == STATE_PREDICTION_SURFACE_COUNT
    )
    for receiver in RECEIVERS:
        k5 = next(
            row
            for row in plan.rows
            if (row.receiver, row.k_shot, row.new_count) == (receiver, 5, 20)
        )
        k10 = next(
            row
            for row in plan.rows
            if (row.receiver, row.k_shot, row.new_count) == (receiver, 10, 20)
        )
        for short_pair, long_pair in zip(k5.scenarios, k10.scenarios, strict=True):
            for short, long in (
                (short_pair.before, long_pair.before),
                (short_pair.after, long_pair.after),
            ):
                assert set(short.support_physical_ids).issubset(long.support_physical_ids)
                assert short.query_physical_root_sha256 == long.query_physical_root_sha256


def test_development_claim_scope_cannot_be_upgraded_by_relabeling() -> None:
    plan = _plan()
    upgraded = replace(
        plan,
        claim_scope=FORMAL_CLAIM_SCOPE,
        formal_launch_authority=True,
    )
    with TemporaryDirectory() as temp:
        with pytest.raises(D105Target25RunnerError, match="candidate method lock"):
            prepare_d105_target25_run(
                upgraded,
                output_root=Path(temp),
                run_id="d105-target25-illegal-upgrade",
                schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
            )


def test_freeze_cannot_relabel_development_candidate_lock_as_formal() -> None:
    with pytest.raises(D105Target25RunnerError, match="candidate method lock"):
        freeze_d105_target25_plan(
            candidate_runtime_manifest_path=_CANDIDATE_RUNTIME_PATH,
            candidate_method_lock_path=_CANDIDATE_LOCK_PATH,
            receivers=RECEIVERS,
            scenario_plans=_plan_inputs(),
            claim_scope=FORMAL_CLAIM_SCOPE,
            formal_launch_authority=True,
        )


def test_plan_rejects_non_nested_k5_support() -> None:
    inputs = _plan_inputs()
    key = (RECEIVERS[0], 5, 20)
    broken = list(inputs[key])
    first = broken[0]
    broken_before = replace(
        first.before,
        support_physical_ids=("replacement-support-id",)
        + first.before.support_physical_ids[1:],
    )
    broken[0] = replace(first, before=broken_before)
    inputs[key] = tuple(broken)
    with pytest.raises(D105Target25RunnerError, match="K5/new20"):
        freeze_d105_target25_plan(
            candidate_runtime_manifest_path=_CANDIDATE_RUNTIME_PATH,
            candidate_method_lock_path=_CANDIDATE_LOCK_PATH,
            receivers=RECEIVERS,
            scenario_plans=inputs,
        )


def test_plan_rejects_k5_k10_state_registry_drift() -> None:
    inputs = _plan_inputs()
    key = (RECEIVERS[0], 5, 20)
    broken = []
    for scenario in inputs[key]:
        drifted_new = ("malicious-new-registry",) + scenario.after.new_classes[1:]
        drifted_after = replace(
            scenario.after,
            registered_classes=scenario.after.old_classes + drifted_new,
            new_classes=drifted_new,
        )
        broken.append(replace(scenario, after=drifted_after))
    inputs[key] = tuple(broken)
    with pytest.raises(D105Target25RunnerError, match="K5/new20"):
        freeze_d105_target25_plan(
            candidate_runtime_manifest_path=_CANDIDATE_RUNTIME_PATH,
            candidate_method_lock_path=_CANDIDATE_LOCK_PATH,
            receivers=RECEIVERS,
            scenario_plans=inputs,
        )


def test_complete_prediction_then_independent_paired_truth_scoring_and_summary() -> None:
    requests: list[D105Target25PredictionRequest] = []
    assert not any("truth" in field.name for field in fields(D105Target25PredictionRequest))
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = prepare_d105_target25_run(
            _plan(),
            output_root=root,
            run_id="d105-target25-local-001",
            schedule=D105Target25GPUSchedule(gpu_ids=(0, 1), workers_per_gpu=2),
        )
        summary = execute_d105_target25_predictions(run, _predictor(requests))
        assert summary.status == "PREDICTIONS_COMPLETE"
        assert summary.scenario_arm_pair_count == SCENARIO_ARM_PAIR_COUNT
        assert summary.state_prediction_surface_count == STATE_PREDICTION_SURFACE_COUNT
        assert len(requests) == SCENARIO_ROW_COUNT * 2
        manifest = verify_d105_target25_prediction_manifest(run)
        assert manifest["claim_scope"] == DEVELOPMENT_CLAIM_SCOPE
        assert manifest["formal_launch_authority"] is False
        assert manifest["scenario_arm_pair_count"] == SCENARIO_ARM_PAIR_COUNT
        assert manifest["state_prediction_surface_count"] == STATE_PREDICTION_SURFACE_COUNT
        first_artifact = json.loads(
            (run.run_root / "predictions" / f"{run.plan.rows[0].row_id}.json").read_text(
                encoding="utf-8"
            )
        )
        first_state = first_artifact["scenario_predictions"][0]["state_predictions"][0]
        expected_context = run.plan.rows[0].scenarios[0].before.prediction_context_sha256
        assert first_state["prediction_context_sha256"] == expected_context
        assert tuple(first_state["arm_prediction_sha256_by_arm"]) == ARMS
        assert expected_context in json.dumps(first_state, sort_keys=True)
        summary_document = summarize_d105_target25_outputs(run)
        assert summary_document["status"] == "PREDICTIONS_COMPLETE"
        assert summary_document["state_prediction_surface_count"] == 600
        truth_manifest = build_d105_target25_truth_side_manifest(
            run, truth_catalog_sha256=_sha("independently-managed-truth-catalog")
        )
        truth_manifest_path = root / "truth_side_manifest.json"
        write_d105_target25_truth_side_manifest(truth_manifest_path, truth_manifest)
        truth_calls = []

        def truth_provider(request):
            assert (run.run_root / "prediction_manifest.json").exists()
            assert not hasattr(request, "predictions")
            truth_calls.append(request)
            labels = tuple(
                request.old_classes[index % len(request.old_classes)]
                if physical_id.startswith("oldq-")
                else request.new_classes[index % len(request.new_classes)]
                for index, physical_id in enumerate(request.query_physical_ids)
            )
            return D105Target25TruthLabels(
                query_physical_ids=request.query_physical_ids,
                labels=labels,
            )

        score_path = score_d105_target25_truth_side(
            run,
            truth_manifest,
            truth_provider,
            score_root=root / "scores",
        )
        score_manifest = json.loads(score_path.read_text(encoding="utf-8"))
        assert score_manifest["claim_scope"] == DEVELOPMENT_CLAIM_SCOPE
        assert score_manifest["formal_launch_authority"] is False
        assert score_manifest["status"] == "SCORES_COMPLETE"
        assert score_manifest["scenario_arm_pair_count"] == SCENARIO_ARM_PAIR_COUNT
        assert score_manifest["state_prediction_surface_count"] == STATE_PREDICTION_SURFACE_COUNT
        assert len(score_manifest["rows"]) == OUTER_ROW_COUNT
        assert len(truth_calls) == SCENARIO_ROW_COUNT * 2
        score_row = json.loads(next((root / "scores" / "rows").glob("*.json")).read_text())
        pair_score = score_row["scenario_pairs"][0]["arm_pair_scores"]["M0"]
        assert {"B_old", "A_old", "N", "H_old_new", "forgetting"} <= set(pair_score)
        catalog_path = root / "truth_catalog.json"
        catalog_sha = seal_d105_target25_truth_catalog_manifest(
            catalog_path, _truth_catalog_document(run)
        )
        loaded_catalog = load_d105_target25_truth_catalog_manifest(
            run, catalog_path, expected_file_sha256=catalog_sha
        )
        assert loaded_catalog.file_sha256 == catalog_sha
        assert len(loaded_catalog.labels_by_state) == SCENARIO_ROW_COUNT * 2
        formal_score_path = score_d105_target25_from_catalog_file(
            run,
            truth_catalog_path=catalog_path,
            expected_truth_catalog_sha256=catalog_sha,
            score_root=root / "formal-scores",
        )
        assert json.loads(formal_score_path.read_text(encoding="utf-8"))[
            "status"
        ] == "SCORES_COMPLETE"
        with pytest.raises(FileExistsError, match="immutable run ID"):
            prepare_d105_target25_run(
                _plan(),
                output_root=root,
                run_id="d105-target25-local-001",
                schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
            )


def test_readonly_and_artifact_sha_tamper_fail_closed_before_truth_open() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = prepare_d105_target25_run(
            _plan(),
            output_root=root,
            run_id="d105-target25-local-002",
            schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
        )
        execute_d105_target25_predictions(run, _predictor([]))
        artifact_path = next((run.run_root / "predictions").glob("*.json"))
        assert not artifact_path.stat().st_mode & stat.S_IWUSR
        _make_writable(artifact_path)
        with pytest.raises(D105Target25RunnerError, match="immutable JSON file is writable"):
            verify_d105_target25_prediction_manifest(run)
        tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
        tampered["scenario_predictions"][0]["state_predictions"][0]["arm_predictions"][
            "M0"
        ][0] = "old-b"
        artifact_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(D105Target25RunnerError, match="artifact SHA drift"):
            verify_d105_target25_prediction_manifest(run)


def test_truth_catalog_rejects_external_sha_and_plan_registry_drift() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = prepare_d105_target25_run(
            _plan(),
            output_root=root,
            run_id="d105-target25-local-catalog-negative",
            schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
        )
        valid_path = root / "valid-truth-catalog.json"
        valid_sha = seal_d105_target25_truth_catalog_manifest(
            valid_path, _truth_catalog_document(run)
        )
        with pytest.raises(D105Target25RunnerError, match="external file SHA drift"):
            load_d105_target25_truth_catalog_manifest(
                run,
                valid_path,
                expected_file_sha256=_sha("wrong-external-file"),
            )

        root_drift = _truth_catalog_document(run)
        root_drift["states"][0]["query_physical_root_sha256"] = _sha(
            "malicious-query-root"
        )
        root_drift_path = root / "root-drift-truth-catalog.json"
        root_drift_sha = seal_d105_target25_truth_catalog_manifest(
            root_drift_path, root_drift
        )
        with pytest.raises(D105Target25RunnerError, match="state/plan binding drift"):
            load_d105_target25_truth_catalog_manifest(
                run,
                root_drift_path,
                expected_file_sha256=root_drift_sha,
            )

        registry_drift = _truth_catalog_document(run)
        registry_drift["states"][0]["registered_classes"][0] = "malicious-class"
        registry_drift_path = root / "registry-drift-truth-catalog.json"
        registry_drift_sha = seal_d105_target25_truth_catalog_manifest(
            registry_drift_path, registry_drift
        )
        with pytest.raises(D105Target25RunnerError, match="state/plan binding drift"):
            load_d105_target25_truth_catalog_manifest(
                run,
                registry_drift_path,
                expected_file_sha256=registry_drift_sha,
            )

        label_drift = _truth_catalog_document(run)
        label_drift["states"][0]["labels"][0] = "malicious-class"
        label_drift_path = root / "label-drift-truth-catalog.json"
        label_drift_sha = seal_d105_target25_truth_catalog_manifest(
            label_drift_path, label_drift
        )
        with pytest.raises(D105Target25RunnerError, match="outside the frozen"):
            load_d105_target25_truth_catalog_manifest(
                run,
                label_drift_path,
                expected_file_sha256=label_drift_sha,
            )


def test_two_distinct_same_fingerprint_stops_dispatch_without_truth_or_scores() -> None:
    calls = []
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = prepare_d105_target25_run(
            _plan(),
            output_root=root,
            run_id="d105-target25-local-003",
            schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
        )

        def deterministic_failure(request):
            calls.append(request)
            raise RuntimeError(f"decoder fault 0xdeadbeef row {request.row_id}")

        summary = execute_d105_target25_predictions(run, deterministic_failure)
        assert summary.stop_dispatch is True
        assert summary.launched_outer_rows == 2
        assert summary.succeeded_outer_rows == 0
        assert summary.failed_outer_rows == 2
        assert len(calls) == 2
        assert not (run.run_root / "prediction_manifest.json").exists()
        assert (run.run_root / "partial_prediction_manifest.json").exists()
        truth_manifest = build_d105_target25_truth_side_manifest(
            run, truth_catalog_sha256=_sha("truth-catalog")
        )
        provider_called = False

        def forbidden_provider(_request):
            nonlocal provider_called
            provider_called = True
            raise AssertionError("truth provider must not be opened")

        with pytest.raises(D105Target25RunnerError, match="expected regular immutable JSON"):
            score_d105_target25_truth_side(
                run,
                truth_manifest,
                forbidden_provider,
                score_root=root / "scores",
            )
        unopened_catalog = root / "must-not-open-truth-catalog.json"
        unopened_catalog.write_text("{}", encoding="utf-8")
        unopened_sha = hashlib.sha256(unopened_catalog.read_bytes()).hexdigest()
        with pytest.raises(D105Target25RunnerError, match="expected regular immutable JSON"):
            score_d105_target25_from_catalog_file(
                run,
                truth_catalog_path=unopened_catalog,
                expected_truth_catalog_sha256=unopened_sha,
                score_root=root / "formal-scores",
            )
        assert provider_called is False


def test_row_exit_and_score_artifacts_are_independently_fail_closed() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = prepare_d105_target25_run(
            _plan(),
            output_root=root,
            run_id="d105-target25-local-004",
            schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
        )
        execute_d105_target25_predictions(run, _predictor([]))
        first_row = run.plan.rows[0]
        exit_path = run.run_root / "row_logs" / first_row.row_id / "exit.json"
        _make_writable(exit_path)
        exit_document = json.loads(exit_path.read_text(encoding="utf-8"))
        exit_document["exit_code"] = 9
        exit_path.write_text(json.dumps(exit_document), encoding="utf-8")
        with pytest.raises(D105Target25RunnerError, match="log/exit SHA drift"):
            verify_d105_target25_prediction_manifest(run)

    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = prepare_d105_target25_run(
            _plan(),
            output_root=root,
            run_id="d105-target25-local-005",
            schedule=D105Target25GPUSchedule(gpu_ids=(0,)),
        )
        execute_d105_target25_predictions(run, _predictor([]))
        truth_manifest = build_d105_target25_truth_side_manifest(
            run, truth_catalog_sha256=_sha("score-tamper-truth")
        )

        def truth_provider(request):
            labels = tuple(
                request.old_classes[0]
                if physical_id.startswith("oldq-")
                else request.new_classes[0]
                for physical_id in request.query_physical_ids
            )
            return D105Target25TruthLabels(
                query_physical_ids=request.query_physical_ids,
                labels=labels,
            )

        score_root = root / "scores"
        score_d105_target25_truth_side(
            run, truth_manifest, truth_provider, score_root=score_root
        )
        score_file = next((score_root / "rows").glob("*.json"))
        _make_writable(score_file)
        score_document = json.loads(score_file.read_text(encoding="utf-8"))
        score_document["scenario_pairs"][0]["arm_pair_scores"]["M0"]["before_old"][
            "correct_count"
        ] = 99
        score_file.write_text(json.dumps(score_document), encoding="utf-8")
        with pytest.raises(D105Target25RunnerError, match="row score SHA drift"):
            verify_d105_target25_score_manifest(run, score_root)
