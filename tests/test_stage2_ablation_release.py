from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from cvsrffi.full_ablation_spec import (
    DESIGN_ID,
    DESIGN_SCHEMA,
    LEO_SCENARIOS,
    PROTOCOL_SCHEMA,
    PHASE2_T1_ARMS,
    SeedBundle,
    build_phase2_rows,
)
from cvsrffi.stage2_ablation_release import (
    BINDING_REGISTRY_SCHEMA,
    Stage2AblationReleaseError,
    seal_stage2_plan,
    validate_sealed_stage2_plan,
)
from scripts.run_full_ablation_stage2 import (
    Stage2AblationProtocolError,
    _validate_request_hash,
    dry_run_commands,
    is_p0_protocol_failure,
    normalize_exception_fingerprint,
)


SHA = "a" * 64
GIT = "b" * 40


def _row(
    ablation_id: str,
    *,
    suffix: str = "",
    physical_config_id: str | None = None,
) -> dict:
    return {
        "design_id": DESIGN_ID,
        "design_schema": DESIGN_SCHEMA,
        "phase": "stage2c",
        "stage": "screening",
        "ablation_id": ablation_id,
        "evidence_level": "M",
        "mechanism_family": "test",
        "comparison_target": "P2-FULL",
        "physical_config_id": (
            physical_config_id or ablation_id
        ),
        "git_commit": GIT,
        "protocol_schema": PROTOCOL_SCHEMA,
        "receiver_id": "20-1",
        "k_shot": 5,
        "old_class_count": 6,
        "new_class_count": 20,
        "scenarios": list(LEO_SCENARIOS),
        "train_seed": 840001,
        "support_seed": 840002,
        "query_seed": 840003,
        "method_seed": 840001,
        "phase1_bundle_training_seed": None,
        "new_class_draw_seed": 840004,
        "data_binding_status": "UNBOUND_FAIL_CLOSED",
        "executor_status": "LOCAL_IMPLEMENTED_PENDING_REVIEW",
        "formal_launch_authority": False,
        "worker": {"gpu": 0, "slot": 0},
        "row_key": (
            f"{ablation_id}__rx_20_1__k_5__new_20"
            f"__support_840002__query_840003__draw_840004{suffix}"
        ),
    }


def _plan(rows: list[dict]) -> dict:
    return {
        "schema": "cvs.full_ablation.plan.v1",
        "design_id": DESIGN_ID,
        "phase": "phase2",
        "stage": "screening",
        "git_commit": GIT,
        "rows": rows,
    }


def _binding(
    row_key: str,
    *,
    cache_sha: str = SHA,
    mode: str = "execute",
    reuse: str = "",
    reuse_sha: str = "",
    reuse_physical_id: str = "",
) -> dict:
    return {
        "row_key": row_key,
        "mode": mode,
        "feature_cache_payload": "/cache/features.npz",
        "feature_cache_payload_sha256": cache_sha,
        "feature_cache_manifest": "/cache/manifest.json",
        "feature_cache_manifest_sha256": "c" * 64,
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-test",
        "split_id": "split-test",
        "phase1_bundle_sha256": "f" * 64,
        "phase1_prototype_sha256": "0" * 64,
        "scoring_manifest": "/cache/truth.manifest.json",
        "scoring_manifest_sha256": "d" * 64,
        "reuse_row_execution_receipt": reuse,
        "reuse_row_execution_receipt_sha256": reuse_sha,
        "reuse_physical_execution_id": reuse_physical_id,
    }


def _registry(bindings: list[dict]) -> dict:
    return {
        "schema": BINDING_REGISTRY_SCHEMA,
        "candidate_lock_sha256": "e" * 64,
        "bindings": bindings,
    }


def _seal(
    tmp_path: Path,
    rows: list[dict],
    bindings: list[dict],
) -> dict:
    return seal_stage2_plan(
        _plan(rows),
        _registry(bindings),
        run_id="phase2_test_v1",
        request_root=tmp_path / "requests",
        run_root=tmp_path / "run",
        log_root=tmp_path / "logs",
        python_environment_id="CVS-RFFI",
        review_p0_count=0,
        review_p1_count=0,
        write_requests=True,
    )


def test_full_and_f3_share_one_physical_prediction(
    tmp_path: Path,
) -> None:
    full = _row("P2-FULL", physical_config_id="P2-FULL")
    alias = _row("P2-F3", physical_config_id="P2-FULL")
    plan = _seal(
        tmp_path,
        [full, alias],
        [_binding(full["row_key"]), _binding(alias["row_key"])],
    )
    validate_sealed_stage2_plan(plan)
    assert plan["logical_row_count"] == 2
    assert plan["physical_execution_count"] == 1
    assert plan["alias_logical_count"] == 1
    physical = plan["physical_rows"][0]
    assert physical["representative_ablation_id"] == "P2-FULL"
    assert Path(physical["prediction_request_path"]).is_file()
    logical = {
        item["ablation_id"]: item
        for item in physical["logical_rows"]
    }
    assert logical["P2-FULL"]["alias_of"] is None
    assert logical["P2-F3"]["alias_of"] == full["row_key"]
    for item in logical.values():
        request = json.loads(
            Path(item["score_request_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert request["physical_execution_id"] == physical[
            "physical_execution_id"
        ]


def test_different_launch_cache_bindings_are_not_forced_equal(
    tmp_path: Path,
) -> None:
    first = _row("P2-FULL", suffix="__first")
    second = _row("P2-F0", suffix="__second")
    plan = _seal(
        tmp_path,
        [first, second],
        [
            _binding(first["row_key"], cache_sha="1" * 64),
            _binding(second["row_key"], cache_sha="2" * 64),
        ],
    )
    assert plan["physical_execution_count"] == 2


def test_reuse_prediction_skips_predictor_but_keeps_scorer(
    tmp_path: Path,
) -> None:
    row = _row("P2-FULL")
    plan = _seal(
        tmp_path,
        [row],
        [
            _binding(
                row["row_key"],
                mode="reuse_prediction",
                reuse="/prior/row_execution_receipt.json",
                reuse_sha="9" * 64,
                reuse_physical_id="prior_physical_1",
            )
        ],
    )
    commands = dry_run_commands(
        plan,
        python=Path("/env/bin/python"),
        predictor_script=Path("/repo/predict.py"),
        scorer_script=Path("/repo/score.py"),
    )
    assert commands["reused_physical_count"] == 1
    assert commands["commands"][0]["predictor"] is None
    assert len(commands["commands"][0]["scorers"]) == 1
    assert commands["commands"][0]["physical_execution_id"] == (
        "prior_physical_1"
    )


def test_release_fails_closed_without_zero_p0_p1_review(
    tmp_path: Path,
) -> None:
    row = _row("P2-FULL")
    with pytest.raises(
        Stage2AblationReleaseError,
        match="P0=0 and P1=0",
    ):
        seal_stage2_plan(
            _plan([row]),
            _registry([_binding(row["row_key"])]),
            run_id="bad",
            request_root=tmp_path / "requests",
            run_root=tmp_path / "run",
            log_root=tmp_path / "logs",
            python_environment_id="CVS-RFFI",
            review_p0_count=0,
            review_p1_count=1,
        )


def test_failure_fingerprint_normalizes_paths_and_numbers() -> None:
    first = normalize_exception_fingerprint(
        "RuntimeError at /a/run/123/file.py line 91"
    )
    second = normalize_exception_fingerprint(
        "RuntimeError at /b/run/999/file.py line 44"
    )
    assert first == second
    assert is_p0_protocol_failure(
        'receipt={"query_truth_opened":true}',
        None,
    )


def test_child_request_tampering_is_a_p0_release_failure(
    tmp_path: Path,
) -> None:
    row = _row("P2-FULL")
    plan = _seal(
        tmp_path,
        [row],
        [_binding(row["row_key"])],
    )
    physical = plan["physical_rows"][0]
    request_path = Path(physical["prediction_request_path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["output_root"] = str(tmp_path / "escaped")
    request_path.chmod(0o600)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(
        Stage2AblationProtocolError,
        match="request content drift",
    ):
        _validate_request_hash(
            request_path,
            physical["prediction_request_sha256"],
        )


def test_full_t1_screening_seals_1575_logical_to_1500_physical(
    tmp_path: Path,
) -> None:
    rows = build_phase2_rows(
        stage="screening",
        arms=PHASE2_T1_ARMS,
        seed_bundles=(
            SeedBundle(840101, 840102, 840103),
            SeedBundle(840201, 840202, 840203),
            SeedBundle(840301, 840302, 840303),
        ),
        class_draw_seeds=(840901,),
        git_commit=GIT,
    )
    bindings = []
    for row in rows:
        identity = json.dumps(
            {
                "receiver": row["receiver_id"],
                "k_shot": row["k_shot"],
                "new_class_count": row["new_class_count"],
                "method_seed": row["method_seed"],
                "support_seed": row["support_seed"],
                "query_seed": row["query_seed"],
                "draw": row["new_class_draw_seed"],
            },
            sort_keys=True,
        )
        cache_sha = hashlib.sha256(identity.encode()).hexdigest()
        binding = _binding(
            row["row_key"],
            cache_sha=cache_sha,
        )
        binding["feature_cache_payload"] = (
            f"/cache/{cache_sha}.npz"
        )
        binding["feature_cache_manifest"] = (
            f"/cache/{cache_sha}.manifest.json"
        )
        binding["feature_cache_manifest_sha256"] = hashlib.sha256(
            (identity + ":manifest").encode()
        ).hexdigest()
        binding["scoring_manifest"] = (
            f"/scorer/{cache_sha}.manifest.json"
        )
        binding["scoring_manifest_sha256"] = hashlib.sha256(
            (identity + ":scorer").encode()
        ).hexdigest()
        bindings.append(binding)
    sealed = seal_stage2_plan(
        _plan(rows),
        _registry(bindings),
        run_id="full_t1_screening_test",
        request_root=tmp_path / "requests",
        run_root=tmp_path / "run",
        log_root=tmp_path / "logs",
        python_environment_id="CVS-RFFI",
        review_p0_count=0,
        review_p1_count=0,
        write_requests=False,
    )
    assert sealed["logical_row_count"] == 1575
    assert sealed["physical_execution_count"] == 1500
    assert sealed["alias_logical_count"] == 75
    assert {
        (
            physical["worker"]["gpu"],
            physical["worker"]["slot"],
        )
        for physical in sealed["physical_rows"]
    } == {
        (gpu, slot)
        for gpu in range(8)
        for slot in range(2)
    }
