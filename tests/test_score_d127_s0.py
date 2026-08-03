from __future__ import annotations

from argparse import Namespace
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from cvsrffi import stage2_d127_s0_scorer as scorer


def _sign(document: dict, field: str) -> dict:
    unsigned = dict(document)
    unsigned.pop(field, None)
    document[field] = scorer.canonical_sha256(unsigned)
    return document


def _arm(classes: list[str], predictions: list[str]) -> dict:
    return {"classes": classes, "predictions": predictions}


def _paired_fixture(monkeypatch) -> tuple[dict, dict]:
    from cvsrffi import stage2_d127_s0_package_adapter as adapter

    monkeypatch.setattr(adapter, "validate_d127_s0_prediction_pairs", lambda prediction, *, prepared_plan: None)
    states: dict[str, dict] = {"before": {"rows": []}, "after": {"rows": []}}
    bindings: list[dict] = []
    index = 0
    for receiver in ("r0", "r1", "r2"):
        for k_shot in (1, 5):
            for scene in ("s0", "s1", "s2"):
                row_id = f"{receiver}.k{k_shot}.{scene}"
                before_ids = [f"q-{index}-old0", f"q-{index}-old1"]
                after_ids = [*before_ids, f"q-{index}-new0"]
                before_classes, after_classes = ["old0", "old1"], ["old0", "old1", "new0"]

                def state_row(state: str) -> dict:
                    classes = before_classes if state == "before" else after_classes
                    ids = before_ids if state == "before" else after_ids
                    common = ["old0", "old1"] if state == "before" else ["old0", "old0", "old0"]
                    da = ["old0", "old1"] if state == "before" else (["old0", "old1", "new0"] if k_shot == 1 else ["old0", "old1", "old0"])
                    joint = ["old0", "old1"] if state == "before" else ["old0", "old1", "new0"]
                    return {
                        "row_id": row_id, "receiver": receiver, "k_shot": k_shot, "scene": scene,
                        "opaque_query_ids": ids,
                        "common_arms": {"M0": _arm(classes, common), "M_L92": _arm(classes, common)},
                        "candidates": {
                            candidate: {"arms": {"M_DA": _arm(classes, da), "M_JOINT": _arm(classes, joint)}}
                            for candidate in scorer.CANDIDATE_IDS
                        },
                    }

                states["before"]["rows"].append(state_row("before"))
                states["after"]["rows"].append(state_row("after"))
                bindings.append(
                    {
                        "row_id": row_id, "receiver": receiver, "k_shot": k_shot, "scene": scene,
                        "before": {}, "after": {},
                        "formal_d92_reference": {"source_d92_job_id": f"d92-{row_id}", "d92_retry2_manifest_sha256": "b" * 64},
                    }
                )
                index += 1
    paired = {
        "schema": "cvs.stage2.d127.s0.paired_prediction.v1", "truth_loaded": False,
        "method_lock_sha256": "a" * 64, "candidate_ids": list(scorer.CANDIDATE_IDS),
        "pair_manifest": {"pair_manifest_sha256": "c" * 64}, "pair_bindings": bindings,
        "states": states, "paired_prediction_sha256": "d" * 64,
    }
    plan = {"method_lock_sha256": "a" * 64, "prepared_plan_sha256": "e" * 64}
    return paired, plan


def _truth_and_formal(normalized: dict) -> tuple[dict, dict]:
    queries = []
    formal_rows = []
    for index, row in enumerate(normalized["rows"]):
        queries.extend(
            [
                {"opaque_query_id": row["after_query_ids"][0], "label": "old0", "role": "old"},
                {"opaque_query_id": row["after_query_ids"][1], "label": "old1", "role": "old"},
                {"opaque_query_id": row["after_query_ids"][2], "label": "new0", "role": "new"},
            ]
        )
        formal_rows.append(
            {
                "row_id": row["row_id"], "receiver_id": row["receiver_id"], "k_shot": row["k_shot"], "scene": row["scene"],
                "source_d92_job_id": row["formal_d92_source_job_id"],
                "d92_retry2_manifest_sha256": row["formal_d92_retry2_manifest_sha256"],
                "formal_d92_score_row_key": f"score-{row['row_id']}",
                "formal_d92_score_row_sha256": f"{index + 1:064x}",
            }
        )
    binding = {field: normalized[field] for field in ("paired_prediction_sha256", "prepared_plan_sha256", "method_lock_sha256", "pair_manifest_sha256", "normalized_prediction_sha256")}
    truth = {"schema": scorer.PAIRED_TRUTH_CATALOG_SCHEMA, "truth_open": True, **binding, "query_count": len(queries), "queries": queries}
    _sign(truth, "truth_catalog_sha256")
    formal = {"schema": scorer.PAIRED_FORMAL_D92_REFERENCE_SCHEMA, **binding, "pipeline_receipt_sha256": "f" * 64, "row_count": 18, "rows": formal_rows}
    _sign(formal, "formal_d92_reference_sha256")
    return truth, formal


def test_normalizes_existing_paired_prediction_and_scores_all_arms(monkeypatch, tmp_path: Path) -> None:
    paired, plan = _paired_fixture(monkeypatch)
    normalized = scorer.normalize_d127_s0_paired_prediction(paired_prediction=paired, prepared_plan=plan, method_lock_sha256="a" * 64)
    assert normalized["row_count"] == 18
    assert set(normalized["rows"][0]["arms_by_state"]["after"]) == set(scorer.CANDIDATE_IDS)
    truth, formal = _truth_and_formal(normalized)
    event = scorer.build_d127_s0_truth_open_event(normalized)
    result = scorer.score_d127_s0_paired(normalized_prediction=normalized, truth_open_event=event, truth_catalog=truth, formal_d92_reference=formal)
    assert result["metric_row_count"] == 18 * 3 * 4
    assert all(item["all_three_direction_pass"] for item in result["s0_direction_decisions"])
    event_path = tmp_path / "truth-open.json"
    scorer.write_d127_s0_truth_open_event_exclusive(event_path, event)
    with pytest.raises(scorer.D127S0ScorerError, match="already exists"):
        scorer.write_d127_s0_truth_open_event_exclusive(event_path, event)
    score_path = tmp_path / "score.json"
    scorer.write_d127_s0_paired_score_exclusive(score_path, result)
    with pytest.raises(scorer.D127S0ScorerError, match="already exists"):
        scorer.write_d127_s0_paired_score_exclusive(score_path, result)


def test_paired_arm_or_hash_drift_is_rejected(monkeypatch) -> None:
    paired, plan = _paired_fixture(monkeypatch)
    del paired["states"]["after"]["rows"][0]["candidates"][scorer.CANDIDATE_IDS[-1]]["arms"]["M_JOINT"]
    with pytest.raises(scorer.D127S0ScorerError, match="adapted-arm closure"):
        scorer.normalize_d127_s0_paired_prediction(paired_prediction=paired, prepared_plan=plan, method_lock_sha256="a" * 64)


def test_prepare_reads_canonical_sorted_paired_json_without_mapping_order_dependency(monkeypatch, tmp_path: Path) -> None:
    paired, plan = _paired_fixture(monkeypatch)
    from cvsrffi import stage2_d127_s0_package_adapter as adapter

    # The actual package writer canonicalizes with sort_keys=True, which reads
    # the state mapping back as after,before instead of its frozen semantic
    # iteration order.  Scoring must validate exact keys, then access STATES.
    paired_path = tmp_path / "paired.json"
    paired_path.write_bytes(adapter._canonical_bytes(paired) + b"\n")
    paired_file_sha = hashlib.sha256(paired_path.read_bytes()).hexdigest()
    plan_with_checkpoint = {
        **plan,
        "checkpoint_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        adapter,
        "load_d127_s0_prepared_plan",
        lambda path, *, expected_sha256: (plan_with_checkpoint, expected_sha256),
    )
    monkeypatch.setattr(
        adapter,
        "load_d127_s0_method_lock",
        lambda path, *, expected_sha256: (
            {"checkpoint": {"sha256": "9" * 64}},
            "a" * 64,
            {},
        ),
    )
    prepared = scorer.prepare_d127_s0_scoring_inputs(
        paired_prediction_path=paired_path,
        expected_paired_prediction_sha256=paired_file_sha,
        prepared_plan_path=tmp_path / "plan.json",
        expected_prepared_plan_sha256="e" * 64,
        method_lock_path=tmp_path / "lock.json",
        expected_method_lock_sha256="a" * 64,
    )
    assert prepared["normalized_prediction"]["row_count"] == 18
    assert prepared["normalized_prediction"]["rows"][0]["arms_by_state"]["before"]


def test_truth_and_formal_are_not_read_when_truth_free_prepare_fails(monkeypatch, tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "code" / "scripts" / "score_d127_s0.py"
    spec = spec_from_file_location("score_d127_s0_test_module", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[str] = []

    def fail_prepare(**kwargs):
        raise scorer.D127S0ScorerError("paired closure failed")

    def forbidden_read(*args, **kwargs):
        calls.append("read")
        raise AssertionError("truth/formal must not be read before closure")

    monkeypatch.setattr(module.scorer, "prepare_d127_s0_scoring_inputs", fail_prepare)
    monkeypatch.setattr(module.scorer, "_load_pinned_json", forbidden_read)
    args = Namespace(
        paired_prediction=tmp_path / "paired.json", paired_prediction_sha256="a" * 64,
        prepared_plan=tmp_path / "plan.json", prepared_plan_sha256="b" * 64,
        method_lock=tmp_path / "lock.json", method_lock_sha256="c" * 64,
        truth_open_event=tmp_path / "open.json", truth_open_event_sha256="f" * 64,
        truth_catalog=tmp_path / "truth.json", truth_catalog_sha256="d" * 64,
        formal_d92_reference=tmp_path / "formal.json", formal_d92_reference_sha256="e" * 64,
        score_output=tmp_path / "score.json",
    )
    with pytest.raises(scorer.D127S0ScorerError, match="paired closure"):
        module._score(args)
    assert calls == [] and not args.truth_open_event.exists()


def test_cli_writes_open_event_before_reading_runner_truth_and_writes_exclusive_score(monkeypatch, tmp_path: Path) -> None:
    paired, plan = _paired_fixture(monkeypatch)
    normalized = scorer.normalize_d127_s0_paired_prediction(paired_prediction=paired, prepared_plan=plan, method_lock_sha256="a" * 64)
    truth, formal = _truth_and_formal(normalized)
    script_path = Path(__file__).resolve().parents[1] / "code" / "scripts" / "score_d127_s0.py"
    spec = spec_from_file_location("score_d127_s0_success_module", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.scorer, "prepare_d127_s0_scoring_inputs", lambda **kwargs: {"normalized_prediction": normalized})
    open_args = Namespace(
        paired_prediction=tmp_path / "paired.json", paired_prediction_sha256="a" * 64,
        prepared_plan=tmp_path / "plan.json", prepared_plan_sha256="b" * 64,
        method_lock=tmp_path / "lock.json", method_lock_sha256="c" * 64,
        truth_open_event_output=tmp_path / "open.json",
    )
    open_status = module._open(open_args)
    assert open_status["status"] == "D127_S0_TRUTH_OPENED"
    args = Namespace(
        paired_prediction=tmp_path / "paired.json", paired_prediction_sha256="a" * 64,
        prepared_plan=tmp_path / "plan.json", prepared_plan_sha256="b" * 64,
        method_lock=tmp_path / "lock.json", method_lock_sha256="c" * 64,
        truth_open_event=open_args.truth_open_event_output,
        truth_open_event_sha256=open_status["truth_open_event_sha256"],
        truth_catalog=tmp_path / "truth.json", truth_catalog_sha256="d" * 64,
        formal_d92_reference=tmp_path / "formal.json", formal_d92_reference_sha256="e" * 64,
        score_output=tmp_path / "score.json",
    )

    original_read = module.scorer._load_pinned_json

    def read_after_event(path, *, expected_sha256, name):
        if name == "D127 truth-open event":
            return original_read(path, expected_sha256=expected_sha256, name=name)
        assert args.truth_open_event.exists()
        return (truth if "truth" in name else formal), expected_sha256

    monkeypatch.setattr(module.scorer, "_load_pinned_json", read_after_event)
    status = module._score(args)
    assert status["row_count"] == 18 and args.score_output.exists()
    with pytest.raises(scorer.D127S0ScorerError, match="already exists"):
        module._score(args)


def test_score_rejects_truth_open_event_file_hash_before_truth_reads(monkeypatch, tmp_path: Path) -> None:
    paired, plan = _paired_fixture(monkeypatch)
    normalized = scorer.normalize_d127_s0_paired_prediction(paired_prediction=paired, prepared_plan=plan, method_lock_sha256="a" * 64)
    script_path = Path(__file__).resolve().parents[1] / "code" / "scripts" / "score_d127_s0.py"
    spec = spec_from_file_location("score_d127_s0_hash_module", script_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.scorer, "prepare_d127_s0_scoring_inputs", lambda **kwargs: {"normalized_prediction": normalized})
    event_path = tmp_path / "open.json"
    scorer.write_d127_s0_truth_open_event_exclusive(event_path, scorer.build_d127_s0_truth_open_event(normalized))
    calls: list[str] = []
    original_read = module.scorer._load_pinned_json

    def read_event_only(path, *, expected_sha256, name):
        if name != "D127 truth-open event":
            calls.append(name)
        return original_read(path, expected_sha256=expected_sha256, name=name)

    monkeypatch.setattr(module.scorer, "_load_pinned_json", read_event_only)
    args = Namespace(
        paired_prediction=tmp_path / "paired.json", paired_prediction_sha256="a" * 64,
        prepared_plan=tmp_path / "plan.json", prepared_plan_sha256="b" * 64,
        method_lock=tmp_path / "lock.json", method_lock_sha256="c" * 64,
        truth_open_event=event_path, truth_open_event_sha256="0" * 64,
        truth_catalog=tmp_path / "truth.json", truth_catalog_sha256="d" * 64,
        formal_d92_reference=tmp_path / "formal.json", formal_d92_reference_sha256="e" * 64,
        score_output=tmp_path / "score.json",
    )
    with pytest.raises(scorer.D127S0ScorerError, match="SHA mismatch"):
        module._score(args)
    assert calls == [] and not args.score_output.exists()
