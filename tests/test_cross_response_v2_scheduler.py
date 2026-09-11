"""Attempt transactions, direction evidence and strict resumability (CPU)."""
import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from cvsrffi.cross_response.schema import BlockCandidate, CrossBlock, SampleRecord
from cvsrffi.cross_response.roles import make_roles
from cvsrffi.cross_response.scheduler import FeedbackScheduler


def block(direction=0, block_id=0):
    candidate = BlockCandidate(tuple(range(4)), tuple(range(4)), 1, "clean", 2)
    records = tuple(SampleRecord(i, t, r, 1, "clean", (t, r, k))
                    for i, (t, r, k) in enumerate((t, r, k) for t in range(4)
                                                  for r in range(4) for k in range(2)))
    return CrossBlock(candidate, records, tuple(range(32)),
                      make_roles(candidate.tx_ids, candidate.rx_ids, direction,
                                 policy="balanced_partitions_v2"), block_id, 1.)


def scheduler(**kwargs):
    return FeedbackScheduler(update_interval=20, feedback_version="transaction_v2",
                             role_policy_version="balanced_partitions_v2", **kwargs)


def run_fixture(reverse=False, resume_at=None):
    s = scheduler()
    for step in range(20):
        values = [0., 0., 0., 10.] if step == 19 else [0.] * 4
        ordered = list(enumerate(values))
        if reverse:
            ordered.reverse()
        for index, value in ordered:
            s.collect_block_feedback(block(block_id=index), step, response=value)
        s.finish_step_and_maybe_flush(step)
        if step == resume_at:
            restored = scheduler()
            restored.load_state_dict(s.state_dict())
            s = restored
    return s


def test_twenty_steps_four_blocks_single_flush_order_invariant():
    a, b = run_fixture(), run_fixture(reverse=True)
    assert a.history[block().candidate.key]["response"] == pytest.approx(.025)
    assert a.history[block().candidate.key]["sample_count"] == 80
    assert a.flush_count == b.flush_count == 1
    assert a.state_dict() == b.state_dict()
    assert a.state_dict() == run_fixture(resume_at=8).state_dict()


def test_failed_attempt_and_invalid_block_never_enter_history():
    s = scheduler()
    s.collect_block_feedback(block(), 0, response=100.)
    assert not s.coverage.exposed_physical
    event = s.finish_step_and_maybe_flush(0, success=False)
    assert not event["success"] and event["accepted_blocks"] == 0
    assert not s.pending and not s.history
    assert len(s.coverage.exposed_physical) == 32
    s.collect_block_feedback(block(), 1, response=float("nan"))
    s.collect_block_feedback(block(), 1, response=100., success=False)
    s.finish_step_and_maybe_flush(1)
    assert not s.pending
    s.collect_block_feedback(block(), 2, response=5.)
    s.finish_step_and_maybe_flush(2)
    s.finish_step_and_maybe_flush(19, success=False)
    assert s.history[block().candidate.key]["response"] == 1.
    assert s.history[block().candidate.key]["sample_count"] == 1
    assert s.flush_count == 1


def test_direction_measurements_remain_distinct_and_prior_explicit():
    s = scheduler(direction_shrinkage=2.)
    for step, direction, value in ((0, 0, 10.), (1, 1, 2.)):
        s.collect_block_feedback(block(direction), step, response=value)
        s.finish_step_and_maybe_flush(step)
    s.finish_step_and_maybe_flush(19)
    a = s.directional_estimate(block().candidate, block().roles)
    b = s.directional_estimate(block(1).candidate, block(1).roles)
    unseen = s.directional_estimate(block(2).candidate, block(2).roles)
    assert a["raw"]["response"] == 2.
    assert b["raw"]["response"] == pytest.approx(.4)
    assert a["raw_sample_count"] == 1 and a["direction_weight"] == pytest.approx(1 / 3)
    assert a["fallback_reason"] == "sparse_direction_shrinkage"
    assert unseen["raw"] is None and unseen["raw_sample_count"] == 0
    assert unseen["fallback_reason"] == "unobserved_direction"
    assert unseen["estimate"]["response"] == pytest.approx(1.2)
    fresh = scheduler().directional_estimate(block().candidate, block().roles)
    assert fresh["estimate"] is None and fresh["fallback_reason"] == "no_candidate_history"


def test_strict_resume_duplicate_finish_and_pending_tail():
    s = scheduler()
    s.collect_block_feedback(block(), 0, response=3.)
    resumed = scheduler()
    resumed.load_state_dict(s.state_dict())
    s.finish_step_and_maybe_flush(0)
    resumed.finish_step_and_maybe_flush(0)
    assert s.state_dict() == resumed.state_dict()
    assert s.pending and not s.history and s.flush_count == 0
    for obj in (s, resumed):
        with pytest.raises(ValueError, match="already finished"):
            obj.finish_step_and_maybe_flush(0)
        with pytest.raises(ValueError, match="already finished"):
            obj.collect_block_feedback(block(), 0)
        with pytest.raises(ValueError, match="version"):
            obj.load_state_dict(FeedbackScheduler().state_dict())
    with pytest.raises(ValueError, match="version"):
        FeedbackScheduler().load_state_dict(s.state_dict())
    with pytest.raises(ValueError, match="configuration"):
        scheduler(direction_shrinkage=5.).load_state_dict(s.state_dict())
    damaged = copy.deepcopy(s.state_dict())
    del damaged["direction_pending"]
    with pytest.raises(ValueError, match="incomplete"):
        scheduler().load_state_dict(damaged)
    with pytest.raises(RuntimeError, match="requires"):
        s.commit(block(), 1)


def test_active_step_must_finish_before_next_attempt():
    s = scheduler()
    s.collect_block_feedback(block(), 0)
    with pytest.raises(ValueError, match="active step"):
        s.collect_block_feedback(block(), 1)
    with pytest.raises(ValueError, match="active step"):
        s.finish_step_and_maybe_flush(1)
    with pytest.raises(ValueError, match="role policy"):
        FeedbackScheduler(feedback_version="transaction_v2").collect_block_feedback(block(), 0)


def test_legacy_api_and_first_update_are_preserved():
    s = FeedbackScheduler(update_interval=20)
    old_block = block()
    s.commit(old_block, 19, response=10.)
    assert s.history[old_block.candidate.key]["response"] == 10.
    assert "state_version" not in s.state_dict()
    restored = FeedbackScheduler(update_interval=20)
    restored.load_state_dict(s.state_dict())
    assert restored.state_dict() == s.state_dict()


def evidence_config():
    """Synthetic interface fixture; never a claim of real source qualification."""
    names = ("coverage", "query_gap", "new_physical", "staleness", "decision", "response")
    return dict(frozen=True, source_evidence=dict(source_only=True, target_used=False,
                joint_objective_validated=True, evidence_id="SYNTHETIC_TEST_ONLY",
                gate_result=dict(source_role="source_validation", **{
                    name: dict(passed=True, sample_count=8, metrics={"synthetic_margin": 1.})
                    for name in ("capability", "necessity", "update_value")})),
                weights={name: 1. for name in names},
                scales={name: 1. for name in names + ("noise", "cost")},
                new_physical_cap=32., unknown_reliability=1.)


def evidence_scheduler(cfg=None):
    return scheduler(mode="guided", gain_strategy="reliable_evidence",
                     evidence_config=evidence_config() if cfg is None else cfg)


def evidence_inputs():
    candidates = [block().candidate, replace(block().candidate, day_id=2)]
    roles = {c.key: block().roles for c in candidates}
    physical = {c.key: dict(new_physical_estimate=32, record_budget=c.size) for c in candidates}
    return candidates, roles, physical


@pytest.mark.parametrize("failure", ["not_frozen", "target_used", "not_joint", "missing_gate",
                                    "failed_capability", "failed_necessity", "failed_update_value",
                                    "zero_samples", "missing_metrics", "missing_scale", "missing_cap"])
def test_reliable_gain_requires_source_freeze_and_three_measured_conditions(failure):
    cfg = evidence_config()
    src = cfg["source_evidence"]
    if failure == "not_frozen":
        cfg["frozen"] = False
    elif failure == "target_used":
        src["target_used"] = True
    elif failure == "not_joint":
        src["joint_objective_validated"] = False
    elif failure == "missing_gate":
        del src["gate_result"]
    elif failure.startswith("failed_"):
        src["gate_result"][failure.removeprefix("failed_")]["passed"] = False
    elif failure == "zero_samples":
        src["gate_result"]["necessity"]["sample_count"] = 0
    elif failure == "missing_metrics":
        del src["gate_result"]["update_value"]["metrics"]
    elif failure == "missing_scale":
        del cfg["scales"]["noise"]
    else:
        del cfg["new_physical_cap"]
    with pytest.raises(ValueError):
        evidence_scheduler(cfg)


@pytest.mark.parametrize("component", ["coverage", "query_gap", "new_physical", "staleness",
                                      "decision", "response", "reliability", "noise", "cost"])
def test_each_reliable_component_changes_actual_probability(component):
    cfg = evidence_config()
    term = component if component in cfg["weights"] else "coverage"
    cfg["weights"] = {name: float(name == term) for name in cfg["weights"]}
    s = evidence_scheduler(cfg)
    candidates, roles, physical = evidence_inputs()
    if component == "cost":
        candidates[0] = replace(candidates[0], k=4)
        roles[candidates[0].key] = block().roles
        physical[candidates[0].key] = dict(new_physical_estimate=32, record_budget=64)
    for c in candidates:
        s.history[c.key] = dict(response=0., decision=0., reliability=1., noise=0., sample_count=8)
    c = candidates[0]
    role = roles[c.key]
    expected_first_larger = component in ("decision", "response")
    if component == "coverage":
        for key in s.coverage.keys(c, role)[0]:
            s.coverage.joint[key] = dict(successes=100)
    elif component == "query_gap":
        for key in s.coverage.query_keys(c, role):
            s.coverage.response_query_rectangles[key] = dict(valid_tasks=1)
    elif component == "new_physical":
        physical[c.key]["new_physical_estimate"] = 0
    elif component == "staleness":
        for key in s.coverage.query_keys(c, role):
            s.coverage.response_query_rectangles[key] = dict(last_success_step=100)
    elif component in ("decision", "response", "noise"):
        s.history[c.key][component] = 10.
    elif component == "reliability":
        s.history[c.key]["reliability"] = .1
    probabilities = s.probabilities(candidates, 0, 128, role_plans=roles,
                                    candidate_evidence=physical, step_id=100)
    assert (probabilities[0] > probabilities[1]) == expected_first_larger
    assert probabilities[0] != probabilities[1]
    assert sum(probabilities) == pytest.approx(1.)
    assert min(probabilities) >= s.exploration / len(candidates)
    assert all(0 <= value <= 1 for row in s.last_probability_audit for value in row["normalized"].values())
    assert [r["probability"] for r in s.last_probability_audit] == probabilities


def test_reliable_caps_novelty_records_actual_new_reads_and_strict_resume():
    cfg = evidence_config()
    cfg["new_physical_cap"] = 5
    s = evidence_scheduler(cfg)
    candidates, roles, physical = evidence_inputs()
    s.probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical)
    assert s.last_probability_audit[0]["raw"]["new_physical"] == 5
    assert s.last_probability_audit[0]["novelty_estimate_before_cap"] == 32
    s.collect_block_feedback(block(), 0)
    first = s.finish_step_and_maybe_flush(0)
    assert first["new_physical_records"] == 32
    s.collect_block_feedback(block(), 1)
    second = s.finish_step_and_maybe_flush(1, success=False)
    assert second["new_physical_records"] == 0
    assert second["block_physical_evidence"][0]["actual_new_physical"] == 0
    resumed = evidence_scheduler(cfg)
    resumed.load_state_dict(s.state_dict())
    assert resumed.state_dict() == s.state_dict()
    with pytest.raises(ValueError, match="version"):
        scheduler().load_state_dict(s.state_dict())
    cfg["weights"]["response"] = 5
    with pytest.raises(ValueError, match="configuration"):
        evidence_scheduler(cfg).load_state_dict(s.state_dict())


def test_reliable_requires_prospective_metadata_and_never_changes_legacy_default():
    candidates, roles, physical = evidence_inputs()
    with pytest.raises(ValueError, match="prospective"):
        evidence_scheduler().probabilities(candidates, 0, 128)
    physical[candidates[0].key]["record_budget"] = 1
    with pytest.raises(ValueError, match="physical evidence"):
        evidence_scheduler().probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical)
    legacy = scheduler(mode="guided")
    expected = legacy.probabilities(candidates, 0, 128, role_plans=roles)
    assert legacy.probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical) == expected
    assert legacy.gain_strategy == "legacy_gain" and not legacy.last_probability_audit


def test_reliable_actually_uses_directional_shrinkage_and_freeze_is_immutable():
    cfg = evidence_config()
    cfg["weights"] = {name: float(name == "response") for name in cfg["weights"]}
    s = scheduler(mode="guided", gain_strategy="reliable_evidence", evidence_config=cfg,
                  direction_shrinkage=2.)
    candidates, roles, physical = evidence_inputs()
    for candidate in candidates:
        s.history[candidate.key] = dict(response=2., decision=0., noise=0., reliability=1., sample_count=8)
    first = candidates[0]
    s.direction_history[s.direction_key(first, roles[first.key])] = dict(
        response=10., decision=0., noise=0., reliability=1., sample_count=2)
    p = s.probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical)
    assert p[0] > p[1]
    assert s.last_probability_audit[0]["raw"]["response"] == 6.
    assert s.last_probability_audit[0]["direction"]["direction_weight"] == .5
    assert s.last_probability_audit[1]["direction"]["raw"] is None
    s.evidence_config["weights"]["response"] = 2.
    with pytest.raises(ValueError, match="frozen evidence configuration changed"):
        s.probabilities(candidates, 0, 128, role_plans=roles, candidate_evidence=physical)
