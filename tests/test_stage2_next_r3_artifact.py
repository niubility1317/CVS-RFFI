from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest

from cvsrffi import stage2_next_r3_artifact as artifact
from cvsrffi import stage2_next_r3_matrix as matrix
from cvsrffi import stage2_next_r3_score as score


CLASSES = tuple(f"tx-{index}" for index in range(6))


@dataclass
class FakeCache:
    representation: str
    registration_state: str
    registered_classes: tuple[str, ...]
    support_labels: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    receipt: dict

    @property
    def cache_sha256(self):
        return self.receipt["cache_sha256"]


@dataclass
class FakeArm:
    arm_id: str
    cache: FakeCache
    predictions: tuple[str, ...]
    receipt: dict


@dataclass
class FakeRegistration:
    registration_state: str
    caches: dict[str, FakeCache]
    arms: dict[str, FakeArm]
    receipt: dict


@dataclass
class FakeBridge:
    row_id: str
    binding_sha256: str

    def as_dict(self):
        return {"schema": "fake.bridge.v1", "row_id": self.row_id, "binding_sha256": self.binding_sha256}


@dataclass
class FakeRuntime:
    bridge: FakeBridge
    reg0: FakeRegistration
    reg1: FakeRegistration
    four_state: dict
    four_state_receipt: dict
    resource_receipt: dict
    runtime_receipt: dict
    da1_reg0_state_sha256: str
    da1_reg1_state_sha256: str


def _fake_runtime(
    row: dict,
    query_cache: dict[str, tuple[str, ...]],
    truth: dict[str, str],
    support_cache: dict[str, tuple[str, ...]],
):
    registrations = {}
    four_state = {}
    state_receipts = {}
    for registration_id in matrix.REGISTRATION_IDS:
        classes = tuple(row["retained_classes"] if registration_id == "REG0" else row["all_registered_classes"])
        query_ids = query_cache[registration_id]
        support_ids = support_cache[registration_id][: len(classes) * row["active_k"]]
        labels = tuple(cls for cls in classes for _ in range(row["active_k"]))
        caches = {}
        arms = {}
        for representation in ("R0", "R1"):
            cache = FakeCache(
                representation,
                registration_id,
                classes,
                labels,
                support_ids,
                query_ids,
                {"schema": "fake.cache.v1", "cache_sha256": f"{row['row_id']}-{registration_id}-{representation}"},
            )
            caches[representation] = cache
            for head in ("Q", "F", "L"):
                arm_id = f"{representation}{head}"
                preds = tuple(truth[qid] for qid in query_ids)
                arms[arm_id] = FakeArm(arm_id, cache, preds, {"arm_id": arm_id, "cache_sha256": cache.cache_sha256})
        registrations[registration_id] = FakeRegistration(registration_id, caches, arms, {"registration_state": registration_id})
        for state_id in (("DA0_REG0", "DA1_REG0") if registration_id == "REG0" else ("DA0_REG1", "DA1_REG1")):
            prefix = "R0" if state_id.startswith("DA0") else "R1"
            four_state[state_id] = {head: arms[f"{prefix}{head}"] for head in ("Q", "F", "L")}
            state_receipts[state_id] = {"state_id": state_id, "registration_state": registration_id, "representation": prefix}
    return FakeRuntime(
        bridge=FakeBridge(row["row_id"], "b" * 64),
        reg0=registrations["REG0"],
        reg1=registrations["REG1"],
        four_state=four_state,
        four_state_receipt={"states": state_receipts},
        resource_receipt={"query_rows_used_for_fit": 0, "query_state_updates": 0},
        runtime_receipt={"query_rows_used_for_fit": 0, "query_truth_input_exists": False, "source_runtime_access": False},
        da1_reg0_state_sha256="a" * 64,
        da1_reg1_state_sha256="a" * 64,
    )


def _fixtures():
    plan = dict(matrix.build_next_r3_proxy24_plan(CLASSES))
    # K1/K5 share query streams per receiver/held class, as required by score.
    query_by_pair: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
    truth: dict[str, str] = {}
    row_results = {}
    for row in plan["rows"]:
        pair = (row["held_receiver"], row["held_class"])
        if pair not in query_by_pair:
            query_by_pair[pair] = {}
            old_classes = tuple(row["retained_classes"])
            new_class = row["held_class"]
            old_ids = tuple(
                f"{pair[0]}|{pair[1]}|REG|{cls}|q{index}"
                for cls in old_classes
                for index in range(matrix.QUERY_PER_CLASS)
            )
            new_ids = tuple(
                f"{pair[0]}|{pair[1]}|REG|{new_class}|q{index}"
                for index in range(matrix.QUERY_PER_CLASS)
            )
            query_by_pair[pair]["REG0"] = old_ids
            query_by_pair[pair]["REG1"] = old_ids + new_ids
            truth.update({qid: qid.split("|")[3] for qid in old_ids + new_ids})
        support_cache = {}
        for registration_id in matrix.REGISTRATION_IDS:
            classes = tuple(row["retained_classes"] if registration_id == "REG0" else row["all_registered_classes"])
            support_cache[registration_id] = tuple(
                f"{pair[0]}|{pair[1]}|{registration_id}|s{i}"
                for i in range(len(classes) * 5)
            )
        row_results[row["row_id"]] = _fake_runtime(row, query_by_pair[pair], truth, support_cache)
    return plan, row_results, truth


def test_builder_emits_complete_truth_free_artifact_that_scores():
    plan, row_results, truth = _fixtures()
    prediction = artifact.build_next_r3_prediction_artifact(plan, row_results)
    assert prediction["schema"] == score.PREDICTION_SCHEMA
    assert prediction["row_count"] == 24
    assert prediction["state_prediction_count"] == 96
    assert prediction["arm_prediction_count"] == 288
    assert "truth" not in prediction
    result = score.score_next_r3_proxy24(prediction=prediction, plan=plan, truth_by_query_id=truth)
    assert result["row_count"] == 24
    assert result["arm_prediction_count"] == 288


def test_builder_rejects_missing_row():
    plan, row_results, _ = _fixtures()
    row_results = dict(row_results)
    row_results.pop(next(iter(row_results)))
    with pytest.raises(artifact.NextR3ArtifactError, match="coverage drift"):
        artifact.build_next_r3_prediction_artifact(plan, row_results)


def test_builder_and_scorer_reject_truth_pollution():
    plan, row_results, truth = _fixtures()
    polluted = dict(row_results)
    first_id = next(iter(polluted))
    polluted[first_id] = {**polluted[first_id].__dict__, "truth": {"q": "tx-0"}}
    with pytest.raises(artifact.NextR3ArtifactError, match="forbidden"):
        artifact.build_next_r3_prediction_artifact(plan, polluted)

    prediction = artifact.build_next_r3_prediction_artifact(plan, row_results)
    bad = deepcopy(prediction)
    bad["query_truth"] = "must-not-be-here"
    with pytest.raises(score.NextR3ScoreError, match="forbidden"):
        score.score_next_r3_proxy24(prediction=bad, plan=plan, truth_by_query_id=truth)
