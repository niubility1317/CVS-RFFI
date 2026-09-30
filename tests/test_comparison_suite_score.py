"""Independent truth-last scorer checks using synthetic sealed predictions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from comparison_suite import score as s


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def prediction(split_id, mode, ids, pred, classes):
    return dict(split_id=split_id, capsule_id="test-capsule", mode=mode, query_ids=ids, predicted_indices=pred,
                classes=classes, query_fit=False, receiver=0, scenario="high", k=1,
                support_seed=2026092711, paired_old_split_id="old-split",
                C_inherits_B=mode.startswith("C"))


def p2_fixture(tmp_path, *, closed=False, two_rows=False):
    truths = {
        "old-a": dict(transmitter="o1", old=True, pool_role="query", receiver=0, scenario="high"),
        "old-b": dict(transmitter="o0", old=True, pool_role="query", receiver=0, scenario="high"),
        "old-c": dict(transmitter="o1", old=True, pool_role="query", receiver=0, scenario="high"),
        "new-a": dict(transmitter="n0", old=False, pool_role="query", receiver=0, scenario="high"),
        "new-b": dict(transmitter="n1", old=False, pool_role="query", receiver=0, scenario="high"),
    }
    truth_path = tmp_path / "truth.json"
    put(truth_path, truths)
    records = [prediction("old-split", "A_source_frozen", ["old-a", "old-b"], [1, 1], ["o1", "o0"]),
               prediction("old-split", "B_old_support", ["old-a", "old-b"], [0, 1], ["o1", "o0"])]
    if not closed:
        records.append(prediction("registered-split", "C_prototype_append_extension", ["old-a", "old-b", "new-a", "new-b"],
                                  [2, 1, 2, 3], ["o1", "o0", "n0", "n1"]))
    p1_capsule, p2_capsule = tmp_path / "p1_capsule", tmp_path / "p2_capsule"
    put(p1_capsule / "manifest.json", dict(classes=["o1", "o0"]))
    put(p2_capsule / "manifest.json", dict(capsule_id="test-capsule"))
    np.savez(p2_capsule / "received.npz", ids=np.asarray(["old-a", "old-b", "new-a", "new-b", "old-c"]))
    for split_id, registered, indices in [("old-split", ["o1", "o0"], [0, 1]),
                                          ("registered-split", ["o1", "o0", "n0", "n1"], [0, 1, 2, 3])]:
        put(p2_capsule / "splits" / (split_id + ".json"), dict(split_id=split_id,
            registered_classes=registered, query_indices=indices, receiver=0, scenario="high", k=1,
            support_seed=2026092711))
    config = tmp_path / "config.json"
    put(config, dict(p1_capsule=str(p1_capsule), p2_capsule=str(p2_capsule)))
    rows = []
    for index in range(2 if two_rows else 1):
        output = tmp_path / f"row{index}"
        output.mkdir()
        (output / "phase2_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        put(output / "phase2_complete.json", dict(status="PREDICTIONS_COMPLETE", truth_read=False,
                                                 predictions=len(records)))
        rows.append(dict(stage="phase12", row_id=f"r{index}", method="dadda" if closed else "protonet",
                         model_seed=2026092701 + index, config=str(config), output_root=str(output)))
    return dict(rows=rows, runtime_root=str(tmp_path), p2_truth=str(truth_path)), records


def replace_records(spec, index, records):
    output = Path(spec["rows"][index]["output_root"])
    (output / "phase2_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def forbid_truth(monkeypatch, truth_path):
    original = s.read
    calls = []
    def wrapped(path):
        if Path(path) == Path(truth_path):
            calls.append(path)
            raise AssertionError("truth must remain closed until every prediction passes preflight")
        return original(path)
    monkeypatch.setattr(s, "read", wrapped)
    return calls


@pytest.mark.parametrize("change", ["incomplete", "contaminated", "missing"])
def test_all_row_markers_preflight_blocks_truth_and_outputs(tmp_path, monkeypatch, change):
    spec, _ = p2_fixture(tmp_path, two_rows=True)
    marker = Path(spec["rows"][1]["output_root"]) / "phase2_complete.json"
    if change == "missing":
        marker.unlink()
    else:
        value = s.read(marker)
        value["status" if change == "incomplete" else "truth_read"] = "FAILED" if change == "incomplete" else True
        put(marker, value)
    calls = forbid_truth(monkeypatch, spec["p2_truth"])
    with pytest.raises((ValueError, FileNotFoundError)):
        s.score(spec, "p2")
    assert not calls
    assert not (tmp_path / "phase2_scored_results.json").exists()


@pytest.mark.parametrize("change", ["duplicate_physical_id", "query_fit", "out_of_range", "float_index", "duplicate_record", "count"])
def test_every_row_frozen_artifact_checked_before_truth(tmp_path, monkeypatch, change):
    spec, records = p2_fixture(tmp_path, two_rows=True)
    corrupt = copy.deepcopy(records)
    if change == "duplicate_physical_id":
        corrupt[0]["query_ids"] = ["old-a", "old-a"]
    elif change == "query_fit":
        corrupt[0]["query_fit"] = True
    elif change == "out_of_range":
        corrupt[0]["predicted_indices"] = [2, 1]
    elif change == "float_index":
        corrupt[0]["predicted_indices"] = [0.0, 1]
    elif change == "duplicate_record":
        corrupt.append(copy.deepcopy(corrupt[0]))
    else:
        corrupt.pop()
    replace_records(spec, 1, corrupt)
    calls = forbid_truth(monkeypatch, spec["p2_truth"])
    with pytest.raises(ValueError):
        s.score(spec, "p2")
    assert not calls


def test_A_B_C_pairing_counts_new_class_competition_and_reports_all_metrics(tmp_path):
    spec, _ = p2_fixture(tmp_path)
    results, pairs = s.p2(spec, s.preflight(spec, "p2"))
    assert len(results) == 3 and len(pairs) == 2
    paired = next(row for row in pairs if row["C_inherits_B"])
    n0 = next(row for row in pairs if not row["C_inherits_B"])
    assert n0["new_class_count"] == 0
    assert n0["C_old_accuracy"] is None and n0["C_new_accuracy"] is None
    assert paired["A_old_accuracy"] == 0.5
    assert paired["B_old_accuracy"] == 1.0
    # old-a is mistaken for registered new class n0; full competition must penalize it.
    assert paired["C_old_accuracy"] == 0.5
    assert paired["C_new_accuracy"] == 1.0
    assert paired["adaptation_gain_pp"] == 50
    assert paired["registration_old_drop_pp"] == 50
    assert paired["old_new_gap_pp"] == 50
    assert paired["H"] == pytest.approx(2 / 3)
    assert paired["C_inherits_B"] is True
    c = next(row for row in results if row["mode"].startswith("C"))
    assert c["accuracy"] == 0.75
    assert c["macro_f1"] == pytest.approx(2 / 3)
    assert c["macro_accuracy"] == 0.75
    assert c["old_class_count"] == 2 and c["new_class_count"] == 2
    assert np.asarray(c["confusion"]).shape == (4, 4)


@pytest.mark.parametrize("stage", ["A_B", "B_C"])
def test_paired_old_physical_ids_must_match(tmp_path, stage):
    spec, records = p2_fixture(tmp_path)
    if stage == "A_B":
        records[1]["query_ids"] = ["old-c", "old-b"]
    else:
        records[2]["query_ids"] = ["old-c", "old-b", "new-a", "new-b"]
    replace_records(spec, 0, records)
    with pytest.raises(ValueError, match="query identity mismatch|old physical query mismatch|capsule identity mismatch"):
        s.p2(spec, s.preflight(spec, "p2"))


def test_closed_set_methods_report_registration_NA_without_filling_other_method(tmp_path):
    spec, _ = p2_fixture(tmp_path, closed=True)
    results, pairs = s.p2(spec, s.preflight(spec, "p2"))
    assert len(results) == 2 and len(pairs) == 1
    assert pairs[0]["C_mode"] == "N/A_closed_set_method"
    for key in ("C_old_accuracy", "C_new_accuracy", "registration_old_drop_pp", "old_new_gap_pp", "H"):
        assert pairs[0][key] is None
    assert pairs[0]["C_inherits_B"] is False


def test_fresh_four_main_and_historical_seed_are_separate_model_level_averages():
    results = []
    for seed, accuracy in zip([392005, 2026092701, 2026092702, 2026092703, 2026092704],
                              [0.9, 0.1, 0.2, 0.3, 0.4]):
        for rx in [0, 2]:
            results.append(dict(method="protonet", mode="B_old", scenario="high", k=5,
                                new_class_count=0, receiver=rx, model_seed=seed,
                                accuracy=accuracy, macro_f1=accuracy, old_accuracy=accuracy,
                                new_accuracy=None, harmonic_mean=None))
    summary = {row["seed_scope"]: row for row in s.aggregate(results, "p2")}
    fresh = summary["fresh_four"]
    assert fresh["model_seeds"] == [2026092701, 2026092702, 2026092703, 2026092704]
    assert fresh["accuracy_mean"] == pytest.approx(0.25)
    assert fresh["accuracy_seed_sd"] == pytest.approx(np.std([.1, .2, .3, .4], ddof=1))
    assert fresh["new_accuracy_mean"] is None
    assert summary["historical_392005"]["accuracy_mean"] == 0.9
    assert summary["historical_392005"]["accuracy_seed_sd"] is None
    assert summary["all_five_sensitivity"]["accuracy_mean"] == pytest.approx(.38)


def test_macro_F1_uses_all_registered_classes_and_zero_denominator_is_defined():
    value = s.metrics(np.array([0, 0, 1]), np.array([0, 1, 2]), 4)
    # Class0 F1=2/3, all other registered class F1=0, including class3 absent in truth.
    assert value["accuracy"] == pytest.approx(1 / 3)
    assert value["macro_f1"] == pytest.approx(1 / 6)
    assert value["macro_accuracy"] == pytest.approx(1 / 4)
    assert value["query_count"] == 3


def test_P1_all_rows_identity_validation_precedes_truth_read(tmp_path, monkeypatch):
    capsule = tmp_path / "p1_capsule"
    capsule.mkdir()
    ids = np.asarray([f"q{i}" for i in range(6)])
    np.savez(capsule / "index.npz", ids=ids, scenes=np.asarray([0, 0, 1, 1, 2, 2]))
    put(capsule / "manifest.json", dict(classes=["o0", "o1"], scenes=["high", "mid", "urban"]))
    config = tmp_path / "config.json"
    put(config, dict(p1_capsule=str(capsule)))
    rows = []
    for i in range(2):
        output = tmp_path / f"row{i}"
        output.mkdir()
        put(output / "phase1_complete.json", dict(status="PREDICTIONS_COMPLETE", truth_read=False))
        np.savez(output / "phase1_predictions.npz", ids=ids if i == 0 else ids[::-1],
                 clean=np.asarray([0, 1, 0, 1, 0, 1]), satellite=np.asarray([0, 1, 0, 1, 0, 1]))
        rows.append(dict(stage="phase12", output_root=str(output), config=str(config)))
    truth = tmp_path / "truth.json"
    put(truth, {})
    spec = dict(rows=rows, runtime_root=str(tmp_path), p1_truth=str(truth))
    calls = forbid_truth(monkeypatch, truth)
    with pytest.raises(ValueError, match="identity mismatch"):
        s.score(spec, "p1")
    assert not calls
    assert not (tmp_path / "phase1_scored_results.json").exists()


def test_score_writes_reproducible_readable_outputs_and_is_idempotent(tmp_path):
    spec, _ = p2_fixture(tmp_path)
    marker = s.score(spec, "p2")
    assert marker["status"] == "SCORED_COMPLETE"
    assert marker["records"] == 3 and marker["paired_records"] == 2
    saved = (tmp_path / "phase2_scored_results.json").read_bytes()
    assert s.score(spec, "p2") == marker
    assert (tmp_path / "phase2_scored_results.json").read_bytes() == saved
    assert (tmp_path / "paired_results.csv").exists()
    assert s.read(tmp_path / "phase2_scored_results.json")["target_feedback_forbidden"] is True
