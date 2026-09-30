"""Synthetic pipeline integration; fixtures are not scientific run results.

Real ProtoNet factory/registration runs on synthetic IQ. The fake 200-epoch
metadata exercises checkpoint checks, without implying these random weights
have actually been trained for 200 epochs. No scientific dataset is opened.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from comparison_suite import pipeline as p
from comparison_suite.models import factory


def dump(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def splits_fixture(classes):
    result = []
    for n in (0, 2, 5, 10, 20):
        count = len(classes) + n
        result.append({"split_id": f"synthetic-n{n}", "capsule_id": "synthetic-residual-noeq",
            "protocol_schema": "p2_min_v1", "phase2_data_status": "VALIDATED_ONCE",
            "receiver": "target-rx", "scenario": "practical_high", "k": 1,
            "support_seed": 123, "registered_classes": [f"tx{i}" for i in range(count)],
            "support_indices": [3*i for i in range(count)], "support_labels": list(range(count)),
            "query_indices": [3*i+j for i in range(count) for j in (1, 2)]})
    return result


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(91)
    classes = [f"tx{i}" for i in range(6)]
    raw_contract = {"role_ids": {"L_s": ["synthetic-source-L"], "U_s": ["synthetic-source-U"], "V": ["synthetic-source-V"]},
        "source_rxs": [1, 3, 4, 6, 8], "source_days": [1, 2, 3],
        "ratios": [.07, .63, .30], "split_seed": 392005, "num_classes": 6}
    contract = dict(raw_contract, classes=classes, equalized=1, out_len=256, normalize=True,
                    physical_roles="EXACT_MATCH")
    source = tmp_path / "source"
    source.mkdir()
    raw_path = tmp_path / "source-input-contract.json"
    dump(raw_path, raw_contract)
    initial = {"status": "SCRATCH", "checkpoint": None, "ancestors": [], "model_seed": 91,
        "source_contract": str(raw_path), "physical_roles": "EXACT_MATCH",
        "selection": "fixed_last_epoch", "target_access": False, "scratch_only": True,
        "checkpoint_sources": [], "target_contact": False}
    dump(source / "initialization.json", initial)
    dump(source / "completion.json", {"status": "SOURCE_TRAINED", "epoch": 200,
         "target_access": False, "target_evaluated": False})
    dump(source / "source_contract.json", contract)
    model = factory("protonet", 6, 12).eval()
    with torch.no_grad():
        model.source_prototypes.copy_(torch.randn_like(model.source_prototypes))
        model.prototypes_ready.fill_(True)
    torch.save({"epoch": 200, "method": "protonet", "classes": classes,
        "source_contract": contract, "initialization": initial, "selection": "fixed_last_epoch",
        "num_classes": 6, "num_receivers": 12, "model_kwargs": {}, "model": model.state_dict()}, source / "last.pt")
    torch.save({"source_role": "L_s", "source_class_ids": list(range(6)),
        "old_prototypes": model.source_prototypes.detach(), "computed_before_target_adaptation": True}, source / "base_state.pt")
    capsule = tmp_path / "phase2"
    (capsule / "splits").mkdir(parents=True)
    dump(capsule / "manifest.json", {"phase2_data_status": "VALIDATED_ONCE", "capsule_id": "synthetic-residual-noeq",
        "channel": {"route": "residual", "mode": "post_sync", "equalization_enabled": False, "fs_hz": 25000000.}})
    rng = np.random.default_rng(91)
    iq = rng.normal(size=(78, 2, 256)).astype(np.float32)
    np.savez(capsule / "received.npz", iq=iq, ids=np.asarray([f"synthetic-id{i}" for i in range(78)]))
    for split in splits_fixture(classes):
        dump(capsule / "splits" / (split["split_id"] + ".json"), split)
    # An attempted truth read would fail; neither fit nor prediction needs it.
    (capsule / "truth.json").write_text("FORBIDDEN_SYNTHETIC_TRUTH_NOT_JSON", encoding="utf-8")
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    dump(phase1 / "manifest.json", {"status": "VALIDATED_ONCE", "classes": classes, "channel": "residual/post_sync/noeq"})
    np.savez(phase1 / "index.npz", ids=np.asarray(["synthetic-p1-a", "synthetic-p1-b"]))
    np.save(phase1 / "clean.npy", iq[:2])
    np.save(phase1 / "satellite.npy", iq[:2] * .97)
    return {"method": "protonet", "model_seed": 91, "source_contract": str(raw_path),
        "source_output": str(source), "output_root": str(tmp_path / "output"),
        "p2_capsule": str(capsule), "p1_capsule": str(phase1), "device": "cpu", "batch_size": 8,
        "registration": {"old_adaptation_epochs": 1}, "synthetic_fixture": True}


def test_real_protonet_end_to_end_predictions_paired_inherit_without_truth(fixture):
    p.run(fixture)
    out = Path(fixture["output_root"])
    done = p.read(out / "phase2_complete.json")
    assert done["status"] == "PREDICTIONS_COMPLETE" and done["predictions"] == 6
    assert done["truth_read"] is False and done["paired_old_tasks"] == 1
    records = [json.loads(row) for row in (out / "phase2_predictions.jsonl").read_text().splitlines()]
    assert [len(row["classes"]) for row in records] == [6, 6, 8, 11, 16, 26]
    assert records[0]["query_ids"] == records[1]["query_ids"]
    for row in records[2:]:
        assert row["query_ids"][:12] == records[1]["query_ids"]
        assert row["C_inherits_B"] is True and row["paired_old_split_id"] == "synthetic-n0"
        assert len(row["predicted_indices"]) == len(row["query_ids"])
        assert all(0 <= y < len(row["classes"]) for y in row["predicted_indices"])
    resources = [json.loads(row) for row in (out / "resources.jsonl").read_text().splitlines()]
    assert all(isinstance(row["inference_seconds"], float) and row["inference_seconds"] >= 0 for row in resources)
    phase1 = np.load(out / "phase1_predictions.npz", allow_pickle=False)
    assert len(phase1["clean"]) == len(phase1["satellite"]) == len(phase1["ids"]) == 2
    assert p.read(out / "provenance.json")["status"] == "VERIFIED"


@pytest.mark.parametrize("mutation", ["overlap", "unpaired_old_support", "unpaired_old_query", "duplicate_support", "bad_labels", "truncated_labels", "duplicate_classes"])
def test_group_rejects_broken_physical_pair_or_mapping(mutation):
    classes = [f"tx{i}" for i in range(6)]
    splits = splits_fixture(classes)
    target = splits[-1]
    if mutation == "overlap": target["query_indices"][0] = target["support_indices"][0]
    elif mutation == "unpaired_old_support": target["support_indices"][0] = 76
    elif mutation == "unpaired_old_query": target["query_indices"][0] = 77
    elif mutation == "duplicate_support": target["support_indices"][-1] = target["support_indices"][-2]
    elif mutation == "bad_labels": target["support_labels"][-1] = -1
    elif mutation == "truncated_labels": target["support_labels"] = target["support_labels"][:-1]
    else: target["registered_classes"][-1] = target["registered_classes"][-2]
    with pytest.raises(ValueError):
        p.group_splits(splits, classes, "synthetic-residual-noeq")


@pytest.mark.parametrize("field,value", [("status", "RESUMED"), ("checkpoint", "ancestor.pt"),
    ("ancestors", ["unknown-parent"]), ("target_access", True), ("target_contact", True),
    ("checkpoint_sources", ["old.pt"]), ("scratch_only", False)])
def test_rejects_checkpoint_origin_or_target_contamination(fixture, field, value):
    source = Path(fixture["source_output"])
    initial = p.read(source / "initialization.json")
    initial[field] = value
    dump(source / "initialization.json", initial)
    with pytest.raises(ValueError):
        p.provenance(fixture, source)


@pytest.mark.parametrize("field,value", [("equalized", 0), ("out_len", 128), ("normalize", False), ("num_classes", 7)])
def test_rejects_checkpoint_input_view_contract_mismatch(fixture, field, value):
    source = Path(fixture["source_output"])
    contract = p.read(source / "source_contract.json")
    contract[field] = value
    dump(source / "source_contract.json", contract)
    with pytest.raises(ValueError):
        p.provenance(fixture, source)


def test_infer_is_per_sample_and_rejects_nonfinite():
    iq = np.ones((7, 2, 256), dtype=np.float32)
    def score(x):
        return torch.stack([x.sum((1, 2)), -x.sum((1, 2))], 1)
    assert p.infer(score, iq, "cpu", 1) == p.infer(score, iq, "cpu", 7) == [0]*7
    with pytest.raises(FloatingPointError):
        p.infer(lambda x: torch.full((len(x), 2), float("nan")), iq, "cpu")
    with pytest.raises(ValueError):
        p.infer(lambda x: torch.ones((len(x), 2)), iq, "cpu", expected_classes=26)
    with pytest.raises(ValueError):
        p.infer(lambda x: torch.ones((1, 2)), iq, "cpu", expected_classes=2)


def test_pipeline_rejects_phase1_unpaired_array(fixture):
    np.save(Path(fixture["p1_capsule"]) / "satellite.npy", np.ones((1, 2, 256), dtype=np.float32))
    with pytest.raises(ValueError, match="paired IQ shape"):
        p.run(fixture)


def test_pipeline_rejects_different_residual_processing_version(fixture):
    capsule = Path(fixture["p2_capsule"])
    manifest = p.read(capsule / "manifest.json")
    manifest["channel"]["equalization_enabled"] = True
    dump(capsule / "manifest.json", manifest)
    with pytest.raises(ValueError, match="channel version"):
        p.run(fixture)
