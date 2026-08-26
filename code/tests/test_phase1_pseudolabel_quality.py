import importlib.util
import math
from pathlib import Path

import torch

from cvsrffi.phase1_pseudolabel_quality import (
    build_truth_blind_records,
    score_truth_blind_records,
)


def _records():
    return build_truth_blind_records(
        physical_sample_ids=["p0", "p1", "p2", "p3"],
        receivers=["rx0", "rx0", "rx1", "rx1"],
        days=["d0", "d0", "d1", "d1"],
        routes=["H", "P", "P", "R"],
        pseudo_labels=[0, 1, 2, 1],
        candidate_masks=torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=torch.bool
        ),
        fused_probabilities=torch.tensor(
            [[0.90, 0.05, 0.05], [0.10, 0.55, 0.35], [0.45, 0.10, 0.45], [0.30, 0.40, 0.30]]
        ),
        p_correct=[0.99, 0.20, 0.30, 0.10],
        p_set_safe=[0.99, 0.97, 0.96, 0.20],
        sample_weights=[1.0, 0.8, 0.7, 0.0],
    )


def test_truth_blind_artifact_contains_decisions_but_never_tx_truth():
    records = _records()

    assert records[1] == {
        "physical_sample_id": "p1",
        "receiver": "rx0",
        "day": "d0",
        "route": "P",
        "top1_pseudo_label": 1,
        "candidate_set": [1, 2],
        "p_correct": 0.2,
        "p_set_safe": 0.97,
        "partial_conditional_distribution": [0.0, 0.611111, 0.388889],
        "sample_weight": 0.8,
    }
    assert all("truth" not in key and "label" not in key.replace("pseudo_label", "") for key in records[0])


def test_independent_scorer_reports_h_precision_pset_coverage_and_rank_accuracy():
    summary = score_truth_blind_records(
        _records(),
        truth_by_physical_id={"p0": 0, "p1": 2, "p2": 1, "p3": 1},
    )

    assert summary["counts"] == {"all": 4, "H": 1, "P": 2, "R": 1}
    assert summary["overall"]["h_precision"] == 1.0
    assert summary["overall"]["h_coverage"] == 0.25
    assert summary["overall"]["p_set_coverage"] == 0.5
    assert summary["overall"]["p_mean_set_size"] == 2.0
    assert summary["overall"]["p_p95_set_size"] == 2.0
    assert summary["overall"]["p_rank_accuracy_when_set_safe"] == 0.0
    assert math.isfinite(summary["overall"]["h_aurc"])
    assert summary["by_receiver"]["rx0"]["p_set_coverage"] == 1.0
    assert summary["by_receiver"]["rx1"]["p_set_coverage"] == 0.0
    assert summary["worst_receiver"]["p_set_coverage"] == 0.0


def test_independent_scorer_rejects_missing_or_extra_truth_ids():
    records = _records()

    try:
        score_truth_blind_records(records, {"p0": 0})
    except ValueError as exc:
        assert "truth ids must exactly match artifact ids" in str(exc)
    else:
        raise AssertionError("missing truth ids must fail closed")


def test_generator_cli_cannot_accept_truth_and_truth_export_is_a_separate_command():
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase1_fasttrust_vselect_quality.py"
    spec = importlib.util.spec_from_file_location("phase1_fasttrust_vselect_quality", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    parser = module.build_parser()

    generate_args = parser.parse_args(
        ["generate", "--checkpoint", "model.pth", "--artifact-out", "artifact.jsonl"]
    )
    truth_args = parser.parse_args(
        ["extract-truth", "--checkpoint", "model.pth", "--truth-out", "truth.json"]
    )

    assert not hasattr(generate_args, "truth_out")
    assert truth_args.func is module.extract_truth
