from __future__ import annotations

import inspect

import torch

from paper_reproduction.cvs_aligned.adv3b02_ci_heads import (
    METHODS,
    fit_incremental_head,
    predict_incremental_head,
    prototype_baseline,
)


def _support(k: int = 1):
    generator = torch.Generator().manual_seed(17)
    labels = torch.arange(11).repeat_interleave(k)
    centers = torch.randn(11, 160, generator=generator)
    features = centers[labels] + 0.01 * torch.randn(11 * k, 160, generator=generator)
    query = centers.repeat_interleave(2, dim=0) + 0.02 * torch.randn(22, 160, generator=generator)
    return features, labels, query


def test_all_ci_heads_fit_k1_without_query_and_predict_all_classes():
    support, labels, query = _support(1)
    signature = inspect.signature(fit_incremental_head)
    assert all("query" not in name for name in signature.parameters)
    for method in METHODS:
        fitted = fit_incremental_head(
            method, support, labels, old_count=6, seed=713101, steps=1
        )
        before, after = predict_incremental_head(fitted, query)
        assert before.shape == after.shape == (22,)
        assert int(before.max()) < 6
        assert int(after.max()) < 11
        assert fitted.resource["query_rows_used_for_training"] == 0
        assert fitted.resource["trainable_parameters"] <= 50_000
        assert fitted.resource["persistent_state_bytes"] <= 256 * 1024


def test_identity_prototype_baseline_is_per_sample_all_registered_classes():
    support, labels, query = _support(2)
    predicted = prototype_baseline(support, labels, query, class_count=11)
    assert predicted.shape == (22,)
    assert int(predicted.min()) >= 0
    assert int(predicted.max()) < 11


def test_truth_free_predictor_source_has_no_truth_or_quota_cli():
    source = (
        __import__(
            "paper_reproduction.scripts.run_adv3b02_ci_truth_free_predictor",
            fromlist=["dummy"],
        )
    )
    parser_source = inspect.getsource(source.parse_args)
    assert "truth" not in parser_source.lower()
    assert "quota" not in parser_source.lower()
    predict_source = inspect.getsource(source.predict)
    assert "hungarian" not in predict_source.lower()
    assert "optimal_transport" not in predict_source.lower()
    assert predict_source.index("enrolled_head_sha256 =") < predict_source.index(
        "roles[f\"query:{scenario}\"]"
    )
    assert '"query_members_opened_before_head_lock": False' in predict_source
