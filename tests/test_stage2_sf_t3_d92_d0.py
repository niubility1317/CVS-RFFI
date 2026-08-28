from __future__ import annotations

import inspect

import pytest
import torch

from cvsrffi.stage2_sf_t3_d92_d0 import (
    D0H6CompactDeltaError,
    build_d0_h6_compact_candidate_spec,
    build_support_only_t3_norm_delta,
)


def test_candidate_spec_locks_d0_h6_compact_to_d92_e0_without_query_or_rf32() -> None:
    spec = build_d0_h6_compact_candidate_spec()

    assert spec["sf_tapft_row"] == "D0"
    assert spec["adapter_execution"] == "H6_COMPACT"
    assert spec["persistent_parameter_names"] == (
        "model.t3.norm.weight",
        "model.t3.norm.bias",
    )
    assert spec["temporary_target_head_policy"] == "discard_after_support_fit"
    assert spec["method_lock"] == "D92-E0-NORF32"
    assert spec["d92_enabled"] is True
    assert spec["e0_locked"] is True
    assert spec["rf32_used"] is False
    assert spec["support_only"] is True
    assert spec["query_rows_used"] == 0


def test_support_only_delta_keeps_only_t3_norm_and_discards_temporary_head() -> None:
    base = {
        "t3.norm.weight": torch.tensor([1.0, 2.0]),
        "t3.norm.bias": torch.tensor([0.25, -0.25]),
        "unrelated.weight": torch.tensor([9.0]),
    }
    adapted = {
        "model.t3.norm.weight": torch.tensor([1.5, 1.0]),
        "model.t3.norm.bias": torch.tensor([0.50, -0.75]),
        "head.weight": torch.tensor([[7.0, 8.0]]),
        "head.bias": torch.tensor([3.0]),
    }

    result = build_support_only_t3_norm_delta(
        base,
        adapted,
        support_rows_used=60,
    )

    assert tuple(result.model_deltas) == (
        "model.t3.norm.weight",
        "model.t3.norm.bias",
    )
    torch.testing.assert_close(
        result.model_deltas["model.t3.norm.weight"], torch.tensor([0.5, -1.0])
    )
    torch.testing.assert_close(
        result.model_deltas["model.t3.norm.bias"], torch.tensor([0.25, -0.50])
    )
    assert result.audit["temporary_target_head_discarded"] is True
    assert result.audit["discarded_target_head_names"] == (
        "head.bias",
        "head.weight",
    )
    assert result.audit["persisted_parameter_names"] == tuple(result.model_deltas)
    assert result.audit["support_rows_used"] == 60
    assert result.audit["query_rows_used"] == 0
    assert result.audit["method_lock"] == "D92-E0-NORF32"
    assert result.audit["rf32_used"] is False

    adapted["model.t3.norm.weight"].add_(100.0)
    torch.testing.assert_close(
        result.model_deltas["model.t3.norm.weight"], torch.tensor([0.5, -1.0])
    )


def test_support_only_delta_rejects_missing_or_invalid_t3_norm_state() -> None:
    base = {
        "t3.norm.weight": torch.ones(2),
        "t3.norm.bias": torch.zeros(2),
    }
    with pytest.raises(D0H6CompactDeltaError, match="missing adapted parameter"):
        build_support_only_t3_norm_delta(
            base,
            {"model.t3.norm.weight": torch.ones(2), "head.weight": torch.ones(1, 2)},
            support_rows_used=60,
        )
    with pytest.raises(D0H6CompactDeltaError, match="finite floating tensor"):
        build_support_only_t3_norm_delta(
            base,
            {
                "model.t3.norm.weight": torch.tensor([float("nan"), 1.0]),
                "model.t3.norm.bias": torch.zeros(2),
            },
            support_rows_used=60,
        )
    with pytest.raises(D0H6CompactDeltaError, match="support_rows_used"):
        build_support_only_t3_norm_delta(
            base,
            {
                "model.t3.norm.weight": torch.ones(2),
                "model.t3.norm.bias": torch.zeros(2),
            },
            support_rows_used=0,
        )


def test_support_only_interface_exposes_no_query_input() -> None:
    parameters = inspect.signature(build_support_only_t3_norm_delta).parameters
    assert "query_rows" not in parameters
    assert "query_labels" not in parameters
    assert "query_truth" not in parameters
    assert "query_roles" not in parameters
