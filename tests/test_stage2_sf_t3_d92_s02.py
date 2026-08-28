from __future__ import annotations

import inspect
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from cvsrffi.stage2_sf_t3_d92_s02 import (
    S02PersistentDelta,
    adapt_s02_support_only,
    build_s02_long_horizon_spec,
)
from cvsrffi.target_only_progressive_adapt import TargetOnlyAdaptationDataset


class _TinyT3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t3 = nn.Module()
        self.t3.norm = nn.LayerNorm(3)
        self.unrelated = nn.Linear(3, 3)


def _support() -> TargetOnlyAdaptationDataset:
    return TargetOnlyAdaptationDataset(
        received_iq=torch.zeros(2, 2, 4),
        labels=torch.tensor([0, 1]),
        physical_ids=("support-0", "support-1"),
    )


def _successful_selector(calls: list[object], *, selected_steps=(317, 0, 0)):
    def select(model, support, config, *, folds, full_support_refit):
        calls.append((model, support, config, folds, full_support_refit))
        fitted = _TinyT3Model()
        fitted.load_state_dict(model.state_dict())
        anchors = {
            "model.t3.norm.weight": model.t3.norm.weight.detach().clone(),
            "model.t3.norm.bias": model.t3.norm.bias.detach().clone(),
            "head.weight": torch.zeros(2, 3),
        }
        with torch.no_grad():
            fitted.t3.norm.weight.add_(torch.tensor([0.25, -0.5, 0.75]))
            fitted.t3.norm.bias.add_(torch.tensor([-0.1, 0.2, 0.3]))
        full_support_result = SimpleNamespace(
            model=fitted,
            head=nn.Linear(3, 2, bias=False),
            base_parameter_anchors=MappingProxyType(anchors),
            audit=SimpleNamespace(
                phase_steps=selected_steps,
                total_steps=sum(selected_steps),
                updated_parameter_names=(
                    "head.weight",
                    "model.t3.norm.weight",
                    "model.t3.norm.bias",
                ),
                nonpermitted_changed_names=(),
                source_loader_opened=False,
                source_samples_opened=False,
                source_cache_opened=False,
                target_eval_opened=False,
                query_opened=False,
                training_sample_count=len(support.physical_ids),
                checkpoint_selection_role="fixed_final_step",
            ),
        )
        return SimpleNamespace(
            selected="adapted",
            fold_rows=tuple(object() for _ in range(4)),
            selected_phase_steps=selected_steps,
            adapted_result=full_support_result,
            full_support_result=full_support_result,
            final_training_sample_count=len(support.physical_ids),
            fold0_as_final=False,
        )

    return select


def test_s02_spec_locks_the_long_horizon_and_t3_affine_only() -> None:
    spec = build_s02_long_horizon_spec()

    assert spec.row_id == "S02"
    assert spec.candidate_id == "SF_TAPFT_NORM_T3_LONG_D92_E0_NORF32"
    assert spec.method_lock == "D92-E0-NORF32"
    assert spec.rf32_used is False
    assert spec.persistent_parameter_names == (
        "model.t3.norm.weight",
        "model.t3.norm.bias",
    )
    assert spec.config.trainability_profile == "p1_head_norm"
    assert spec.config.norm_rules == (("t3", "weight_bias"),)
    assert spec.config.phase_steps == (4500, 0, 0)
    assert spec.selection_mode == "grouped_oof_full_support_refit"
    assert spec.folds == 4
    assert spec.config.scheduler_reference_steps == 0
    assert spec.config.validation_steps == ()
    assert spec.config.checkpoint_average_top_k == 3


def test_support_only_fit_persists_only_t3_norm_delta_and_discards_head() -> None:
    calls: list[object] = []
    model = _TinyT3Model()

    result = adapt_s02_support_only(
        model,
        _support(),
        grouped_selector=_successful_selector(calls),
    )

    assert isinstance(result, S02PersistentDelta)
    assert len(calls) == 1
    _, passed_support, config, folds, full_support_refit = calls[0]
    assert passed_support.physical_ids == ("support-0", "support-1")
    assert config.phase_steps == (4500, 0, 0)
    assert config.checkpoint_average_top_k == 3
    assert folds == 4
    assert full_support_refit is True
    assert tuple(result.parameter_deltas) == (
        "model.t3.norm.weight",
        "model.t3.norm.bias",
    )
    assert torch.equal(
        result.parameter_deltas["model.t3.norm.weight"],
        torch.tensor([0.25, -0.5, 0.75]),
    )
    assert torch.allclose(
        result.parameter_deltas["model.t3.norm.bias"],
        torch.tensor([-0.1, 0.2, 0.3]),
    )
    assert "head.weight" not in result.parameter_deltas
    assert result.audit["temporary_target_head_discarded"] is True
    assert result.audit["method_lock"] == "D92-E0-NORF32"
    assert result.audit["d92_variant"] == "E0"
    assert result.audit["rf32_used"] is False
    assert result.audit["query_rows_used"] == 0
    assert result.audit["query_opened"] is False
    assert result.audit["source_opened"] is False
    assert result.audit["selection_mode"] == "grouped_oof_full_support_refit"
    assert result.audit["selection_folds"] == 4
    assert result.audit["selection_phase_steps"] == (4500, 0, 0)
    assert result.audit["selected_phase_steps"] == (317, 0, 0)
    assert result.audit["full_support_refit"] is True


def test_support_only_interface_has_no_query_or_truth_input_and_fails_closed() -> None:
    parameter_names = set(inspect.signature(adapt_s02_support_only).parameters)
    assert not parameter_names & {"query", "query_iq", "query_labels", "truth"}

    def query_using_selector(model, support, config, *, folds, full_support_refit):
        selection = _successful_selector([])(
            model,
            support,
            config,
            folds=folds,
            full_support_refit=full_support_refit,
        )
        selection.full_support_result.audit.query_opened = True
        return selection

    with pytest.raises(ValueError, match="query/source-free"):
        adapt_s02_support_only(
            _TinyT3Model(),
            _support(),
            grouped_selector=query_using_selector,
        )


def test_support_only_interface_emits_zero_delta_for_legal_zero_adapt() -> None:
    def fold_only_selector(model, support, config, *, folds, full_support_refit):
        return SimpleNamespace(
            selected="zero_adapt",
            fold_rows=tuple(object() for _ in range(4)),
            selected_phase_steps=(301, 0, 0),
            adapted_result=None,
            full_support_result=None,
            final_training_sample_count=0,
            fold0_as_final=False,
        )

    result = adapt_s02_support_only(
        _TinyT3Model(),
        _support(),
        grouped_selector=fold_only_selector,
    )

    assert all(torch.count_nonzero(value) == 0 for value in result.parameter_deltas.values())
    assert result.audit["selected"] == "zero_adapt"
    assert result.audit["zero_delta_fallback"] is True
    assert result.audit["full_support_refit"] is False
