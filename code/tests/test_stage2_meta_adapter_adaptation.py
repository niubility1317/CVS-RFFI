from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import ResidualMetaAdapter  # noqa: E402
from cvsrffi.stage2_meta_adapter_adaptation import (  # noqa: E402
    AdaptedMetaAdapterHandle,
    MetaAdapterPhase2Config,
    MetaAdapterPhase2DiagnosticConfig,
    ValidatedTargetSupportBatch,
    adapt_meta_adapter_diagnostic_on_support,
    adapt_meta_adapter_on_support,
    predict_with_frozen_meta_adapter,
)
from model import build_model  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


class _ToyPhase2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.meta_adapter_time = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_freq = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_fusion = ResidualMetaAdapter(dim=4, rank=2)
        self.fixed_backbone_parameter = nn.Parameter(torch.zeros(10_000))
        self.register_buffer("query_counter", torch.zeros((), dtype=torch.long))

    def forward(self, x, y=None, return_aux=False):
        del y, return_aux
        z = self.meta_adapter_time(x)
        z = self.meta_adapter_freq(z)
        z = self.meta_adapter_fusion(z)
        self.query_counter.add_(1)
        return {"feat_cls": z}


class _ToyPartialPhase2Model(_ToyPhase2Model):
    def forward(self, x, y=None, return_aux=False):
        del y, return_aux
        return {"feat_cls": self.meta_adapter_time(x)}


def _context(capsule_id: str = "capsule-test-01", split_id: str = "split-test-01") -> dict[str, str]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": capsule_id,
        "split_id": split_id,
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    torch.manual_seed(23)
    support_iq = torch.randn(6, 4)
    support_labels = torch.tensor([10, 20, 30, 10, 20, 30], dtype=torch.long)
    prototypes = torch.eye(4, dtype=torch.float32)[:3]
    class_ids = [10, 20, 30]
    return support_iq, support_labels, prototypes, class_ids


def _carrier(
    support_iq: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    capsule_id: str = "capsule-test-01",
    split_id: str = "split-test-01",
) -> ValidatedTargetSupportBatch:
    return ValidatedTargetSupportBatch(
        received_iq=support_iq,
        labels=support_labels,
        support_physical_ids=tuple(f"rx7-physical-{index}" for index in range(support_iq.size(0))),
        receiver_id=7,
        context=_context(capsule_id, split_id),
    )


def _config(capsule_id: str = "capsule-test-01", split_id: str = "split-test-01") -> MetaAdapterPhase2Config:
    return MetaAdapterPhase2Config(
        expected_capsule_id=capsule_id,
        expected_split_id=split_id,
    )


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_validated_support_carrier_is_typed_and_rejects_invalid_physical_ids():
    support_iq, support_labels, _, _ = _inputs()
    carrier = _carrier(support_iq, support_labels)
    assert isinstance(carrier, ValidatedTargetSupportBatch)
    assert carrier.received_iq.shape[0] == len(carrier.support_physical_ids)
    assert len(set(carrier.support_physical_ids)) == support_iq.shape[0]
    with pytest.raises(ValueError, match="physical"):
        ValidatedTargetSupportBatch(
            received_iq=support_iq,
            labels=support_labels,
            support_physical_ids=("same",) * support_iq.size(0),
            receiver_id=7,
            context=_context(),
        )
    with pytest.raises(ValueError, match="provenance"):
        ValidatedTargetSupportBatch(
            received_iq=support_iq,
            labels=support_labels,
            support_physical_ids=carrier.support_physical_ids,
            receiver_id=7,
            context=_context(),
            provenance="clean_source_iq",
        )


def test_formal_api_accepts_only_validated_carrier_and_frozen_expected_handles():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    carrier = _carrier(support_iq, support_labels)
    handle = adapt_meta_adapter_on_support(model, carrier, prototypes, class_ids, _config())
    assert isinstance(handle, AdaptedMetaAdapterHandle)
    assert handle.audit.diagnostic is False
    with pytest.raises(TypeError, match="ValidatedTargetSupportBatch"):
        adapt_meta_adapter_on_support(model, support_iq, prototypes, class_ids, _config())
    with pytest.raises(ValueError, match="capsule_id"):
        adapt_meta_adapter_on_support(
            _ToyPhase2Model(),
            _carrier(support_iq, support_labels, capsule_id="other-capsule"),
            prototypes,
            class_ids,
            _config(),
        )
    with pytest.raises(TypeError, match="steps"):
        MetaAdapterPhase2Config(expected_capsule_id="c", expected_split_id="s", steps=3)


def test_phase2_updates_only_adapter_and_exactly_three_real_support_steps():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    carrier = _carrier(support_iq, support_labels)
    before = _state(model)
    learned_steps_before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name.endswith("log_step_size")
    }

    handle = adapt_meta_adapter_on_support(model, carrier, prototypes, class_ids, _config())
    audit = handle.audit

    assert audit.steps == 3
    assert audit.support_loss_evaluations == 3
    assert audit.gradient_updates == 3
    assert not hasattr(audit, "backward_count")
    assert audit.trainable_fraction <= 0.01
    assert audit.updated_parameter_names
    assert all("meta_adapter" in name for name in audit.updated_parameter_names)
    for name, value in model.named_parameters():
        if name.endswith("log_step_size"):
            assert torch.equal(value.detach(), learned_steps_before[name])
        if not name.startswith("meta_adapter_"):
            assert torch.equal(value.detach(), before[name])
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_formal_phase2_applies_bundle_selected_prototype_logit_scale():
    model_default = _ToyPhase2Model()
    model_scaled = _ToyPhase2Model()
    model_scaled.load_state_dict(model_default.state_dict())
    support_iq, support_labels, prototypes, class_ids = _inputs()
    carrier = _carrier(support_iq, support_labels)

    default_handle = adapt_meta_adapter_on_support(
        model_default,
        carrier,
        prototypes,
        class_ids,
        _config(),
    )
    scaled_handle = adapt_meta_adapter_on_support(
        model_scaled,
        carrier,
        prototypes,
        class_ids,
        MetaAdapterPhase2Config(
            expected_capsule_id="capsule-test-01",
            expected_split_id="split-test-01",
            adaptation_objective="frozen_prototype_cosine_ce_v1",
            support_logit_scale=16.0,
        ),
    )

    assert scaled_handle.audit.adaptation_objective == "frozen_prototype_cosine_ce_v1"
    assert scaled_handle.audit.support_logit_scale == 16.0
    assert any(
        not torch.equal(default_handle.fast_state.parameters[name], value)
        for name, value in scaled_handle.fast_state.parameters.items()
    )


def test_diagnostic_api_is_explicit_bounded_and_not_query_eligible():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    carrier = _carrier(support_iq, support_labels)
    diagnostic = adapt_meta_adapter_diagnostic_on_support(
        model,
        carrier,
        prototypes,
        class_ids,
        MetaAdapterPhase2DiagnosticConfig(
            steps=5,
            expected_capsule_id="capsule-test-01",
            expected_split_id="split-test-01",
        ),
    )
    assert diagnostic.audit.diagnostic is True
    assert diagnostic.audit.support_loss_evaluations == 5
    with pytest.raises(ValueError, match="formal"):
        predict_with_frozen_meta_adapter(diagnostic, support_iq[:2], prototypes, class_ids)
    with pytest.raises(ValueError, match="5"):
        adapt_meta_adapter_diagnostic_on_support(
            _ToyPhase2Model(),
            carrier,
            prototypes,
            class_ids,
            MetaAdapterPhase2DiagnosticConfig(
                steps=6,
                expected_capsule_id="capsule-test-01",
                expected_split_id="split-test-01",
            ),
        )


def test_formal_query_accepts_only_frozen_handle_and_restores_original_python_state():
    model = _ToyPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    carrier = _carrier(support_iq, support_labels)
    handle = adapt_meta_adapter_on_support(model, carrier, prototypes, class_ids, _config())
    before = _state(handle.model)
    prediction = predict_with_frozen_meta_adapter(handle, support_iq[:3], prototypes, class_ids)
    assert prediction.shape == (3,)
    assert prediction.dtype == torch.long
    assert set(prediction.tolist()).issubset(set(class_ids))
    for name, value in handle.model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert handle.model.training is False
    assert all(not parameter.requires_grad for parameter in handle.model.parameters())
    with pytest.raises(TypeError, match="AdaptedMetaAdapterHandle"):
        predict_with_frozen_meta_adapter(model, support_iq[:2], prototypes, class_ids)


def test_unreachable_adapter_stays_frozen_without_fake_update_names():
    model = _ToyPartialPhase2Model()
    support_iq, support_labels, prototypes, class_ids = _inputs()
    carrier = _carrier(support_iq, support_labels)
    before = _state(model)
    handle = adapt_meta_adapter_on_support(model, carrier, prototypes, class_ids, _config())
    assert handle.audit.updated_parameter_names
    assert all(name.startswith("meta_adapter_time.") for name in handle.audit.updated_parameter_names)
    for name, value in model.state_dict().items():
        if name.startswith(("meta_adapter_freq.", "meta_adapter_fusion.")):
            assert torch.equal(value, before[name]), name


def test_real_single_adv3b02_support_adaptation_and_formal_prediction():
    torch.manual_seed(101)
    model = build_model(
        num_classes=3,
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="base",
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    )
    support_iq = torch.randn(3, 2, 64)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        embedding_dim = int(model(support_iq, return_aux=True)["feat_cls"].size(1))
    prototypes = torch.randn(3, embedding_dim)
    carrier = _carrier(support_iq, labels)
    handle = adapt_meta_adapter_on_support(model, carrier, prototypes, [0, 1, 2], _config())
    assert handle.audit.gradient_updates == 3
    assert handle.audit.trainable_fraction <= 0.01
    prediction = predict_with_frozen_meta_adapter(handle, support_iq[:1], prototypes, [0, 1, 2])
    assert prediction.shape == (1,)


def test_real_dual_adv3b02_updates_identity_and_preserves_domain_adapter():
    torch.manual_seed(102)
    model = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="base",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
    )
    support_iq = torch.randn(3, 2, 64)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        embedding_dim = int(model(support_iq, return_aux=True)["z_id"].size(1))
    prototypes = torch.randn(3, embedding_dim)
    carrier = _carrier(support_iq, labels)
    before = _state(model)
    handle = adapt_meta_adapter_on_support(model, carrier, prototypes, [0, 1, 2], _config())
    assert handle.audit.gradient_updates == 3
    assert any(name.startswith("id_backbone.meta_adapter_") for name in handle.audit.updated_parameter_names)
    assert all(not name.startswith("dom_backbone.meta_adapter_") for name in handle.audit.updated_parameter_names)
    for name, value in model.state_dict().items():
        if name.startswith("dom_backbone.meta_adapter_"):
            assert torch.equal(value, before[name]), name
    prediction = predict_with_frozen_meta_adapter(handle, support_iq[:1], prototypes, [0, 1, 2])
    assert prediction.shape == (1,)
