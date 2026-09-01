import copy

import pytest
import torch
from torch import nn

from cvsrffi.marc_ot_source_experts import MARCOTSourceExpertConfig, build_source_expert_bank
from cvsrffi.meta_weight_bank import DeltaTaskKey, WeightDeltaBank


class _ToySourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.t3 = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2, bias=False)
        self.register_buffer("fixed_buffer", torch.tensor([3.0]))
        self.forward_start_weights = []
        with torch.no_grad():
            self.id_backbone.t3.weight.copy_(torch.eye(2))
            self.classifier.weight.copy_(torch.tensor([[1.0, -0.5], [-0.5, 1.0]]))

    def forward(self, iq):
        self.forward_start_weights.append(self.id_backbone.t3.weight.detach().clone())
        return {"tx_logits": self.classifier(self.id_backbone.t3(iq))}


class _BufferMutatingSourceModel(_ToySourceModel):
    def forward(self, iq):
        self.fixed_buffer.add_(1.0)
        return super().forward(iq)


class _ConflictingLogitsModel(_ToySourceModel):
    def forward(self, iq):
        logits = self.classifier(self.id_backbone.t3(iq))
        return {"logits": logits, "tx_logits": logits + 1.0}


class _NonFiniteLogitsModel(_ToySourceModel):
    def forward(self, iq):
        logits = self.classifier(self.id_backbone.t3(iq))
        return {"logits": logits * torch.tensor(float("nan"))}


def _config(**overrides):
    values = dict(
        trainable_prefixes=("id_backbone.t3.",),
        base_checkpoint_id="phase1-base-001",
        steps=3,
        lr=0.2,
        max_rank=8,
    )
    values.update(overrides)
    return MARCOTSourceExpertConfig(**values)


def _batches():
    return {
        DeltaTaskKey("rx_a", "d1", "leo_clear_weak", 10): (torch.tensor([[2.0, 0.0], [1.5, 0.0]]), torch.tensor([1, 1])),
        DeltaTaskKey("rx_b", "d1", "leo_rain_weak", 10): (torch.tensor([[0.0, 2.0], [0.0, 1.5]]), torch.tensor([0, 0])),
        DeltaTaskKey("rx_c", "d1", "leo_low_elev_weak", 10): (torch.tensor([[1.0, 1.0], [1.2, 0.8]]), torch.tensor([1, 0])),
    }


def _split_batches():
    result = {}
    for task_key, (iq, _labels) in _batches().items():
        labels = torch.tensor([0, 1])
        result[task_key] = {
            "expert_fit": {
                "iq": iq,
                "clean_iq": iq,
                "leo_iq": iq.flip(-1),
                "labels": labels,
                "physical_ids": ("fit-0", "fit-1"),
            },
            "expert_select": {
                "iq": iq.flip(0),
                "labels": labels.flip(0),
                "physical_ids": ("select-0", "select-1"),
            },
        }
    return result


def test_source_experts_change_only_allowlisted_identity_parameters_per_distinct_task():
    """Would fail if the frozen classifier/buffer changes or task experts collapse to one delta."""
    model = _ToySourceModel()
    base = copy.deepcopy(model.state_dict())

    result = build_source_expert_bank(model, _batches(), _config())

    assert set(result.task_losses) == set(_batches())
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.task_losses.values())
    assert set(result.updated_parameter_names) == {"id_backbone.t3.weight"}
    deltas = [delta["id_backbone.t3.weight"] for delta in result.task_deltas.values()]
    assert all(torch.count_nonzero(delta).item() > 0 for delta in deltas)
    assert not torch.equal(deltas[0], deltas[1])
    assert torch.equal(model.classifier.weight, base["classifier.weight"])
    assert torch.equal(model.fixed_buffer, base["fixed_buffer"])
    assert torch.equal(model.id_backbone.t3.weight, base["id_backbone.t3.weight"])


def test_each_task_starts_at_same_base_state_and_the_call_restores_mode_and_grad_flags():
    """Would fail if one expert trains from the previous expert or leaves model mutation behind."""
    model = _ToySourceModel().eval()
    original_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    base_identity = model.id_backbone.t3.weight.detach().clone()
    config = _config(steps=2)

    result = build_source_expert_bank(model, _batches(), config)

    assert len(result.task_deltas) == 3
    starts = model.forward_start_weights[:: config.steps]
    assert len(starts) == 3
    assert all(torch.equal(weight, base_identity) for weight in starts)
    assert model.training is False
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == original_requires_grad
    assert all(torch.equal(state["fixed_buffer"], torch.tensor([3.0])) for state in result.adapted_states.values())


def test_delta_bank_uses_existing_blockwise_type_with_rank_bounded_by_unique_task_count():
    """Would fail if source experts substitute a private bank or rank exceeds D-1."""
    result = build_source_expert_bank(_ToySourceModel(), _batches(), _config(max_rank=99))

    assert isinstance(result.bank, WeightDeltaBank)
    assert result.bank.base_checkpoint_id == "phase1-base-001"
    assert len(result.bank.task_keys) == 3
    assert len(result.bank.entries) == 1
    entry = result.bank.entries[0]
    assert entry.spec.name == "t3"
    assert entry.effective_rank <= 2
    assert entry.basis.is_leaf and entry.basis.requires_grad is True
    assert entry.task_coefficients.requires_grad is False
    assert entry.task_coefficients.shape[0] == 3


def test_experts_select_checkpoint_on_disjoint_holdout_and_record_selection() -> None:
    result = build_source_expert_bank(
        _ToySourceModel(),
        _split_batches(),
        _config(
            steps=3,
            paired_consistency_weight=0.05,
            expert_mode="stratified_select",
        ),
    )

    assert set(result.selected_steps) == set(_split_batches())
    assert set(result.select_losses) == set(_split_batches())
    assert all(0 <= step <= 3 for step in result.selected_steps.values())
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.select_losses.values())


def test_expert_split_fails_closed_without_all_classes_or_disjoint_physical_ids() -> None:
    batches = _split_batches()
    first = next(iter(batches))
    batches[first]["expert_select"]["labels"] = torch.tensor([0, 0])
    with pytest.raises(ValueError, match="all old classes"):
        build_source_expert_bank(
            _ToySourceModel(), batches, _config(expert_mode="stratified_select")
        )

    batches = _split_batches()
    batches[first]["expert_select"]["physical_ids"] = ("fit-0", "select-1")
    with pytest.raises(ValueError, match="disjoint"):
        build_source_expert_bank(
            _ToySourceModel(), batches, _config(expert_mode="stratified_select")
        )


def test_legacy_mode_rejects_split_batches_and_records_fixed_final_step() -> None:
    result = build_source_expert_bank(_ToySourceModel(), _batches(), _config(steps=3))
    assert set(result.selected_steps.values()) == {3}
    with pytest.raises(ValueError, match="legacy|single batch"):
        build_source_expert_bank(_ToySourceModel(), _split_batches(), _config())


def test_buffer_drift_is_rejected_and_model_is_restored_instead_of_silently_rewritten():
    """Would fail if an adapted buffer is overwritten before strict blockwise extraction."""
    model = _BufferMutatingSourceModel()
    base_buffer = model.fixed_buffer.detach().clone()

    with pytest.raises(ValueError, match="unallowlisted tensor changed"):
        build_source_expert_bank(model, _batches(), _config())

    assert torch.equal(model.fixed_buffer, base_buffer)


@pytest.mark.parametrize(
    ("model", "config", "batches", "message"),
    [
        (_ToySourceModel(), _config(trainable_prefixes=("unknown.",)), _batches(), "allowlisted"),
        (_ToySourceModel(), _config(), {DeltaTaskKey("only", "d1", "leo_clear_weak", 10): next(iter(_batches().values()))}, "at least two"),
        (_NonFiniteLogitsModel(), _config(), _batches(), "non-finite"),
        (_ConflictingLogitsModel(), _config(), _batches(), "conflicting"),
    ],
)
def test_source_expert_builder_rejects_invalid_training_contracts(model, config, batches, message):
    """Would fail if invalid prefixes, task cardinality, or unsafe logits are silently accepted."""
    with pytest.raises((ValueError, FloatingPointError), match=message):
        build_source_expert_bank(model, batches, config)
