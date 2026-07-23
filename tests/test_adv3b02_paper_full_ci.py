import copy

import torch
from torch import nn

from paper_reproduction.cvs_aligned.adv3b02_paper_full_ci import (
    _semantic_layer_hierarchical_regularization,
    fit_csil_paper_full,
    fit_mopc_hr_paper_full,
    predict_after,
    zero_bias_logits,
)


class TinyBackbone(nn.Module):
    def __init__(self, dim=6, old_count=2):
        super().__init__()
        self.block1 = nn.Linear(dim, dim)
        self.block2 = nn.Linear(dim, dim)
        self.old_head = nn.Linear(dim, old_count)

    def forward(self, x):
        feature = torch.tanh(self.block2(torch.relu(self.block1(x))))
        return feature, self.old_head(feature)


def feature_fn(backbone, x):
    return backbone(x)


def support(class_count, shots=2, dim=6):
    generator = torch.Generator().manual_seed(713101)
    labels = torch.arange(class_count).repeat_interleave(shots)
    centers = torch.eye(class_count, dim)
    x = centers[labels] + 0.05 * torch.randn(
        len(labels), dim, generator=generator
    )
    return x, labels


def test_zero_bias_matches_five_cosine_plus_five():
    x = torch.tensor([[1.0, 0.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(zero_bias_logits(x, w), torch.tensor([[10.0, 5.0]]))


def test_csil_updates_backbone_but_masks_old_fingerprint_blocks():
    torch.manual_seed(9)
    x, y = support(4)
    backbone = TinyBackbone()
    initial = copy.deepcopy(backbone.state_dict())
    fitted = fit_csil_paper_full(
        backbone,
        x,
        y,
        feature_fn=feature_fn,
        old_count=2,
        seed=19,
    )
    model = fitted.current_model
    assert fitted.resource["backbone_frozen"] is False
    assert fitted.resource["optimizer_steps"] == 3
    assert fitted.resource["fisher_max"] >= fitted.resource["fisher_min_active"] >= 1.0
    assert any(
        not torch.equal(initial[name], value)
        for name, value in model.backbone.state_dict().items()
    )
    assert torch.count_nonzero(model.fingerprints[:2, model.feature_dim :]) == 0
    assert torch.count_nonzero(model.fingerprints[2:, : model.feature_dim]) == 0


def test_csil_accepts_original_base_fingerprints_and_fisher():
    torch.manual_seed(12)
    x, y = support(4)
    backbone = TinyBackbone()
    with torch.no_grad():
        base_features = feature_fn(backbone, x[y < 2])[0]
    base_fingerprints = torch.stack(
        [base_features[y[y < 2] == class_id].mean(0) for class_id in range(2)]
    )
    fisher = {
        name: torch.ones_like(parameter)
        for name, parameter in backbone.named_parameters()
    }
    fitted = fit_csil_paper_full(
        backbone,
        x,
        y,
        feature_fn=feature_fn,
        old_count=2,
        seed=31,
        base_old_fingerprints=base_fingerprints,
        base_fisher=fisher,
    )
    assert fitted.resource["fisher_source"] == "original_base_source_training_state"
    assert fitted.resource["old_fingerprint_source"] == (
        "original_base_source_training_state"
    )


def test_semantic_hr_uses_one_decay_per_module_and_squared_l2():
    previous = {
        "a.weight": torch.zeros(2),
        "a.bias": torch.zeros(1),
        "b.weight": torch.zeros(1),
    }
    current = {name: value + 1.0 for name, value in previous.items()}
    # layer a lambda=1.0 for three elements; layer b lambda=0.5 for one.
    value = _semantic_layer_hierarchical_regularization(
        current, previous, lambda_max=1.0
    )
    assert torch.allclose(value, torch.tensor(3.5))


def test_mopc_uses_sequential_five_class_stages_and_classifier_query():
    torch.manual_seed(10)
    x, y = support(12, shots=1)
    backbone = TinyBackbone()
    fitted = fit_mopc_hr_paper_full(
        backbone,
        x,
        y,
        feature_fn=feature_fn,
        old_count=2,
        seed=23,
        epochs=1,
        batch_size=4,
    )
    assert fitted.resource["incremental_stage_sizes"] == [5, 5]
    assert fitted.resource["query_decision"] == (
        "current_model_all_registered_classifier_logits"
    )
    assert len(fitted.loss_trace) == 4
    assert fitted.after_prototypes.shape == (12, 6)
    query = x[:3]
    expected = fitted.current_model(query)[0][:, :12].argmax(1)
    assert torch.equal(predict_after(fitted, query), expected)


def test_mopc_k1_small_batch_still_runs_twenty_epochs():
    torch.manual_seed(11)
    x, y = support(4, shots=1)
    fitted = fit_mopc_hr_paper_full(
        TinyBackbone(),
        x,
        y,
        feature_fn=feature_fn,
        old_count=2,
        seed=29,
    )
    assert fitted.resource["optimizer_steps"] == 20
    assert fitted.resource["batch_size"] == 16
    assert all(row["registered_class_count"] == 4 for row in fitted.loss_trace)
