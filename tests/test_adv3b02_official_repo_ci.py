import torch

from paper_reproduction.cvs_aligned.adv3b02_official_repo_ci import (
    _manual_sgdm_step,
    build_csil_base_state,
    csil_distillation,
    csil_ewc,
    csil_official_fisher_objective,
    fit_csil_official_repo,
    fit_mopc_hr_official_repo,
    mopc_correct_prototypes,
    mopc_parameter_hr,
    zero_bias_logits,
)


def _feature_fn(model, rows):
    feature = model(rows)
    return feature, feature[:, :2]


def _dummy_backbone():
    torch.manual_seed(3)
    return torch.nn.Linear(2, 3)


def test_zero_bias_matches_public_layer_formula():
    features = torch.tensor([[3.0, 4.0], [1.0, -1.0]])
    fingerprints = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    expected = 5.0 * (
        features / torch.sqrt(features.square().sum(1, keepdim=True) + 1e-18)
    ) @ (
        fingerprints
        / torch.sqrt(fingerprints.square().sum(1, keepdim=True) + 1e-18)
    ).t() + 5.0
    assert torch.allclose(
        zero_bias_logits(features, fingerprints), expected, atol=1e-7
    )


def test_csil_fisher_objective_uses_global_minimum_shift():
    probabilities = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    expected = torch.log(probabilities - probabilities.min() + 1e-5).mean()
    assert torch.equal(csil_official_fisher_objective(probabilities), expected)


def test_csil_ewc_and_kd_use_sum_and_fixed_32_divisor():
    previous = {"weight": torch.tensor([[1.0, 2.0]])}
    current = {"weight": torch.tensor([[2.0, 4.0], [9.0, 9.0]])}
    fisher = {"weight": torch.tensor([[2.0, 3.0]])}
    assert torch.equal(csil_ewc(current, previous, fisher), torch.tensor(7.0))
    old = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new = old + 2.0
    assert torch.equal(csil_distillation(old, new), torch.tensor(0.5))


def test_csil_sgdm_scales_current_gradient_by_current_learning_rate():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    model.weight.grad = torch.tensor([[3.0]])
    velocity = {"weight": torch.tensor([[4.0]])}
    _manual_sgdm_step(
        model,
        velocity,
        learning_rate=0.1,
        momentum=0.9,
        l2_factor=0.05,
    )
    expected_velocity = 0.9 * 4.0 + 0.1 * (3.0 + 2.0 * 0.05 * 2.0)
    assert torch.allclose(velocity["weight"], torch.tensor([[expected_velocity]]))
    assert torch.allclose(model.weight, torch.tensor([[2.0 - expected_velocity]]))


def test_mopc_hr_is_per_parameter_unsquared_l2():
    previous = [torch.tensor([0.0, 0.0]), torch.tensor([1.0])]
    current = [torch.tensor([3.0, 4.0]), torch.tensor([3.0])]
    # lambda sequence is 1, 0.5: 5 + 0.5*2
    assert torch.equal(mopc_parameter_hr(current, previous), torch.tensor(6.0))


def test_mopc_correction_is_raw_dot_softmax_not_cosine():
    old = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    new_previous = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    new_current = new_previous + torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.softmax(old @ new_previous.t(), dim=1)
    expected = 0.97 * old + 0.03 * (
        weights @ (new_current - new_previous)
    )
    actual = mopc_correct_prototypes(old, new_previous, new_current)
    assert torch.allclose(actual, expected)
    cosine_weights = torch.softmax(
        torch.nn.functional.normalize(old, dim=1)
        @ torch.nn.functional.normalize(new_previous, dim=1).t(),
        dim=1,
    )
    cosine_result = 0.97 * old + 0.03 * (
        cosine_weights @ (new_current - new_previous)
    )
    assert not torch.allclose(actual, cosine_result)


def test_csil_increment_freezes_backbone_and_old_head_blocks():
    backbone = _dummy_backbone()
    fc_weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    fc_bias = torch.zeros(2)
    fingerprints = torch.eye(2)
    fisher = {
        "fingerprints": torch.ones(2, 2),
        "backbone.weight": torch.ones_like(backbone.weight),
        "backbone.bias": torch.ones_like(backbone.bias),
        "fc_bf_fp.weight": torch.ones_like(fc_weight),
        "fc_bf_fp.bias": torch.ones_like(fc_bias),
    }
    base = {
        "csil": {
            "backbone_state": backbone.state_dict(),
            "fc_weight": fc_weight,
            "fc_bias": fc_bias,
            "fingerprints": fingerprints,
            "fisher": fisher,
        }
    }
    old_x = torch.randn(4, 2)
    new_x = torch.randn(10, 2) + 2
    support_x = torch.cat([old_x, new_x])
    support_y = torch.tensor([0, 0, 1, 1] + [2] * 10)
    state = fit_csil_official_repo(
        backbone,
        support_x,
        support_y,
        feature_fn=_feature_fn,
        old_count=2,
        seed=17,
        base_state=base,
    )
    assert state.resource["optimizer_steps"] == 3
    for key, value in backbone.state_dict().items():
        assert torch.equal(state.current_model.backbone.state_dict()[key], value)
    assert torch.equal(state.current_model.fc_bf_fp.weight[:2], fc_weight)
    assert torch.equal(state.current_model.fingerprints[:2, :2], fingerprints)
    assert state.current_model.fc_bf_fp.weight.shape[0] == 3


def test_mopc_increment_uses_classifier_query_and_kd_is_diagnostic_only():
    backbone = _dummy_backbone()
    classifier = torch.nn.Linear(3, 4)
    base = {
        "mopc_hr": {
            "backbone_state": backbone.state_dict(),
            "classifier_weight": classifier.weight.detach().clone(),
            "classifier_bias": classifier.bias.detach().clone(),
            "old_prototypes": torch.randn(2, 3),
        }
    }
    support_x = torch.randn(12, 2)
    support_y = torch.tensor([0, 0, 1, 1] + [2] * 8)
    state = fit_mopc_hr_official_repo(
        backbone,
        support_x,
        support_y,
        feature_fn=_feature_fn,
        old_count=2,
        seed=19,
        base_state=base,
    )
    assert state.resource["optimizer_steps"] == 20
    assert state.resource["effective_batch_size"] == 8
    assert state.resource["kd_in_total_loss"] is False
    assert "knowledge_distillation_not_in_total" in state.loss_trace[0]


def test_csil_base_keeps_trainnetwork_once_shuffle_and_tail_batch():
    backbone = _dummy_backbone()
    source_x = torch.randn(5, 2)
    source_y = torch.tensor([0, 0, 1, 1, 1])
    state = build_csil_base_state(
        backbone,
        source_x,
        source_y,
        feature_fn=_feature_fn,
        old_count=2,
        seed=23,
        epochs=2,
        batch_size=4,
    )
    assert state["optimizer_steps"] == 4
    assert state["tail_batch_retained"] is True
    assert state["shuffle"] == "once"
