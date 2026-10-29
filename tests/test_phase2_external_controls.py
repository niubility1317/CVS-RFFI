"""Synthetic, source-only checks of the declared Phase2 combination controls."""
import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.pamf_phase2_practical.external_controls import build_control, NATURE


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyNative(nn.Module):
    """A distinct nonlinear native prehead, including a frozen BN buffer."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.bn = nn.BatchNorm1d(4)
        self.prehead = nn.Sequential(nn.Linear(4, 5), nn.Tanh())
        self.output = nn.Linear(5, 2)
        self.register_buffer("temperature", torch.tensor(1.7))
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor([[1., 0., .1], [0., 1., .2],
                                                    [.1, .3, 1.], [-.4, .7, .2]]))
            self.encoder.bias.copy_(torch.tensor([.1, -.2, .3, .4]))
            self.prehead[0].weight.copy_(torch.arange(20).reshape(5, 4) / 31 - .3)
            self.prehead[0].bias.copy_(torch.linspace(-.2, .2, 5))
            self.output.weight.copy_(torch.tensor([[2., 0., .1, .2, -.1],
                                                   [0., 3., -.2, .1, .4]]))
            self.output.bias.copy_(torch.tensor([.3, -.4]))

    def embedding(self, x):
        return self.bn(self.encoder(x))

    def head_input(self, x):
        return self.prehead(self.embedding(x))

    def forward(self, x):
        return self.output(self.head_input(x)) / self.temperature


def _embedding(model, x):
    return model.embedding(x)


def _logits(model, x):
    return model(x)


def _expand_head(state, x, compact_y, class_count, seed):
    model = state["model"]
    old = model.output
    with torch.no_grad():
        v = model.head_input(x)
        new = nn.Linear(old.in_features, class_count).to(old.weight)
        means = torch.stack([v[compact_y == i].mean(0) for i in range(class_count)])
        new.weight.copy_(F.normalize(means, dim=1, eps=1e-12)
                         * old.weight.norm(dim=1).median())
        new.bias.zero_()
        new.weight[:old.out_features].copy_(old.weight)
        new.bias[:old.out_features].copy_(old.bias)
    model.output = new
    return dict(model=model, logits_fn=_logits, trainable_parameters=list(new.parameters()))


@pytest.fixture
def case():
    model = TinyNative().train()
    x = torch.tensor([[1., .2, -.1], [.8, .3, -.2], [-.1, 1., .1],
                      [.1, .9, .3], [.1, -.2, 1.], [.2, -.1, .9]])
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    state = dict(model=model, features_fn=_embedding, score_fn=_logits,
                 classes=(0, 1), resources={"B_steps": 7},
                 native_state={"fixed": torch.tensor([4.])}, expand_head=_expand_head)
    return state, x, y


def _snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _unchanged(model, expected):
    assert expected.keys() == model.state_dict().keys()
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())


@pytest.mark.parametrize("kind", ["ncm", "joint_ce100", "native_output_expand_ft100"])
def test_controls_fork_b_without_any_parameter_buffer_or_metadata_change(case, kind):
    state, x, y = case
    expected = _snapshot(state["model"])
    expected_flags = [p.requires_grad for p in state["model"].parameters()]
    state_copy = copy.deepcopy(state["native_state"])
    rows = []
    result = build_control(kind, state, x, y, (0, 1, 2), 11, logger=rows.append)
    assert result["model"] is not state["model"]
    assert state["model"].training
    assert expected_flags == [p.requires_grad for p in state["model"].parameters()]
    _unchanged(state["model"], expected)
    assert torch.equal(state["native_state"]["fixed"], state_copy["fixed"])
    assert state["resources"] == {"B_steps": 7}
    assert result["classes"] == (0, 1, 2)
    assert result["nature"] == NATURE
    assert result["resources"]["optimizer_steps"] == (0 if kind == "ncm" else 100)
    assert len(rows) == result["resources"]["optimizer_steps"]
    assert not result["query_fit"] and not result["source_in_C"]
    assert not result["target_U_in_C"]
    assert result["resources"]["support_training_exposures"] == len(x) * len(rows)
    assert result["resources"]["trainable_parameters"] == sum(
        p.numel() for p in result["model"].parameters() if p.requires_grad)


def test_ncm_is_normalized_sample_mean_cosine_in_declared_class_order(case):
    state, x, y = case
    # Noncompact IDs ensure the control does not infer columns from label max.
    state["classes"] = (10, 4)
    actual_y = torch.tensor([10, 10, 4, 4, 21, 21])
    result = build_control("ncm", state, x, actual_y, (10, 4, 21), 3)
    reference = copy.deepcopy(state["model"]).eval()
    with torch.no_grad():
        u = F.normalize(reference.embedding(x), dim=1, eps=1e-12)
        prototypes = F.normalize(torch.stack([u[y == i].mean(0) for i in range(3)]),
                                 dim=1, eps=1e-12)
        expected = u @ prototypes.t()
    assert torch.allclose(result["score_fn"](result["model"], x), expected, atol=1e-7)


@pytest.mark.parametrize("kind", ["ncm", "joint_ce100", "native_output_expand_ft100"])
def test_query_scoring_is_samplewise_no_fit_no_buffer_updates(case, kind):
    state, x, y = case
    result = build_control(kind, state, x, y, (0, 1, 2), 2)
    before = _snapshot(result["model"])
    # This synthetic query is introduced only after fitting has returned.
    query = torch.tensor([[.4, -.1, .7], [-.5, .2, .3]], requires_grad=True)
    batched = result["score_fn"](result["model"], query)
    separate = torch.cat([result["score_fn"](result["model"], row[None]) for row in query])
    assert batched.shape == (2, 3)
    assert torch.allclose(batched, separate, atol=2e-7)
    assert not batched.requires_grad
    _unchanged(result["model"], before)


def test_joint_and_native_initialize_different_heads_and_only_native_inherits_b(case, monkeypatch):
    state, x, y = case
    created = []
    real_adam = torch.optim.Adam

    def capture(params, **kwargs):
        parameters = list(params)
        created.append([p.detach().clone() for p in parameters])
        assert kwargs == {"lr": .001, "weight_decay": 0.0}
        return real_adam(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", capture)
    joint = build_control("joint_ce100", state, x, y, (0, 1, 2), 9)
    native = build_control("native_output_expand_ft100", state, x, y, (0, 1, 2), 9)
    reference = copy.deepcopy(state["model"]).eval()
    with torch.no_grad():
        u = F.normalize(reference.embedding(x), dim=1, eps=1e-12)
        means = F.normalize(torch.stack([u[y == i].mean(0) for i in range(3)]), dim=1, eps=1e-12)
        v = reference.head_input(x)
        new_direction = F.normalize(v[y == 2].mean(0), dim=0, eps=1e-12)
        expected_new = new_direction * reference.output.weight.norm(dim=1).median()
    joint_w, joint_b = created[0]
    native_w, native_b = created[1]
    assert joint_w.shape == (3, 4) and native_w.shape == (3, 5)
    assert torch.allclose(joint_w, means)
    assert torch.equal(joint_b, torch.zeros(3))
    assert torch.equal(native_w[:2], state["model"].output.weight)
    assert torch.equal(native_b[:2], state["model"].output.bias)
    assert torch.allclose(native_w[2], expected_new)
    assert native_b[2] == 0
    assert not joint["B_native_decision_inherited"]
    assert native["B_native_decision_inherited"]
    assert any(not torch.equal(native["model"].output.weight[i], native_w[i]) for i in range(3))
    for name, value in state["model"].state_dict().items():
        if not name.startswith("output."):
            assert torch.equal(native["model"].state_dict()[name], value)


def test_each_c_invocation_is_an_independent_b_fork(case):
    state, x, y = case
    first = build_control("joint_ce100", state, x, y, (0, 1, 2), 8)
    with torch.no_grad():
        first["model"].backbone.encoder.weight.fill_(999)
        first["model"].head.weight.fill_(999)
    second = build_control("joint_ce100", state, x, y, (0, 1, 2), 8)
    third = build_control("joint_ce100", state, x, y, (0, 1, 2), 8)
    assert torch.equal(second["model"].backbone.encoder.weight, state["model"].encoder.weight)
    assert torch.equal(second["model"].head.weight, third["model"].head.weight)


def test_zero_mean_rows_are_zero_without_query_radius_or_norm_fixes():
    state = dict(model=nn.Identity(), features_fn=lambda model, x: model(x),
                 classes=(0,), resources={})
    x = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    y = torch.tensor([0, 0, 1, 1])
    result = build_control("ncm", state, x, y, (0, 1), 1)
    assert torch.equal(result["model"].prototypes, torch.zeros(2, 2))
    scores = result["score_fn"](result["model"], x)
    assert torch.equal(scores, torch.zeros(4, 2))
    assert torch.equal(scores.argmax(1), torch.zeros(4, dtype=torch.long))


@pytest.mark.parametrize("change", ["missing", "old_rows", "buffer", "extra_trainable"])
def test_native_missing_or_nonpreserving_bridge_fails_without_ncm_fallback(case, change):
    state, x, y = case
    expected = _snapshot(state["model"])
    if change == "missing":
        state.pop("expand_head")
    else:
        def broken(private, *args):
            result = _expand_head(private, *args)
            with torch.no_grad():
                if change == "old_rows":
                    result["model"].output.weight[0].zero_()
                elif change == "buffer":
                    result["model"].bn.running_mean.add_(1)
                else:
                    result["trainable_parameters"].append(result["model"].prehead[0].weight)
            return result
        state["expand_head"] = broken
    with pytest.raises(ValueError):
        build_control("native_output_expand_ft100", state, x, y, (0, 1, 2), 1)
    _unchanged(state["model"], expected)


@pytest.mark.parametrize("kwargs", [dict(steps=1), dict(lr=.01), dict(weight_decay=.1),
                                    dict(batch_size=2), dict(optimizer="SGD")])
def test_fixed_full_batch_budget_cannot_be_silently_changed(case, kwargs):
    state, x, y = case
    with pytest.raises(ValueError):
        build_control("joint_ce100", state, x, y, (0, 1, 2), 1, config=kwargs)


@pytest.mark.parametrize("failure", ["missing_class", "future_label", "class_order", "nan"])
def test_support_coverage_and_stage_order_are_required(case, failure):
    state, x, y = case
    classes = (0, 1, 2)
    if failure == "missing_class":
        x, y = x[:4], y[:4]
    elif failure == "future_label":
        y = y.clone()
        y[-1] = 8
    elif failure == "class_order":
        classes = (1, 0, 2)
    else:
        x = x.clone()
        x[0, 0] = float("nan")
    with pytest.raises(ValueError):
        build_control("ncm", state, x, y, classes, 1)


class NestedActive(nn.Module):
    def __init__(self, active):
        super().__init__()
        self.active = active

    def forward(self, x):
        return self.active(x)


def _active_logits(model, x):
    return model(x)["tx_logits"]


def _active_expand(state, x, y, count, seed):
    model = state["model"]
    active = model.active if isinstance(model, NestedActive) else model
    expanded = _expand_head(dict(model=active.base), x, y, count, seed)
    active.base = expanded["model"]
    active.columns = torch.arange(count, dtype=torch.long, device=x.device)
    path = "active.columns" if isinstance(model, NestedActive) else "columns"
    return dict(model=model, logits_fn=_active_logits,
                trainable_parameters=expanded["trainable_parameters"],
                declared_class_mapping_keys=[path])


def _active_state(state, nested=False):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import _ActiveClasses
    result = dict(state)
    active = _ActiveClasses(copy.deepcopy(state["model"]), (0, 1))
    result.update(model=NestedActive(active) if nested else active, expand_head=_active_expand)
    return result


@pytest.mark.parametrize("nested", [False, True])
def test_declared_actual_da_wrapper_mapping_grows_preserving_b(case, nested):
    state, x, y = case
    state = _active_state(state, nested)
    before = _snapshot(state["model"])
    result = build_control("native_output_expand_ft100", state, x, y, (0, 1, 2), 5)
    path = "active.columns" if nested else "columns"
    assert torch.equal(result["model"].state_dict()[path], torch.arange(3))
    assert result["resources"]["optimizer_steps"] == 100
    assert result["score_fn"](result["model"], x[:1]).shape == (1, 3)
    _unchanged(state["model"], before)
    for name, value in before.items():
        if name != path and ".output." not in name:
            assert torch.equal(result["model"].state_dict()[name], value)


@pytest.mark.parametrize("failure", ["undeclared", "old_prefix", "new_index", "dtype",
                                    "length", "other_declared", "other_changed", "nonexistent"])
def test_mapping_exception_rejects_wrong_columns_and_any_other_buffer_relaxation(case, failure):
    state, x, y = case
    state = _active_state(state)
    before = _snapshot(state["model"])

    def broken(private, *args):
        result = _active_expand(private, *args)
        model = result["model"]
        if failure == "undeclared":
            result.pop("declared_class_mapping_keys")
        elif failure == "old_prefix":
            model.columns[0] = 1
        elif failure == "new_index":
            model.columns[-1] = 7
        elif failure == "dtype":
            model.columns = model.columns.float()
        elif failure == "length":
            model.columns = model.columns[:2]
        elif failure == "other_declared":
            result["declared_class_mapping_keys"].append("base.bn.running_mean")
        elif failure == "other_changed":
            model.base.bn.running_mean.add_(1)
        else:
            result["declared_class_mapping_keys"] = ["base.columns"]
        return result

    state["expand_head"] = broken
    with pytest.raises(ValueError):
        build_control("native_output_expand_ft100", state, x, y, (0, 1, 2), 5)
    _unchanged(state["model"], before)


def test_arbitrary_columns_buffer_is_not_a_da_class_mapping(case):
    state, x, y = case
    state["model"].register_buffer("columns", torch.arange(2))

    def forged(private, *args):
        result = _expand_head(private, *args)
        result["model"].columns = torch.arange(3)
        result["declared_class_mapping_keys"] = ["columns"]
        return result

    state["expand_head"] = forged
    with pytest.raises(ValueError, match="only actual B _ActiveClasses"):
        build_control("native_output_expand_ft100", state, x, y, (0, 1, 2), 5)
