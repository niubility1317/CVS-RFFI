import copy
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import (  # noqa: E402
    ResidualMetaAdapter,
    iter_inner_adapter_parameters,
)
from cvsrffi.meta_inner_loop import (  # noqa: E402
    FastAdapterState,
    MetaInnerLoopError,
    first_order_adapt,
    functional_forward,
)
from model import build_model  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _tiny_model(**kwargs):
    model_variant = kwargs.pop("model_variant", "lite_h")
    return build_model(
        num_classes=3,
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant=model_variant,
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
        **kwargs,
    )


def _tiny_dual_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
        **kwargs,
    )


class _ToyAdapterModel(nn.Module):
    def __init__(self, use_all=True):
        super().__init__()
        self.meta_adapter_time = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_freq = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_fusion = ResidualMetaAdapter(dim=4, rank=2)
        self.head = nn.Linear(4, 3)
        self.use_all = bool(use_all)
        self.register_buffer("fixed_buffer", torch.tensor([1.0, 2.0]))

    def forward(self, x, y=None, return_aux=False):
        z = self.meta_adapter_time(x)
        if self.use_all:
            z = self.meta_adapter_freq(z)
            z = self.meta_adapter_fusion(z)
        logits = self.head(z)
        return {"logits": logits, "feat_cls": z}


class _BatchNormAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta_adapter_time = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_freq = ResidualMetaAdapter(dim=4, rank=2)
        self.meta_adapter_fusion = ResidualMetaAdapter(dim=4, rank=2)
        self.batch_norm = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 3)
        self.fail = False

    def forward(self, x, y=None, return_aux=False):
        z = self.meta_adapter_time(x)
        z = self.batch_norm(z)
        z = self.meta_adapter_freq(z)
        z = self.meta_adapter_fusion(z)
        if self.fail:
            raise RuntimeError("intentional forward failure")
        logits = self.head(z)
        return {"logits": logits, "feat_cls": z}


def _toy_loss(outputs, labels, _fast):
    return F.cross_entropy(outputs["logits"], labels)


def _toy_inputs(seed=13):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(5, 4, generator=generator)
    y = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    return x, y


def _state_snapshot(model):
    return (
        OrderedDict((name, value.detach().clone()) for name, value in model.named_parameters()),
        OrderedDict((name, value.detach().clone()) for name, value in model.named_buffers()),
        tuple((id(module), module.training) for module in model.modules()),
    )


def _assert_snapshot_equal(model, snapshot):
    parameters, buffers, training = snapshot
    assert tuple((id(module), module.training) for module in model.modules()) == training
    for name, before in parameters.items():
        assert torch.equal(dict(model.named_parameters())[name].detach(), before), name
    for name, before in buffers.items():
        assert torch.equal(dict(model.named_buffers())[name].detach(), before), name


def test_three_steps_use_real_forward_change_fast_state_and_preserve_model():
    torch.manual_seed(3)
    model = _ToyAdapterModel()
    model.eval()
    x, y = _toy_inputs()
    snapshot = _state_snapshot(model)
    seen = []

    def support_loss(outputs, labels, fast):
        assert isinstance(outputs, dict)
        seen.append(tuple(fast.keys()))
        return _toy_loss(outputs, labels, fast)

    fast = first_order_adapt(model, x, y, support_loss, steps=3)
    expected = OrderedDict(iter_inner_adapter_parameters(model))
    assert isinstance(fast, FastAdapterState)
    assert fast.steps == 3
    assert len(fast.support_losses) == 3
    assert tuple(fast.parameters) == tuple(expected)
    assert set(fast.parameters) == set(expected)
    assert seen == [tuple(expected)] * 3
    assert any(
        not torch.equal(fast.parameters[name].detach(), expected[name].detach())
        for name in expected
    )
    _assert_snapshot_equal(model, snapshot)


def test_query_backward_reaches_initial_adapter_and_module_step_size():
    torch.manual_seed(5)
    model = _ToyAdapterModel()
    model.eval()
    support_x, support_y = _toy_inputs(21)
    query_x, _ = _toy_inputs(22)
    fast = first_order_adapt(model, support_x, support_y, _toy_loss, steps=1)

    query = functional_forward(model, fast, query_x)
    outer_loss = query["logits"].square().mean()
    outer_loss.backward()

    adapter_grad = model.meta_adapter_time.up.weight.grad
    step_grad = model.meta_adapter_time.log_step_size.grad
    assert adapter_grad is not None and torch.isfinite(adapter_grad).all()
    assert step_grad is not None and torch.isfinite(step_grad).all()
    assert bool(adapter_grad.abs().sum() > 0)
    assert bool(step_grad.abs().sum() > 0)


def test_inner_state_excludes_head_step_size_and_non_adapter_state():
    model = _ToyAdapterModel()
    names = tuple(name for name, _ in iter_inner_adapter_parameters(model))
    assert names
    assert all(name.startswith("meta_adapter_") for name in names)
    assert all(name.endswith(("down.weight", "down.bias", "up.weight", "up.bias", "gate")) for name in names)
    assert all("log_step_size" not in name for name in names)
    assert all("head" not in name and "fixed_buffer" not in name for name in names)


def test_zero_steps_returns_initial_functional_state_without_calling_support_loss():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()

    def fail_if_called(*_args):
        raise AssertionError("steps=0 must not evaluate support loss")

    fast = first_order_adapt(model, x, y, fail_if_called, steps=0)
    expected = OrderedDict(iter_inner_adapter_parameters(model))
    assert fast.steps == 0
    assert fast.support_losses == ()
    assert tuple(fast.parameters) == tuple(expected)
    for name, parameter in expected.items():
        assert fast.parameters[name] is not parameter
        assert fast.parameters[name].data_ptr() != parameter.data_ptr()
        assert torch.equal(fast.parameters[name], parameter)


def test_zero_step_fast_storage_isolated_and_outer_gradient_reaches_initialization():
    torch.manual_seed(17)
    model = _ToyAdapterModel()
    model.eval()
    x, y = _toy_inputs(18)
    fast = first_order_adapt(model, x, y, _toy_loss, steps=0)
    before = OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in iter_inner_adapter_parameters(model)
    )
    for name, parameter in iter_inner_adapter_parameters(model):
        assert fast.parameters[name].data_ptr() != parameter.data_ptr()
        assert torch.equal(fast.parameters[name], before[name])

    with torch.no_grad():
        fast.parameters["meta_adapter_time.up.weight"].add_(0.25)
    assert torch.equal(model.meta_adapter_time.up.weight.detach(), before["meta_adapter_time.up.weight"])

    output = functional_forward(model, fast, x, y)
    output["logits"].square().mean().backward()
    gradient = model.meta_adapter_time.up.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert bool(gradient.abs().sum() > 0)


def test_fast_state_parameter_mapping_is_read_only_but_tensors_are_independent():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()
    fast = first_order_adapt(model, x, y, _toy_loss, steps=0)
    key = next(iter(fast.parameters))
    with pytest.raises(TypeError, match="read-only"):
        fast.parameters[key] = fast.parameters[key]
    with pytest.raises(TypeError, match="read-only"):
        del fast.parameters[key]
    with pytest.raises(TypeError, match="read-only"):
        fast.parameters.update({key: fast.parameters[key]})


def test_support_loss_receives_read_only_fast_mapping():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()
    observed = []

    def support_loss(outputs, labels, fast):
        key = next(iter(fast))
        with pytest.raises(TypeError, match="read-only"):
            fast[key] = fast[key]
        observed.append(tuple(fast))
        return _toy_loss(outputs, labels, fast)

    first_order_adapt(model, x, y, support_loss, steps=1)
    assert observed


def test_public_functional_forward_accepts_only_fast_adapter_state():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()
    mapping = OrderedDict(iter_inner_adapter_parameters(model))
    with pytest.raises(TypeError, match="FastAdapterState"):
        functional_forward(model, mapping, x, y)


def test_batchnorm_buffers_and_training_state_survive_normal_and_failed_forward():
    model = _BatchNormAdapterModel()
    model.train()
    x, y = _toy_inputs(23)
    fast = first_order_adapt(model, x, y, _toy_loss, steps=0)

    snapshot = _state_snapshot(model)
    output = functional_forward(model, fast, x, y)
    assert output["logits"].shape == (x.size(0), 3)
    _assert_snapshot_equal(model, snapshot)

    model.fail = True
    with pytest.raises(RuntimeError, match="intentional forward failure"):
        functional_forward(model, fast, x, y)
    _assert_snapshot_equal(model, snapshot)


@pytest.mark.parametrize("steps", [-1, 11])
def test_inner_steps_are_bounded(steps):
    model = _ToyAdapterModel()
    x, y = _toy_inputs()
    with pytest.raises(ValueError, match=r"V1 source meta inner steps must be in \[0, 10\]"):
        first_order_adapt(model, x, y, _toy_loss, steps=steps)


def test_nonfinite_support_loss_is_explicit_failure():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()

    def nonfinite_loss(outputs, labels, _fast):
        return _toy_loss(outputs, labels, _fast) * torch.tensor(float("nan"))

    with pytest.raises(MetaInnerLoopError, match="support loss"):
        first_order_adapt(model, x, y, nonfinite_loss, steps=1)


class _NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.input_shape = tuple(value.shape)
        return value.sum() * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(
            ctx.input_shape,
            float("nan"),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )


def test_nonfinite_gradient_is_explicit_failure():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()

    def nonfinite_gradient_loss(outputs, labels, fast):
        finite = _toy_loss(outputs, labels, fast) * 0.0
        first = next(iter(fast.values()))
        return finite + _NaNGradient.apply(first)

    with pytest.raises(MetaInnerLoopError, match="gradient"):
        first_order_adapt(model, x, y, nonfinite_gradient_loss, steps=1)


def test_unused_adapter_gradient_is_explicit_failure():
    model = _ToyAdapterModel(use_all=False)
    x, y = _toy_inputs()
    with pytest.raises(MetaInnerLoopError, match="gradient"):
        first_order_adapt(model, x, y, _toy_loss, steps=1)


def test_functional_forward_rejects_extra_missing_and_forged_keys():
    model = _ToyAdapterModel()
    x, y = _toy_inputs()
    valid = OrderedDict(iter_inner_adapter_parameters(model))

    extra = OrderedDict(valid)
    extra["meta_adapter_fake.up.weight"] = torch.zeros(1)
    with pytest.raises(MetaInnerLoopError, match="key"):
        functional_forward(model, FastAdapterState(extra, 0, ()), x, y)

    missing = OrderedDict(valid)
    missing.popitem()
    with pytest.raises(MetaInnerLoopError, match="key"):
        functional_forward(model, FastAdapterState(missing, 0, ()), x, y)

    forged = OrderedDict(valid)
    forged.pop(next(iter(forged)))
    forged["meta_adapter_time.log_step_size"] = torch.tensor(0.0)
    with pytest.raises(MetaInnerLoopError, match="key"):
        functional_forward(model, FastAdapterState(forged, 0, ()), x, y)


@pytest.mark.parametrize("bad_value", ["shape", "dtype", "device"])
def test_functional_forward_rejects_fast_tensor_contract(bad_value):
    model = _ToyAdapterModel()
    x, y = _toy_inputs()
    valid = OrderedDict(iter_inner_adapter_parameters(model))
    key = next(iter(valid))
    value = valid[key]
    if bad_value == "shape":
        replacement = torch.zeros((value.numel() + 1,), dtype=value.dtype)
    elif bad_value == "dtype":
        replacement = value.detach().double()
    else:
        replacement = value.detach().to(device="meta")
    valid[key] = replacement
    with pytest.raises(MetaInnerLoopError, match="must match"):
        functional_forward(model, FastAdapterState(valid, 0, ()), x, y)


def test_functional_forward_preserves_state_and_training_mode():
    model = _ToyAdapterModel()
    model.train()
    x, y = _toy_inputs()
    snapshot = _state_snapshot(model)
    fast = first_order_adapt(model, x, y, _toy_loss, steps=0)
    output = functional_forward(model, fast, x, y)
    assert set(output) == {"logits", "feat_cls"}
    assert output["logits"].shape == (x.size(0), 3)
    _assert_snapshot_equal(model, snapshot)


def test_single_real_adv3b02_forward_consumes_fast_state():
    torch.manual_seed(31)
    model = _tiny_model()
    model.eval()
    x = torch.randn(2, 2, 64)
    y = torch.tensor([0, 1], dtype=torch.long)
    fast = first_order_adapt(
        model,
        x,
        y,
        lambda outputs, labels, _fast: F.cross_entropy(outputs["logits"], labels),
        steps=1,
    )
    output = functional_forward(model, fast, x, y)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2, 3)
    assert output["feat_cls"].shape[0] == 2


def test_real_dual_forward_uses_explicit_y_tx_signature():
    torch.manual_seed(37)
    model = _tiny_dual_model()
    model.eval()
    x = torch.randn(2, 2, 64)
    y = torch.tensor([0, 1], dtype=torch.long)

    def dual_loss(outputs, labels, _fast):
        return F.cross_entropy(outputs["tx_logits"], labels) + outputs["dom_logits"].square().mean()

    fast = first_order_adapt(model, x, y, dual_loss, steps=1)
    output = functional_forward(model, fast, x, y)
    assert isinstance(output, dict)
    assert output["tx_logits"].shape == (2, 3)
    assert output["dom_logits"].shape == (2, 2)


def test_same_seed_and_input_produce_same_fast_state():
    x, y = _toy_inputs(71)
    torch.manual_seed(73)
    model_a = _ToyAdapterModel()
    model_a.eval()
    torch.manual_seed(73)
    model_b = _ToyAdapterModel()
    model_b.eval()
    fast_a = first_order_adapt(model_a, x, y, _toy_loss, steps=3)
    fast_b = first_order_adapt(model_b, x, y, _toy_loss, steps=3)
    assert fast_a.support_losses == pytest.approx(fast_b.support_losses)
    for name in fast_a.parameters:
        assert torch.equal(fast_a.parameters[name], fast_b.parameters[name]), name
