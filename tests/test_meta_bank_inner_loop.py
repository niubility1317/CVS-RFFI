from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _initial_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("id_backbone.t3.weight", torch.tensor(2.0, requires_grad=True)),
            ("id_backbone.t3.bias", torch.tensor(1.0, requires_grad=True)),
        )
    )


def test_inner_loop_uses_explicit_bank_initial_state_and_first_order_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading an unrelated module state or enabling second-order grads breaks this test."""
    from cvsrffi.meta_bank_inner_loop import first_order_bank_adapt

    unrelated_module = torch.nn.Linear(1, 1)
    with torch.no_grad():
        unrelated_module.weight.fill_(100.0)
        unrelated_module.bias.fill_(100.0)
    del unrelated_module

    calls: list[bool] = []
    real_grad = torch.autograd.grad

    def recording_grad(*args, **kwargs):
        calls.append(bool(kwargs.get("create_graph")))
        return real_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", recording_grad)
    initial = _initial_state()
    block_lr = torch.tensor(0.1, requires_grad=True)

    def support_loss(fast, support):
        x, target = support
        prediction = fast["id_backbone.t3.weight"] * x + fast["id_backbone.t3.bias"]
        return (prediction - target).square()

    state = first_order_bank_adapt(
        support_loss,
        initial,
        {"t3": block_lr},
        (torch.tensor(1.0), torch.tensor(0.0)),
        steps=1,
    )

    assert calls == [False]
    assert state.parameters["id_backbone.t3.weight"].detach().item() == pytest.approx(1.4)
    assert state.parameters["id_backbone.t3.bias"].detach().item() == pytest.approx(0.4)
    assert state.block_lrs["t3"] is block_lr


def test_inner_loop_keeps_first_order_outer_paths_to_initial_state_and_learned_lr() -> None:
    """Detaching the initial state or the learned LR would remove these outer gradients."""
    from cvsrffi.meta_bank_inner_loop import first_order_bank_adapt

    initial = _initial_state()
    block_lr = torch.tensor(0.1, requires_grad=True)

    def support_loss(fast, support):
        x, target = support
        prediction = fast["id_backbone.t3.weight"] * x + fast["id_backbone.t3.bias"]
        return (prediction - target).square()

    state = first_order_bank_adapt(
        support_loss,
        initial,
        {"t3": block_lr},
        (torch.tensor(1.0), torch.tensor(0.0)),
        steps=1,
    )
    query_prediction = (
        state.parameters["id_backbone.t3.weight"] * 2.0
        + state.parameters["id_backbone.t3.bias"]
    )
    query_prediction.square().backward()

    for value in (*initial.values(), block_lr):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())
        assert bool(torch.count_nonzero(value.grad))


def test_inner_loop_rejects_unused_or_nonfinite_gradients_instead_of_skipping() -> None:
    """Silent unused-gradient or nonfinite-gradient repair is forbidden."""
    from cvsrffi.meta_bank_inner_loop import MetaBankInnerLoopError, first_order_bank_adapt

    initial = _initial_state()

    with pytest.raises(MetaBankInnerLoopError, match="gradient"):
        first_order_bank_adapt(
            lambda fast, support: fast["id_backbone.t3.weight"].square(),
            initial,
            {"t3": torch.tensor(0.1)},
            object(),
            steps=1,
        )

    with pytest.raises(MetaBankInnerLoopError, match="non-finite"):
        first_order_bank_adapt(
            lambda fast, support: fast["id_backbone.t3.weight"] * torch.tensor(float("nan")),
            initial,
            {"t3": torch.tensor(0.1)},
            object(),
            steps=1,
        )
