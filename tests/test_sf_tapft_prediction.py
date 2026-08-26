from __future__ import annotations

import inspect

import torch
from torch import nn

from cvsrffi.sf_tapft_prediction import predict_sf_tapft_rows


class _SyntheticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.normalization = nn.BatchNorm1d(2)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))

    def forward(
        self,
        received_iq: torch.Tensor,
        *,
        return_aux: bool = False,
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert return_aux is True
        assert y is None
        assert self.training is False
        assert torch.is_grad_enabled() is False
        return {"embedding": self.normalization(self.projection(received_iq))}


class _SyntheticHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        )
        self.class_ids = (11, 23, 47)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert self.training is False
        assert torch.is_grad_enabled() is False
        return embeddings @ self.weight.transpose(0, 1)


def _rows() -> torch.Tensor:
    return torch.tensor([[3.0, 1.0], [1.0, 4.0], [-2.0, -1.0]])


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def test_clean_single_prediction_is_independent_per_row_and_accepts_no_truth_role() -> None:
    model = _SyntheticModel()
    head = _SyntheticHead()

    assert set(inspect.signature(predict_sf_tapft_rows).parameters) == {
        "model",
        "head",
        "received_iq",
    }
    first = predict_sf_tapft_rows(model, head, _rows()[:1])
    batched = predict_sf_tapft_rows(model, head, _rows())

    assert torch.equal(first.logits, batched.logits[:1])
    assert torch.equal(first.predictions, batched.predictions[:1])
    assert batched.logits.shape == (3, 3)
    assert torch.equal(batched.predictions, torch.tensor([11, 23, 47]))
    assert batched.query_truth_opened is False
    assert batched.query_role_opened is False


def test_clean_single_prediction_reordering_only_reorders_outputs() -> None:
    model = _SyntheticModel()
    head = _SyntheticHead()
    order = torch.tensor([2, 0, 1])

    original = predict_sf_tapft_rows(model, head, _rows())
    reordered = predict_sf_tapft_rows(model, head, _rows()[order])

    assert torch.equal(reordered.logits, original.logits[order])
    assert torch.equal(reordered.predictions, original.predictions[order])


def test_clean_single_prediction_is_eval_no_grad_and_preserves_tensor_state() -> None:
    model = _SyntheticModel()
    head = _SyntheticHead()
    model_state = _state(model)
    head_state = _state(head)

    result = predict_sf_tapft_rows(model, head, _rows())

    assert model.training is False
    assert head.training is False
    assert all(torch.equal(value, model_state[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value, head_state[name]) for name, value in head.state_dict().items())
    assert result.logits.device.type == "cpu"
    assert result.predictions.device.type == "cpu"
    assert result.logits.grad_fn is None
    assert result.predictions.grad_fn is None
    assert result.logits.requires_grad is False
    assert result.predictions.requires_grad is False
