from __future__ import annotations

import numpy as np
import pytest
import torch

from cvsrffi import stage2_d127_torch_compat as compat


def _disable_numpy_abi_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise TypeError("simulated NumPy2/Torch2.1 ABI mismatch")

    monkeypatch.setattr(torch, "from_numpy", _blocked)
    monkeypatch.setattr(torch, "as_tensor", _blocked)


def test_float32_copy_uses_python_values_and_is_finite_detached(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_numpy_abi_bridge(monkeypatch)
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7.0
    copied = compat.numpy_to_torch_copy(
        values,
        dtype=torch.float32,
        device="cpu",
        name="IQ",
        detach=True,
    )
    assert copied.shape == values.shape
    assert copied.dtype == torch.float32
    assert copied.requires_grad is False
    assert torch.equal(copied, torch.tensor(values.tolist(), dtype=torch.float32))
    values[0, 0, 0] = -999.0
    assert copied[0, 0, 0].item() != -999.0


def test_integral_label_copy_uses_requested_device_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_numpy_abi_bridge(monkeypatch)
    copied = compat.numpy_to_torch_copy(
        (0, 5, 3, 1),
        dtype=torch.long,
        device=torch.device("cpu"),
        name="labels",
    )
    assert copied.dtype == torch.long
    assert copied.device.type == "cpu"
    assert copied.tolist() == [0, 5, 3, 1]
    assert copied.requires_grad is False


def test_float64_target_copy_preserves_finite_float32_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_numpy_abi_bridge(monkeypatch)
    values = np.asarray([[0.125, -0.25], [0.5, 1.0]], dtype=np.float32)
    copied = compat.numpy_to_torch_copy(
        values,
        dtype=torch.float64,
        device="cpu",
        name="qKNN decoded support",
    )
    assert copied.shape == values.shape
    assert copied.dtype == torch.float64
    assert torch.equal(copied, torch.tensor(values.tolist(), dtype=torch.float64))


def test_float32_copy_preserves_scalar_empty_and_noncontiguous_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_numpy_abi_bridge(monkeypatch)
    scalar = np.asarray(1.25, dtype=np.float32)
    empty_rank2 = np.empty((0, 3), dtype=np.float32)
    empty_rank3 = np.empty((2, 0, 3), dtype=np.float32)
    noncontiguous = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 0, 1)
    assert not noncontiguous.flags.c_contiguous

    for value in (scalar, empty_rank2, empty_rank3, noncontiguous):
        copied = compat.numpy_to_torch_copy(
            value,
            dtype=torch.float32,
            device="cpu",
            name="shape-preservation",
        )
        assert tuple(copied.shape) == value.shape
        assert copied.dtype == torch.float32
        assert copied.device.type == "cpu"
        assert torch.equal(copied, torch.tensor(value.tolist(), dtype=torch.float32).reshape(value.shape))


def test_torch_copy_preserves_scalar_empty_and_noncontiguous_shapes_without_tensor_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise TypeError("simulated NumPy2/Torch2.1 ABI mismatch")

    monkeypatch.setattr(torch.Tensor, "numpy", _blocked)
    scalar = torch.tensor(1.25, dtype=torch.float32, requires_grad=True)
    empty_rank2 = torch.empty((0, 3), dtype=torch.float32)
    empty_rank3 = torch.empty((2, 0, 3), dtype=torch.float32)
    noncontiguous = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 2)
    assert not noncontiguous.is_contiguous()

    for value in (scalar, empty_rank2, empty_rank3, noncontiguous):
        copied = compat.torch_to_numpy_copy(
            value,
            dtype=np.float32,
            name="shape-preservation",
        )
        expected = np.asarray(value.detach().cpu().contiguous().tolist(), dtype=np.float32).reshape(value.shape)
        assert copied.shape == tuple(value.shape)
        assert copied.dtype == np.float32
        np.testing.assert_array_equal(copied, expected)


@pytest.mark.parametrize(
    ("value", "dtype", "message"),
    [
        (np.asarray(["x"], dtype=object), torch.float32, "object"),
        (np.asarray([1.0], dtype=np.float64), torch.float32, "float32"),
        (np.asarray([np.nan], dtype=np.float32), torch.float32, "finite"),
        (np.asarray([np.nan], dtype=np.float64), torch.float64, "finite"),
        (np.asarray([1], dtype=np.int64), torch.float64, "float32"),
        (np.asarray([1.5], dtype=np.float32), torch.long, "integral"),
        (np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64), torch.long, "int64"),
    ],
)
def test_copy_rejects_unsupported_or_unsafe_inputs(
    value: np.ndarray,
    dtype: torch.dtype,
    message: str,
) -> None:
    with pytest.raises(compat.D127TorchCompatError, match=message):
        compat.numpy_to_torch_copy(value, dtype=dtype, name="bad")
