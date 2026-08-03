"""Small explicit-copy bridge for D127 under the NumPy2/Torch2.1 ABI split.

The N607 runtime currently combines NumPy 2.x with Torch 2.1.  D127 only
needs finite float32 IQ/decoded-asset arrays and integral class labels at this
boundary, so converting through Python values is intentionally bounded and
does not depend on the incompatible C-ABI hand-off.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


class D127TorchCompatError(ValueError):
    """Raised when a D127 NumPy-to-Torch copy would change its typed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127TorchCompatError(message)


def numpy_to_torch_copy(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
    name: str,
    detach: bool = False,
) -> Tensor:
    """Copy one supported D127 numeric array through Python values.

    ``torch.tensor`` receives a Python scalar/list rather than a NumPy object.
    This is deliberately a copy: callers cannot mutate a NumPy buffer after
    construction, and the returned tensor never acquires a gradient relation
    to that buffer.  The public D127 call sites need only finite ``float32``
    arrays and finite integral labels represented as ``torch.long``.
    """

    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise D127TorchCompatError(f"{name} is not a rectangular numeric array") from exc
    _require(raw.dtype != np.dtype(object), f"{name} may not use object dtype")

    if dtype == torch.float32:
        _require(raw.dtype == np.dtype(np.float32), f"{name} must be float32")
        _require(bool(np.isfinite(raw).all()), f"{name} must be finite")
        copied = torch.tensor(raw.tolist(), dtype=torch.float32, device=device)
    elif dtype == torch.float64:
        _require(
            raw.dtype in {np.dtype(np.float32), np.dtype(np.float64)},
            f"{name} must be float32 or float64",
        )
        _require(bool(np.isfinite(raw).all()), f"{name} must be finite")
        copied = torch.tensor(raw.tolist(), dtype=torch.float64, device=device)
    elif dtype == torch.long:
        _require(raw.dtype.kind in {"i", "u"}, f"{name} must use an integral dtype")
        if raw.dtype.kind == "u":
            _require(
                not bool(np.any(raw > np.iinfo(np.int64).max)),
                f"{name} exceeds int64 range",
            )
        copied = torch.tensor(raw.tolist(), dtype=torch.long, device=device)
    else:
        raise D127TorchCompatError(f"{name} requests unsupported Torch dtype {dtype}")

    # ``torch.tensor([])`` cannot infer trailing empty dimensions from a
    # Python list.  The NumPy source shape is part of D127's typed contract,
    # including scalar and empty batch forms, so restore it explicitly after
    # the ABI-independent value copy.
    copied = copied.reshape(tuple(int(dimension) for dimension in raw.shape))
    return copied.detach() if detach else copied


def torch_to_numpy_copy(value: Tensor, *, dtype: Any, name: str) -> np.ndarray:
    """Copy a D127 floating tensor to an explicitly typed NumPy array.

    The conversion first detaches and materializes an ordinary contiguous CPU
    tensor, then transfers only Python values.  This preserves the source
    scalar/empty dimensions after ``np.asarray`` has reconstructed its dtype,
    without using the incompatible Torch-to-NumPy C-ABI path.
    """

    _require(torch.is_tensor(value), f"{name} must be a Torch tensor")
    _require(value.dtype.is_floating_point, f"{name} must be a floating Torch tensor")
    try:
        target_dtype = np.dtype(dtype)
    except Exception as exc:
        raise D127TorchCompatError(f"{name} has an invalid NumPy dtype") from exc
    _require(
        target_dtype in {np.dtype(np.float32), np.dtype(np.float64)},
        f"{name} requests unsupported NumPy dtype {target_dtype}",
    )
    source = value.detach().cpu().contiguous()
    shape = tuple(int(dimension) for dimension in source.shape)
    try:
        return np.asarray(source.tolist(), dtype=target_dtype).reshape(shape)
    except Exception as exc:
        raise D127TorchCompatError(f"{name} could not be copied to NumPy") from exc


__all__ = ["D127TorchCompatError", "numpy_to_torch_copy", "torch_to_numpy_copy"]
