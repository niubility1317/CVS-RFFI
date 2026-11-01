"""Exact-valued DLPack transfer compatible with NumPy 2 and PyTorch 2.1."""
import numpy as np


def to_tensor(value, device, dtype):
    import torch
    array = np.asarray(value, dtype=dtype)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=dtype, order='C', copy=True)
    return torch.utils.dlpack.from_dlpack(array).to(device)


def to_numpy(value, dtype):
    return np.array(np.from_dlpack(value.detach().cpu()), dtype=dtype, copy=True)
