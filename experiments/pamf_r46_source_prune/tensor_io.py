"""Exact float32 buffer transfer, compatible with Torch2.1/NumPy2 without its C bridge."""
import numpy as np
import torch

def array_tensor(value,device):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype('float32') or value.ndim!=3 or value.shape[1:]!=(2,256):
        raise ValueError('Native float32 paired IQ[B,2,256] required')
    contiguous=np.ascontiguousarray(value)
    if not contiguous.flags.writeable:contiguous=contiguous.copy()
    return torch.frombuffer(memoryview(contiguous),dtype=torch.float32).reshape(contiguous.shape).to(device)
