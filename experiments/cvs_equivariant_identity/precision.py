"""Registered per-process numerical policy; no model/data/optimizer changes."""
from contextlib import contextmanager
import torch

FULL_FP32_POLICY = dict(cudnn_allow_tf32=False, cuda_matmul_allow_tf32=False,
                       cudnn_benchmark=False, cudnn_deterministic=False,
                       matmul_precision='highest')


def actual_flags():
    return dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_precision=torch.get_float32_matmul_precision())


@contextmanager
def numerical_context(policy):
    if policy is None:
        yield
        return
    if policy != FULL_FP32_POLICY:
        raise ValueError('Unregistered equivariant numerical policy')
    original = torch.backends.cudnn.allow_tf32
    try:
        # Only this flag changes; all other preregistered defaults must already match.
        torch.backends.cudnn.allow_tf32 = False
        if actual_flags() != policy:
            raise ValueError('Actual backend flags differ from fixed numerical policy')
        yield
        if actual_flags() != policy:
            raise ValueError('Numerical policy changed during execution')
    finally:
        torch.backends.cudnn.allow_tf32 = original
