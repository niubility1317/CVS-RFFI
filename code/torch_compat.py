"""Narrow compatibility helpers that retain deterministic execution."""
import torch

def deterministic_cumsum(value,dim):
    try:
        return torch.cumsum(value,dim=dim)
    except RuntimeError as exc:
        if not (value.is_cuda and torch.are_deterministic_algorithms_enabled()
                and 'cumsum_cuda_kernel does not have a deterministic implementation' in str(exc)):
            raise
        return torch.cumsum(value.cpu(),dim=dim).to(value.device)
