"""Deterministic per-record standard normal draws for source sampling."""
import numpy as np
import torch

def source_noise(ids, components, dim):
 z=[]; e=[]
 for i in ids:
  rng=np.random.default_rng(2026165054+int(i))
  z.append(rng.standard_normal((components,dim)).astype(np.float32))
  e.append(rng.standard_normal((components,dim)).astype(np.float32))
 z=np.stack(z); e=np.stack(e)
 return (torch.frombuffer(memoryview(z),dtype=torch.float32).reshape(z.shape),
         torch.frombuffer(memoryview(e),dtype=torch.float32).reshape(e.shape))


def tensor_array(value):
 value=value.detach().cpu().contiguous()
 if value.dtype==torch.bool:return np.from_dlpack(value.to(torch.uint8)).astype(np.bool_)
 return np.from_dlpack(value)
