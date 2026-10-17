"""Exact existing array-to-IQ bridge;no U channel generation in this experiment."""
import numpy as np
import torch
def tensor_iq(a,device):
 a=np.ascontiguousarray(a,dtype=np.float32);return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
