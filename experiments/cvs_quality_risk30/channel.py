"""Existing byte-exact IQ bridge;never use NumPy/Torch ndarray-identity conversion."""
import numpy as np
import torch
def tensor_iq(a,device):
 a=np.ascontiguousarray(a,dtype=np.float32);return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
