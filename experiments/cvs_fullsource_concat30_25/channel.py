"""Copy fixed observed IQ into Torch without the ndarray ABI bridge."""
def tensor_iq(a,device):
 import numpy as np,torch
 a=np.ascontiguousarray(a,dtype=np.float32)
 return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
