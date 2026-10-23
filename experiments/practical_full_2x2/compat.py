"""Equivalent FP32/TF32-off execution on the two existing Torch environments."""
def numerical_policy():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    if hasattr(torch,'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')

def tensor(array,device='cpu'):
    import numpy as np
    import torch
    value=np.array(array,copy=True,order='C')
    dtype={np.dtype('float32'):torch.float32,np.dtype('int64'):torch.int64,np.dtype('bool'):torch.bool}.get(value.dtype)
    if dtype is None:raise ValueError('Explicit FP32/int64/bool input required')
    if hasattr(torch,'frombuffer'):return torch.frombuffer(value,dtype=dtype).reshape(value.shape).to(device)
    return torch.tensor(value,dtype=dtype,device=device)
