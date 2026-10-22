import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def tensor(array, device='cpu'):
    import numpy as np
    import torch
    # Torch2.1 may coexist with NumPy2: do not require its disabled NumPy bridge.
    value = np.array(array, copy=True, order='C')
    dtype = {np.dtype('float32'): torch.float32, np.dtype('int64'): torch.int64,
             np.dtype('bool'): torch.bool}.get(value.dtype)
    if dtype is None:
        raise ValueError('Explicit float32/int64/bool array dtype required')
    return torch.frombuffer(value, dtype=dtype).reshape(value.shape).to(device)
