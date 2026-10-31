from types import SimpleNamespace
import numpy as np
import torch
from experiments.pamf_phase2_practical.milkr_nested_source import source_iq_rows


def test_source_iq_boundary_does_not_require_torch_numpy_index_interop(monkeypatch):
    native=torch.Tensor.__getitem__
    def old_torch_index(tensor,key):
        if isinstance(key,np.ndarray):raise RuntimeError('Could not infer dtype of numpy.int64')
        return native(tensor,key)
    monkeypatch.setattr(torch.Tensor,'__getitem__',old_torch_index)
    x=torch.arange(4*2*256,dtype=torch.float32).reshape(4,2,256)
    rows=source_iq_rows(SimpleNamespace(x=x),np.asarray([3,1],dtype=np.int64))
    assert rows.dtype==np.float32 and rows.shape==(2,2,256)
    assert rows[0,0,0]==3*2*256 and rows[1,0,0]==2*256
