"""Capture one real frozen output_projection input in the normal F forward."""
from .features import branches
from .io import require


def capture(reader,x):
    import torch
    values=[]
    handle=reader.sequence_pool.output_projection.register_forward_pre_hook(lambda module,args:values.append(args[0]))
    try:features,logits=branches(reader,x)
    finally:handle.remove()
    require(len(values)==1 and values[0].shape==(len(x),336) and bool(torch.isfinite(values[0]).all()),
            'Expected one actual three-view signature descriptor')
    return dict(features,sig_views=values[0]),logits
