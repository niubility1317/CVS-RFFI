"""Call-local capture of the existing deployed SIG_TIME branch before gain."""
from contextlib import contextmanager
import torch

PATH = 'id_backbone.encoder.sequence_pool'


@contextmanager
def capture(model):
    pool = model.get_submodule(PATH)
    if getattr(pool, 'arm', None) != 'SIG_TIME':
        raise ValueError('Only actual SIG_TIME sequence branch')
    observed = {}
    def hook(module, args, output):
        if observed:
            raise ValueError('Expected exactly one existing SIG_TIME forward')
        if (not isinstance(output, torch.Tensor) or output.ndim != 2
                or output.shape[1] != 160 or not bool(torch.isfinite(output).all())):
            raise ValueError('Actual pre-gain160 shape/finite boundary differs')
        observed['z'] = output
    handle = pool.output_norm.register_forward_hook(hook)
    try:
        yield observed
        if set(observed) != {'z'}:
            raise ValueError('Existing SIG_TIME branch was not called')
    finally:
        handle.remove()
