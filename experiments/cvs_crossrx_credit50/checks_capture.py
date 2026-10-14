"""Hook lifetime/deepcopy/actual attached feature contract without data."""
import copy
import json
import torch
from .capture import capture


class Pool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arm = 'SIG_TIME'
        self.output_norm = torch.nn.LayerNorm(160)
    def forward(self, x):
        return self.output_norm(x)


def check():
    m = torch.nn.Module()
    m.id_backbone = torch.nn.Module()
    m.id_backbone.encoder = torch.nn.Module()
    m.id_backbone.encoder.sequence_pool = Pool()
    pool = m.id_backbone.encoder.sequence_pool
    x = torch.arange(320, dtype=torch.float32).reshape(2, 160).requires_grad_()
    with capture(m) as observed:
        z = pool(x)
    assert observed['z'] is z and observed['z'].requires_grad
    assert not pool.output_norm._forward_hooks
    clone = copy.deepcopy(m)
    assert not clone.id_backbone.encoder.sequence_pool.output_norm._forward_hooks
    try:
        with capture(m):
            raise RuntimeError('caller failure')
    except RuntimeError:
        pass
    assert not pool.output_norm._forward_hooks
    try:
        with capture(m):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError('Missing actual forward accepted')
    assert not pool.output_norm._forward_hooks
    return dict(status='VERIFIED', graph_attached=True, hook_removed_on_success_and_exception=True,
                deepcopy_has_no_closure=True, missing_call_rejected=True, data_reads=0)


if __name__ == '__main__':
    print(json.dumps(check()))
