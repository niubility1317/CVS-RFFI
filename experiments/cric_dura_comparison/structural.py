import torch
from torch import nn
from torch.nn import functional as F


class NoSequence(nn.Module):
    def forward(self, h):
        return h.new_zeros(len(h), 160)


class ZeroProjection(nn.Module):
    def forward(self, value):
        return value.new_zeros(*value.shape[:-1], 160)


def no_time_descriptor(self, h):
    from experiments.cvs_feature_sequence50.features import path_signature
    u = torch.tanh(self.local_projection(h)).transpose(1, 2)
    b, t, _ = u.shape
    result = path_signature(u.reshape(b * 4, t // 4, 6), False)
    # Fixed zero padding keeps the original fusion width and classifier.
    return F.pad(result, (0, 7)).reshape(b, 112)


def remove_branch(ref, ablation):
    if ablation == 'F09':
        ref.core.id_backbone.use_pa_path = False
        ref.core.id_backbone.pa_stats_proj = ZeroProjection()
        # Remove the now-disabled original trainable PA branch itself.
        for name in ('pa1', 'pa2', 'pa_proj'):
            if hasattr(ref.core.id_backbone, name):
                delattr(ref.core.id_backbone, name)
    elif ablation == 'F10':
        from experiments.cvs_feature_sequence50.features import SequenceReference
        class WithoutReference(SequenceReference):
            def features(self, x):
                found = {}
                handle = self.core.id_backbone.t2.register_forward_hook(lambda m, a, v: found.update(h=v))
                try:
                    value = self.core.features(x)
                finally:
                    handle.remove()
                return value + self.sequence_pool(found['h'])
            def contract(self):
                return dict(reference_response=False, output_embedding=160)
        ref.__class__ = WithoutReference
        del ref.response
        del ref.response_projection
        del ref.response_gain
    else:
        raise ValueError('Registered structural removal required')
