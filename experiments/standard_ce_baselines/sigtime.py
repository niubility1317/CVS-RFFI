"""Frozen SIG_TIME architecture port; pure CE uses the shared E200 trainer.

Feature definition follows committed SIG_TIME (eda57144d), not its oldgate/EMA
training recipe. No extra loss, fitted head, feature mask, or domain network.
"""
import math
import torch
from torch import nn
from experiments.cvs_reference_identity.model import ReferenceCVS, ExcitationResponse


class NoExplicitCFO(ExcitationResponse):
    def forward(self, x):
        values = super().forward(x)
        return torch.cat((values[:, :27], torch.zeros_like(values[:, 27:28]), values[:, 28:]), 1)

    def contract(self):
        return dict(super().contract(), explicit_CFO_identity_input=False,
                    constant_zero_coordinate=27, internal_CFO_compensation=True)


def path_signature(u):
    if u.ndim != 3 or u.shape[2] != 6 or u.shape[1] < 2:
        raise ValueError('Six-dimensional multi-position path required')
    t = torch.linspace(0, 1, u.shape[1], device=u.device, dtype=u.dtype)
    u = torch.cat((u, t[None, :, None].expand(len(u), -1, -1)), -1)
    delta = u[:, 1:] - u[:, :-1]
    before = delta.cumsum(1) - delta
    area = .5 * (before[:, :, :, None]*delta[:, :, None, :] -
                 delta[:, :, :, None]*before[:, :, None, :]).sum(1)
    ij = torch.triu_indices(7, 7, offset=1, device=u.device)
    return torch.cat((delta.sum(1), area[:, ij[0], ij[1]]), -1)


class SignaturePool(nn.Module):
    def __init__(self, channels, seed):
        super().__init__()
        self.channels = channels
        # CPU-only generator seeding does not reset CUDA/global dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed)+80411)
            self.local_projection = nn.Conv1d(channels, 6, 1, bias=False)
            self.output_projection = nn.Linear(112, 160, bias=False)
        self.output_norm = nn.LayerNorm(160)
        self.gain_raw = nn.Parameter(torch.full((160,), math.atanh(.05/.2)))

    def forward(self, h):
        if h.ndim != 3 or h.shape[1] != self.channels or h.shape[2] < 8 or h.shape[2] % 4:
            raise ValueError('Four equal native t2 windows required')
        u = torch.tanh(self.local_projection(h)).transpose(1, 2)
        b, t, _ = u.shape
        descriptor = path_signature(u.reshape(b*4, t//4, 6)).reshape(b, 112)
        return .2*self.gain_raw.tanh()*self.output_norm(self.output_projection(descriptor))


class SIGTimeCVS(ReferenceCVS):
    def __init__(self, seed):
        super().__init__('reference_response')
        self.response.__class__ = NoExplicitCFO
        self.sequence_pool = SignaturePool(self.core.id_backbone.t2.pw.out_channels, seed)
        self._architecture = dict(name='SIG_TIME', layer='t2 after native dropout',
            windows=4, local_dimension=6, descriptor_width=28, output_embedding=160,
            gain_initial=.05, gain_bound=.2, internal_reference_response=True,
            explicit_CFO_coordinate=False, default_classifier_jointly_trained=True,
            extra_feature_parameters=sum(p.numel() for p in self.sequence_pool.parameters()),
            architectural_source_commit='eda57144de34d479288f42d14c037eb7802c1ecd',
            extra_losses=[], teacher=False, domain_backbone=False)

    def features(self, x):
        local = {}
        handle = self.core.id_backbone.t2.register_forward_hook(
            lambda module, args, output: local.update(h=output))
        try:
            base = super().features(x)
        finally:
            handle.remove()
        if set(local) != {'h'}:
            raise ValueError('Actual t2 feature branch bypassed')
        return base+self.sequence_pool(local['h'])
