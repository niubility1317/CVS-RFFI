"""Small per-packet residual heads around the unchanged reference_response."""
import torch
from torch import nn
from experiments.cvs_reference_identity.model import build
from experiments.cvs_receiver_residual_v2.design import ARMS, RECIPE


class ReceiverResidualCVS(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown residual arm')
        self.arm = arm
        self.encoder = build('reference_response')
        # Additional initialization cannot change the common backbone/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026100801)
            if arm in ('displacement', 'combined'):
                self.displacement = nn.Sequential(nn.LayerNorm(29), nn.Linear(29, 64),
                    nn.SiLU(), nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 160, bias=False))
                nn.init.zeros_(self.displacement[-1].weight)
            if arm in ('contribution', 'combined'):
                torch.random.default_generator.manual_seed(2026100802)
                self.contribution = nn.Sequential(nn.LayerNorm(29), nn.Linear(29, 64),
                    nn.SiLU(), nn.Linear(64, 3))
                nn.init.zeros_(self.contribution[-1].weight)
                nn.init.zeros_(self.contribution[-1].bias)
        self.register_buffer('strength', torch.tensor(0. if arm != 'baseline' else 1.))

    def contract(self):
        return dict(variant='reference_response', arm=self.arm,
            architecture=self.encoder.contract(), mechanism=RECIPE,
            identity_parameters=sum(p.numel() for p in self.encoder.parameters()),
            added_parameters=sum(p.numel() for n, p in self.named_parameters() if not n.startswith('encoder.')),
            inference_inputs='one IQ packet only; no RX/TX labels, source centroids, query fitting or batch summaries')

    def base_details(self, x):
        values = {}
        head = self.encoder.core.id_backbone.cls_head
        modules = {'tf': head.base_norm, 'pa': head.pa_norm,
                   'reference': self.encoder.response_projection, 'q': self.encoder.response}
        hooks = [m.register_forward_hook(
            lambda module, inputs, output, name=n: values.__setitem__(name, output))
            for n, m in modules.items()]
        try:
            z = self.encoder.features(x)
        finally:
            for h in hooks:
                h.remove()
        branches = torch.stack((values['tf'], head.gain.tanh()*values['pa'],
            self.encoder.response_gain.tanh()*values['reference']), dim=1)
        return dict(base=z, branches=branches, q=values['q'])

    def details(self, x):
        d = self.base_details(x)
        q = d['q'].detach()  # Fixed received observables, no auxiliary gradient into IQ.
        z = d['base']
        zero = z.new_zeros(())
        d.update(correction=None, contribution_prediction=None, gate=None,
                 correction_norm_ratio=zero, gate_abs_delta=zero)
        if hasattr(self, 'contribution'):
            p = self.contribution(q)
            gate = 1. + self.strength*RECIPE['gate_relative_cap']*torch.tanh(
                p/RECIPE['contribution_temperature'])
            # Residual arithmetic gives bitwise base identity when p == 0.
            z = z + ((gate-1.)[:, :, None]*d['branches']).sum(1)
            d.update(contribution_prediction=p, gate=gate, gate_abs_delta=(gate-1.).abs().mean())
        if hasattr(self, 'displacement'):
            n = self.displacement(q)
            # Bound the inference correction while retaining the full auxiliary prediction.
            norm = torch.sqrt(n.square().sum(1, keepdim=True)+1e-8)
            base_norm = torch.sqrt(d['base'].detach().square().sum(1, keepdim=True)+1e-8)
            applied = self.strength*n*torch.clamp(
                RECIPE['correction_relative_norm_cap']*base_norm/norm, max=1.)
            z = z - applied
            d.update(correction=n, correction_norm_ratio=(applied.norm(dim=1)/base_norm[:, 0]).mean())
        d.update(features=z, logits=self.encoder.classify_features(z))
        return d

    def forward(self, x):
        if self.arm == 'baseline':
            return self.encoder(x)
        return self.details(x)['logits']
