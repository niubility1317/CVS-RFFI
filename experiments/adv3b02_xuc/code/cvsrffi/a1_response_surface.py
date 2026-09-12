"""Source-trained ECRS response encoder with a fixed physical estimator.

The estimator is the historical compact Schur-ridge/complex-anchor route.
Its cross-fit remains diagnostic; no reconstruction-learning benefit is claimed.
Unlike the historical frozen-backbone adaptation, fused CE retains the raw
identity gradient in this scratch-training experiment.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .a1_response_ridge import ECRSV2PhysicalEstimator


class A1ResponseSurface(nn.Module):
    def __init__(self, num_classes, identity_dim=160, rho=.05):
        super().__init__()
        if not 0 <= rho <= .25:
            raise ValueError('Response fusion rho must be in [0,.25]')
        self.physical = ECRSV2PhysicalEstimator(cache_fixed_operator=False)
        self.encoder = nn.Sequential(nn.Linear(48, 96), nn.SiLU(), nn.Linear(96, 64))
        self.classifier = nn.Linear(64, num_classes)
        self.projection = nn.Linear(64, identity_dim, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.register_buffer('rho', torch.tensor(float(rho)))

    def forward(self, x, raw_identity=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            physical = self.physical(x.float())
            anchor = physical['resp_anchor'].flatten(1).detach()
            encoded = F.normalize(self.encoder(anchor.float()), dim=1)
            response = torch.where(physical['quality_valid'][:, None], encoded, torch.zeros_like(encoded))
            out = {'z_response': response, 'response_logits': self.classifier(response),
                   'response_valid': physical['quality_valid'], 'response_anchor': anchor}
            if raw_identity is not None:
                raw = F.normalize(raw_identity.float(), dim=1)
                projected = self.projection(response.detach())
                tangent = projected - (projected * raw).sum(1, keepdim=True) * raw
                # Smooth norm cap, with nonzero derivative at zero initialization.
                tangent = tangent / (1. + tangent.norm(dim=1, keepdim=True))
                out['z_fused'] = F.normalize(raw + self.rho * tangent, dim=1) * raw_identity.float().norm(dim=1, keepdim=True)
            return out


def response_pair_loss(branch, clean, leo):
    # The clean target does not update a teacher, BN, or model state.
    with torch.no_grad():
        target = branch(clean)['z_response']
    student = branch(leo)
    valid = student['response_valid'] & (target.norm(dim=1) > 0)
    per_row = 1. - (student['z_response'] * target).sum(1)
    loss = (per_row * valid).sum() / valid.sum().clamp_min(1)
    return loss, int(valid.sum().detach())
