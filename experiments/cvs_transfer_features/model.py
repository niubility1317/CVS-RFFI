"""Class-independent joint160 export for the reference / no-explicit-CFO CVS path.

Only the ReferenceCVS encoder is copied. The six Phase1 class rows, nuisance
network, teacher, runtime metadata and diagnostic h349 are not exported.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_reference_identity.model import ReferenceCVS
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
from experiments.cvs_source_identity_readout.input_policy import NoExplicitCFO


class TransferFeatures(nn.Module):
    dimension = 160
    normalization_epsilon = 1e-4

    def __init__(self, reference):
        super().__init__()
        if type(reference) is not ReferenceCVS:
            raise ValueError('Transfer export requires the ReferenceCVS encoder')
        if type(reference.response) is not NoExplicitCFO:
            raise ValueError('Transfer export requires the existing NoExplicitCFO input policy')
        backbone = reference.core.id_backbone
        head = backbone.cls_head
        if type(head) is not ResidualPhysicalHead or backbone.nmfdu_gate is not None:
            raise ValueError('Transfer export requires residual fusion without NMFDu')
        if head.weight.shape != (6, self.dimension) or head.scale != 30.:
            raise ValueError('Phase1 classifier interface differs')
        self.encoder = deepcopy(reference)
        # Keep head.components, both LayerNorms and channel gain: they create z.
        # Native feature_only returns before classify, so no class rows are needed.
        del self.encoder.core.id_backbone.cls_head.weight
        self.train(reference.training)

    def forward_features(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Transfer features require IQ[B,2,256]')
        with torch.autocast(device_type=x.device.type, enabled=False):
            joint = self.encoder.features(x.float())
        if joint.shape != (len(x), self.dimension):
            raise ValueError('Transfer feature boundary is joint160')
        return joint

    def forward(self, x):
        return self.forward_features(x)

    def normalized_features(self, x):
        return F.normalize(self.forward_features(x), dim=1,
                           eps=self.normalization_epsilon)


def from_phase1(model):
    """Extract only a recognized ReferenceCVS identity path from a Phase1 model.

    Frequency-context blocks live inside this encoder and are copied unchanged.
    Wrappers with an additional representation branch must use their own export;
    silently dropping such a branch would change the feature definition.
    """
    if type(model) is ReferenceCVS:
        return TransferFeatures(model)
    if hasattr(model, 'nonlinear') or hasattr(model, 'nonlinear_gain'):
        raise ValueError('Additional ReportIdentity representation branches are unsupported')
    from experiments.cvs_phase1_overlay.model import Phase1IdentityAdapter
    identity = getattr(model, 'id_backbone', None)
    if type(identity) is not Phase1IdentityAdapter:
        raise ValueError('Unsupported Phase1 identity architecture for transfer export')
    return TransferFeatures(identity.encoder)
