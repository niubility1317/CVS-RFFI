"""User-requested best tested fixed CVS structure; scratch construction only."""
from torch import nn
from experiments.cvs_residual_identity.model import build as residual_build,ResidualPhysicalHead
VARIANT='residual_fusion'

class FixedResidualCVS(nn.Module):
    def __init__(self):
        super().__init__();self.identity=residual_build(VARIANT)
    def forward(self,x):return self.identity(x)
    def contract(self):
        b=self.identity.id_backbone
        return dict(variant=VARIANT,identity_only=True,domain_backbone=False,
            native_time_frequency_pa=True,no_dac=not b.use_dac_path,
            head_type=type(b.cls_head).__name__,residual_head=isinstance(b.cls_head,ResidualPhysicalHead),
            classifier_scale=30.,ordinary_ce=True,label_margin=False,
            total_parameters=sum(p.numel() for p in self.parameters()))

def build():return FixedResidualCVS()
