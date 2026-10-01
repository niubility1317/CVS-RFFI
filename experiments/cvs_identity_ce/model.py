"""The native CVS lite_d identity backbone, with no domain network."""
from pathlib import Path
import sys
import torch.nn as nn

NATIVE = Path(__file__).resolve().parents[1] / 'adv3b02_xuc' / 'code'
sys.path.insert(0, str(NATIVE))
from model import build_model


class IdentityOnlyCVS(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.id_backbone = build_model(num_classes=num_classes, model_size='M',
            dataset='wisig', input_len=256, sample_rate_hz=25e6,
            model_variant='lite_d', branch_ablation='no_dac', mixstyle_on=False,
            use_crra=False, physical_gate_variant='none')

    def forward(self, x):
        # y=None deliberately disables the native label-dependent CosFace margin.
        # The native cosine classifier (scale30) remains part of the architecture.
        return self.id_backbone(x, y=None, return_aux=False)

    def features(self, x):
        return self.id_backbone(x, y=None, return_aux=True)['feat_joint']
