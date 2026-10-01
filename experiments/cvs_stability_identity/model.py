"""Supplement native identity cues with phase increments and spectral residuals."""
from torch import nn
from experiments.cvs_identity_ce.model import IdentityOnlyCVS,build_model
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
VARIANTS=('phase_delta','phase_dsq')

class StabilityCVS(IdentityOnlyCVS):
    def __init__(self,variant):
        nn.Module.__init__(self)
        self.id_backbone=build_model(num_classes=6,model_size='M',dataset='wisig',input_len=256,sample_rate_hz=25e6,
            model_variant='lite_d',branch_ablation='no_dac',mixstyle_on=False,use_crra=False,physical_gate_variant='none',
            time_stability_mode='phase_delta',freq_stability_mode='dsq' if variant=='phase_dsq' else 'off',
            time_stability_channels=8,freq_stability_channels=4)
        b=self.id_backbone
        if b.use_dac_path or not b.use_pa_path or b.nmfdu_gate is not None:raise ValueError('Native identity/PA contract changed')
        b.con_proj=nn.Identity();b.cls_head=ResidualPhysicalHead(b.emb_dim)

def build(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered physical stability variant')
    return StabilityCVS(variant)
