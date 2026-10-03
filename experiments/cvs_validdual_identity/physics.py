"""Public-only whole-network constant-phase diagnostic, not identity accuracy."""
import torch
import numpy as np
from experiments.cvs_reference_identity.physics import TX_ROWS,RX_ROWS,received
from experiments.cvs_equivariant_identity.model import rotate_pair
from experiments.cvs_channel_order_identity.model import _diagnostic_mode
@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    p=next(model.parameters());x=torch.tensor(np.stack([received(t,r) for t in TX_ROWS for r in RX_ROWS]).tolist(),dtype=p.dtype,device=p.device)
    with _diagnostic_mode(model,x):
        base=model(x);rows=[]
        for angle in (.37,-1.2,2.9):
            value=model(rotate_pair(x,x.new_full((len(x),),angle)))
            rows.append(dict(angle=angle,max_logit_error=float((base-value).abs().max())))
        diagnostics=model.diagnostics(x)
    return dict(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE',public_only=True,target_access=False,
        model_updated=False,phase_audit=rows,diagnostics=diagnostics,
        claim='Numerical report only; not channel invariance or actual transmitter accuracy')
