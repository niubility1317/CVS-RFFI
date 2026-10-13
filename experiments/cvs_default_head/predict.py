"""Direct inference through the jointly trained default CVS classifier.

No optimizer, new class weights, source fit, prototype or alternative readout.
The caller supplies its already frozen model and legally obtained IQ tensors.
"""
import torch
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
READOUTS=('ORIGINAL_HEAD',)

def logits(model,iq):
 if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Default inference requires the entire model frozen after joint training')
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 if type(head) is not ResidualPhysicalHead or head.scale!=30. or head.weight.ndim!=2 or head.weight.shape[1]!=160:raise ValueError('Use the original default residual cosine classifier')
 if not torch.is_tensor(iq) or iq.ndim!=3 or iq.shape[1:]!=(2,256) or len(iq)==0 or iq.dtype!=torch.float32 or iq.requires_grad:raise ValueError('Expected fixed float32 IQ[N,2,256]')
 if not torch.isfinite(iq).all():raise ValueError('Default inference rejects nonfinite IQ')
 with torch.no_grad():value=model.forward_identity_only(iq)['tx_logits']
 if value.shape!=(len(iq),head.weight.shape[0]) or not torch.isfinite(value).all():raise ValueError('Default classifier output differs')
 return value
