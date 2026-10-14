import torch
from . import design as d
from .provenance import own_model
from .model import classifier
def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));m=own_model(c,'cuda:0');before={k:v.detach().clone() for k,v in m.state_dict().items()};x=torch.zeros(2,2,256,device='cuda:0')
 with torch.no_grad():
  z=m.forward_identity_only(x)['id_feat_joint'];h=classifier(m);w=h.weight
  try:
   h.weight=torch.nn.Parameter(torch.zeros_like(w),requires_grad=False);assert torch.equal(z,m.id_backbone.encoder.features(x));del h.weight;assert torch.equal(z,m.id_backbone.encoder.features(x))
  finally:h.weight=w
 if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen PA noquery export changed model')
 d.write(d.BASE/row/'diagnostics/summary.json',dict(status='SOURCE_DEFAULT_DIAGNOSTICS_COMPLETE',W_independent_export=True,parameters_buffers_unchanged=True,head_fit_updates=0,query_access=False,synthetic_check_not_performance=True))
