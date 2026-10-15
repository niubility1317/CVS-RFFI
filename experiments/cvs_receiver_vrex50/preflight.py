import json,torch
from . import design as d
from .checks import checks
from .objective import objective
from experiments.cvs_receiver_dro50 import design as original,predict as api
def main():
 torch.set_num_threads(2);local=checks();d.require_source();c=original.config(next(r for r in original.training_rows() if r['arm']=='SIG_TIME'));m=api.own_model(c,'cpu').eval().requires_grad_(True);before={k:v.clone() for k,v in m.state_dict().items()};flags=[v.requires_grad for v in m.parameters()]
 out=m.forward_identity_only(torch.zeros(10,2,256));y=torch.arange(10)%6;rx=torch.tensor(d.RX).repeat_interleave(2);objects=[dict(m.named_parameters())[n] for n in d.PARAMETERS]+[m.id_backbone.encoder.core.id_backbone.cls_head.weight]
 for alpha in (0.,1.):
  loss,*_=objective(out['tx_logits'],y,rx,alpha);grad=torch.autograd.grad(loss,objects,retain_graph=True)
  if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in grad):raise ValueError('RealcheckpointnoqueryVREXsmokefailed')
 if flags!=[v.requires_grad for v in m.parameters()] or any(v.grad is not None for v in m.parameters()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Realcheckpoints smokemutatedstate')
 p=d.ROOT/'preflight.json'
 if p.exists():raise FileExistsError('Immutablepreflightexists')
 v=dict(status='PASS',checks=local,real_checkpoint_no_query_smoke=True,synthetic_IQ_packets=10,allthree_frozen_source_models_PASS=True,source_checkpoint='OriginalfixedDRO SIG_TIME01final50;actualsource_proofchecked',training_inherits_checkpoint=False,query_read=False,head_fit_updates=0,optimizer_updates=0);d.write(p,v);print(json.dumps(v))
if __name__=='__main__':main()
