import json,torch
from . import design as d
from experiments.cvs_signature_order50.observer import objective,capture
from experiments.cvs_source_signature_order.probe import model
from experiments.cvs_source_layer_order.objective import relation_indices
from experiments.cvs_source_layer_order import design as sd
def main():
 torch.set_num_threads(2);m=model().eval().requires_grad_(True);saved={k:v.clone() for k,v in m.state_dict().items()}
 y=torch.tensor([t for t in range(6) for dd in sd.DAYS for r in sd.RX]);rx=torch.tensor([r for t in range(6) for dd in sd.DAYS for r in sd.RX]);day=torch.tensor([dd for t in range(6) for dd in sd.DAYS for r in sd.RX]);q,groups=relation_indices(y,rx,day,list(range(90)))
 with capture(m) as maps:out=m.forward_identity_only(torch.zeros(90,2,256))
 losses,stats,clock=objective(maps,q);head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
 if any(not torch.isfinite(v) or torch.autograd.grad(v,head,allow_unused=True,retain_graph=True)[0] is not None for v in losses.values()) or any(not torch.equal(v,m.state_dict()[k]) for k,v in saved.items()):raise ValueError('Real checkpoint signature no-query smoke failed')
 v=dict(status='PASS',real_checkpoint_no_query_smoke=True,zero_IQ_packets=90,source_checkpoint='fixedfirstSIG_TIMEZERO final50, provenance verified by original own_model',training_inherits_checkpoint=False,query_read=False,head_fit=False,optimizer_updates=0,clock=clock)
 d.write(d.ROOT/'preflight.json',v);print(json.dumps(v))
if __name__=='__main__':main()
