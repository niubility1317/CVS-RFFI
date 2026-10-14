import json,torch
from . import design as d
from .objective import calculate
from experiments.cvs_source_signature_order.probe import model

def main():
 torch.set_num_threads(2);m=model().eval().requires_grad_(True);saved={k:v.clone() for k,v in m.state_dict().items()};flags=[p.requires_grad for p in m.parameters()]
 y=torch.tensor([t for t in range(6) for dd in d.DAYS for r in d.RX for j in range(2)]);rx=torch.tensor([r for t in range(6) for dd in d.DAYS for r in d.RX for j in range(2)]);day=torch.tensor([dd for t in range(6) for dd in d.DAYS for r in d.RX for j in range(2)])
 losses,detail,matrices=calculate(m,torch.zeros(180,2,256),y,rx,day,list(range(180)));params=[dict(m.named_parameters())[n] for n in d.PARAMETERS];W=m.id_backbone.encoder.core.id_backbone.cls_head.weight
 for loss in losses.values():
  gg=torch.autograd.grad(loss,params+[W],allow_unused=True,retain_graph=True)
  if not torch.isfinite(loss) or gg[-1] is not None or any(g is not None and not torch.isfinite(g).all() for g in gg[:-1]):raise ValueError('Real checkpoint second-derivative no-query smoke failed')
 if flags!=[p.requires_grad for p in m.parameters()] or any(not torch.equal(v,m.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Checkpoint smoke mutated model')
 v=dict(status='PASS',real_checkpoint_no_query_smoke=True,zero_IQ_packets=180,actual_t1_t2_second_derivatives=True,source_checkpoint='fixedfirstSIG_TIMEZERO final50, provenance verified by original own_model',training_inherits_checkpoint=False,query_read=False,head_fit=False,optimizer_updates=0)
 d.write(d.ROOT/'preflight.json',v);print(json.dumps(v))
if __name__=='__main__':main()
