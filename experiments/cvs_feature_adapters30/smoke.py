"""Strict real parent and zero-output extended deployment replay; no source/query IQ."""
import torch
from . import design as d,model as api
from experiments.cvs_equivariant_identity.precision import numerical_context

def run():
 torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
  p=api.parent('cpu');state={k:v.clone() for k,v in p.state_dict().items()};x=torch.zeros(2,2,256)
  with torch.no_grad():z=p.forward_identity_only(x)['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real parent no-query smoke failed')
  rows=[]
  for style in ('FULL','LORA','TEMP'):
   c=d.config(next(r for r in d.rows() if r['style']==style and r['As']));m=api.from_state(c,state)
   with torch.no_grad():a=m.forward_identity_only(x)
   if not torch.equal(a['tx_logits'],z) or a['id_feat_joint'].shape!=(2,160):raise ValueError('Real zero-output adapter must preserve original default prediction')
   rows.append(dict(style=style,**api.counts(m)))
  if any(not torch.equal(v,p.state_dict()[k]) for k,v in state.items()):raise ValueError('Real parent smoke mutated state')
 d.write(d.BASE/'checkpoint_smoke.json',dict(status='PASS',actual_trained_checkpoint='verified original pureF B0 scratch fixed FINAL20',rows=rows,NN=8,updates=0,query_read=False,source_IQ_read=False,parent_inherited_by_all_six=True))
 print('SMOKE PASS strict real B0 parent; zero-output features/default head; no query. Continue bank and training.',flush=True)
