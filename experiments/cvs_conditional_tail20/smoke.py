"""Read-only real trained F checkpoint; never inherited by scratch training."""
import torch
from . import design as d
from experiments.cvs_F_nontrue20 import design as old,model as api
from experiments.cvs_equivariant_identity.precision import numerical_context

def run():
 torch.set_num_threads(2)
 with torch.random.fork_rng(devices=[]),numerical_context(d.FULL_FP32_POLICY):
  c=old.config(next(r for r in old.rows() if r['arm']=='B1'));m=api.own_model(c,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Strict real F/defaulthead no-query smoke failed')
  if sum(p.numel() for p in m.parameters())!=900722:raise ValueError('Fixed F parameter count differs')
 d.write(d.BASE/'checkpoint_smoke.json',dict(status='PASS',actual_trained_checkpoint='verified original pureF B1 fixedFINAL20',NN=2,parameter_updates=0,query_read=False,source_IQ_opened=False,inherited_into_training=False))
 print('SMOKE PASS real trained pureF/defaulthead; no query; scratch training follows',flush=True)
