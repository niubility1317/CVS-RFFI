"""Original50 execution plus every scheduled weak-pair feature VJP witness."""
import gzip,json
from collections import Counter
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
from .core import active
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN:raise ValueError('Original source physical RX/day map differs')
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;scheduled=sum(active(s) for s in range(1,steps+1))
  if m._pc_calls!=steps or len(m._pc_records)!=steps or m._pc_scheduled!=scheduled or m._pc_second_captures!=scheduled or m._pc_vjp_calls!=3*m._pc_eligible or m._pc_pending is not None:raise ValueError('Actual block routing calendar differs')
  d.write(out/'weak_pair_progress.json',dict(epoch=len(rows),steps=steps,scheduled=scheduled,eligible=m._pc_eligible,changed=m._pc_changed_steps,extra_feature_VJP_calls=m._pc_vjp_calls,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False))
 def finish(self):
  result=super().finish();m=self.model
  if m._pc_calls!=11100 or len(m._pc_records)!=11100 or m._pc_scheduled!=2498 or m._pc_second_captures!=2498 or m._pc_vjp_calls!=3*m._pc_eligible or not 0<=m._pc_changed_steps<=m._pc_eligible<=2498 or m._pc_pending is not None:raise ValueError('Full original50/weak pair ordered budget required')
  with gzip.open(self.out/'weak_pair_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in m._pc_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  conflicts=Counter();changes=Counter();errors=[]
  for r in m._pc_records:
   for b in r['blocks']:
    conflicts[b['block']]+=int(b['conflict']);changes[b['block']]+=int(b['delta_norm']>0);errors.append(b['fp32_norm_match_error'])
  d.write(self.out/'weak_pair_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,scheduled=2498,second_logit_captures=m._pc_second_captures,eligible=m._pc_eligible,eligible_fraction=m._pc_eligible/2498,changed_steps=m._pc_changed_steps,nonzero_fraction=m._pc_changed_steps/2498,extra_feature_VJP_calls=m._pc_vjp_calls,block_conflict_calls=dict(conflicts),block_nonzero_calls=dict(changes),maximum_norm_match_error=max(errors,default=0.),extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,weak_indices=[1,3],head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,full_deployed_features=True,paired_CE_L_contributions=True,surrogate_scalar_value=0.,Sinc_common_scale_invariant=True))
  return result
