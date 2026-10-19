"""Preserve original witnesses; count current-L meta calls and actual cost."""
import gzip,json
from collections import Counter
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
from .objective import active
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN:raise ValueError('Original source RX/day mapping differs')
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;scheduled=sum(active(s) for s in range(1,steps+1))
  if len(m._metarx_records)!=steps or sum(r['scheduled'] for r in m._metarx_records)!=scheduled or m._metarx_applied>scheduled:raise ValueError('Actual ordered meta calendar differs')
  d.write(out/'metarx_progress.json',dict(epoch=len(rows),steps=steps,scheduled=scheduled,applied=m._metarx_applied,extra_L_packets=m._metarx_packets,head_fit_updates=0,U_truth_read=False))
 def finish(self):
  result=super().finish();m=self.model;scheduled=sum(r['scheduled'] for r in m._metarx_records);eligible=m._metarx_applied
  if len(m._metarx_records)!=11100 or scheduled!=624 or not 0<=eligible<=624:raise ValueError('Complete eligible meta source budget required')
  with gzip.open(self.out/'metarx_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in m._metarx_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  counts=Counter(r['anchor_RX'] for r in m._metarx_records if r.get('eligible'))
  d.write(self.out/'metarx_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,scheduled=624,applied=eligible,eligible_fraction=eligible/624,anchor_RX_calls=dict(counts),extra_model_forwards=2*eligible,extra_L_packets=m._metarx_packets,extra_optimizer_updates=0,extra_parameters=0,weak_indices=[1,3],head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,virtual_Sinc_update=False,inner_alpha=self.c['meta_inner_alpha']))
  return result
