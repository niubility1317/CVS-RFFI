"""Original physical Sinc and paired RDrop/SupCon; extra challenge counts."""
import gzip,json
from collections import Counter
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
from .mechanism import active
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN:raise ValueError('Original RX/day source map differs')
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  steps=len(rows)*222;expected=sum(active(s) for s in range(1,steps+1))
  if len(m._challenge_records)!=steps or m._challenge_pending is not None or m._challenge_applied!=expected:raise ValueError('Scheduled weak-TX challenge missing/escaped')
  d.write(out/'challenge_progress.json',dict(epoch=len(rows),steps=steps,scheduled=expected,applied=m._challenge_applied,extra_L_packets=m._challenge_packets,U_truth_read=False,head_fit_updates=0))
 def finish(self):
  result=super().finish();m=self.model
  count=sum(active(s) for s in range(1,11101));layers=Counter(r['level'] for r in m._challenge_records if r['active'])
  if len(m._challenge_records)!=11100 or count!=2498 or m._challenge_applied!=count or dict(layers)!={'time_down':1249,'t2.pool':1249}:raise ValueError('Complete challenge calendar differs')
  with gzip.open(self.out/'challenge_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in m._challenge_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  d.write(self.out/'challenge_training_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,scheduled=2498,applied=m._challenge_applied,layer_calls=dict(layers),extra_model_forwards=2498,extra_L_packets=m._challenge_packets,extra_optimizer_updates=0,extra_parameters=0,weak_indices=[1,3],head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False))
  return result
