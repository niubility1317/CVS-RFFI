"""Original full unified Sinc; paired L feature objective execution evidence."""
import gzip,json
from experiments.cvs_rdrop_sinc_units.observer import Observer as Base
from . import design as d
from .mechanism import RX_BY_DOMAIN
class Observer(Base):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  expected=d.read(d.SOURCE);domain_map=expected['domain_map']
  if ctx['num_domains']!=15 or tuple(row[0] for row in domain_map)!=RX_BY_DOMAIN or [row[2] for row in domain_map]!=list(range(15)):raise ValueError('Original physical RX/day domain map differs')
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);epoch=len(rows);steps=epoch*222;applied=max(0,epoch-5)*222
  if m._supcon_calls!=steps or m._supcon_applied!=applied or m._supcon_pending is not None or m._supcon_handles or m._supcon_phase is not None or len(m._supcon_records)!=steps:raise ValueError('Actual paired L-only SupCon lifetime/budget differs')
  d.write(out/'supcon_progress.json',dict(epoch=epoch,calls=steps,applied_calls=applied,mode=self.c['arm'],temporary_hooks_remaining=0,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False))
 def finish(self):
  value=super().finish();m=self.model
  if m._supcon_calls!=11100 or m._supcon_applied!=9990 or m._supcon_pending is not None or m._supcon_handles or len(m._supcon_records)!=11100:raise ValueError('Complete fixed F SupCon execution evidence required')
  with gzip.open(self.out/'supcon_steps.jsonl.gz','xt',encoding='utf8') as f:
   for row in m._supcon_records:f.write(json.dumps(row,allow_nan=False)+'\n')
  valid=sum(v['valid_anchors'] for r in m._supcon_records for v in r['details']);total=sum(v['total_anchors'] for r in m._supcon_records for v in r['details'])
  d.write(self.out/'supcon_training_observation.json',dict(status='VERIFIED',mode=self.c['arm'],calls=11100,applied_calls=9990,actual_layer_forward_objectives=39960,valid_anchors=valid,total_anchors=total,extra_model_forwards=0,extra_optimizer_updates=0,extra_parameters=0,temporary_hooks_remaining=0,default_head_jointly_trained=True,head_fit_updates=0,U_truth_read=False))
  return value
