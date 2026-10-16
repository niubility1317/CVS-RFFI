"""Full50 F updates; native RDrop plus physical/adaptive-step witnesses."""
import gzip,json,torch
from experiments.cvs_rdrop_sinc_units.observer import Observer as Base
from . import design as d
from .optimizer import attach
class Observer(Base):
 def __init__(self,c):
  d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def optimizer_created(self,opt):
  self.policy=attach(opt,self.model,self.c['arm']);super().optimizer_created(opt)
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  if self.optimizer._band_gain_steps!=len(rows)*222:raise ValueError('Actual F band policy budget differs')
  if len(rows) in d.SINC_DIAGNOSTIC_EPOCHS:
   witness=[r for r in self.optimizer._band_gain_raw if r['epoch']==len(rows)]
   if len(witness)!=13:raise ValueError('Missing F physical gain witnesses')
   torch.save(witness,out/f'gain_update_witness_e{len(rows)}.pt')
 def finish(self):
  value=super().finish()
  if self.optimizer._band_gain_steps!=11100 or len(self.optimizer._band_gain_records)!=11100 or len(self.optimizer._band_gain_raw)!=52:raise ValueError('Complete F gain trajectory required')
  with gzip.open(self.out/'gain_steps.jsonl.gz','xt',encoding='utf8') as stream:
   for row in self.optimizer._band_gain_records:stream.write(json.dumps(row,allow_nan=False)+'\n')
  d.write(self.out/'gain_policy.json',dict(status='VERIFIED',steps=11100,policy=self.policy,physical_decay_unscaled=True,physical_clip_calls=self.optimizer._sinc_physical_clip_calls,head_fit_updates=0,extra_optimizer_updates=0))
  return value
