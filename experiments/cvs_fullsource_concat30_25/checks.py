"""Focused full-role, batch, loss and truth-boundary negatives; synthetic only."""
import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .model import paired_terms,validate_labels
def run():
 c=d.config(d.rows()[0]);d.validate(c)
 negatives=0
 for change in ({'held_RX':1},{'epochs':50},{'epochs':20},{'label_epochs':13},{'pseudo_epochs':7},{'updates':4440},{'checkpoint_sources':['oldE20.pth']},{'channel':dict(c['channel'],reference_snr_db=40.)},{'source_roles':dict(L_s=5040,U_s=45360,V=21600)},{'train_scenes':['practical_low_urban']}):
  bad=copy.deepcopy(c);bad.update(change)
  try:d.validate(bad)
  except ValueError:negatives+=1
  else:raise AssertionError("Altered role/budget accepted")
 rng=torch.Generator().manual_seed(2201);y=torch.arange(128)%6
 a=torch.randn(256,6,generator=rng,requires_grad=True);b=torch.randn(256,6,generator=rng,requires_grad=True)
 value,ce,kl,parts=paired_terms(a,b,y)
 yy=torch.cat((y,y));p,q=a.log_softmax(1),b.log_softmax(1)
 direct=.5*(F.cross_entropy(a,yy,label_smoothing=.01)+F.cross_entropy(b,yy,label_smoothing=.01))+.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
 assert torch.allclose(value,direct,atol=1e-6,rtol=1e-6)
 value.backward();assert bool((a.grad.abs().sum(1)>0).all()) and bool((b.grad.abs().sum(1)>0).all())
 for labels,n in ((y[:127],128),(torch.full((128,),-1),128),(y.float(),128),(y,256)):
  try:validate_labels(labels,n)
  except ValueError:negatives+=1
  else:raise AssertionError("Hidden U/incorrect label shape accepted")
 assert d.config(d.rows()[1])['As'] and not c['As']
 assert c['source_RX']==[1,3,4,6,8] and c['held_RX'] is None
 assert len(c['test_views'])==7 and c['source_roles']==dict(L_s=6300,U_s=56700,V=27000)
 a=d.make_args(c,'cpu')
 from SSDG.train_ssdg import _resolve_epoch_schedule
 assert _resolve_epoch_schedule(a)==30 and (a.epochs,a.label_epochs,a.pseudo_epochs)==(30,20,10)
 from .source import require_budget
 require_budget(6660,6660)
 for budget in [(4440,4440),(6660,6659)]:
  try:require_budget(*budget)
  except ValueError:negatives+=1
  else:raise AssertionError('E20 or skipped update budget accepted')
 from experiments.cvs_phase1_repair.design import cosine_lr
 assert abs(cosine_lr(a.lr,1,30,a.phase1_lr_min)-a.lr)<1e-12
 assert abs(cosine_lr(a.lr,30,30,a.phase1_lr_min)-a.phase1_lr_min)<1e-12
 return dict(status="PASS",protocol_negatives=negatives,synthetic_logits=512,NN=0,real_IQ=0,CP=0,optimizer_updates=0)
if __name__=="__main__":print(json.dumps(run()))
