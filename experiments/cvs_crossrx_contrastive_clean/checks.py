import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .contrastive import capture,penalty,masks,loss
from .source import objective
from experiments.cvs_cross_rx_feedback.checks import AcceptedBank
from experiments.cvs_report_mechanisms.feedback import objective as uniform
from experiments.cvs_cross_rx_feedback.method import parameter_selection
def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(7);x=torch.randn(120,2,256,device=device);y=torch.arange(6,device=device).repeat_interleave(20);rx=torch.tensor(d.RX,device=device).repeat_interleave(4).repeat(6);records=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c,device);pg,_=groups(m,c);opt=torch.optim.AdamW(pg,weight_decay=.0001);m.train()
  pos,den=masks(y,rx,arm)
  if not torch.equal(pos.sum(1),torch.full((120,),19 if arm=='ALL_TX' else 16,device=device)) or (pos&~den).any() or den.diagonal().any():raise AssertionError('Positive/denominator grouping')
  with capture(m) as maps:logits=m(x)
  term,details=penalty(maps,y,rx,c);gw=torch.autograd.grad(term,[m.head.weight],allow_unused=True,retain_graph=True)[0]
  map_gradients=torch.autograd.grad(term,list(maps.values()),retain_graph=True)
  if any(not torch.isfinite(g).all() or not g.abs().sum()>0 for g in map_gradients):raise AssertionError('Actual loss does not reach both shallow and middle maps')
  perm=torch.tensor([5,3,4,2,0,1],device=device);relabel,_=penalty(maps,perm[y],rx,c)
  if gw is not None or not term>=0 or not torch.allclose(term,relabel,atol=1e-6):raise AssertionError('Cos exclusion/label permutation')
  (F.cross_entropy(logits,y)+.1*term).backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in pg for p in g['params']):raise AssertionError('Actual contrastive gradients')
  for h in maps.values():
   if not h.requires_grad:raise AssertionError('Actual shallow/middle loss disconnected')
  m.eval()
  with torch.no_grad():
   out=m.forward_identity_only(x[:2]);h=out['id_feat_joint'];expected=30*F.linear(F.normalize(h,dim=1,eps=1e-4),F.normalize(m.head.weight,dim=1,eps=1e-4));refit=F.linear(F.normalize(h,dim=1),F.normalize(torch.randn(9,160,device=device),dim=1))
  if not torch.equal(out['tx_logits'],expected) or refit.shape!=(2,9):raise AssertionError('Common cos/refit')
  bad=copy.deepcopy(c);bad['epochs']=31
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Budget guard')
  records.append(dict(arm=arm,actual_layers=details,ordinary_cos=True,parameters=177025))
 # Disallow all-one-class or unpaired crossRX metadata.
 for yy,rr in ((torch.zeros_like(y),rx),(y,torch.ones_like(rx))):
  try:masks(yy,rr,'CROSS_RX_TX')
  except ValueError:pass
  else:raise AssertionError('Illegal empty positive/negative grouping accepted')
 z=torch.randn(120,12,device=device);a,_=loss(z,y,rx,'CROSS_RX_TX');b,_=loss(z*3,y,rx,'CROSS_RX_TX')
 if not torch.allclose(a,b,atol=1e-6):raise AssertionError('Feature scale invariance')
 c=d.config(d.rows()[0]);m=build(c,device);pg,_=groups(m,c);opt=torch.optim.AdamW(pg,weight_decay=.0001);_,fp=parameter_selection(m,pg);xx=torch.randn(128,2,256,device=device);yy=torch.arange(128,device=device)%6;rr=torch.tensor(d.RX,device=device).repeat(26)[:128];ids=[str(i) for i in range(128)];passes=torch.full((128,),2);m.train();torch.manual_seed(71)
 a,_,_=objective(m,xx,yy,rr,xx,ids,passes,AcceptedBank(),c,1111,5);ga=torch.autograd.grad(a,fp,allow_unused=True)
 torch.manual_seed(71);b,_,_=uniform(m,xx,yy,xx[-25:],yy[-25:],xx,ids,passes,AcceptedBank(),opt,fp,d.control(c['model_seed']),1111);gb=torch.autograd.grad(b,fp,allow_unused=True)
 if not torch.allclose(a,b,atol=1e-7,rtol=1e-7) or any(aa is None and bb is not None or aa is not None and (bb is None or not torch.allclose(aa,bb,atol=1e-6,rtol=1e-5)) for aa,bb in zip(ga,gb)):raise AssertionError('Matched UNIFORM loss/gradient changed')
 return dict(status='PASS',records=records,matched_UNIFORM_loss_gradient=True,sourceVperformance_gate=False,query_access=False,persistent_moment_state=False)
if __name__=='__main__':print(json.dumps(run()))
