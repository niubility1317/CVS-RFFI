import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .alignment import capture,penalty
from .source import objective
from experiments.cvs_cross_rx_feedback.checks import AcceptedBank
from experiments.cvs_report_mechanisms.feedback import objective as uniform
from experiments.cvs_cross_rx_feedback.method import parameter_selection
def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(7);x=torch.randn(120,2,256,device=device);y=torch.arange(6,device=device).repeat_interleave(20);rx=torch.tensor(d.RX,device=device).repeat_interleave(4).repeat(6);records=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c,device);pg,_=groups(m,c);opt=torch.optim.AdamW(pg,weight_decay=.0001);m.train()
  with capture(m) as maps:logits=m(x)
  align,details=penalty(maps,y,rx,c);gw=torch.autograd.grad(align,[m.head.weight],allow_unused=True,retain_graph=True)[0]
  if gw is not None or align<=0 or len(details)!=2 or any(a['RX_pairs']!=(60 if arm=='TX_CONDITIONAL' else 10) for a in details):raise AssertionError('Actual conditionalmean grouping/head scope')
  (F.cross_entropy(logits,y)+.1*align).backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in pg for p in g['params']):raise AssertionError('Actualshallowmiddle gradients')
  m.eval()
  with torch.no_grad():
   out=m.forward_identity_only(x[:2]);h=out['id_feat_joint'];expected=30*F.linear(F.normalize(h,dim=1,eps=1e-4),F.normalize(m.head.weight,dim=1,eps=1e-4));refit=F.linear(F.normalize(h,dim=1),F.normalize(torch.randn(9,160,device=device),dim=1))
  if not torch.equal(out['tx_logits'],expected) or refit.shape!=(2,9):raise AssertionError('Commoncos/refitinterface')
  bad=copy.deepcopy(c);bad['epochs']=31
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Epochbudget guard')
  records.append(dict(arm=arm,actual_layers=details,ordinarycos=True,parameters=177025))
 # The no-alignment objective is exactly UNIFORM_SSL apart from unused diagnostics.
 c=d.config(d.rows()[0]);m=build(c,device);pg,_=groups(m,c);opt=torch.optim.AdamW(pg,weight_decay=.0001);_,fp=parameter_selection(m,pg);xx=torch.randn(128,2,256,device=device);yy=torch.arange(128,device=device)%6;rr=torch.tensor(d.RX,device=device).repeat(26)[:128];ids=[str(i) for i in range(128)];passes=torch.full((128,),2);m.train();torch.manual_seed(71)
 a,_,_=objective(m,xx,yy,rr,xx,ids,passes,AcceptedBank(),c,1111,5);ga=torch.autograd.grad(a,fp,allow_unused=True)
 mc=d.control(c['model_seed']);torch.manual_seed(71);b,_,_=uniform(m,xx,yy,xx[-25:],yy[-25:],xx,ids,passes,AcceptedBank(),opt,fp,mc,1111);gb=torch.autograd.grad(b,fp,allow_unused=True)
 if not torch.allclose(a,b,atol=1e-7,rtol=1e-7) or any(aa is None and bb is not None or aa is not None and (bb is None or not torch.allclose(aa,bb,atol=1e-6,rtol=1e-5)) for aa,bb in zip(ga,gb)):raise AssertionError('MatchedUNIFORM loss/gradient changed')
 return dict(status='PASS',records=records,matched_UNIFORM_loss_gradient=True,sourceVperformance_gate=False,query_access=False,persistent_moment_state=False)
if __name__=='__main__':print(json.dumps(run()))
