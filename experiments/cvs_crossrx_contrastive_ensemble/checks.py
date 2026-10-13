import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .source import objective,dropout_seed
from experiments.cvs_ensemble_final30.model import ProbabilityEnsemble
def run(device='cpu'):
 torch.set_num_threads(2);row=d.rows()[0];cs=[d.member_config(r) for r in d.member_rows() if r['row_id']==row['row_id']];models=[build(c,device) for c in cs]
 for model,c in zip(models,cs):groups(model,c)
 if d.readouts(cs[0])!=('FINAL',) or d.readouts(d.config(row))!=('ENSEMBLE',):raise AssertionError('Member/group frozen readout contract differs')
 a,b=[{k:v.detach().clone() for k,v in m.named_parameters() if v.requires_grad} for m in models];different=sum(not torch.equal(a[k],b[k]) for k in a)
 if different<1 or len(a)!=67 or any(c['epochs']!=30 or c['updates']!=6660 for c in cs):raise AssertionError('Actual identity initialization/budget differs')
 if dropout_seed(cs[0],1333)!=cs[0]['member_init_seed']+1333*1979 or dropout_seed(cs[0],1333)==dropout_seed(cs[1],1333):raise AssertionError('Actual member dropout seed differs')
 x=torch.randn(120,2,256,device=device);y=torch.arange(6,device=device).repeat_interleave(20);rx=torch.tensor(d.RX,device=device).repeat_interleave(4).repeat(6);m=models[0];m.train()
 term,metrics,detail=objective(m,x,y,rx,None,None,None,None,cs[0],1333,6);pg,_=groups(m,cs[0]);term.backward()
 if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in pg for p in g['params']) or metrics['contrastive']<=0 or len(detail['actual_layers'])!=2:raise AssertionError('Actual member loss/gradient differs')
 from experiments.cvs_cross_rx_feedback.checks import AcceptedBank
 observed=[];hook=m.register_forward_pre_hook(lambda module,inputs:observed.append(torch.initial_seed()))
 try:
  objective(m,x,y,rx,torch.randn(128,2,256,device=device),[str(i) for i in range(128)],torch.full((128,),2),AcceptedBank(),cs[0],1333,6)
 finally:hook.remove()
 if len(observed)!=3 or observed[-1]!=dropout_seed(cs[0],1333):raise AssertionError('Actual U strong forward did not consume member dropout seed')
 for c in cs:
  bad=copy.deepcopy(c);bad['member_init_seed']+=1
  try:d.validate_member(bad)
  except ValueError:pass
  else:raise AssertionError('Unregistered initseed accepted')
 m0=copy.deepcopy(models[0]);ensemble=ProbabilityEnsemble([m0,*models]).eval()
 with torch.no_grad():
  logits=[m(x[:3]) for m in ensemble.models];expected=torch.stack([z.softmax(1) for z in logits]).mean(0).clamp_min(1e-12).log();out=ensemble(x[:3])
  if not torch.equal(out,expected):raise AssertionError('Fixed T1 equal probability differs')
  for model in ensemble.models:
   h=model.forward_identity_only(x[:2])['id_feat_joint'];plain=30*F.linear(F.normalize(h,dim=1,eps=1e-4),F.normalize(model.head.weight,dim=1,eps=1e-4))
   if not torch.equal(plain,model(x[:2])) or F.linear(F.normalize(h,dim=1),F.normalize(torch.randn(9,160,device=device),dim=1)).shape!=(2,9):raise AssertionError('Ordinary cos/refit differs')
 return dict(status='PASS',actual_trainable_tensors=len(a),differently_initialized_tensors=different,member_epochs=30,new_models=12,new_updates=79920,total_updates=119880,equal_probability_T1=True,ordinarycos_each_member=True,query_access=False)
if __name__=='__main__':print(json.dumps(run()))
