import argparse,json,time
from pathlib import Path
from copy import deepcopy
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .method import objective,parameter_selection,preconditioner,scores,bounded_weights
from .sampling import RotatingStreams
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
class AcceptedBank:
 def update(self,ids,prediction,confidence,passes):return torch.ones_like(prediction,dtype=torch.bool),dict(test_only=True)
def run(device='cpu'):
 torch.set_num_threads(2);records=[];started=time.perf_counter()
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));model=build(c,device);pg,_=groups(model,c);params=[p for g in pg for p in g['params']];names,fp=parameter_selection(model,pg);opt=torch.optim.AdamW(pg,weight_decay=.0001)
  torch.manual_seed(57);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;ux=torch.randn_like(x);ids=[str(i) for i in range(128)];passes=torch.full((128,),2)
  loss,m,detail=objective(model,x,y,x[-32:],y[-32:],None,None,None,None,opt,fp,c,1);loss.backward();opt.step();opt.zero_grad(set_to_none=True)
  if detail['teacher_forwards'] or detail['feedback_gradients']:raise AssertionError('CE warmup executes U/meta')
  if arm!='CE_MATCHED':
   before={k:v.clone() for k,v in model.state_dict().items()};loss,m,detail=objective(model,x,y,x[-32:],y[-32:],ux,ids,passes,AcceptedBank(),opt,fp,c,1111)
   if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise AssertionError('Feedback weights mutate parameters/buffers')
   if detail['feedback_gradients']!=1 or detail['U_gradient_groups']!=4 or not all(torch.isfinite(torch.tensor(detail[k])) for k in ('P_gradient_M_norm','P_gradient_U_norm')):raise AssertionError('Meta gradients invalid')
   loss.backward()
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise AssertionError('Finite actual native loss gradient')
   if arm=='UNIFORM_SSL' and detail['group_weights']!=[1.]*4:raise AssertionError('Uniform costmatched control weights')
   if arm=='META_SSL' and (not all(0<=v<=2 for v in detail['group_weights']) or detail['mean_weight']>1.000001):raise AssertionError('Nonnegative bounded weights')
   opt.step()
  model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
  with torch.no_grad():
   out=model.forward_identity_only(x[:8]);h=out['id_feat_joint'];expected=30*F.linear(F.normalize(h,dim=1,eps=1e-4),F.normalize(model.head.weight,dim=1,eps=1e-4));single=torch.cat([model(a[None]) for a in x[:8]])
   replaced=30*F.linear(F.normalize(h,dim=1,eps=1e-4),F.normalize(torch.randn(9,160,device=device),dim=1,eps=1e-4))
  if not torch.equal(expected,out['tx_logits']) or not torch.allclose(single,out['tx_logits'],atol=1e-4,rtol=1e-4) or replaced.shape!=(8,9) or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise AssertionError('Ordinarycos replacement/perpacket inference')
  if sum(p.numel() for p in params)!=177025 or id(model.head.weight) in {id(p) for p in fp}:raise AssertionError('Classifier capacity/feedbackgradient scope')
  for k,v in [('epochs',100),('held_rx',0),('training_target_access',True)]:
   bad=deepcopy(c);bad[k]=v
   try:d.validate(bad)
   except ValueError:pass
   else:raise AssertionError('Forbidden source config')
  bad=deepcopy(c);bad['classifier']['routing']=True
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Head routing accepted')
  records.append(dict(arm=arm,parameters=177025,feedback_parameters=len(fp),cosines=detail['gradient_cosines'],weights=detail['group_weights'],ordinarycos=True,refit9class=True))
 if bounded_weights([-1.,0.,-2.,-.1],device).sum()!=0:raise AssertionError('Conflict-only U must be off')
 if not torch.equal(bounded_weights([1.,1.,1.,1.],device),torch.ones(4,device=device)):raise AssertionError('Equal useful gradients meanweight1')
 # For a frozen diagonal P, central difference of virtual SGD feedback risk.
 theta=torch.tensor([.2,-.3],dtype=torch.float64);gm=theta-torch.tensor([1.,2.],dtype=torch.float64);gu=torch.tensor([-.7,.4],dtype=torch.float64);pre=torch.tensor([2.,.5],dtype=torch.float64);eps=1e-5
 risk=lambda a:.5*(theta-eps*a*pre*gu-torch.tensor([1.,2.],dtype=torch.float64)).square().sum()
 numeric=-(risk(1.)-risk(-1.))/(2*eps);expected=(gm*pre*gu).sum()
 if not torch.allclose(numeric,expected,atol=1e-9):raise AssertionError('FrozenP directional derivative formula')
 fake=dict(rx=torch.tensor([3,4,6,8]).repeat_interleave(120),day=torch.tensor([1,2,3]).repeat(160));ls=RotatingStreams(fake,5)
 for step in range(4):
  mr=sorted(ls.indices)[step];inner=[r for r in ls.indices if r!=mr];parts=[ls.batch(r,32)[0] for r in inner+[mr]]
  if any(not (fake['rx'][ix]==r).all() for r,ix in zip(inner+[mr],parts)) or sum(map(len,parts))!=128:raise AssertionError('FeedbackRX sampler')
 labeled=dict(rx=torch.tensor([3,4,6,8]).repeat_interleave(180),y=torch.arange(180).remainder(6).repeat(4),day=(torch.arange(180)//6).remainder(3).add(1).repeat(4));stream=RotatingStreams(labeled,5,labeled=True)
 for r in stream.indices:
  ix,_=stream.batch(r,32)
  if len(ix)!=32 or not (labeled['rx'][ix]==r).all() or labeled['y'][ix].bincount(minlength=6).min()<5:raise AssertionError('L feedback TX/day balanced exposure')
 try:validate_unlabeled({'y':torch.zeros(1)})
 except ValueError:pass
 else:raise AssertionError('U truth schema accepted')
 return dict(status='PASS',records=records,negative_cases=13,frozenP_derivative=True,seconds=time.perf_counter()-started)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
