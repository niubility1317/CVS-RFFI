"""Actual F feature boundary and weighted-anchor reference checks, no data/training."""
import copy,json,sys,torch
from torch.nn import functional as F
from . import feature as new
from experiments.cvs_f_supcon50 import mechanism as old,design as original,model as api
from experiments.cvs_f_supcon50.checks import fixture,W,S
from experiments.cvs_equivariant_identity.precision import numerical_context

def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(19);weights=(.9752826424296159,1.0737087120372426,.8243183329638664,1.5,1.0178452671248062,.6088450454444692)
 z=torch.randn(24,16,dtype=torch.float64,device=device,requires_grad=True);y=torch.arange(24,device=device)%6;rx=torch.tensor([1]*6+[3]*6+[4]*6+[6]*6,device=device)
 a,_=old.loss(z,y,rx,'CROSS_RX');b,_=new.loss(z,y,rx,'CROSS_RX',(1.,)*6)
 if not torch.equal(a,b):raise ValueError('Equal weighting must exactly preserve original feature loss')
 g1=torch.autograd.grad(a,z,retain_graph=True)[0];g2=torch.autograd.grad(b,z,retain_graph=True)[0]
 if not torch.equal(g1,g2):raise ValueError('Equal-weight original feature gradient changed')
 v,_=new.loss(z,y,rx,'CROSS_RX',weights);pos,den,valid=old.masks(y,rx,'CROSS_RX');unit=F.normalize(z,dim=1,eps=1e-4);logits=(unit@unit.T/.2)[valid];p=pos[valid];n=p.sum(1);ell=torch.logsumexp(logits.masked_fill(~den[valid],-torch.inf),1)-(logits*p).sum(1)/n-n.to(z.dtype).log();w=torch.tensor(weights,device=device,dtype=z.dtype)[y[valid]];expected=(ell*w).sum()/w.sum()
 if not torch.allclose(v,expected,atol=1e-12,rtol=1e-12):raise ValueError('Weighted anchor formula differs')
 perm=torch.tensor([4,1,5,2,0,3],device=device);inverse=perm.argsort();v2,_=new.loss(z,inverse[y],rx,'CROSS_RX',torch.tensor(weights,device=device,dtype=z.dtype)[perm])
 if not torch.allclose(v,v2,atol=1e-12,rtol=1e-12):raise ValueError('Class label renumbering changed mechanism')
 empty,_=new.loss(z,torch.zeros_like(y),rx,'CROSS_RX',weights)
 if float(empty.detach())!=0 or not torch.isfinite(torch.autograd.grad(empty,z,retain_graph=True)[0]).all():raise ValueError('Invalid-anchor handling changed')
 negative=0
 for bad in ((1.,)*5,(2.,1.,1.,1.,1.,0.),(.5,)*6,(float('nan'),1.,1.,1.,1.,1.)):
  try:new.loss(z,y,rx,'CROSS_RX',bad)
  except ValueError:negative+=1
  else:raise ValueError('Invalid weight vector accepted')
 with numerical_context(original.FULL_FP32_POLICY):
  c=original.config(next(r for r in original.training_rows() if r['arm']=='CROSS_RX'));m=api.build(c,device).train();state={n:v.clone() for n,v in m.state_dict().items()};new.install(m,weights);captured=[];penalty=new.penalty
  def capture(*a,**kw):
   v=penalty(*a,**kw);captured.append(v[0]);return v
  new.penalty=capture
  try:total,_=fixture(m,6,device)
  finally:new.penalty=penalty
  if len(captured)!=2 or m._supcon_pending is not None or m._supcon_handles or m._supcon_applied!=1:raise ValueError('Original two actual L forwards/cache lifetime changed')
  aux=torch.stack(captured).mean();params=dict(m.named_parameters());gw,gq=torch.autograd.grad(aux,[params[W],params[S]],retain_graph=True,allow_unused=True);head=torch.autograd.grad(total,params[W],retain_graph=True)[0]
  if gw is not None or gq is None or not torch.isfinite(gq).all() or gq.norm()<=0 or head.norm()<=0:raise ValueError('Feature Sinc/defaulthead gradient boundary differs')
  if any(not torch.equal(v,m.state_dict()[n]) for n,v in state.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Synthetic check unexpectedly updated model state')
  m.eval();x=torch.randn(4,2,256,device=device)
  with torch.no_grad():f=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
  if not torch.equal(f,exported) or sum(p.numel() for p in m.parameters())!=900722:raise ValueError('Actual inference feature graph changed')
 return dict(status='PASS',device=device,negative_cases=negative,equal_weight_loss_and_gradient_exact=True,label_permutation_equivariant=True,source_weighted_formula_verified=True,actual_F_Sinc_gradient_norm=float(gq.norm()),aux_direct_head_gradient=None,joint_default_head_gradient_norm=float(head.norm()),actual_L_forwards_reused=2,extra_model_forwards=0,inference_feature_export_exact=True,model_state_updated=False,optimizer_updates=0,source_or_target_IQ_reads=0,head_fit_updates=0)
if __name__=='__main__':print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'cpu'),allow_nan=False))
