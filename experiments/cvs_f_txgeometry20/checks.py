"""Focused geometry direction, protocol-negative, real F/defaulthead checks."""
import copy,json,sys,tempfile
from pathlib import Path
import torch
from . import design as d,mechanism as me,model as api
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY
from experiments.cvs_f_txrisk.feature_checks import run as risk_checks
from experiments.cvs_f_supcon50.checks import W,S

def run(device='cpu'):
 torch.set_num_threads(2);negative=0;c=d.config(d.rows()[0])
 for key,value in (('epochs',51),('U_used',True),('EMA',True),('separate_head_training',True),('checkpoint_sources',['unknown']),('test_views',['clean'])):
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise ValueError('Forbidden recipe accepted')
 z=torch.tensor([[1.,.4],[1.,-.4],[-1.,.4],[-1.,-.4]],dtype=torch.float64,requires_grad=True,device=device);y=torch.tensor([0,0,1,1],device=device);rx=torch.tensor([1,3,1,3],device=device)
 def stats(v):
  v=torch.nn.functional.normalize(v,dim=1);cos=v@v.T
  return (cos[0,1]+cos[2,3])/2,cos[:2,2:].mean()
 old=stats(z);v,_=me.loss(z,y,rx,'CROSS_RX',(1.,)*6);g=torch.autograd.grad(v,z)[0];z2=z.detach()-.05*g;new=stats(z2);v2,_=me.loss(z2,y,rx,'CROSS_RX',(1.,)*6)
 if not (new[0]>old[0] and new[1]<old[1] and v2<v):raise ValueError('Metric gradient does not compact positives and repel negatives')
 with numerical_context(FULL_FP32_POLICY):
  c=d.config(next(r for r in d.rows() if r['arm']=='BOTH'));m=api.build(c,device).train();torch.manual_seed(101);x=torch.randn(256,2,256,device=device);y=torch.arange(128,device=device)%6;rx=torch.tensor([1,3,4,6,8],device=device)[torch.arange(128,device=device)%5];before={n:v.clone() for n,v in m.state_dict().items()};captured=[];original=me.loss
  def capture(*args,**kwargs):
   v=original(*args,**kwargs);captured.append(v[0]);return v
  me.loss=capture
  try:total,metrics=me.objective(m,x,y,rx,6,c)
  finally:me.loss=original
  if len(captured)!=2 or metrics['paired_forwards']!=2 or metrics['extra_forwards_for_metric']:raise ValueError('Actual paired final feature path missing')
  aux=torch.stack(captured).mean();p=dict(m.named_parameters());gw,gq=torch.autograd.grad(aux,[p[W],p[S]],allow_unused=True,retain_graph=True);hw=torch.autograd.grad(total,p[W],retain_graph=True)[0]
  if gw is not None or gq is None or gq.norm()<=0 or not torch.isfinite(gq).all() or hw.norm()<=0:raise ValueError('Final feature loss/defaulthead gradient boundary failed')
  if any(not torch.equal(v,m.state_dict()[n]) for n,v in before.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('No-update check changed model')
  m.eval()
  with torch.no_grad():a=m.forward_identity_only(x[:4])['id_feat_joint'];b=api.api.export(m)(x[:4])
  if not torch.equal(a,b) or sum(p.numel() for p in m.parameters())!=900722:raise ValueError('Actual deployed features or parameter count changed')
  if any(m.get_submodule(me.PREFIX+k)._forward_hooks for k in me.LEVELS):raise ValueError('Temporary hooks escaped source call')
 return dict(status='PASS',device=device,protocol_negative_cases=negative,positive_cosine_before=float(old[0].detach()),positive_cosine_after=float(new[0]),negative_cosine_before=float(old[1].detach()),negative_cosine_after=float(new[1]),geometry_loss_decreased=True,actual_joint160_Sinc_gradient_norm=float(gq.norm()),geometry_direct_defaulthead_gradient=None,defaulthead_joint_gradient_norm=float(hw.norm()),paired_forward_metrics=metrics,actual_feature_export_exact=True,extra_inference_parameters=0,optimizer_updates=0,source_or_target_reads=0,head_fit_updates=0)
if __name__=='__main__':print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'cpu'),allow_nan=False))
