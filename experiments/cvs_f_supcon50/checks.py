"""Actual F paired-flow, no data reads, no fitted head or optimizer update."""
import copy,json,sys,torch
from torch.nn import functional as F
from . import design as d,model as api,mechanism as me
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_equivariant_identity.precision import numerical_context
W='id_backbone.encoder.core.id_backbone.cls_head.weight'
S='id_backbone.encoder.core.id_backbone.sinc.low_coordinate'
def protocol():
 c=d.config(d.training_rows()[0]);count=0
 for key,value in (('epochs',51),('label_epochs',32),('test_views',['practical_high']),('separate_head_training',True),('output_root','foreign_or_target_root'),('baseline_ckpt','oldweight')):
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:count+=1
  else:raise ValueError('Forbidden source/target/defaulthead recipe accepted')
 try:d.make_args(d.config(d.rows()[0]),'cpu')
 except ValueError:count+=1
 else:raise ValueError('Existing F reference retrained')
 return count
def fixture(m,epoch,device,seed=7061):
 torch.manual_seed(seed);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;domain=torch.arange(128,device=device)%15;rx=torch.tensor(me.RX_BY_DOMAIN,device=device)[domain];prior=(epoch-1)*222;m._feature_l_steps=prior;m._legacy_calls=prior;m._legacy_extra_packets=prior*128
 if hasattr(m,'_supcon_calls'):m._supcon_calls=prior
 with fresh_parameter_caches(m):
  with m._phase1_labeled_feature_context():first=m(x,y_tx=y,return_aux=True,domain_labels=domain,update_crra_support=False)
  ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01);correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=first,labels=y,extra={'rx_i':rx},domains=domain,optimizer_step=prior+1,identity_loss=ce,labeled_count=128)
 return ce+correction,metrics
def run(device='cpu'):
 torch.set_num_threads(2);negative=protocol();objective=[];flows=[]
 torch.manual_seed(55);z=torch.randn(12,20,dtype=torch.float64,requires_grad=True,device=device);y=torch.arange(12,device=device)%6;rx=torch.tensor([1]*6+[3]*6,device=device)
 for mode in ('ALL_TX','CROSS_RX'):
  value,info=me.loss(z,y,rx,mode);pos,den,valid=me.masks(y,rx,mode);unit=F.normalize(z,dim=1,eps=1e-4);logits=unit@unit.T/.2;expected=(torch.logsumexp(logits.masked_fill(~den,-torch.inf),1)-(logits*pos).sum(1)/pos.sum(1)-pos.sum(1).to(z.dtype).log()).mean()
  if not torch.allclose(value,expected,atol=1e-12,rtol=1e-12):raise ValueError('Historical SupCon formula changed')
  if mode=='CROSS_RX' and bool((den&((y[:,None]==y[None,:])&(rx[:,None]==rx[None,:]))).any()):raise ValueError('SameTX sameRX became a negative')
  zero,_=me.loss(z,torch.zeros_like(y),rx,mode);g=torch.autograd.grad(zero,z,retain_graph=True)[0]
  if float(zero)!=0 or not torch.isfinite(g).all() or torch.count_nonzero(g):raise ValueError('No valid anchor must give finite zero gradient')
  objective.append(dict(mode=mode,value=float(value.detach()),valid=info['valid_anchors']))
 for bady,badrx,badmode in ((y-7,rx,'ALL_TX'),(y,rx*0+2,'CROSS_RX'),(y,rx,'UNKNOWN')):
  try:me.loss(z,bady,badrx,badmode)
  except ValueError:negative+=1
  else:raise ValueError('Illegal labels/RX/mode accepted')
 with numerical_context(d.FULL_FP32_POLICY):
  base=api.build(d.config(d.rows()[0]),device).train();new=api.build(d.config(d.training_rows()[0]),device).train()
  if any(not torch.equal(v,new.state_dict()[n]) for n,v in base.state_dict().items()):raise ValueError('Added objective changed F initialization')
  a,_=fixture(base,1,device);b,_=fixture(new,1,device);diff=abs(float(a.detach()-b.detach()))
  if diff>(0. if device=='cpu' else 1e-5):raise ValueError('Warmup zero objective changed original RDrop value')
  for mode in ('ALL_TX','CROSS_RX'):
   c=d.config(next(r for r in d.training_rows() if r['arm']==mode));m=api.build(c,device).train();saved={n:v.clone() for n,v in m.state_dict().items()};captured=[];original=me.penalty
   def record(*args,**kwargs):
    value=original(*args,**kwargs);captured.append(value[0]);return value
   me.penalty=record
   try:total,metrics=fixture(m,6,device)
   finally:me.penalty=original
   if len(captured)!=2 or m._supcon_pending is not None or m._supcon_handles or m._legacy_extra_packets!=(5*222+1)*128 or m._supcon_applied!=1:raise ValueError('SupCon did not reuse exactly the two original L forwards')
   aux=torch.stack(captured).mean();named=dict(m.named_parameters());aw,asinc=torch.autograd.grad(aux,[named[W],named[S]],retain_graph=True,allow_unused=True)
   if aw is not None or asinc is None or not torch.isfinite(asinc).all() or float(asinc.norm())<=0:raise ValueError('Feature loss must reach actual Sinc without direct classW gradient')
   mainW=torch.autograd.grad(total,named[W],retain_graph=True)[0]
   if not torch.isfinite(mainW).all() or float(mainW.norm())<=0 or any(p.grad is not None for p in m.parameters()):raise ValueError('Default head not joint or unexpected .grad accumulation')
   if any(not torch.equal(v,m.state_dict()[n]) for n,v in saved.items()):raise ValueError('Check changed model state without optimizer')
   teacher=copy.deepcopy(m).eval();before=teacher._supcon_calls
   with torch.no_grad(),fresh_parameter_caches(teacher):v=teacher.forward_identity_only(torch.zeros(4,2,256,device=device))
   if teacher._supcon_calls!=before or teacher._supcon_pending is not None or teacher._supcon_handles:raise ValueError('EMA/eval received feature objective')
   m.eval();x=torch.randn(4,2,256,device=device)
   with torch.no_grad(),fresh_parameter_caches(m):f=m.forward_identity_only(x)['id_feat_joint'];exported=api.export(m)(x)
   with torch.no_grad():named[W].zero_()
   with torch.no_grad(),fresh_parameter_caches(m):zeroW=m.forward_identity_only(x)['id_feat_joint']
   if not torch.equal(f,exported) or not torch.equal(f,zeroW) or sum(p.numel() for p in m.parameters())!=900722 or len(m.id_backbone.encoder.sequence_pool.views)!=3:raise ValueError('Actual F feature export/defaultW boundary changed')
   flows.append(dict(mode=mode,actual_calls=m._supcon_calls,aux_direct_W_gradient=None,aux_Sinc_gradient_norm=float(asinc.norm()),main_W_gradient_norm=float(mainW.norm()),feature_export_exact=True,EMA_eval_untouched=True,temporary_hooks_remaining=0,real_paired_details=m._supcon_records[-1]['details']))
 return dict(status='PASS',device=device,protocol_negative_cases=negative,objective=objective,warmup_loss_difference=diff,actual_F_flows=flows,source_or_target_IQ_reads=0,head_fit_updates=0,optimizer_updates=0,network_parameters=900722,default_head_shape=[6,160],source_qualification_gate=False)
if __name__=='__main__':print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'cpu'),allow_nan=False))
