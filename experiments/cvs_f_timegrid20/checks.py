"""Focused synthetic protocol/operator/actual default-head training checks."""
import copy,io,math,random
import torch

def run():
 from . import design as d,model as api
 from .iq import Runtime,transform
 from experiments.cvs_additive_as20 import model as parent
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 from SSDG import train_ssdg as native
 torch.set_num_threads(2);neg=0
 def reject(fn):
  nonlocal neg
  try:fn()
  except (ValueError,TypeError,KeyError):neg+=1
  else:raise AssertionError('Illegal input accepted')
 for k,v in [('epochs',50),('model_seed',2026092702),('target_access',True),('checkpoint_sources',['old']),('model_initialization','pretrained'),('separate_head_training',True)]:
  c=copy.deepcopy(d.config(d.rows()[0]));c[k]=v;reject(lambda:d.validate(c))
 for k,v in [('rho_max',.01),('rho_min',0.),('U_or_V',True),('normalization','RMS'),('session','TX')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c['timegrid'][k]=v;reject(lambda:d.validate(c))
 g=torch.Generator().manual_seed(17702);x=torch.randn(128,2,256,generator=g);saved=x.clone();labels=torch.arange(128)%6;rx=[d.FIT[(i//3)%4] for i in range(128)];day=[i%3+1 for i in range(128)];domains=torch.tensor([3+d.FIT.index(r)*3+t-1 for r,t in zip(rx,day)]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.tensor(rx),day_i=torch.tensor(day),sig_i=torch.arange(128),base_index=torch.arange(128))
 assert torch.equal(transform(x,[0.]*128),x)
 reject(lambda:transform(x[:,:1],[d.RHO_MIN]*128));reject(lambda:transform(x*float('nan'),[d.RHO_MIN]*128));reject(lambda:transform(x,[-1.]*128));reject(lambda:transform(x,[.01]*128));reject(lambda:transform(x,[d.RHO_MIN]));reject(lambda:transform(x,[float('nan')]*128))
 identity=torch.eye(256,dtype=torch.double);matrix_proofs=[]
 for rho in (d.RHO_MIN,d.RHO_MAX):
  basis=torch.stack((identity,identity),1);mapped=transform(basis,[rho]*256)[:,0].T;E=mapped-identity;bound=math.sqrt(float(E.abs().sum(0).max()*E.abs().sum(1).max()));diagonal=mapped.diag();assert bool((diagonal>(mapped.abs().sum(1)-diagonal.abs())).all()) and bound<=d.L2_BOUND+1e-12 and bool((mapped>=0).all()) and torch.allclose(mapped.sum(1),torch.ones(256,dtype=torch.double),atol=1e-12)
  original=torch.randn(256,4,generator=g,dtype=torch.double);back=torch.linalg.solve(mapped,mapped@original);assert float((back-original).abs().max())<1e-12
  matrix_proofs.append(dict(rho=rho,inverse_max_error=float((back-original).abs().max()),L2_operator_bound=bound,diagonal_min=float(diagonal.min()),nonnegative_convex_rows=True))
 zero=torch.zeros_like(x);assert not bool(transform(zero,[d.RHO_MAX]*128).any())
 t=Runtime(io.StringIO());cpu=torch.get_rng_state().clone();prng=random.getstate();v=t.apply(x,meta,labels,1,1);t2=Runtime(io.StringIO());v2=t2.apply(x,meta,labels.roll(1),1,1);assert torch.equal(v,v2) and t.sessions==t2.sessions and not torch.equal(v,x) and torch.equal(cpu,torch.get_rng_state()) and prng==random.getstate() and torch.equal(saved,x) and copy.deepcopy(t) is None
 assert set(t.sessions)=={(1,1,r,dd) for r,dd in zip(rx,day)}
 changed=t.apply(x,meta,labels,2,2);assert t.sessions[(1,1,3,1)]!=t.sessions[(2,1,3,1)]
 for change in [dict(rx_i=torch.ones(128,dtype=torch.long)),dict(day_i=torch.zeros(128,dtype=torch.long)),dict(eq_i=torch.zeros(128,dtype=torch.long))]:reject(lambda:Runtime(io.StringIO()).apply(x,dict(meta,**change),labels,1,1))
 reject(lambda:Runtime(io.StringIO()).apply(x,meta,labels-9,1,1));reject(lambda:Runtime(io.StringIO()).apply(x,meta,labels,1,2));reject(lambda:Runtime(io.StringIO()).apply(x,meta,labels,0,1))
 args=d.make_args(d.config(d.rows()[0]),'cpu');basecfg=native.build_aug_base_cfg(args);aug=native.make_augmentor(basecfg);native.configure_augmentor_for_epoch(aug,basecfg,6,args);random.seed(17703);torch.manual_seed(17703);ordinary=aug(x.clone(),labels=labels,no_pa=False)
 initial=[];grads=[];losses=[];ends=[];reports=[];orig=api.original.hook
 for arm in ('D_raw','TIMEGRID'):
  torch.manual_seed(d.SEED);m=(parent.build(d.template(),'cpu') if arm=='D_raw' else api.build(d.config(d.rows()[0]),'cpu')).train();initial.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
  if arm=='TIMEGRID':m._timegrid_runtime=Runtime(io.StringIO());m._timegrid_runtime.calls=1110
  seen=[];native_view=m.id_backbone.encoder._view_features
  def trace(value):seen.append(value.detach().clone());return native_view(value)
  m.id_backbone.encoder._view_features=trace;graph=[]
  def capture(obj,**kw):
   corr,met=orig(obj,**kw);graph.append(corr+kw['identity_loss']);return corr,met
  api.original.hook=capture;torch.manual_seed(17704);random.seed(17704)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01);corr,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+corr;api.original.hook=orig
  assert len(graph)==1;losses.append(graph[0].detach());grads.append(torch.autograd.grad(graph[0],tuple(m.parameters()),allow_unused=True,retain_graph=True));ends.append((torch.get_rng_state().clone(),random.getstate()))
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(saved,x)
  if arm=='TIMEGRID':
   rho=[m._timegrid_runtime.sessions[(6,1,r,dd)] for r,dd in zip(rx,day)];assert torch.equal(seen[6],transform(x,rho)) and not torch.equal(seen[6],x)
  else:assert torch.equal(seen[6],x)
  r=m._joint_details[-1];assert r['extra_As_packets']==m._joint_As_packets==0 and m._joint_L_packets==1024 and m._additive_L_packets==256 and m._supcon_applied==1 and math.isclose(float(total.detach()),r['base_total_L']+.25*r['extra_pair_loss'],abs_tol=1e-6)
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total.backward();clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W))
  m.id_backbone.encoder._view_features=native_view;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert f.shape==(2,160) and torch.equal(f,export(x[:2]));assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  reports.append(dict(arm=arm,L_forwards=8,common_inputs_exact=True,fourth_same_IQ=True,default_head_joint_update=True,export160_exact=True,new_parameters=0))
 assert all(torch.equal(a,initial[1][k]) for k,a in initial[0].items()) and torch.equal(losses[0],losses[1]) and all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(grads[0],grads[1])) and torch.equal(ends[0][0],ends[1][0]) and ends[0][1]==ends[1][1]
 return dict(status='PASS',negative_cases=neg,matrix_proofs=matrix_proofs,rows=reports,all_initial_states_exact=True,common_loss_and_all_parameter_gradients_exact=True,RNG_endstates_equal=True,TX_not_used_in_session_rho=True,session_refresh=True,no_padding_or_RMS=True,synthetic_NN_packets=2060,synthetic_optimizer_updates=2,real_IQ_reads=0,real_checkpoint_loads=0,target_access=False)
