"""Synthetic protocol and numerical checks;0 real IQ/checkpoint/target."""
import copy,io,math,random
from types import SimpleNamespace
import torch

def run():
 from . import design as d,model as api
 from .pairing import Bank,Runtime,KEYS
 from .iqmix import coherent_mix
 from experiments.cvs_additive_as20 import model as parent
 from experiments.cvs_crossrx_satmix20 import supcon
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 from SSDG import train_ssdg as native
 torch.set_num_threads(2);neg=0
 def reject(fn):
  nonlocal neg
  try:fn()
  except (ValueError,TypeError):neg+=1
  else:raise AssertionError('Illegal override/input accepted')
 for k,v in [('epochs',40),('model_seed',2026092701),('target_access',True),('checkpoint_sources',['old']),('model_initialization','pretrained')]:
  c=copy.deepcopy(d.config(d.rows()[0]));c[k]=v;reject(lambda:d.validate(c))
 for k,v in [('mixture_fraction',.5),('max_relative_L2_change',.2),('reference_window',[0,256]),('U_or_V_donors',True)]:
  c=copy.deepcopy(d.config(d.rows()[0]));c['coherent_mix'][k]=v;reject(lambda:d.validate(c))
 g=torch.Generator().manual_seed(20261016962);allx=torch.randn(5040,2,256,generator=g);entries=[]
 for i in range(5040):
  tx=i//840;r=d.FIT[(i//210)%4];day=(i//70)%3+1;meta=dict(eq_i=1,rx_i=r,day_i=day,sig_i=0,base_index=i);domain=3+d.FIT.index(r)*3+day-1;entries.append((allx[i],tx,domain,meta))
 bank=Bank(entries);indices=(torch.arange(128)*37)%5040;x=torch.stack([entries[int(i)][0] for i in indices]);labels=torch.tensor([entries[int(i)][1] for i in indices]);domains=torch.tensor([entries[int(i)][2] for i in indices]);meta={k:torch.tensor([entries[int(i)][3][k] for i in indices]) for k in KEYS};saved=x.clone()
 for mode in d.ARMS:
  b=Bank(entries);p,dd,rec=b.select(labels,meta,1,mode,x);v,obs=coherent_mix(x,p);assert len(rec)==len(obs)==128 and not torch.equal(v,x) and max(o['relative_L2'] for o in obs)<=.10001 and all(abs(o['norm_ratio']-1)<1e-5 for o in obs)
  assert all(z['primary']!=z['partner'] and z['primary'][2]==z['partner'][2] and ((z['primary'][1]==z['partner'][1])==(mode=='SAME_RX')) for z in rec)
  v2,_=coherent_mix(x*7,p*7);assert torch.allclose(v2,v*7,atol=3e-6,rtol=1e-6)
 phase=torch.stack((-2*x[:,1],2*x[:,0]),1);same,ob=coherent_mix(x,phase);assert torch.allclose(same,x,atol=1e-6,rtol=1e-6)
 reject(lambda:coherent_mix(x[:,:1],x));reject(lambda:coherent_mix(x*float('nan'),x));reject(lambda:coherent_mix(x.double(),x));reject(lambda:coherent_mix(torch.zeros_like(x),x))
 bad=entries.copy();bad[0]=(bad[0][0],-1,bad[0][2],bad[0][3]);reject(lambda:Bank(bad))
 bad=entries.copy();bad[0]=(bad[0][0],0,0,dict(bad[0][3],rx_i=1));reject(lambda:Bank(bad))
 reject(lambda:Bank(entries[:-1]));reject(lambda:Bank(entries).select(labels-9,meta,1,'CROSS_RX',x));reject(lambda:Bank(entries).select(labels,meta,2,'CROSS_RX',x));reject(lambda:Bank(entries).select(labels,meta,1,'unknown',x));reject(lambda:Bank(entries).select(labels,meta,1,'SAME_RX',x+1))
 args=d.make_args(d.config(d.rows()[0]),'cpu');base_cfg=native.build_aug_base_cfg(args);aug=native.make_augmentor(base_cfg);native.configure_augmentor_for_epoch(aug,base_cfg,6,args);random.seed(962);torch.manual_seed(962);ordinary=aug(x.clone(),labels=labels,no_pa=False)
 reports=[];initials=[];ends=[];common_grads=[];common_values=[]
 original_hook=api.original.hook
 for arm in ('reused_D_raw',*d.ARMS):
  torch.manual_seed(d.SEED);m=(parent.build(d.template(),'cpu') if arm=='reused_D_raw' else api.build(d.config(next(r for r in d.rows() if r['arm']==arm)),'cpu')).train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
  seen=[];view=m.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());return view(v)
  m.id_backbone.encoder._view_features=trace
  if arm!='reused_D_raw':
   b=Bank(entries);b.calls=1110;m._mix_runtime=Runtime(b,io.StringIO());assert copy.deepcopy(m._mix_runtime) is None
  graph=[]
  def capture(obj,**kw):
   corr,met=original_hook(obj,**kw);graph.append(corr+kw['identity_loss']);return corr,met
  api.original.hook=capture
  torch.manual_seed(963);random.seed(963)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
  corr,met=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+corr;api.original.hook=original_hook
  assert len(graph)==1;common_values.append(graph[0].detach());common_grads.append(torch.autograd.grad(graph[0],tuple(m.parameters()),allow_unused=True,retain_graph=True));ends.append((torch.get_rng_state().clone(),random.getstate()))
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(v,x) for v in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(x,saved)
  r=m._joint_details[-1];assert r['extra_As_packets']==m._joint_As_packets==0 and len(r['supcon_details'])==4 and m._supcon_applied==1 and m._joint_L_packets==1024 and m._additive_L_packets==256
  if arm!='reused_D_raw':assert r['coherent_packets']==128 and m._mix_packets==128 and not torch.equal(seen[6],x)
  else:assert torch.equal(seen[6],x)
  assert math.isclose(float(total.detach()),r['base_total_L']+.25*r['extra_pair_loss'],abs_tol=1e-6)
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';old=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total.backward();clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(old,m.get_parameter(W))
  m.id_backbone.encoder._view_features=view;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(x[:2]);e=api.export(m).eval();assert f.shape==(2,160) and torch.equal(f,e(x[:2]));assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  reports.append(dict(arm=arm,L_forwards=8,common_inputs_exact=True,fourth_same_IQ=True,default_head_joint_update=True,export160_exact=True,new_parameters=0,total_pair_weight=r['total_pair_weight']))
 assert all(all(torch.equal(v,s[k]) for k,v in initials[0].items()) for s in initials[1:]) and all(torch.equal(common_values[0],z) for z in common_values[1:])
 assert all(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(common_grads[0],g)) for g in common_grads[1:])
 assert all(torch.equal(ends[0][0],v[0]) and ends[0][1]==v[1] for v in ends[1:])
 return dict(status='PASS',negative_cases=neg,rows=reports,identical_initial_states=True,common_loss_and_all_parameter_gradients_exact=True,CPU_Python_RNG_endstates_equal=True,no_training_As=True,source_bank_only=True,synthetic_NN_packets=3090,synthetic_optimizer_updates=3,synthetic_operator_packets=128*7,real_IQ_reads=0,real_checkpoint_loads=0,target_access=False)
