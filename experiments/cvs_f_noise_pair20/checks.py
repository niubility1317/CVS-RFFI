"""Real original forward with synthetic IQ;contract,correlation and matched controls."""
import copy,io,json,math,random,subprocess,sys
from types import SimpleNamespace
import numpy as np
import torch

def run():
 from . import design as d,model as api
 from .channel import TrainingChannel,KEYS
 from .runtime import Runtime
 from experiments.cvs_additive_as20 import model as parent
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 from SSDG import train_ssdg as native
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid source/recipe override accepted')
 d.parent.old.prior.SEED=d.SEED;d.parent.old.SEED=d.SEED;d.parent.SEED=d.SEED
 for row in d.rows():
  c=d.config(row);a=d.make_args(c,'cpu');assert a.seed==2026092701 and (a.epochs,a.label_epochs,a.pseudo_epochs)==(20,13,7) and not a.baseline_ckpt and not a.teacher_ckpt
  for path,value in ((('epochs',),50),(('model_seed',),2026092702),(('target_access',),True),(('checkpoint_sources',),['old']),(('baseline_reuse',),True),(('noise_pair','noise_seed'),9),(('noise_pair','cross_noise_consistency'),True),(('noise_pair','U_or_teacher_views'),True),(('additive','extra_weight'),.5),(('training_mechanism','L_forwards_per_packet'),8),(('fold','held_RX'),6),(('channel','equalization_enabled'),True)):
   b=copy.deepcopy(c);part=b
   for k in path[:-1]:part=part[k]
   part[path[-1]]=value;reject(lambda b=b:d.validate(b))
 g=torch.Generator().manual_seed(20261017503);x=torch.randn(128,2,256,generator=g);saved=x.clone();labels=torch.arange(128)%6;rx=torch.tensor([d.FIT[i%4] for i in range(128)]);day=torch.arange(128)%3+1;domains=torch.tensor([3+3*d.FIT.index(int(r))+int(t)-1 for r,t in zip(rx,day)]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=day,sig_i=torch.zeros(128,dtype=torch.long),base_index=torch.arange(128))
 channels={};views={};physics={}
 for arm in ('INDEP','ANTI'):
  channels[arm]=TrainingChannel();views[arm],physics[arm]=channels[arm].apply(x,meta,step=1111,mode=arm)
 assert torch.equal(views['INDEP'][0],views['ANTI'][0]) and not torch.equal(views['ANTI'][0],views['ANTI'][1]) and not torch.equal(views['INDEP'][1],views['ANTI'][1]) and torch.equal(x,saved)
 assert all(a==b for a,b in zip(physics['INDEP'][::2],physics['ANTI'][::2]))
 for arm in ('INDEP','ANTI'):
  assert channels[arm].calls==1 and channels[arm].physical_presentations==128 and channels[arm].noise_observations==256
  assert all(p['G']==1 and p['EQ'] is False and p['AGC_recomputed_per_view'] for p in physics[arm])
 reject(lambda:TrainingChannel().apply(x,meta,step=0,mode='ANTI'));reject(lambda:TrainingChannel().apply(x,meta,step=1,mode='unknown'));reject(lambda:TrainingChannel().apply(x*float('nan'),meta,step=1,mode='ANTI'));bad=dict(meta,rx_i=torch.ones(128,dtype=torch.long));reject(lambda:TrainingChannel().apply(x,bad,step=1,mode='ANTI'))
 args=d.make_args(d.config(d.rows()[0]),'cpu');cfg=native.build_aug_base_cfg(args);augmentor=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(augmentor,cfg,6,args);torch.manual_seed(17504);random.seed(17504);ordinary=augmentor(x.clone(),labels=labels,no_pa=False)
 snapshots=[];values=[];base_grad=[];states=[];reports=[];original_hook=api.original.hook
 for row in d.rows():
  c=d.config(row);torch.manual_seed(d.SEED);model=api.build(c,'cpu').train();snapshots.append({k:v.detach().clone() for k,v in model.state_dict().items()});assert len(snapshots[-1])==206
  model._feature_l_steps=model._joint_calls=model._legacy_calls=1110
  class Fixed:
   def apply(self,*args,**kw):return views[row['arm']],copy.deepcopy(physics[row['arm']])
  model._noise_runtime=Runtime(io.StringIO(),None if row['arm']=='RAW10' else Fixed());assert copy.deepcopy(model._noise_runtime) is None
  seen=[];original_features=model.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());return original_features(v)
  model.id_backbone.encoder._view_features=trace;graph=[]
  def capture(obj,**kw):
   corr,met=original_hook(obj,**kw);graph.append(corr+kw['identity_loss']);return corr,met
  api.original.hook=capture;torch.manual_seed(17505);random.seed(17505)
  try:
   with model._phase1_labeled_feature_context():out=model(ordinary,return_aux=True)
   ce=torch.nn.functional.cross_entropy(out['tx_logits'],labels,label_smoothing=.01)
   corr,metrics=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=labels,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  finally:api.original.hook=original_hook
  total=ce+corr;values.append(graph[0].detach());base_grad.append(torch.autograd.grad(graph[0],tuple(model.parameters()),allow_unused=True,retain_graph=True));states.append((torch.get_rng_state().clone(),random.getstate()));record=model._joint_details[-1]
  assert len(seen)==10 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(v,x) for v in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(seen[8],seen[9]) and torch.equal(x,saved)
  if row['arm']=='RAW10':assert all(torch.equal(v,x) for v in seen[6:]) and record['extra_As_packets']==0
  else:assert torch.equal(seen[6],views[row['arm']][0]) and torch.equal(seen[8],views[row['arm']][1]) and record['extra_As_packets']==256
  assert model._joint_L_packets==1280 and model._additive_L_packets==512 and model._noise_calls==1 and len(record['supcon_details'])==4 and model._supcon_applied==1
  assert math.isclose(float(total.detach()),record['base_total_L']+.25*.5*(record['extra_pair0_loss']+record['extra_pair1_loss']),abs_tol=1e-6) and record['cross_noise_KL'] is False
  head='id_backbone.encoder.core.id_backbone.cls_head.weight';before=model.get_parameter(head).detach().clone();optimizer=build(model,list(model.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total.backward();clip_physical(optimizer,model.parameters(),5.);optimizer.step();assert not torch.equal(before,model.get_parameter(head))
  model.id_backbone.encoder._view_features=original_features;model.eval()
  with torch.no_grad(),fresh_parameter_caches(model):
   f=model.id_backbone.encoder.features(x[:2]);exported=api.export(model).eval();assert f.shape==(2,160) and torch.equal(f,exported(x[:2])) and model.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  reports.append(dict(arm=row['arm'],L_forwards=10,common_inputs_exact=True,each_RD_same_IQ=True,no_cross_noise_KL=True,default_head_joint_update=True,export160_exact=True,new_parameters=0))
 assert all(all(torch.equal(v,s[k]) for k,v in snapshots[0].items()) for s in snapshots[1:]) and all(torch.equal(values[0],v) for v in values[1:])
 assert all(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(base_grad[0],grad)) for grad in base_grad[1:])
 assert all(torch.equal(states[0][0],state[0]) and states[0][1]==state[1] for state in states[1:])
 # Exact fixed threshold boundary and one-sided fail,without data or model selection.
 results={}
 for arm in d.ARMS:
  a=.8 if arm!='ANTI' else .8;practical=.4 if arm=='RAW10' else (.43 if arm=='INDEP' else .45);low=.3 if arm=='RAW10' else (.34 if arm=='INDEP' else .35)
  results[arm]=dict(actual_steps=4440,L_forward_packets=5683200,extra_L_forward_packets=2273280,noise_pair_steps=4440,supcon_applied=3330,supcon_layer_forward_terms=13320,final_base_L_objective=.5,noise_pair_physical_presentations=568320,As_training_observations=0 if arm=='RAW10' else 1136640,teacher_noise_pair_calls=0,head_fit_updates=0,U_truth_read=False,target_access=False,checkpoint_sources=[],held_RX=1,fit_RX=list(d.FIT),seen_V=dict(source_val_accuracy=.98),views={v:dict(accuracy=a if v=='clean' else (low if v==d.SCENES[-1] else practical),macro_F1=.78) for v in ('clean',*d.SCENES)})
 assert d.qualify(results)['qualified'];bad=copy.deepcopy(results);bad['ANTI']['views']['clean']['accuracy']-=.00001;assert not d.qualify(bad)['qualified']
 direct=subprocess.run([sys.executable,str(__import__('pathlib').Path(__file__).with_name('runner.py')),'--epochs','50'],capture_output=True,text=True);assert direct.returncode and 'No data/noise/loss/epoch/seed overrides' in direct.stderr
 return dict(status='PASS',negative_cases=negative,rows=reports,identical_initial_states=True,common_loss_and_all_parameter_gradients_exact=True,CPU_Python_RNG_endstates_equal=True,noise0_bitwise_equal_between_As_arms=True,physics_exact_and_AGC_perview=True,scientific_threshold_boundary_PASS=True,synthetic_NN_packets=3858,synthetic_optimizer_updates=3,real_IQ_reads=0,real_checkpoint_loads=0,target_access=False)
