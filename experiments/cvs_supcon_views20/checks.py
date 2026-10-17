"""Focused real-F synthetic execution and fixed protocol/scientific counterexamples."""
import copy,json,math,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from experiments.cvs_scene_cycle20 import model as legacy
 from experiments.cvs_crossrx_satmix20 import supcon
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=0
 def reject(f):
  nonlocal negative
  try:f()
  except (ValueError,KeyError):negative+=1
  else:raise AssertionError('Invalid candidate accepted')
 for key,value in [('epochs',40),('model_seed',2026092702),('model_initialization','parent'),('classifier','replacement'),('test_views',['clean'])]:
  c=copy.deepcopy(d.config(d.rows()[1]));c[key]=value;reject(lambda c=c:d.validate(c))
 for sec,key,value in [('supcon_view_routing','satellite_positions',32),('supcon_view_routing','selection_seed',1),('supcon_view_routing','policy','satellite'),('supcon_view_routing','temperature',.1),('supcon_view_routing','new_loss_terms',1),('supcon_view_routing','extra_NN_forwards',1),('training_mechanism','U_rule','changed'),('training_mechanism','Ag',True),('additive','extra_weight',.1),('scene_schedule','mode','balanced')]:
  c=copy.deepcopy(d.config(d.rows()[1]));c[sec][key]=value;reject(lambda c=c:d.validate(c))
 for pol,st in [('bad',1),('mixed',0),('mixed',4441)]:reject(lambda pol=pol,st=st:api.positions(pol,st))
 a0=d.make_args(d.config(d.rows()[0]),'cpu');args1=vars(d.make_args(d.config(d.rows()[1]),'cpu'));args0=vars(a0);assert {k for k,v in args0.items() if v!=args1.get(k) and not(isinstance(v,float) and isinstance(args1.get(k),float) and math.isnan(v) and math.isnan(args1[k]))}=={'candidate_id','output_dir'}
 x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(202610176011));saved=x.clone();y=torch.arange(128)%6;domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128));base=native.build_aug_base_cfg(a0);aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,6,a0);random.seed(202610176012);torch.manual_seed(202610176012);ordinary=aug(x.clone(),labels=y,no_pa=False)
 def execute(m,inspect=False):
  m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];maps=[];handles=[];orig=m.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());maps.append({});return orig(v)
  m.id_backbone.encoder._view_features=trace
  if inspect:
   for level in supcon.LEVELS:
    def capture(mod,args,out,level=level):maps[len(seen)-1][level]=out
    handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(capture))
  torch.manual_seed(202610176013);random.seed(202610176013)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  for h in handles:h.remove()
  m.id_backbone.encoder._view_features=orig
  return ce+correction,seen,maps,(torch.get_rng_state().clone(),random.getstate())
 torch.manual_seed(202610176014);old=legacy.build(d.legacy_config(),'cpu').train();old_loss,old_inputs,_,old_rng=execute(old);old_loss.backward()
 torch.manual_seed(202610176014);control=api.build(d.config(d.rows()[0]),'cpu').train();control._route_config=copy.deepcopy(control._route_config);control._route_config['supcon_view_routing']['policy']='ordinary';control_loss,control_inputs,_,control_rng=execute(control);control_loss.backward()
 assert torch.equal(old_loss.detach(),control_loss.detach()) and all(torch.equal(a,b) for a,b in zip(old_inputs,control_inputs)) and set(old.state_dict())==set(control.state_dict())
 compared=0
 for (name,p),(other,q) in zip(old.named_parameters(),control.named_parameters()):
  assert name==other and ((p.grad is None)==(q.grad is None))
  if p.grad is not None:assert torch.allclose(p.grad,q.grad,rtol=1e-5,atol=2e-6),name;compared+=1
 ends=[old_rng,control_rng];initial=[];reports=[];manual_aux=0
 for row in d.rows():
  torch.manual_seed(202610176014);m=api.build(d.config(row),'cpu').train();initial.append({k:v.detach().clone() for k,v in m.state_dict().items()});loss,seen,maps,rng=execute(m,True);ends.append(rng);r=m._joint_details[-1];sel=api.positions(d.config(row)['supcon_view_routing']['policy'],1111)
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(seen[6],old_inputs[6]) and torch.equal(x,saved)
  values=[]
  for k in range(2):
   chosen=maps[6+k] if row['arm']=='Q_sat' else api.route_maps(maps[k],maps[6+k],sel);value,_=supcon.penalty(chosen,y,rx,'CROSS_RX');values.append(value)
  assert math.isclose(r['supcon'],float(torch.stack(values).mean().detach()),abs_tol=1e-6);manual_aux+=1
  expected=.25*(r['ordinary_CE']+r['ordinary_KL'])+.5*(r['raw_CE']+r['raw_KL'])+.25*(r['s_slot_CE']+r['s_slot_KL'])+r['supcon_weight']*r['supcon']+.25*r['extra_pair_loss'];assert math.isclose(float(loss.detach()),expected,abs_tol=1e-6)
  assert r['route_selected_As_positions']==sel and len(sel)==(128 if row['arm']=='Q_sat' else 64) and len(r['supcon_details'])==4 and m._route_handles==[] and m._supcon_handles==[] and m._supcon_pending is None and m._joint_L_packets==1024 and m._additive_L_packets==256 and m._joint_As_packets==128 and sum(m._joint_coverage.values())==128
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);loss.backward();assert m.get_parameter(W).grad is not None;clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   f=m.id_backbone.encoder.features(x[:2]);ex=api.export(m).eval();assert f.shape==(2,160) and torch.equal(f,ex(x[:2]));assert m.forward_identity_only(x[:2])['tx_logits'].shape==(2,6)
  reports.append(dict(arm=row['arm'],same_IQ_RDrop=True,L_forwards=8,As_packets=128,SupCon_terms=4,As_positions=len(sel),default_head_updated=True,export160_exact=True))
 assert all(torch.equal(ends[0][0],z[0]) and ends[0][1]==z[1] for z in ends[1:]) and all(torch.equal(initial[0][k],initial[1][k]) for k in initial[0])
 fake={a:dict(actual_steps=4440,L_forward_packets=4546560,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,head_fit_updates=0,target_access=False,As_training_observations=0 if a=='Q0' else 568320,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=(.85 if a=='Q_mix' else .84) if s=='clean' else .42 if a=='Q_mix' else .40 if a in ('Q_ord','Q_sat') else .30,macro_F1=.85 if a=='Q_mix' else .84) for s in ('clean',*d.SCENES)}) for a in d.ARMS};assert d.qualify(fake)['qualified']
 for kind in ('clean','F1','low','lowF1','mid','suburban','V','budget','As','loss'):
  q=copy.deepcopy(fake);r=q['Q_mix']
  if kind=='clean':r['views']['clean']['accuracy']=.83
  elif kind=='F1':r['views']['clean']['macro_F1']=.83
  elif kind=='low':r['views'][d.SCENES[-1]]['accuracy']=.405
  elif kind=='lowF1':r['views'][d.SCENES[-1]]['macro_F1']=.83
  elif kind=='mid':r['views'][d.SCENES[-2]]['accuracy']=.39
  elif kind=='suburban':r['views'][d.SCENES[0]]['accuracy']=.39
  elif kind=='V':r['seen_V']['source_val_accuracy']=.89
  elif kind=='budget':r['actual_steps']=4439
  elif kind=='As':r['As_training_observations']=568319
  else:r['final_base_L_objective']=.6
  assert not d.qualify(q)['qualified'];negative+=1
 bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_supcon_views20/runner.py'),'--unregistered'],capture_output=True,text=True);assert bad.returncode and 'No budget/data/operator' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;negative+=1
 return dict(status='PASS',protocol_scientific_negatives=negative,rows=reports,legacy_ordinary_route_loss_exact=True,all_parameter_gradients_compared=compared,all_gradients_close=True,same_fullAs_IQ_and_CPU_Python_RNG=True,same_scratch_state=True,no_new_model_parameters=True,synthetic_raw_packets=128,ordinary_generated=128,fullpractical_generated=512,synthetic_train_NN_packets=4096,synthetic_eval_NN_packets=12,synthetic_NN_packets=4108,synthetic_optimizer_updates=2,manual_aux_recomputations=manual_aux,real_IQ_reads=0,checkpoint_loads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 v=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',v);print({k:x for k,x in v.items() if k!='rows'})
