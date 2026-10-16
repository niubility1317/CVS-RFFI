"""Focused protocol negatives and actual original-network/defaulthead checks."""
import copy,math,random,subprocess,sys
from pathlib import Path
import numpy as np
import torch

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from .channel import SceneChannel
 from experiments.cvs_crossrx_satmix20.channel import PairedChannel
 from experiments.cvs_additive_as20 import design as legacy_design,model as legacy_api
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 from experiments.cvs_crossrx_satmix20 import supcon
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError):negative+=1
  else:raise AssertionError('Forbidden config/input accepted')
 for key,value in [('epochs',40),('model_seed',2026092702),('model_initialization','pretrained'),('test_views',['clean']),('classifier','replacement')]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[key]=value;reject(lambda:d.validate(c))
 for section,key,value in [('scene_schedule','cycle_length',5),('scene_schedule','permutation_seed',1),('scene_schedule','mode','iid'),('scene_schedule','scenes',list(reversed(d.SCENES))),('training_mechanism','U_rule','changed'),('training_mechanism','Ag',True),('training_mechanism','L_forwards_per_packet',6),('additive','extra_weight',.1)]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[section][key]=value;reject(lambda:d.validate(c))
 x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(202610175011));saved=x.clone();y=torch.arange(128)%6;domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128));namespace='train/epoch6/step1111'
 reference=PairedChannel();iid=SceneChannel('iid');cpu=torch.get_rng_state();py=random.getstate();npr=np.random.get_state();v,info=iid.apply(x,meta,namespace=namespace);rv,ri=reference.apply(x,meta,namespace=namespace)
 assert torch.equal(v,rv) and info==ri and torch.equal(x,saved) and torch.equal(cpu,torch.get_rng_state()) and py==random.getstate() and all(np.array_equal(a,b) for a,b in zip(npr,np.random.get_state()))
 bal=SceneChannel('balanced');m2={k:t[:2] for k,t in meta.items()};bv,bi=bal.apply(x[:2],m2,namespace='source_eval/RX1/final20',scene=d.SCENES[-1]);pv,pi=reference.apply(x[:2],m2,namespace='source_eval/RX1/final20',scene=d.SCENES[-1]);assert torch.equal(bv,pv) and bi==pi and not bal.visits and not bal.trace
 schedule=SceneChannel('balanced');choices={}
 for visit in range(113):
  for pid in [(3,1,0,i,i) for i in range(17)]:
   selected,before=schedule.choose(pid,'train/epoch1/step'+str(visit+1));assert before==visit;choices.setdefault(pid,[]).append(selected);assert max(schedule.counts[pid])-min(schedule.counts[pid])<=1
 for values in choices.values():assert all(sorted(values[i:i+6])==list(range(6)) for i in range(0,108,6))
 for mode in ('bad',):reject(lambda:SceneChannel(mode))
 for a,b,ns,scene in [(x[:,:,:128],meta,namespace,None),(x*float('nan'),meta,namespace,None),(x,{},namespace,None),(x,dict(meta,rx_i=torch.ones(128,dtype=torch.long)),namespace,None),(x,meta,'target_eval/unknown',None),(x,meta,namespace,d.SCENES[0])]:reject(lambda:SceneChannel('iid').apply(a,b,namespace=ns,scene=scene))
 a=d.make_args(d.config(d.rows()[0]),'cpu');base=native.build_aug_base_cfg(a);aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,6,a);random.seed(202610175012);torch.manual_seed(202610175012);ordinary=aug(x.clone(),labels=y,no_pa=False)
 def objective(m):
  torch.manual_seed(202610175013);random.seed(202610175013)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);return ce+correction
 reports=[];initials=[];ends=[];manual_physics=0
 for row in d.rows():
  torch.manual_seed(202610175014);m=api.build(d.config(row),'cpu').train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];orig=m.id_backbone.encoder._view_features
  def trace(t):seen.append(t.detach().clone());return orig(t)
  m.id_backbone.encoder._view_features=trace;legacy=None
  if row['arm']=='B00':
   torch.manual_seed(202610175014);lc=legacy_design.config(next(r for r in legacy_design.rows() if r['arm']=='D_raw'));legacy=legacy_api.build(lc,'cpu').train();legacy._feature_l_steps=legacy._joint_calls=legacy._legacy_calls=1110;legacy_loss=objective(legacy);legacy_loss.backward()
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total=objective(m);ends.append((torch.get_rng_state().clone(),random.getstate()));record=m._joint_details[-1];hasAs=row['arm']!='B00'
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(x,saved)
  if row['arm']=='B00':assert torch.equal(seen[6],x)
  if row['arm']=='C_IID':assert torch.equal(seen[6],v)
  if row['arm']=='B01':
   for i in range(4):
    o=m._joint_channel.trace[0]['scene_indices'][i];name=d.SCENES[o];pid=tuple(int(meta[k][i]) for k in ('rx_i','day_i','eq_i','sig_i','base_index'));cfg=reference.configs[name];session='rx%d:day%d:eq%d'%(pid[0],pid[1],pid[2]);rec,_=reference.ch.normalize_input(np.array(saved[i,0].tolist())+1j*np.array(saved[i,1].tolist()));out,_=reference.residual.ResidualChannel(cfg,reference.ch.stable_seed(20261016101,namespace,name,pid),reference.ch.receiver_for_session(cfg,20261016102,session)).process(rec);expected=torch.tensor(np.stack((out.real,out.imag)).tolist(),dtype=torch.float32);assert torch.equal(expected,seen[6][i]);manual_physics+=1
  assert math.isclose(float(total.detach()),record['base_total_L']+.25*record['extra_pair_loss'],abs_tol=1e-6) and record['base_raw_total_weight']==.75 and record['base_ordinary_weight']==.25 and record['total_pair_weight']==1.25 and m._joint_L_packets==1024 and m._additive_L_packets==256 and m._additive_calls==1 and m._supcon_applied==1
  assert m._joint_As_packets==(128 if hasAs else 0) and sum(m._joint_coverage.values())==m._joint_As_packets and m._joint_channel.summary()['training_calls']==int(hasAs) and record['As_packets']==0 and record['sat_raw_MSE']==0 and m._supcon_pending is None and not m._supcon_handles and len(record['supcon_details'])==4
  total.backward()
  if legacy is not None:assert torch.equal(total.detach(),legacy_loss.detach()) and torch.equal(m.get_parameter(W).grad,legacy.get_parameter(W).grad) and set(m.state_dict())==set(legacy.state_dict())
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.id_backbone.encoder._view_features=orig;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   feat=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert feat.shape==(2,160) and torch.equal(feat,export(x[:2]));z=m.forward_identity_only(x[:2])['tx_logits'];assert z.shape==(2,6) and m._additive_calls==1
  reports.append(dict(arm=row['arm'],L_forwards=8,same_IQ_pairs=True,As_generated=m._joint_As_packets,total_pair_weight=1.25,tracked_BN_modules=len(m._additive_bn_names),default_head_joint_update=True,export160_exact=True))
 assert all(all(torch.equal(v,q[k]) for k,v in initials[0].items()) for q in initials[1:]) and all(torch.equal(ends[0][0],q[0]) and ends[0][1]==q[1] for q in ends[1:])
 aa=[vars(d.make_args(d.config(r),'cpu')) for r in d.rows()]
 for a in aa[1:]:assert {k for k,v in aa[0].items() if v!=a.get(k) and not(isinstance(v,float) and isinstance(a.get(k),float) and math.isnan(v) and math.isnan(a[k]))}=={'candidate_id','output_dir'}
 fake={arm:dict(actual_steps=4440,L_forward_packets=4546560,additive_steps=4440,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,As_training_observations=0 if arm=='B00' else 568320,teacher_additive_calls=0,head_fit_updates=0,target_access=False,scene_schedule_mode={'B00':'none','C_IID':'iid','B01':'balanced'}[arm],scene_schedule_training_calls=0 if arm=='B00' else 4440,balanced_coverage_max_minus_min=1,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=(.84 if arm=='B01' else .82) if s=='clean' else .40 if arm=='B01' else .38 if arm=='C_IID' else .30,macro_F1=.82 if arm=='B01' else .80) for s in ('clean',*d.SCENES)}) for arm in d.ARMS};assert d.qualify(fake)['qualified']
 for kind in ('clean','F1','low','lowF1','mid','suburban','V','budget','base','teacher','As','mode','balance','iid_control'):
  f=copy.deepcopy(fake);r=f['B01']
  if kind=='clean':r['views']['clean']['accuracy']=.81
  elif kind=='F1':r['views']['clean']['macro_F1']=.79
  elif kind=='low':r['views'][d.SCENES[-1]]['accuracy']=.385
  elif kind=='lowF1':r['views'][d.SCENES[-1]]['macro_F1']=.79
  elif kind=='mid':r['views'][d.SCENES[-2]]['accuracy']=.37
  elif kind=='suburban':r['views'][d.SCENES[0]]['accuracy']=.37
  elif kind=='V':r['seen_V']['source_val_accuracy']=.89
  elif kind=='budget':r['actual_steps']=4439
  elif kind=='base':r['final_base_L_objective']=.6
  elif kind=='teacher':r['teacher_additive_calls']=1
  elif kind=='As':r['As_training_observations']=568319
  elif kind=='mode':r['scene_schedule_mode']='iid'
  elif kind=='balance':r['balanced_coverage_max_minus_min']=2
  else:f['C_IID']['views'][d.SCENES[-1]]['accuracy']=.399
  assert not d.qualify(f)['qualified'];negative+=1
 bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_scene_cycle20/runner.py'),'--unregistered'],capture_output=True,text=True);assert bad.returncode and 'No budget/data/operator' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;negative+=1
 return dict(status='PASS',negative_cases=negative,rows=reports,identical_scratch_states=True,B00_loss_default_head_gradient_exact_existing_additive_raw=True,CPU_Python_RNG_endstates_equal=True,IID_fullIQ_and_metadata_exact_native=True,balanced_fullphysics_manual_packets_exact=manual_physics,source_eval_fullIQ_exact_native_no_schedule_state=True,balanced_counts_gap_at_most1_every_visit=True,all6_eachblock6=True,algebraic_schedule_choices=1921,unchanged_native_U_args=True,synthetic_ordinary_generations=128,synthetic_fullpractical_generations=520,synthetic_NN_packets=4114,synthetic_optimizer_updates=3,real_IQ_reads=0,checkpoint_loads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 v=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',v);print({k:x for k,x in v.items() if k!='rows'})
