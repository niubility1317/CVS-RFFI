"""Focused roles/recipe negatives and actual 8-forward original-head objective checks."""
import copy,math,random,subprocess,sys
from pathlib import Path
import torch

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api,noise
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
 for key,value in [('epochs',40),('model_seed',2026092702),('model_initialization','pretrained'),('test_views',['clean'])]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[key]=value;reject(lambda:d.validate(c))
 for section,key,value in [('isolated_noise','SNR_dB',[30,40]),('isolated_noise','extra_pair_weight',.1),('isolated_noise','pair_weight_sum',1.),('training_mechanism','U_rule','changed'),('training_mechanism','As',True),('training_mechanism','L_forwards_per_packet',6)]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[section][key]=value;reject(lambda:d.validate(c))
 x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(202610172311));saved=x.clone();y=torch.arange(128)%6;domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
 a=d.make_args(d.config(d.rows()[0]),'cpu');base=native.build_aug_base_cfg(a);aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,6,a);random.seed(202610172312);torch.manual_seed(202610172312);ordinary=aug(x.clone(),labels=y,no_pa=False);module=noise.native_module();rng=torch.get_rng_state();py=random.getstate();v,info=noise.apply(x,1111);assert torch.equal(x,saved) and torch.equal(rng,torch.get_rng_state()) and py==random.getstate()
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(d.NOISE_SEED+1111);s=module._rand_uniform((128,1),x.device,low=20.,high=36.,dtype=x.dtype);canonical=module._from_complex(module._apply_awgn(module._to_complex(x),s))
 assert torch.equal(v,canonical);again,inf2=noise.apply(x,1111);other,inf3=noise.apply(x,1112);assert torch.equal(v,again) and info==inf2 and not torch.equal(v,other) and info!=inf3
 for value,step in [(x,0),(x,4441),(x[:,:,:128],1),(x.double(),1),(torch.zeros_like(x),1),(x*float('nan'),1),(x.clone().requires_grad_(),1)]:reject(lambda:noise.apply(value,step))
 def objective(m):
  torch.manual_seed(202610172313);random.seed(202610172313)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);return ce+correction
 reports=[];initials=[];ends=[]
 for row in d.rows():
  torch.manual_seed(202610172314);m=api.build(d.config(row),'cpu').train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];orig=m.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());return orig(v)
  m.id_backbone.encoder._view_features=trace;legacy=None
  if row['arm']=='B00':
   torch.manual_seed(202610172314);lc=legacy_design.config(next(r for r in legacy_design.rows() if r['arm']=='D_raw'));legacy=legacy_api.build(lc,'cpu').train();legacy._feature_l_steps=legacy._joint_calls=legacy._legacy_calls=1110;legacy_loss=objective(legacy);legacy_loss.backward()
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total=objective(m);ends.append((torch.get_rng_state().clone(),random.getstate()));record=m._joint_details[-1]
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(x,saved);assert torch.equal(seen[6],{'B00':x,'C_ordinary':ordinary,'B10':v}[row['arm']])
  assert math.isclose(float(total.detach()),record['base_total_L']+.25*record['extra_pair_loss'],abs_tol=1e-6) and record['base_raw_total_weight']==.75 and record['base_ordinary_weight']==.25 and record['total_pair_weight']==1.25 and m._joint_L_packets==1024 and m._noise_L_packets==256 and m._noise_calls==1 and m._supcon_applied==1
  assert m._noise_generated==(128 if row['arm']=='B10' else 0) and sum(m._noise_coverage.values())==m._noise_generated and not m._joint_As_packets and record['As_packets']==0 and record['sat_raw_MSE']==0 and m._supcon_pending is None and not m._supcon_handles and len(record['supcon_details'])==4
  total.backward()
  if legacy is not None:assert torch.equal(total.detach(),legacy_loss.detach()) and torch.equal(m.get_parameter(W).grad,legacy.get_parameter(W).grad) and set(m.state_dict())==set(legacy.state_dict())
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.id_backbone.encoder._view_features=orig;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   feat=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert feat.shape==(2,160) and torch.equal(feat,export(x[:2]));z=m.forward_identity_only(x[:2])['tx_logits'];assert z.shape==(2,6) and m._noise_calls==1
  reports.append(dict(arm=row['arm'],L_forwards=8,common_inputs_exact=True,extra_same_IQ=True,noise_generated=m._noise_generated,total_pair_weight=1.25,loss=float(total.detach()),base_loss=record['base_total_L'],tracked_BN_modules=len(m._noise_bn_names),extra_BN_updates=record['extra_BN_batch_updates'],default_head_joint_update=True,export160_exact=True))
 assert all(all(torch.equal(v,q[k]) for k,v in initials[0].items()) for q in initials[1:]) and all(torch.equal(ends[0][0],q[0]) and ends[0][1]==q[1] for q in ends[1:])
 aa=[vars(d.make_args(d.config(r),'cpu')) for r in d.rows()]
 for a in aa[1:]:
  diff={k for k,v in aa[0].items() if v!=a.get(k) and not(isinstance(v,float) and isinstance(a.get(k),float) and math.isnan(v) and math.isnan(a[k]))};assert diff=={'candidate_id','output_dir'}
 fake={arm:dict(actual_steps=4440,L_forward_packets=4546560,noise_steps=4440,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,noise_generated_packets=568320 if arm=='B10' else 0,As_training_observations=0,teacher_noise_calls=0,head_fit_updates=0,target_access=False,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=.82 if s=='clean' and arm=='B10' else .8 if s=='clean' else .31,macro_F1=.82 if arm=='B10' else .8) for s in ('clean',*d.SCENES)}) for arm in d.ARMS};assert d.qualify(fake)['qualified']
 for kind in ('clean','F1','practical','V','budget','base','teacher','As','ordinary_control','generations'):
  f=copy.deepcopy(fake)
  if kind=='clean':f['B10']['views']['clean']['accuracy']=.805
  elif kind=='F1':f['B10']['views']['clean']['macro_F1']=.79
  elif kind=='practical':f['B10']['views'][d.SCENES[-1]]['accuracy']=.30
  elif kind=='V':f['B10']['seen_V']['source_val_accuracy']=.89
  elif kind=='budget':f['B10']['actual_steps']=4439
  elif kind=='base':f['B10']['final_base_L_objective']=.6
  elif kind=='teacher':f['B10']['teacher_noise_calls']=1
  elif kind=='As':f['B10']['As_training_observations']=1
  elif kind=='generations':f['B10']['noise_generated_packets']=568319
  else:f['C_ordinary']['views']['clean']['accuracy']=.815
  assert not d.qualify(f)['qualified'];negative+=1
 bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_isolated_awgn20/runner.py'),'--unregistered'],capture_output=True,text=True);assert bad.returncode and 'No budget/data/operator' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;negative+=1
 return dict(status='PASS',negative_cases=negative,rows=reports,identical_scratch_states=True,B00_loss_default_head_gradient_exactly_match_existing_additive_raw=True,CPU_Python_RNG_endstates_equal=True,native_AWGN_canonical_exact=True,stateless_step_repeatable_different_step_distinct=True,oldordinary_actualtensor_reused=True,unchanged_native_U_args=True,synthetic_ordinary_generations=128,synthetic_native_noise_generations=640,synthetic_NN_packets=4114,synthetic_optimizer_updates=3,real_IQ_reads=0,checkpoint_loads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 v=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',v);print({k:x for k,x in v.items() if k!='rows'})
