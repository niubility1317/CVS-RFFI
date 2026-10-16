"""Native-output/RNG isolation and real six-forward/default-head objective."""
import copy,math,random,sys
from pathlib import Path
import torch
def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from .augmentation import Wrapper
 from experiments.cvs_crossrx_satmix20 import model as old_model
 from experiments.cvs_crossrx_satmix20 import supcon
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);neg=0;synthetic_aug=0
 for key,value in (('epochs',50),('model_initialization','pretrained'),('test_views',['clean']),('model_seed',2026092702)):
  c=copy.deepcopy(d.config(d.rows()[0]));c[key]=value
  try:d.validate(c)
  except ValueError:neg+=1
  else:raise AssertionError('Budget/model/scope override accepted')
 for key,value in (('As',True),('raw_weight',.25),('U_rule','changed')):
  c=copy.deepcopy(d.config(d.rows()[1]));c['training_mechanism'][key]=value
  try:d.validate(c)
  except ValueError:neg+=1
  else:raise AssertionError('Protected training rule changed')
 x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(16291));y=torch.arange(128)%6
 for arm in d.ARMS:
  args=d.make_args(d.config(next(r for r in d.rows() if r['arm']==arm)),'cpu');base=native.build_aug_base_cfg(args);aug=native.make_augmentor(base);records=[];wrapper=Wrapper(aug,arm,records);schedules=[]
  assert args.label_epochs==13 and args.pseudo_epochs==7 and args.epochs==20 and args.aug_enable_pa_normal and args.aug_p_dac==0 and not args.aug_enable_class_signature
  for epoch in range(1,21):
   state=native.configure_augmentor_for_epoch(wrapper,base,min(epoch,13),args);schedules.append(state);random.seed(16292+epoch);torch.manual_seed(16292+epoch);a=random.getstate();b=torch.get_rng_state();expected=aug(x,labels=y,no_pa=False);end_a=random.getstate();end_b=torch.get_rng_state();random.setstate(a);torch.set_rng_state(b);actual=wrapper(x,labels=y,no_pa=False);synthetic_aug+=256
   assert random.getstate()==end_a and torch.equal(torch.get_rng_state(),end_b)
   if arm=='B00':assert torch.equal(actual,expected)
   else:
    random.setstate(a);torch.set_rng_state(b);scope=aug._apply_general_channel.__func__.__globals__;original_shift=scope['_apply_time_shift']
    def expected_bypass(value,maximum):original_shift(value,maximum);return value
    scope['_apply_time_shift']=expected_bypass
    try:expected_without_shift=aug(x,labels=y,no_pa=False)
    finally:scope['_apply_time_shift']=original_shift
    synthetic_aug+=128;assert torch.equal(actual,expected_without_shift) and random.getstate()==end_a and torch.equal(torch.get_rng_state(),end_b)

  assert all(v==schedules[12] for v in schedules[13:]) and len(records)==20 and all(v['bypass_returned']==(arm=='B10') for v in records)
  for bad in ({'labels':None,'no_pa':False},{'labels':y,'no_pa':True}):
   try:wrapper(x,**bad)
   except ValueError:neg+=1
   else:raise AssertionError('Non-native/U/PA mode entered wrapper')
 reports=[];initials=[]
 for row in d.rows():
  torch.manual_seed(16293);c=d.config(row);m=api.build(c,'cpu').train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110
  args=d.make_args(c,'cpu');cfg=native.build_aug_base_cfg(args);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,args);aug.p_pa=1.;aug.p_time_shift=1.;records=[];ordinary=Wrapper(aug,row['arm'],records)(x,labels=y,no_pa=False);synthetic_aug+=128
  domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
  seen=[];original=m.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());return original(v)
  def objective(model):
   torch.manual_seed(16294)
   with model._phase1_labeled_feature_context():out=model(ordinary,return_aux=True)
   ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);correction,_=model._source_labeled_feature_hook(model=model,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);return ce+correction
  legacy=None
  if row['arm']=='B00':
   torch.manual_seed(16293);legacy=old_model.build(d.original_config(),'cpu').train();legacy._feature_l_steps=legacy._joint_calls=legacy._legacy_calls=1110;legacy_loss=objective(legacy);legacy_loss.backward()
  m.id_backbone.encoder._view_features=trace;W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total=objective(m)
  assert len(seen)==6 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:]);r=m._joint_details[0];assert r['As_packets']==0 and m._joint_As_packets==0 and m._supcon_applied==1 and len(r['supcon_details'])==4 and math.isclose(r['supcon_weight'],.01,abs_tol=1e-15)
  total.backward();gradient=m.get_parameter(W).grad
  if legacy is not None:assert torch.equal(total.detach(),legacy_loss.detach()) and torch.equal(gradient,legacy.get_parameter(W).grad)
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.id_backbone.encoder._view_features=original;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):features=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert features.shape==(2,160) and torch.equal(features,export(x[:2]))
  reports.append(dict(arm=row['arm'],L_forwards=6,same_IQ_pairs=True,raw_weight=.5,ordinary_weight=.25,raw_slot_weight=.25,original_ordinary_pair_only_supcon=True,default_head_joint_update=True,export160_exact=True,no_new_deployment_parameters=True,total=float(total.detach())))
 assert all(torch.equal(v,initials[1][k]) for k,v in initials[0].items())
 a,b=[d.make_args(d.config(r),'cpu') for r in d.rows()];differences={k for k,v in vars(a).items() if v!=vars(b).get(k) and not (isinstance(v,float) and isinstance(vars(b).get(k),float) and math.isnan(v) and math.isnan(vars(b)[k]))};assert differences=={'candidate_id','output_dir'}
 fake={arm:dict(actual_steps=4440,L_forward_packets=3409920,supcon_applied=3330,final_L_CE=.2,target_access=False,head_fit_updates=0,As_training_observations=0,shift_native_L_calls=4440,shift_native_L_packets=568320,shift_bypass_returned_packets=568320 if arm=='B10' else 0,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=.8 if arm=='B00' else .82,macro_F1=.8 if arm=='B00' else .82) for s in ('clean',*d.SCENES)}) for arm in d.ARMS};assert d.qualify(fake)['qualified']
 for key in ('clean','practical','V','F1','steps'):
  f=copy.deepcopy(fake)
  if key=='clean':f['B10']['views']['clean']['accuracy']=.8
  elif key=='practical':f['B10']['views'][d.SCENES[-1]]['accuracy']=.79
  elif key=='V':f['B10']['seen_V']['source_val_accuracy']=.89
  elif key=='F1':f['B10']['views']['clean']['macro_F1']=.79
  else:f['B10']['actual_steps']=4439
  assert not d.qualify(f)['qualified'];neg+=1
 import subprocess
 bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_shift_bypass20/runner.py'),'--unregistered'],capture_output=True,text=True)
 assert bad.returncode!=0 and 'No budget/data/shift overrides permitted' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;neg+=1
 return dict(status='PASS',negative_cases=neg,rows=reports,identical_scratch_initialization=True,control_loss_and_head_gradient_exactly_match_previous_B00=True,canonical_native_B00_output_bitwise=True,B10_equals_independent_same_draw_shift_bypass=True,both_arms_native_Python_Torch_RNG_endstates_exact=True,actual_native_schedule_capped_at13=True,synthetic_augmentation_packets=synthetic_aug,synthetic_NN_packets=2312,synthetic_optimizer_steps=2,source_IQ_reads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 value=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print('LOCAL_RESULT '+str({k:v for k,v in value.items() if k!='rows'}))
