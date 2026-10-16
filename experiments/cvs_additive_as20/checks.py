"""New objective/control isolation using real F and unmodified full practical."""
import copy,math,random,subprocess,sys
from pathlib import Path
import torch

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from experiments.cvs_crossrx_satmix20 import model as old,supcon
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=0;reports=[];initials=[];ends=[]
 def reject(c):
  nonlocal negative
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Protected config override accepted')
 for key,value in (('epochs',40),('model_initialization','pretrained'),('model_seed',2026092702),('test_views',['clean'])):
  c=copy.deepcopy(d.config(d.rows()[0]));c[key]=value;reject(c)
 for key,value in (('raw_total_weight',.5),('extra_pair_weight',.1),('L_forwards_per_packet',6),('U_rule','changed')):
  c=copy.deepcopy(d.config(d.rows()[2]));c['training_mechanism'][key]=value;reject(c)
 c=copy.deepcopy(d.config(d.rows()[2]));c['additive']['operator']='ordinary then As';reject(c)
 x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(20261016611));saved=x.clone();y=torch.arange(128)%6
 args=d.make_args(d.config(d.rows()[0]),'cpu');base=native.build_aug_base_cfg(args);aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,6,args);random.seed(20261016612);torch.manual_seed(20261016612);ordinary=aug(x.clone(),labels=y,no_pa=False)
 domains=3+torch.arange(128)%12;rx=torch.tensor(supcon.RX_BY_DOMAIN)[domains];meta=dict(rx_i=rx,day_i=domains%3+1,eq_i=torch.zeros(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
 def objective(m):
  torch.manual_seed(20261016613);random.seed(20261016613)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
  correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  return ce+correction
 for row in d.rows():
  torch.manual_seed(20261016614);c=d.config(row);m=api.build(c,'cpu').train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];channel_calls=[];original_view=m.id_backbone.encoder._view_features;original_apply=m._joint_channel.apply
  def trace(v):seen.append(v.detach().clone());return original_view(v)
  def channel(v,metadata,**kw):channel_calls.append(v.detach().clone());return original_apply(v,metadata,**kw)
  m.id_backbone.encoder._view_features=trace;m._joint_channel.apply=channel
  legacy=None
  if row['arm']=='D00':
   torch.manual_seed(20261016614);legacy=old.build(d.original_config(),'cpu').train();legacy._feature_l_steps=legacy._joint_calls=legacy._legacy_calls=1110;legacy_loss=objective(legacy);legacy_loss.backward()
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total=objective(m);ends.append((torch.get_rng_state().clone(),random.getstate()))
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(x,saved)
  r=m._joint_details[-1];w=c['additive']['extra_weight'];assert math.isclose(float(total.detach()),r['base_total_L']+w*r['extra_pair_loss'],abs_tol=1e-6) and r['base_raw_total_weight']==.75 and r['base_ordinary_weight']==.25 and r['extra_pair_weight']==w and m._joint_L_packets==1024 and m._additive_L_packets==256 and m._additive_calls==1 and m._supcon_applied==1
  if row['arm']=='D_As':assert len(channel_calls)==1 and torch.equal(channel_calls[0],x) and r['extra_As_packets']==128 and m._joint_As_packets==128 and not torch.equal(seen[6],x)
  else:assert not channel_calls and r['extra_As_packets']==0 and torch.equal(seen[6],x)
  assert r['As_packets']==0 and r['sat_raw_MSE']==0 and m._supcon_pending is None and not m._supcon_handles and len(r['supcon_details'])==4
  total.backward()
  if legacy is not None:assert torch.equal(total.detach(),legacy_loss.detach()) and torch.equal(m.get_parameter(W).grad,legacy.get_parameter(W).grad) and set(m.state_dict())==set(legacy.state_dict())
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.id_backbone.encoder._view_features=original_view;m._joint_channel.apply=original_apply;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   feats=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert feats.shape==(2,160) and torch.equal(feats,export(x[:2]));z=m.forward_identity_only(x[:2])['tx_logits'];assert z.shape==(2,6) and m._additive_calls==1
  reports.append(dict(arm=row['arm'],L_forwards=8,common_inputs_exact=True,extra_same_IQ=True,extra_As_packets=r['extra_As_packets'],loss=float(total.detach()),base_loss=r['base_total_L'],weight_sum=r['total_pair_weight'],tracked_BN_modules=len(m._additive_bn_names),extra_BN_batch_updates=r['extra_BN_batch_updates'],default_head_joint_update=True,export160_exact=True,no_new_state_keys=True))
 assert all(all(torch.equal(v,initial[k]) for k,v in initials[0].items()) for initial in initials[1:]) and all(torch.equal(ends[0][0],v[0]) and ends[0][1]==v[1] for v in ends[1:])
 args=[d.make_args(d.config(r),'cpu') for r in d.rows()]
 for a in args[1:]:
  diff={k for k,v in vars(args[0]).items() if v!=vars(a).get(k) and not(isinstance(v,float) and isinstance(vars(a).get(k),float) and math.isnan(v) and math.isnan(vars(a)[k]))};assert diff=={'candidate_id','output_dir'}
 fake={arm:dict(actual_steps=4440,L_forward_packets=4546560,additive_steps=4440,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,final_L_CE=.7,As_training_observations=568320 if arm=='D_As' else 0,teacher_additive_calls=0,head_fit_updates=0,target_access=False,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=.82 if s=='clean' and arm=='D_As' else .8 if s=='clean' else .38 if arm=='D_As' else .3,macro_F1=.82 if arm=='D_As' else .8) for s in ('clean',*d.SCENES)}) for arm in d.ARMS};assert d.qualify(fake)['qualified']
 for kind in ('clean','F1','low','urban','practical','V','budget','base_health','teacher','raw_control'):
  f=copy.deepcopy(fake)
  if kind=='clean':f['D_As']['views']['clean']['accuracy']=.79
  elif kind=='F1':f['D_As']['views']['clean']['macro_F1']=.79
  elif kind=='low':f['D_As']['views'][d.SCENES[-1]]['accuracy']=.34
  elif kind=='urban':
   for s in d.SCENES[3:5]:f['D_As']['views'][s]['accuracy']=.30
  elif kind=='practical':f['D_As']['views'][d.SCENES[0]]['accuracy']=.29
  elif kind=='V':f['D_As']['seen_V']['source_val_accuracy']=.89
  elif kind=='budget':f['D_As']['actual_steps']=4439
  elif kind=='base_health':f['D_As']['final_base_L_objective']=.6
  elif kind=='teacher':f['D_As']['teacher_additive_calls']=1
  else:
   for s in d.SCENES[3:]:f['D_raw']['views'][s]['accuracy']=.38
  assert not d.qualify(f)['qualified'];negative+=1
 bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_additive_as20/runner.py'),'--unregistered'],capture_output=True,text=True);assert bad.returncode and 'No budget/data/' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;negative+=1
 return dict(status='PASS',negative_cases=negative,rows=reports,identical_scratch_states=True,D00_loss_and_default_head_gradient_match_original6forward=True,all3CPU_Python_RNG_endstates_equal=True,unchanged_native_U_args=True,full_unmodified_practical_only_on_raw=True,no_As_outside_L_context=True,synthetic_ordinary_generated_packets=128,synthetic_As_generated_packets=128,synthetic_NN_packets=3858,synthetic_optimizer_updates=3,real_IQ_reads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 value=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print({k:v for k,v in value.items() if k!='rows'})
