"""Focused legality, matched-view, default-head gradient and export tests."""
import copy,math,random,subprocess,sys
from pathlib import Path
import torch

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api,bank
 from experiments.cvs_additive_as20 import design as legacy_design,model as legacy_api
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError):negative+=1
  else:raise AssertionError('Forbidden data/config accepted')
 for key,value in [('epochs',40),('model_seed',2026092702),('model_initialization','pretrained'),('test_views',['clean'])]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[key]=value;reject(lambda:d.validate(c))
 for section,key,value in [('iqmix','coefficient',.2),('iqmix','extra_pair_weight',.1),('iqmix','pair_weight_sum',1.),('training_mechanism','U_rule','changed'),('training_mechanism','As',True),('training_mechanism','L_forwards_per_packet',6)]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[section][key]=value;reject(lambda:d.validate(c))
 labels=[];rx=[];days=[];keys=[];ids=[]
 for y in range(6):
  for r in d.FIT:
   for day in (1,2,3):
    for k in range(70):
     i=len(labels);labels.append(y);rx.append(r);days.append(day);keys.append((1,r,day,k,i));ids.append(f'synthetic-{i:05d}')
 g=torch.Generator().manual_seed(20261016181);data=torch.randn(5040,2,256,generator=g)
 def fresh():return bank.Bank(data,labels,rx,days,keys,ids)
 for args in [(data[:5580] if len(data)>5580 else data[:-1],labels,rx,days,keys,ids),(data,[-1]*5040,rx,days,keys,ids),(data,labels,[1]*5040,days,keys,ids),(data,labels,rx,[0]*5040,keys,ids),(data,labels,rx,days,[keys[0]]*5040,ids),(data,labels,rx,days,keys,[ids[0]]*5040),(data*float('nan'),labels,rx,days,keys,ids)]:reject(lambda:bank.Bank(*args))
 ix=[]
 for j in range(128):
  yy=j%6;ri=(j//6)%4;dd=(j//24)%3;rank=j//72;ix.append(((yy*4+ri)*3+dd)*70+rank)
 x=data[ix].clone();saved=x.clone();y=torch.tensor([labels[i] for i in ix]);domains=torch.tensor([3+3*d.FIT.index(rx[i])+days[i]-1 for i in ix]);meta={k:torch.tensor([keys[i][j] for i in ix]) for j,k in enumerate(bank.KEYS)}
 # Every anchor receives three distinct sameRX and three distinct crossRX variants.
 for mode in ('sameRX','crossRX'):
  b=fresh();chosen=[]
  for step in range(1,4):
   out,info=b.apply(x,y,meta,mode,step);chosen.append(info['donor_indices']);assert torch.equal(x,saved) and info['variant_count_per_anchor']==3
  assert all(len({chosen[s][j] for s in range(3)})==3 for j in range(128)) and b.generated==384
 b=fresh();bad_meta=copy.deepcopy(meta);bad_meta['rx_i'][0]=1;reject(lambda:b.apply(x,y,bad_meta,'crossRX',1));b=fresh();reject(lambda:b.apply(x,torch.full_like(y,-1),meta,'crossRX',1));b=fresh();wrong=y.clone();wrong[0]=(int(wrong[0])+1)%6;reject(lambda:b.apply(x,wrong,meta,'crossRX',1));b=fresh();reject(lambda:b.apply(x,y,meta,'crossRX',2));b=fresh();reject(lambda:b.apply(x,y,meta,'unsupported',1))
 bank._BANK=None;reject(bank.current);bank.install(fresh());reject(lambda:bank.install(fresh()));bank._BANK=None
 args=d.make_args(d.config(d.rows()[0]),'cpu');base=native.build_aug_base_cfg(args);aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,6,args);random.seed(20261016182);torch.manual_seed(20261016182);ordinary=aug(x.clone(),labels=y,no_pa=False)
 def objective(m):
  torch.manual_seed(20261016183);random.seed(20261016183)
  with m._phase1_labeled_feature_context():out=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(out['tx_logits'],y,label_smoothing=.01);correction,_=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  return ce+correction
 reports=[];initials=[];ends=[]
 for row in d.rows():
  bank._BANK=None;b=fresh();b.calls=1110;bank.install(b);torch.manual_seed(20261016184);c=d.config(row);m=api.build(c,'cpu').train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];original_view=m.id_backbone.encoder._view_features
  def trace(v):seen.append(v.detach().clone());return original_view(v)
  m.id_backbone.encoder._view_features=trace;legacy=None
  if row['arm']=='B00':
   torch.manual_seed(20261016184);lc=legacy_design.config(next(r for r in legacy_design.rows() if r['arm']=='D_raw'));legacy=legacy_api.build(lc,'cpu').train();legacy._feature_l_steps=legacy._joint_calls=legacy._legacy_calls=1110;legacy_loss=objective(legacy);legacy_loss.backward()
  W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();opt=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001);total=objective(m);ends.append((torch.get_rng_state().clone(),random.getstate()))
  assert len(seen)==8 and torch.equal(seen[0],ordinary) and torch.equal(seen[0],seen[1]) and all(torch.equal(z,x) for z in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(x,saved)
  record=m._joint_details[-1];assert math.isclose(float(total.detach()),record['base_total_L']+.25*record['extra_pair_loss'],abs_tol=1e-6) and record['base_raw_total_weight']==.75 and record['base_ordinary_weight']==.25 and record['total_pair_weight']==1.25 and m._joint_L_packets==1024 and m._IQmix_L_packets==256 and m._IQmix_calls==1 and m._supcon_applied==1
  assert b.calls==1111 and sum(b.visits)==128 and b.generated==(0 if row['arm']=='B00' else 128) and not m._joint_As_packets and record['As_packets']==0 and record['sat_raw_MSE']==0 and m._supcon_pending is None and not m._supcon_handles and len(record['supcon_details'])==4
  if row['arm']=='B00':assert torch.equal(seen[6],x) and not record['IQmix']['donor_indices']
  else:assert not torch.equal(seen[6],x) and len(record['IQmix']['donor_indices'])==128
  total.backward()
  if legacy is not None:assert torch.equal(total.detach(),legacy_loss.detach()) and torch.equal(m.get_parameter(W).grad,legacy.get_parameter(W).grad) and set(m.state_dict())==set(legacy.state_dict())
  assert not any(v is b or (torch.is_tensor(v) and tuple(v.shape)==(5040,2,256)) for v in vars(m).values())
  clip_physical(opt,m.parameters(),5.);opt.step();assert not torch.equal(before,m.get_parameter(W));m.id_backbone.encoder._view_features=original_view;m.eval()
  with torch.no_grad(),fresh_parameter_caches(m):
   feat=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert feat.shape==(2,160) and torch.equal(feat,export(x[:2]));z=m.forward_identity_only(x[:2])['tx_logits'];assert z.shape==(2,6) and m._IQmix_calls==1
  reports.append(dict(arm=row['arm'],L_forwards=8,common_inputs_exact=True,extra_same_IQ=True,extra_IQmix_packets=b.generated,total_pair_weight=record['total_pair_weight'],loss=float(total.detach()),base_loss=record['base_total_L'],tracked_BN_modules=len(m._IQmix_bn_names),extra_BN_updates=record['extra_BN_batch_updates'],default_head_joint_update=True,export160_exact=True,bank_not_in_model_state=True))
 assert all(all(torch.equal(v,q[k]) for k,v in initials[0].items()) for q in initials[1:]) and all(torch.equal(ends[0][0],q[0]) and ends[0][1]==q[1] for q in ends[1:])
 args=[vars(d.make_args(d.config(r),'cpu')) for r in d.rows()]
 for a in args[1:]:
  diff={k for k,v in args[0].items() if v!=a.get(k) and not(isinstance(v,float) and isinstance(a.get(k),float) and math.isnan(v) and math.isnan(a[k]))};assert diff=={'candidate_id','output_dir'}
 fake={arm:dict(actual_steps=4440,L_forward_packets=4546560,IQmix_steps=4440,extra_L_forward_packets=1136640,supcon_applied=3330,final_base_L_objective=.2,IQmix_bank_unique_L=5040,IQmix_generated_packets=0 if arm=='B00' else 568320,As_training_observations=0,teacher_IQmix_calls=0,head_fit_updates=0,target_access=False,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=.82 if s=='clean' and arm=='B10' else .8 if s=='clean' else .31,macro_F1=.82 if arm=='B10' else .8) for s in ('clean',*d.SCENES)}) for arm in d.ARMS};assert d.qualify(fake)['qualified']
 for kind in ('clean','F1','practical','V','budget','bank','base','teacher','As','same_control'):
  f=copy.deepcopy(fake)
  if kind=='clean':f['B10']['views']['clean']['accuracy']=.805
  elif kind=='F1':f['B10']['views']['clean']['macro_F1']=.79
  elif kind=='practical':f['B10']['views'][d.SCENES[-1]]['accuracy']=.30
  elif kind=='V':f['B10']['seen_V']['source_val_accuracy']=.89
  elif kind=='budget':f['B10']['actual_steps']=4439
  elif kind=='bank':f['B10']['IQmix_bank_unique_L']=6300
  elif kind=='base':f['B10']['final_base_L_objective']=.6
  elif kind=='teacher':f['B10']['teacher_IQmix_calls']=1
  elif kind=='As':f['B10']['As_training_observations']=1
  else:f['C_same']['views']['clean']['accuracy']=.815
  assert not d.qualify(f)['qualified'];negative+=1
 bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_same_tx_iq20/runner.py'),'--unregistered'],capture_output=True,text=True);assert bad.returncode and 'No budget/data/operator' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;negative+=1
 bank._BANK=None
 return dict(status='PASS',negative_cases=negative,rows=reports,identical_scratch_states=True,B00_loss_default_head_gradient_match_original_additive_raw_control=True,CPU_Python_RNG_endstates_equal=True,three_distinct_same_and_cross_donor_variants=True,unchanged_native_U_args=True,bank_not_in_model_teacher_checkpoint=True,synthetic_bank_packets=5040,synthetic_ordinary_generated_packets=128,synthetic_IQmix_generated_packets=1024,synthetic_NN_packets=4114,synthetic_optimizer_updates=3,real_IQ_reads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 value=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print({k:v for k,v in value.items() if k!='rows'})
