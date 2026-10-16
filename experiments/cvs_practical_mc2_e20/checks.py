"""Focused protocol negatives, common gradients, same-IQ pairs and true full-channel sampling."""
import copy,math,random,subprocess,sys
from dataclasses import asdict
from pathlib import Path
import torch

def run():
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root/'experiments/adv3b02_xuc/code'),str(root/'code')]
 from SSDG import train_ssdg as native
 from . import design as d,model as api
 from experiments.cvs_crossrx_satmix20 import model as original
 from experiments.cvs_rdrop_sinc_units.optimizer import build,clip_physical
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 torch.set_num_threads(2);negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError):negative+=1
  else:raise AssertionError('Forbidden MC2 data/config accepted')
 for k,v in [('epochs',40),('model_seed',2026092702),('model_initialization','pretrained'),('test_views',['clean']),('target_access',True)]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[k]=v;reject(lambda:d.validate(c))
 for section,k,v in [('mc2','unique_As_views',4),('mc2','extra_pair_weight',.25),('mc2','pair_weight_sum',1.),('mc2','U','changed'),('mc2','practical_distribution','weakened'),('training_mechanism','L_forwards_per_packet',8),('training_mechanism','base_raw_total_weight',.5),('channel','composition',True)]:
  c=copy.deepcopy(d.config(d.rows()[2]));c[section][k]=v;reject(lambda:d.validate(c))
 g=torch.Generator().manual_seed(20261017131);x=torch.randn(128,2,256,generator=g);saved=x.clone();y=torch.arange(128)%6;rx=torch.tensor([d.FIT[(i//6)%4] for i in range(128)]);day=1+torch.arange(128)%3;domains=torch.tensor([3+3*d.FIT.index(int(rx[i]))+int(day[i])-1 for i in range(128)])
 meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=day,sig_i=torch.arange(128),base_index=torch.arange(128))
 args=d.make_args(d.config(d.rows()[0]),'cpu');assert (args.epochs,args.label_epochs,args.pseudo_epochs,args.seed,args.label_smoothing,args.batch_size)==(20,13,7,d.SEED,.01,128) and args.from_scratch and not args.baseline_ckpt and not args.teacher_ckpt
 base=native.build_aug_base_cfg(args);aug=native.make_augmentor(base);native.configure_augmentor_for_epoch(aug,base,6,args);random.seed(20261017132);torch.manual_seed(20261017132);ordinary=aug(x.clone(),labels=y,no_pa=False)
 def objective(m):
  torch.manual_seed(20261017134);random.seed(20261017134)
  with m._phase1_labeled_feature_context():output=m(ordinary,return_aux=True)
  ce=torch.nn.functional.cross_entropy(output['tx_logits'],y,label_smoothing=.01);correction,_=m._source_labeled_feature_hook(model=m,labeled_output=output,labels=y,extra=(domains,meta),domains=domains,optimizer_step=1111,identity_loss=ce,labeled_count=128,labeled_inputs=x);return ce+correction
 torch.manual_seed(20261017133);reference=original.build(d.original_config(),'cpu').train();reference._feature_l_steps=reference._joint_calls=reference._legacy_calls=1110;ref_loss=objective(reference);ref_grad=torch.autograd.grad(ref_loss,tuple(reference.parameters()),allow_unused=True);ref_state={k:v.detach().clone() for k,v in reference.state_dict().items()};ref_grad={k:None if g is None else g.detach().clone() for (k,p),g in zip(reference.named_parameters(),ref_grad)}
 real_hook=original.hook
 def capture(m,**kw):
  correction,metrics=real_hook(m,**kw);m._test_common_tensor=correction+kw['identity_loss'];return correction,metrics
 capture._accepts_labeled_inputs=True;original.hook=capture
 reports=[];initials=[];ends=[];first_As={};draws={}
 try:
  for row in d.rows():
   torch.manual_seed(20261017133);c=d.config(row);m=api.build(c,'cpu').train();initials.append({k:v.detach().clone() for k,v in m.state_dict().items()});assert set(m.state_dict())==set(reference.state_dict());m._feature_l_steps=m._joint_calls=m._legacy_calls=1110;seen=[];f=m.id_backbone.encoder._view_features
   def trace(v):seen.append(v.detach().clone());return f(v)
   m.id_backbone.encoder._view_features=trace;W='id_backbone.encoder.core.id_backbone.cls_head.weight';before=m.get_parameter(W).detach().clone();optimizer=build(m,list(m.parameters()),'CENTER_UNITS',lr=.0002,weight_decay=.0001)
   channel=m._joint_channel;reject(lambda:channel.apply(x[:,:1],meta,namespace='train/invalid'));reject(lambda:channel.apply(x,{k:v for k,v in meta.items() if k!='rx_i'},namespace='train/invalid'));bad=copy.deepcopy(meta);bad['rx_i']=torch.zeros(128,dtype=torch.long);reject(lambda:channel.apply(x,bad,namespace='train/invalid'));reject(lambda:channel.apply(x,meta,namespace='train/invalid',scene=d.SCENES[0]))
   receiver=channel.ch.receiver_for_session;receiver_seen={}
   def tracked_receiver(cfg,seed,session):
    value=receiver(cfg,seed,session);key=(cfg.scenario,seed,session);state=asdict(value)
    if key in receiver_seen:assert state==receiver_seen[key]
    receiver_seen[key]=state;assert session.startswith('rx') and ':day' in session and ':eq' in session and 'mc' not in session;return value
   channel.ch.receiver_for_session=tracked_receiver
   try:total=objective(m)
   finally:channel.ch.receiver_for_session=receiver
   ends.append((torch.get_rng_state().clone(),random.getstate()));prefix=torch.autograd.grad(m._test_common_tensor,tuple(m.parameters()),retain_graph=True,allow_unused=True)
   assert all((g is None and ref_grad[k] is None) or (g is not None and ref_grad[k] is not None and torch.equal(g,ref_grad[k])) for (k,p),g in zip(m.named_parameters(),prefix));assert torch.equal(m._test_common_tensor.detach(),ref_loss.detach());del m._test_common_tensor
   assert len(seen)==10 and all(torch.equal(q,ordinary) for q in seen[:2]) and all(torch.equal(q,x) for q in seen[2:6]) and torch.equal(seen[6],seen[7]) and torch.equal(seen[8],seen[9]) and torch.equal(x,saved)
   r=m._joint_details[-1];n=c['mc2']['unique_As_views'];assert m._joint_L_packets==1280 and m._MC2_L_packets==512 and m._MC2_calls==1 and m._joint_As_packets==128*n and sum(m._joint_coverage.values())==128*n and sum(m._MC2_coverage.values())==128*n and m._MC2_generated==[128 if n else 0,128 if n==2 else 0] and m._MC2_As_supervised_pair_observations==(256 if n else 0)
   assert r['As_packets']==0 and r['sat_raw_MSE']==0 and r['base_raw_total_weight']==.75 and r['base_ordinary_weight']==.25 and r['total_pair_weight']==1.25 and r['no_IQ_averaging'] and r['no_crossview_consistency'] and r['all_extra_L_supervised'] and len(r['extra_pairs'])==2 and len(r['extra_draws'])==n and len(r['supcon_details'])==4 and not m._supcon_handles and m._supcon_pending is None
   assert math.isclose(float(total.detach()),r['base_total_L']+.125*sum(q['loss'] for q in r['extra_pairs']),abs_tol=1e-6)
   for q in r['extra_pairs']:assert q['weight']==.125 and q['generated_once'] and q['same_IQ_within_pair']
   if n==0:assert all(torch.equal(q,x) for q in seen[6:]) and r['extra_between_MSE']==0 and not receiver_seen
   elif n==1:assert torch.equal(seen[6],seen[8]) and r['extra_between_MSE']==0 and receiver_seen
   else:assert not torch.equal(seen[6],seen[8]) and r['extra_between_MSE']>0 and receiver_seen
   if n:first_As[row['arm']]=seen[6];draws[row['arm']]=r['extra_draws'][0];assert all(q.processing_route=='residual' and q.mode=='post_sync' and q.equalization_enabled is False for q in channel.configs.values())
   total.backward();clip_physical(optimizer,m.parameters(),5.);optimizer.step();assert not torch.equal(before,m.get_parameter(W));m.id_backbone.encoder._view_features=f;m.eval()
   with torch.no_grad(),fresh_parameter_caches(m):
    feat=m.id_backbone.encoder.features(x[:2]);export=api.export(m).eval();assert feat.shape==(2,160) and torch.equal(feat,export(x[:2]));z=m.forward_identity_only(x[:2])['tx_logits'];assert z.shape==(2,6) and m._MC2_calls==1
   reports.append(dict(arm=row['arm'],L_forwards=10,base_loss_all_parameter_gradients_exact=True,extra_same_IQ_within_each_pair=True,unique_generated_As_packets=128*n,As_supervised_pair_observations=256 if n else 0,total_pair_weight=1.25,common_loss=r['base_total_L'],total_loss=float(total.detach()),tracked_BN_modules=len(m._MC2_bn_names),extra_BN_updates=r['extra_BN_batch_updates'],default_head_joint_update=True,export160_exact=True))
 finally:original.hook=real_hook
 assert torch.equal(first_As['M_repeat'],first_As['M_independent']) and draws['M_repeat']==draws['M_independent'] and all(all(torch.equal(q[k],v) for k,v in initials[0].items()) for q in initials[1:]) and all(torch.equal(ends[0][0],q[0]) and ends[0][1]==q[1] for q in ends[1:])
 args=[vars(d.make_args(d.config(r),'cpu')) for r in d.rows()]
 for a in args[1:]:assert {k for k,v in args[0].items() if v!=a.get(k) and not(isinstance(v,float) and isinstance(a.get(k),float) and math.isnan(v) and math.isnan(a[k]))}=={'candidate_id','output_dir'}
 fake={arm:dict(actual_steps=4440,MC2_steps=4440,L_forward_packets=5683200,extra_L_forward_packets=2273280,supcon_applied=3330,final_base_L_objective=.2,As_training_observations=d.config(next(r for r in d.rows() if r['arm']==arm))['mc2']['expected_As_generated_packets'],As_supervised_pair_observations=0 if arm=='M00' else 1136640,teacher_MC2_calls=0,head_fit_updates=0,target_access=False,seen_V=dict(source_val_accuracy=.9),views={s:dict(accuracy=.82 if s=='clean' and arm=='M_independent' else .8 if s=='clean' else .36 if arm=='M_independent' else .35 if arm=='M_repeat' else .30,macro_F1=.82 if arm=='M_independent' else .8) for s in ('clean',*d.SCENES)}) for arm in d.ARMS};assert d.qualify(fake)['qualified']
 for case in ('rawclean','repeatclean','F1','low','urban','suburban','practical','V','steps','NN','Asdraws','Aspairs','teacher','base','head','target'):
  q=copy.deepcopy(fake);c=q['M_independent']
  if case=='rawclean':c['views']['clean']['accuracy']=.79
  elif case=='repeatclean':q['M_repeat']['views']['clean']['accuracy']=.815
  elif case=='F1':c['views']['clean']['macro_F1']=.79
  elif case=='low':c['views'][d.SCENES[-1]]['accuracy']=.34
  elif case=='urban':c['views'][d.SCENES[3]]['accuracy']=.28;c['views'][d.SCENES[4]]['accuracy']=.28
  elif case=='suburban':c['views'][d.SCENES[0]]['accuracy']=.29
  elif case=='practical':q['M_repeat']['views'][d.SCENES[0]]['accuracy']=.37
  elif case=='V':c['seen_V']['source_val_accuracy']=.89
  elif case=='steps':c['actual_steps']=4439
  elif case=='NN':c['L_forward_packets']=4546560
  elif case=='Asdraws':c['As_training_observations']=568320
  elif case=='Aspairs':c['As_supervised_pair_observations']=568320
  elif case=='teacher':c['teacher_MC2_calls']=1
  elif case=='base':c['final_base_L_objective']=.6
  elif case=='head':c['head_fit_updates']=1
  else:c['target_access']=True
  assert not d.qualify(q)['qualified'];negative+=1
 reject(lambda:d.qualify({'M_independent':fake['M_independent']}));bad=subprocess.run([sys.executable,'-S',str(root/'experiments/cvs_practical_mc2_e20/runner.py'),'--unregistered'],capture_output=True,text=True);assert bad.returncode and 'No budget/data/operator' in bad.stderr and 'ModuleNotFoundError' not in bad.stderr;negative+=1
 return dict(status='PASS',negative_cases=negative,rows=reports,all_common_parameter_gradients_exact=True,identical_scratch_states=True,CPU_Python_RNG_endstates_equal=True,MC0_inputs_and_draw_records_exact_between_repeat_independent=True,receiver_session_parameters_stable=True,unchanged_native_U_args=True,no_new_deployment_state_keys=True,synthetic_NN_packets=4626,synthetic_optimizer_updates=3,synthetic_ordinary_generated_packets=128,synthetic_full_As_generated_packets=384,real_IQ_reads=0,target_access=False,head_fit_updates=0)
if __name__=='__main__':
 from . import design as d
 v=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',v);print({k:v for k,v in v.items() if k!='rows'})
