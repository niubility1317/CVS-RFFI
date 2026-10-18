"""Deterministic metadata/toy-log fixtures only; no actual IQ, CP or model."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('u_quality66_analysis',Path(__file__).with_name('cvs_u_quality66_analysis_20261018.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,allow_nan=False),encoding='utf8')

def decision():return dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,model_seed_count=1,NN=0,source_qualified=False,checks=dict(clean=False,midlow=True,low=True,allocation=True,low_allocation=True))

def stat(n,v):
 return dict(count=n,mass=n*v,mean=v,positive=n if v else 0,quantiles=[v]*5,histogram=[n if v==0 else 0,n if 0<v<=.25 else 0,n if .25<v<=.5 else 0,n if .5<v<=.75 else 0,n if v>.75 else 0])

def record(arm=a.CANDIDATE,step=1,native=256):
 n=min(128,native);e=(step-1)//222+1;w=.16*min(e/5.,1.)
 s=dict(scene=a.SCENES[-1],state='LOS',lock='locked',count=n,none_snr_count=0,q_teacher_mass=float(n),q_observation_mass=n*.5,mass_uniform_mass=n*.5,observation_weighted_mass=n*.5,q_mass=n*.5,q_mean=.5,q_teacher_mean=1.,q_observation_mean=.5,positive_q=n,signal_to_added_noise_db_mean=5.)
 return dict(arm=arm,step=step,cumulative_step=4440+step,local_epoch=e,epoch=20+e,native_packets=native,packets=n,selected_teacher='FROZEN_AS20_EMA',weight=w,KL=.5,weighted_KL=.5*w,unweighted_KL=1.,q_mass=n*.5,q_mean=.5,positive_q=n,none_snr_count=0,q_teacher=stat(n,1.),q_observation=stat(n,.5),effective_q=stat(n,.5),mass_match=dict(q_teacher_mass=float(n),observation_weighted_mass=n*.5,mass_uniform_mass=n*.5,difference=0.,absolute_difference=0.,uniform_multiplier=.5),strata=[s],original_selected=2,extra_original_selected=1,original_mask_unchanged=True,old_U_masks_unchanged=True,U_TX_truth_read=False,urban_confidence_filter=False,dropout_rng_restored=True,source_health_gate=False,subset_rule='first_min_128_native_U_batch_no_padding',selected_count_normalization=False,temperature=1.,observation_proxy_is_recoverability_truth=False,extra_student_NN=n,extra_frozen_teacher_NN=n,frozen_teacher_NN=dict(B01=n),selected_physical_ids=[[1,1,1,i,i] for i in range(n)])

def summary(rows):
 n=sum(r['packets'] for r in rows);keys=('KL','weighted_KL','native_packets','original_selected','extra_original_selected','extra_student_NN','extra_frozen_teacher_NN');s={k+'_sum':sum(r[k] for r in rows) for k in keys}
 quality={}
 for key in ('q_teacher','q_observation','effective_q'):
  mass=sum(r[key]['mass'] for r in rows);quality[key]=dict(count=n,mass=mass,mean=mass/n,positive=sum(r[key]['positive'] for r in rows),histogram=[sum(r[key]['histogram'][i] for r in rows) for i in range(5)])
 s.update(steps=len(rows),arm=rows[0]['arm'],packets_sum=n,extra_student_NN=n,extra_frozen_teacher_NN=n,frozen_teacher_NN=dict(B01=n),q_mass_sum=n*.5,positive_q_sum=n,none_snr_count=0,quality_statistics=quality,mass_match={k+'_sum':sum(r['mass_match'][k] for r in rows) for k in ('q_teacher_mass','observation_weighted_mass','mass_uniform_mass','difference')},all_frozen_eval=True,teacher_parameters_registered_in_student=False,student_state_keys=206,student_parameters=900722,extra_teacher_training_updates=0,extra_deployment_parameters=0,native_U_mask_unchanged=True,all_labeled_As_CE_retained=True,U_TX_truth_read=False,source_global_disable=False,student_urban_confidence_gate=False,selected_count_normalization=False,observation_proxy_is_recoverability_truth=False)
 s['mass_match']['max_absolute_difference']=0.
 return s

def predictions_fixture(base):
 specs=a.row_specs(base);ids=np.asarray(['toy'+str(i) for i in range(6)])
 for s in specs:
  root=s['root'];c=dict(row_id=s['row_id']);p=root/'prediction'
  dump(p/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=root.name,count=6,views=list(a.VIEWS),config=c,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=44))
  dump(p/'resolved_config.json',dict(config=c,views=list(a.VIEWS),model_frozen=True,separate_head_fit=False,truth_read=False,query_fit=False,checkpoint=str(root/'source/final_ssdg.pth')))
  dump(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',row_id=root.name,views=list(a.VIEWS),truth_last=True));dump(root/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=root.name))
  dump(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=[dict(view=v,dimension='overall',stratum='ALL',query_count=6) for v in a.VIEWS]))
  np.savez(p/'predictions.npz',ids=ids,**{v:np.arange(6,dtype=np.int64) for v in a.VIEWS})
  for v in a.VIEWS:np.savez(p/(v+'_scores.npz'),ids=ids,logits=np.eye(6,dtype=np.float32)*30)
 t=Path(base)/'toy_truth.json';dump(t,{s:dict(label=i,receiver='toyRX',day='toyday') for i,s in enumerate(ids)})
 return specs,ids,t,lambda s,p:dict(config=dict(row_id=s['row_id']))

def training_metadata_fixture(base):
 specs=a.row_specs(base);runs=Path(base)/'runs';roles=dict(L_s=['toyL'],U_s=['toyU'],V=['toyV']);dump(runs/a.CONTRACT_RUN/'source_contract.json',dict(role_ids=roles))
 for spec in specs:
  parent=runs/a.REF/'B01-s2026092701'/'stage20' if not spec['reused'] else spec['root']/'stage20';parentrow='B01-s2026092701' if not spec['reused'] else spec['row_id']
  for folder,is_parent,e,l,u,steps,row in [(parent,True,20,13,7,4440,parentrow),(spec['root']/'source',False,10,0,10,2220,spec['row_id'])]:
   cp=str(parent/'final_ssdg.pth');sources=[] if is_parent else [cp];c=dict(run_id=a.REF if is_parent else spec['run'],row_id=row,model_seed=a.SEED,epochs=e,label_epochs=l,pseudo_epochs=u,updates=steps,As=spec['key']!='NOAS30',checkpoint_sources=sources,Ag_mode='NONE',new_Ag=False,L_NN_per_step=768,total_L_loss_scale=1.25)
   if not is_parent:c.update(parent_source=str(parent),epoch_offset=20,step_offset=4440)
   if not is_parent and not spec['reused']:c.update(arm=spec['key'],teacher_sources={'B01':cp})
   dump(folder/'completion.json',dict(config=c,epoch=e,optimizer_steps=steps,target_access=False,head_fit_updates=0,native_fitted_probes_disabled=True,final_source_metrics=dict(source_val_count=27000)))
   dump(folder/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,selection='final_only',target_access=False));dump(folder/'initialization.json',dict(checkpoint_sources=sources,scratch_only=is_parent,target_contact=False,source_roles='EXACT_MATCH'));dump(folder/'source_contract.json',dict(role_ids=roles));dump(folder/'result.json',dict(config=c,head_fit_updates=0,U_truth_read=False,target_access=False,source_RX=list(a.RX),source_RX_holdout=None))
 return specs

class Checks(unittest.TestCase):
 def test_single_As20_source_for_student_and_teacher(self):
  with tempfile.TemporaryDirectory() as t:
   specs=training_metadata_fixture(t)
   for s in specs:self.assertEqual(len(a.training_complete(s,t)['config']['checkpoint_sources']),1)
   p=specs[-1]['root']/'source/initialization.json';x=a.read(p);x['checkpoint_sources'].append('wrong_NoAs_teacher');dump(p,x)
   with self.assertRaises(ValueError):a.training_complete(specs[-1],t)

 def test_metrics_empty_margin_and_repair(self):
  y=np.arange(6);z=np.eye(6)*30;r=a.logit_metric(y,z,np.ones(6,bool));self.assertEqual(r['mean_margin_over30'],1.);self.assertEqual(r['macro_f1'],1.)
  self.assertIsNone(a.metric(np.array([],dtype=int),np.array([],dtype=int))['accuracy']);self.assertTrue(a.reconcile_partitions(r,[a.metric(y[:3],y[:3]),a.metric(y[3:],y[3:])]))
  for old in range(16):
   for new in range(16):
    yy=np.zeros(4,dtype=int);p=np.asarray([0 if old>>i&1 else 1 for i in range(4)]);q=np.asarray([0 if new>>i&1 else 1 for i in range(4)]);m=a.paired(yy,p,q)
    self.assertEqual(m['net_repairs'],m['new_correct']-m['control_correct'])

 def test_stored_source_verdict_never_recomputed(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'runs'/a.RUN/'source_matrix_decision.json';d=decision();dump(p,d);self.assertEqual(a.read_source_decision(t),d)
   for change in [dict(formal_scores_read=True),dict(target_feedback=True),dict(NN=1),dict(source_qualified=True),dict(checks={})]:
    dump(p,dict(d,**change))
    with self.subTest(change=change),self.assertRaises(ValueError):a.read_source_decision(t)

 def test_truth_after_four_rows_all_seven_views_once(self):
  with tempfile.TemporaryDirectory() as t:
   specs,ids,truth,checker=predictions_fixture(t);seen=[];read=a.read
   def spy(path):seen.append('truth' if Path(path)==truth else str(path));return read(path)
   with patch.object(a,'read',side_effect=spy):data,meta=a.truth_last_arrays(specs,ids,truth,t,6,checker)
   self.assertEqual(len(data),4);self.assertEqual(seen.count('truth'),1);self.assertEqual(seen[-1],'truth')

 def test_late_missing_or_invalid_view_keeps_truth_closed(self):
  for mode in ('missing','nan','argmax','unfrozen'):
   with tempfile.TemporaryDirectory() as t:
    specs,ids,truth,checker=predictions_fixture(t);p=specs[-1]['root']/'prediction'
    if mode=='missing':(p/'practical_low_urban_scores.npz').unlink()
    elif mode=='unfrozen':v=a.read(p/'complete.json');v['all_parameters_frozen']=False;dump(p/'complete.json',v)
    else:
     z=np.eye(6,dtype=np.float32)*30;z[0,1]=np.nan if mode=='nan' else 31;np.savez(p/'practical_low_urban_scores.npz',ids=ids,logits=z)
    read=a.read;seen=[]
    def spy(path):
     if Path(path)==truth:seen.append(True);raise AssertionError('Truth opened too soon')
     return read(path)
    with patch.object(a,'read',side_effect=spy),self.assertRaises((ValueError,FileNotFoundError)):a.truth_last_arrays(specs,ids,truth,t,6,checker)
    self.assertFalse(seen)

 def test_mass_record_original_U_and_no_extra_gates(self):
  for arm in a.NEW_ARMS:
   for step,native in [(1,256),(222,124),(1111,256),(2220,124)]:
    r=record(arm,step,native);a.validate_quality_record(r,step,native,2,arm)
    for c in [dict(extra_frozen_teacher_NN=256),dict(U_TX_truth_read=True),dict(source_health_gate=True),dict(urban_confidence_filter=True),dict(selected_count_normalization=True),dict(q_mean=float('nan')),dict(observation_proxy_is_recoverability_truth=True)]:
     with self.subTest(change=c),self.assertRaises(ValueError):a.validate_quality_record(dict(r,**c),step,native,2,arm)
  r=record();r['mass_match']['mass_uniform_mass']+=1
  with self.assertRaises(ValueError):a.validate_quality_record(r,1,256,2,a.CANDIDATE)

 def test_whole_tail_and_summary_recount(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);rows=[];details=[dict(mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,phase=phase,level=level) for phase in ('first','second') for level in ('time_down','t2.pool')];names=('reference_steps','shape_steps','step_metrics','unit_steps','u_quality_steps');files={k:gzip.open(p/(k+'.jsonl.gz'),'wt',encoding='utf8') for k in names}
   for i in range(1,2221):
    e=(i-1)//222+1;n=124 if i%222==0 else 256;tr=record(step=i,native=n);rows.append(tr);clock=dict(step=i,cumulative_step=4440+i)
    r=dict(clock,L_forwards=6,As_packets=64,Ag_packets=0,Ag_mode='NONE',extra_control_packets=128,ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,common_CE=.8,common_KL=.2,extra_CE=.8,extra_KL=.2,SupCon=0.,SupCon_weight=.1,base_L_before_shape=1.,total_L=1.25,ordinary_raw_MSE=.01,crossview_mode='ORD128',new_losses=False,extra_U_NN=0,SupCon_details=details)
    shape=dict(clock,mode='NONE',NN=256,packets=128,applied=0,weight=.25,same_IQ_each_pair=True,new_boundary_samples=False,relative_noise_RMS=None,shape_CE=.8,shape_KL=.2,shape_RD=1.)
    native=dict(step=i,epoch=e,metrics={'train/loss':1.4,'train/pseudo_total':n,'train/pseudo_selected':2,'train/grad_norm':1.});unit=dict(clock,physical_clip_calls=i)
    for key,value in zip(names,(r,shape,native,unit,tr)):files[key].write(json.dumps(value)+'\n')
   for f in files.values():f.close()
   s=summary(rows);a.check_quality_summary(s,rows,a.CANDIDATE);self.assertEqual(s['packets_sum'],284120);self.assertEqual(s['native_packets_sum'],567000)
   dump(p/'result.json',dict(L_NN=1704960,L_As=142080,optimizer_steps=2220,Ag_packets=0,Ag_control_NN=568320,extra_U_NN=284120,U_As=284120,native_extra_U_NN=0,native_U_As=0,extra_student_U_NN=284120,extra_frozen_teacher_U_NN=284120,u_quality_summary=s,student_train_NN=1,student_eval_NN=2,teacher_train_NN=0,teacher_eval_NN=3,observed_student_train_NN_including_extra=284121,native_NN=6,training_total_NN=568246,wall_seconds=1.));dump(p/'u_quality_progress.json',s);dump(p/'completion.json',{})
   for name in ('metrics_epoch.jsonl','epoch_metrics.jsonl'):
    with (p/name).open('w',encoding='utf8') as f:
     for e in range(1,11):f.write(json.dumps(dict(epoch=e,lr=.0002-(e-1)*(.0002-.000001)/9,val_tx_acc=98.,train_grad_backbone=1.,epoch_time_s=1.))+'\n')
   x=a.parse_tail(p,a.CANDIDATE);self.assertEqual(len(x['teacher_rows']),2220);self.assertEqual(x['old_U_total'],567000);self.assertAlmostEqual(x['curve'][0]['total_L'],1.25)
   with self.assertRaises(ValueError):a.check_quality_summary(dict(s,extra_frozen_teacher_NN=568240),rows,a.CANDIDATE)

 def test_saved_physics_recounts_q_without_IQ(self):
  rows=[record(step=i,native=124 if i%222==0 else 256) for i in range(1,2221)]
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'metadata.jsonl.gz'
   with gzip.open(p,'wt',encoding='utf8') as f:
    for r in rows:f.write(json.dumps(dict(step=r['cumulative_step'],observations=[dict(sample_id=pid,scene=a.SCENES[-1],state_end='LOS',receiver_lock_status='locked',channel_equalization_applied=False,signal_to_added_noise_db=5.) for pid in r['selected_physical_ids']]))+'\n')
   x=a.physics_recount(p,rows);self.assertEqual(x['packets'],284120);self.assertEqual(x['max_q_observation_mass_error'],0.)
   bad=copy.deepcopy(rows);bad[0]['strata'][0]['q_observation_mass']=0.
   with self.assertRaises(ValueError):a.physics_recount(p,bad)

 def test_actual_cost_deduplicates_single_As20(self):
  with tempfile.TemporaryDirectory() as t:
   specs=a.row_specs(t);base=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=1.)
   for row in specs:
    for stage in ('stage20','source') if row['reused'] else ('source',):
     d=dict(base)
     if not row['reused']:d.update(extra_student_U_NN=284120,extra_frozen_teacher_U_NN=284120,native_NN=3162962,observed_student_train_NN_including_extra=2556080,training_total_NN=3731202)
     dump(row['root']/stage/'result.json',d)
    dump(row['root']/'source_qualification/complete.json',dict(status='SOURCE_READOUT_COMPLETE',NN=12600,wall_seconds=2.));dump(row['root']/'prediction/complete.json',dict(actual_NN=1176002,wall_seconds=3.))
   dump(Path(t)/'runs'/a.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=4));c=a.cost_analysis(t,specs,None)
   self.assertEqual(c['new_total_remote_NN_known'],9839612);self.assertEqual(c['new_updates'],4440);self.assertEqual(c['unique_total_updates_including_reused'],17760);self.assertEqual(c['unique_shared_prerequisite_epochs'],20);self.assertEqual(c['candidate_prerequisites_plus_new_epochs_each'],30)

 def test_immutable_output_precedes_formal_reads(self):
  with tempfile.TemporaryDirectory() as t:
   dump(Path(t)/'runs'/a.RUN/'source_matrix_decision.json',decision());p=Path(t)/'existing';p.mkdir()
   with patch.object(a,'formal_analysis',side_effect=AssertionError('Must not open formal')),self.assertRaises(ValueError):a.run(t,p)

if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not r.wasSuccessful():raise SystemExit(1)
 print(json.dumps(dict(status='PASS',tests=r.testsRun,NN=0,IQ_reads=0,checkpoint_reads=0,real_truth_reads=0,updates=0,remote_calls=0)))
