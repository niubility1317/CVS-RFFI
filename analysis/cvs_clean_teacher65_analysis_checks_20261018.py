"""Small deterministic synthetic fixtures only: no model, IQ, checkpoint or truth dataset."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE=Path(__file__).with_name('cvs_clean_teacher65_analysis_20261018.py')
spec=importlib.util.spec_from_file_location('clean_teacher65_analysis',MODULE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')

def decision(status='SOURCE_READOUT_DECIDED',qualified=False):
 return dict(status=status,formal_scores_read=False,target_feedback=False,model_seed_count=1,NN=0,source_qualified=qualified,checks=dict(clean=True,midlow=qualified,low=True,FC_vs_FA=True,FC_vs_ZERO=True,FC_vs_FA_low=True))

def fixture(base):
 specs=a.row_specs(base);ids=np.asarray(['toy'+str(i) for i in range(6)]);truth={s:dict(label=i,receiver='toyRX',day='toyday') for i,s in enumerate(ids)}
 for s in specs:
  root=s['root'];c=dict(row_id=s['row_id']);p=root/'prediction'
  dump(p/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=root.name,count=6,views=list(a.VIEWS),config=c,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=44))
  dump(p/'resolved_config.json',dict(config=c,views=list(a.VIEWS),model_frozen=True,separate_head_fit=False,truth_read=False,query_fit=False,checkpoint=str(root/'source/final_ssdg.pth')))
  dump(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',row_id=root.name,views=list(a.VIEWS),truth_last=True))
  dump(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=[dict(view=v,dimension='overall',stratum='ALL',query_count=6) for v in a.VIEWS]))
  dump(root/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=root.name))
  np.savez(p/'predictions.npz',ids=ids,**{v:np.arange(6,dtype=np.int64) for v in a.VIEWS})
  z=np.eye(6,dtype=np.float32)*30
  for v in a.VIEWS:np.savez(p/(v+'_scores.npz'),ids=ids,logits=z)
 t=Path(base)/'toy_truth.json';dump(t,truth)
 return specs,ids,t,lambda s,p:dict(config=dict(row_id=s['row_id']))

def teacher_record(arm='FROZEN_NOAS20_EMA',step=1,native_n=256,selected=2):
 n=min(128,native_n);local=(step-1)//222+1;teacher='EVOLVING_AS_EMA' if arm=='ZERO_KL' else arm;weight=0. if arm=='ZERO_KL' else .16*min(local/5.,1.);positive=n//2
 stats=dict(q_mean=.5,q_mass=n*.5,positive_q=positive,pmax_mean=.9,pmax_quantiles=[.5,.6,.97,.99,1.],q_quantiles=[0.,0.,.5,1.,1.],q_histogram=[n-positive,0,0,0,positive],pseudo_counts=[n,0,0,0,0,0],positive_pseudo_counts=[positive,0,0,0,0,0],KL=.5,unweighted_KL=1.)
 return dict(step=step,cumulative_step=4440+step,epoch=20+local,local_epoch=local,arm=arm,packets=n,native_packets=native_n,selected_teacher=teacher,weight=weight,KL=.5,weighted_KL=.5*weight,q_mean=.5,positive_q=positive,original_selected=selected,extra_original_selected=1,original_mask_unchanged=True,old_U_masks_unchanged=True,U_TX_truth_read=False,urban_confidence_filter=False,extra_student_NN=n,extra_frozen_teacher_NN=2*n,frozen_teacher_NN=dict(B01=n,B00=n),teachers={k:copy.deepcopy(stats) for k in a.TEACHERS},teacher_disagreement={str(i):0. for i in range(3)},selected_physical_ids=[[1,1,1,i,i] for i in range(n)],dropout_rng_restored=True,source_health_gate=False,subset_rule='first_min_128_native_U_batch_no_padding',selected_count_normalization=False,temperature=1.)

def teacher_summary(records):
 p=records[0];s={k+'_sum':sum(r[k] for r in records) for k in a.TEACHER_SUMS}
 s.update(steps=len(records),arm=p['arm'],packets_sum=sum(r['packets'] for r in records),extra_student_NN=sum(r['extra_student_NN'] for r in records),extra_frozen_teacher_NN=sum(r['extra_frozen_teacher_NN'] for r in records),frozen_teacher_NN={k:sum(r['frozen_teacher_NN'][k] for r in records) for k in ('B01','B00')},q_mass_sum=sum(r['teachers'][r['selected_teacher']]['q_mass'] for r in records),all_frozen_eval=True,teacher_parameters_registered_in_student=False,student_state_keys=206,student_parameters=900722,extra_teacher_training_updates=0,extra_deployment_parameters=0,native_U_mask_unchanged=True,all_labeled_As_CE_retained=True,U_TX_truth_read=False,source_global_disable=False,student_urban_confidence_gate=False,selected_count_normalization=False)
 return s

def training_metadata_fixture(base):
 specs=a.row_specs(base);runs=Path(base)/'runs';roles={'L_s':['toyL'],'U_s':['toyU'],'V':['toyV']};dump(runs/a.CONTRACT_RUN/'source_contract.json',dict(role_ids=roles))
 for spec in specs:
  parent=runs/a.REF/'B01-s2026092701'/'stage20' if not spec['reused'] else spec['root']/'stage20';parent_row='B01-s2026092701' if not spec['reused'] else spec['row_id']
  for folder,parent_stage,e,l,u,steps,row in [(parent,True,20,13,7,4440,parent_row),(spec['root']/'source',False,10,0,10,2220,spec['row_id'])]:
   parents={'FROZEN_AS20_EMA':str(runs/a.REF/'B01-s2026092701'/'stage20/final_ssdg.pth'),'FROZEN_NOAS20_EMA':str(runs/a.REF/'B00-s2026092701'/'stage20/final_ssdg.pth')}
   sources=[] if parent_stage else ([str(parent/'final_ssdg.pth')] if spec['reused'] else list(parents.values()))
   c=dict(run_id=a.REF if parent_stage else spec['run'],row_id=row,model_seed=a.SEED,epochs=e,label_epochs=l,pseudo_epochs=u,updates=steps,As=spec['key']!='NOAS30',checkpoint_sources=sources,Ag_mode='NONE',new_Ag=False,L_NN_per_step=768,total_L_loss_scale=1.25)
   if not parent_stage:c.update(parent_source=str(parent),epoch_offset=20,step_offset=4440)
   if not parent_stage and not spec['reused']:c.update(arm=spec['key'],teacher_sources=parents)
   dump(folder/'completion.json',dict(config=c,epoch=e,optimizer_steps=steps,target_access=False,head_fit_updates=0,native_fitted_probes_disabled=True,final_source_metrics=dict(source_val_count=27000)))
   dump(folder/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,selection='final_only',target_access=False))
   dump(folder/'initialization.json',dict(checkpoint_sources=sources,scratch_only=parent_stage,target_contact=False,source_roles='EXACT_MATCH'))
   dump(folder/'source_contract.json',dict(role_ids=roles))
   dump(folder/'result.json',dict(config=c,head_fit_updates=0,U_truth_read=False,target_access=False,source_RX=list(a.RX),source_RX_holdout=None))
 return specs

def update_metadata_config(folder,changes):
 for filename in ('completion.json','source_frozen.json','result.json'):
  path=folder/filename;value=a.read(path);value['config'].update(changes);dump(path,value)

class Checks(unittest.TestCase):
 def test_training_checkpoint_sources_accept_new_dual_and_reused_single(self):
  with tempfile.TemporaryDirectory() as t:
   specs=training_metadata_fixture(t)
   for spec in specs:
    actual=a.training_complete(spec,t);expected_count=1 if spec['reused'] else 2;self.assertEqual(len(actual['config']['checkpoint_sources']),expected_count)
   new=specs[-1];folder=new['root']/'source';path=folder/'initialization.json';original=a.read(path);bad=copy.deepcopy(original);bad['checkpoint_sources'].pop();dump(path,bad)
   with self.assertRaises(ValueError):a.training_complete(new,t)
   dump(path,original)
   teachers=a.read(folder/'completion.json')['config']['teacher_sources'];bad=dict(teachers);bad['FROZEN_NOAS20_EMA']=teachers['FROZEN_AS20_EMA'];update_metadata_config(folder,dict(teacher_sources=bad))
   with self.assertRaises(ValueError):a.training_complete(new,t)
   update_metadata_config(folder,dict(teacher_sources=teachers))
   noas=Path(t)/'runs'/a.REF/'B00-s2026092701'/'stage20';update_metadata_config(noas,dict(As=True))
   with self.assertRaises(ValueError):a.training_complete(new,t)

 def test_six_class_metrics_and_empty(self):
  y=np.arange(6,dtype=np.int64);p=y.copy();p[0]=1;m=a.metric(y,p)
  self.assertEqual(m['correct'],5);self.assertAlmostEqual(m['macro_f1'],(0+2/3+4)/6)
  e=a.metric(np.asarray([],dtype=int),np.asarray([],dtype=int));self.assertIsNone(e['accuracy']);self.assertTrue(e['empty_cell'])

 def test_all_correctness_pair_cases(self):
  for old in range(16):
   for new in range(16):
    y=np.zeros(4,dtype=int);p=np.asarray([0 if old>>i&1 else 1 for i in range(4)]);q=np.asarray([0 if new>>i&1 else 1 for i in range(4)])
    r=a.paired(y,p,q,np.asarray([True,False,True,False]));self.assertEqual(r['repairs']-r['new_errors'],r['new_correct']-r['control_correct']);self.assertEqual(r['fixed_NOAS30_clean_correct_count'],2)

 def test_margin_scale_and_partitions(self):
  y=np.arange(6);z=np.eye(6)*30;m=a.logit_metric(y,z,np.ones(6,bool));self.assertEqual(m['mean_margin_over30'],1.)
  self.assertTrue(a.reconcile_partitions(m,[a.metric(y[:3],y[:3]),a.metric(y[3:],y[3:])]))
  with self.assertRaises(ValueError):a.reconcile_partitions(m,[a.metric(y[:3],y[:3])])
  z[0,0]=np.nan
  with self.assertRaises(ValueError):a.logit_metric(y,z,np.ones(6,bool))

 def test_source_producer_complete_and_partial_statuses(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'runs'/a.RUN/'source_matrix_decision.json'
   for status in a.DECISION_COMPLETE|a.DECISION_PARTIAL:
    d=decision(status);dump(p,d);self.assertEqual(a.read_source_decision(t),d)
   for change in [dict(status='SOURCE_BUDGET_READOUT_COMPLETE'),dict(formal_scores_read=True),dict(target_feedback=True),dict(model_seed_count=2),dict(NN=1),dict(source_qualified=True),dict(checks=dict(clean=1))]:
    d=decision();d.update(change);dump(p,d)
    with self.subTest(change=change),self.assertRaises(ValueError):a.read_source_decision(t)

 def test_truth_last_once_after_six_rows_seven_views(self):
  with tempfile.TemporaryDirectory() as t:
   specs,ids,truth,checker=fixture(t);seen=[];read=a.read
   def counted(path):
    if Path(path)==truth:seen.append('truth')
    else:seen.append(str(path))
    return read(path)
   with patch.object(a,'read',side_effect=counted):data,meta=a.truth_last_arrays(specs,ids,truth,t,count=6,checker=checker)
   self.assertEqual(len(data),6);self.assertEqual(seen.count('truth'),1);self.assertEqual(seen[-1],'truth');self.assertEqual(meta['y'].tolist(),list(range(6)))

 def test_missing_late_score_keeps_truth_closed(self):
  with tempfile.TemporaryDirectory() as t:
   specs,ids,truth,checker=fixture(t);(specs[-1]['root']/'prediction'/('practical_low_urban_scores.npz')).unlink();read=a.read;seen=[]
   def checked(path):
    if Path(path)==truth:seen.append('truth');raise AssertionError('Truth opened too early')
    return read(path)
   with patch.object(a,'read',side_effect=checked),self.assertRaises(FileNotFoundError):a.truth_last_arrays(specs,ids,truth,t,count=6,checker=checker)
   self.assertEqual(seen,[])

 def test_prediction_identity_flags_and_logits_negatives(self):
  for mode in ('unfrozen','missing_scoring','missing_score_view','bad_ids','nan','argmax'):
   with tempfile.TemporaryDirectory() as t:
    specs,ids,truth,checker=fixture(t);r=specs[-1]['root'];p=r/'prediction'
    if mode=='unfrozen':d=a.read(p/'complete.json');d['all_parameters_frozen']=False;dump(p/'complete.json',d)
    elif mode=='missing_scoring':(r/'scoring/complete.json').unlink()
    elif mode=='missing_score_view':d=a.read(r/'scoring/scores.json');d['results'].pop();dump(r/'scoring/scores.json',d)
    else:
     z=np.eye(6,dtype=np.float32)*30;iz=ids.copy()
     if mode=='bad_ids':iz=iz[::-1]
     elif mode=='nan':z[0,0]=np.nan
     elif mode=='argmax':z[0,1]=31
     np.savez(p/'clean_scores.npz',ids=iz,logits=z)
    read=a.read;seen=[]
    def checked(path):
     if Path(path)==truth:seen.append('truth');raise AssertionError('Truth opened too early')
     return read(path)
    with patch.object(a,'read',side_effect=checked),self.assertRaises((ValueError,FileNotFoundError)):a.truth_last_arrays(specs,ids,truth,t,count=6,checker=checker)
    self.assertEqual(seen,[])

 def test_fixed_teacher_record_and_negative_policies(self):
  for arm in a.NEW_ARMS:
   for step,native in [(1,256),(222,124),(1111,256),(2220,124)]:
    r=teacher_record(arm,step,native);a.validate_teacher_record(r,step,native,2,arm)
    for change in [dict(weight=.75),dict(packets=129),dict(extra_student_NN=256),dict(extra_frozen_teacher_NN=128),dict(original_mask_unchanged=False),dict(U_TX_truth_read=True),dict(urban_confidence_filter=True),dict(dropout_rng_restored=False),dict(source_health_gate=True),dict(weighted_KL=-1.),dict(q_mean=float('nan')),dict(selected_count_normalization=True)]:
     bad=dict(r,**change)
     with self.subTest(arm=arm,step=step,change=change),self.assertRaises(ValueError):a.validate_teacher_record(bad,step,native,2,arm)
  r=teacher_record();r['teachers'][a.CANDIDATE]['pseudo_counts'][0]-=1
  with self.assertRaises(ValueError):a.validate_teacher_record(r,1,256,2,a.CANDIDATE)

 def test_complete_teacher_summary_counts_unpadded_tail_and_no_fitted_state(self):
  rows=[teacher_record(step=i,native_n=124 if i%222==0 else 256) for i in range(1,2221)];s=teacher_summary(rows);a.check_teacher_summary(s,rows,a.CANDIDATE)
  self.assertEqual(s['packets_sum'],284120);self.assertEqual(s['native_packets_sum'],567000)
  for change in [dict(extra_student_NN=284160),dict(extra_frozen_teacher_NN=568320),dict(student_state_keys=207),dict(teacher_parameters_registered_in_student=True),dict(q_mass_sum=0.)]:
   with self.subTest(change=change),self.assertRaises(ValueError):a.check_teacher_summary(dict(s,**change),rows,a.CANDIDATE)

 def test_entire_2220_step_tail_preserves_L_and_records_every_U_prefix(self):
  arm=a.CANDIDATE
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);teaching=[]
   details=[dict(mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,phase=phase,level=level) for phase in ('first','second') for level in ('time_down','t2.pool')]
   files={n:gzip.open(p/(n+'.jsonl.gz'),'wt',encoding='utf-8') for n in ('reference_steps','shape_steps','step_metrics','unit_steps','clean_teacher_steps')}
   for i in range(1,2221):
    e=(i-1)//222+1;clock=dict(step=i,cumulative_step=4440+i);n=124 if i%222==0 else 256;tr=teacher_record(arm,i,n);teaching.append(tr)
    r=dict(clock,L_forwards=6,As_packets=64,Ag_packets=0,Ag_mode='NONE',extra_control_packets=128,ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,common_CE=.8,common_KL=.2,extra_CE=.8,extra_KL=.2,SupCon=0.,SupCon_weight=.1,base_L_before_shape=1.,total_L=1.25,ordinary_raw_MSE=.01,crossview_mode='ORD128',new_losses=False,extra_U_NN=0,SupCon_details=details)
    shape=dict(clock,mode='NONE',NN=256,packets=128,applied=0,weight=.25,same_IQ_each_pair=True,new_boundary_samples=False,relative_noise_RMS=None,shape_CE=.8,shape_KL=.2,shape_RD=1.)
    native=dict(step=i,epoch=e,metrics={'train/loss':1.4,'train/pseudo_total':n,'train/pseudo_selected':2,'train/grad_norm':1.});unit=dict(clock,physical_clip_calls=i)
    for name,value in zip(files,(r,shape,native,unit,tr)):files[name].write(json.dumps(value)+chr(10))
   for f in files.values():f.close()
   summary=teacher_summary(teaching);native=6
   dump(p/'result.json',dict(L_NN=1704960,L_As=142080,optimizer_steps=2220,Ag_packets=0,Ag_control_NN=568320,extra_U_NN=284120,U_As=284120,native_extra_U_NN=0,native_U_As=0,extra_student_U_NN=284120,extra_frozen_teacher_U_NN=568240,clean_teacher_summary=summary,student_train_NN=1,student_eval_NN=2,teacher_train_NN=0,teacher_eval_NN=3,observed_student_train_NN_including_extra=284121,native_NN=native,training_total_NN=native+284120+568240,wall_seconds=1.))
   dump(p/'clean_teacher_progress.json',summary);dump(p/'completion.json',{})
   for name in ('metrics_epoch.jsonl','epoch_metrics.jsonl'):
    with (p/name).open('w',encoding='utf-8') as f:
     for e in range(1,11):f.write(json.dumps(dict(epoch=e,lr=.0002-(e-1)*(.0002-.000001)/9,val_tx_acc=98.,train_grad_backbone=1.,epoch_time_s=1.))+chr(10))
   x=a.parse_tail(p,arm);self.assertEqual(x['old_U_total'],567000);self.assertEqual(len(x['teacher_rows']),2220);self.assertEqual(x['gradient_telemetry_field_counts']['train/grad_norm'],2220);self.assertLess(x['maximum_L_decomposition_error'],1e-12)
   self.assertAlmostEqual(x['curve'][0]['total_L'],1.25);self.assertAlmostEqual(x['cost']['training_total_NN'],852366)

 def test_cost_deduplicates_parent_teachers_and_extra_forward_aliases(self):
  with tempfile.TemporaryDirectory() as t:
   specs=a.row_specs(t);base=dict(student_train_NN=100,student_eval_NN=20,teacher_train_NN=0,teacher_eval_NN=30,wall_seconds=1.)
   for row in specs:
    for stage in ('stage20','source') if row['reused'] else ('source',):
     data=dict(base)
     if not row['reused']:data.update(extra_student_U_NN=a.EXTRA_STUDENT,extra_frozen_teacher_U_NN=a.EXTRA_FROZEN,extra_U_NN=a.EXTRA_STUDENT,observed_student_train_NN_including_extra=100+a.EXTRA_STUDENT,native_NN=150,training_total_NN=150+a.EXTRA_STUDENT+a.EXTRA_FROZEN)
     dump(row['root']/stage/'result.json',data)
    dump(row['root']/'source_qualification/complete.json',dict(status='SOURCE_READOUT_COMPLETE',NN=12600,wall_seconds=2.))
    dump(row['root']/'prediction/complete.json',dict(actual_NN=1176002,wall_seconds=3.))
   dump(Path(t)/'runs'/a.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=6))
   c=a.cost_analysis(t,specs,None);each=150+a.EXTRA_STUDENT+a.EXTRA_FROZEN
   self.assertEqual(c['new_total_remote_NN_known'],4*(each+12600+1176002)+6);self.assertEqual(c['unique_total_remote_NN_known'],c['new_total_remote_NN_known']+2*(300+12600+1176002));self.assertEqual(c['candidate_lifetime_NN_each'][a.CANDIDATE],each+12600+1176002+300)
   self.assertEqual(c['unique_total_updates_including_reused'],22200);self.assertEqual(c['new_source_readout_wall_seconds'],8.);self.assertEqual(c['new_extra_student_U_NN'],4*a.EXTRA_STUDENT);self.assertEqual(c['new_extra_frozen_teacher_U_NN'],4*a.EXTRA_FROZEN)
   bad=specs[-1]['root']/'source/result.json';data=a.read(bad);data['observed_student_train_NN_including_extra']+=1;dump(bad,data)
   with self.assertRaises(ValueError):a.native_cost(bad.parent)

 def test_output_immutability_before_any_formal_access(self):
  with tempfile.TemporaryDirectory() as t:
   dump(Path(t)/'runs'/a.RUN/'source_matrix_decision.json',decision());out=Path(t)/'existing';out.mkdir()
   with patch.object(a,'formal_analysis',side_effect=AssertionError('No formal read')),self.assertRaises(ValueError):a.run(t,out)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(1)
 print(json.dumps(dict(status='PASS',tests=result.testsRun,NN=0,IQ_reads=0,checkpoint_reads=0,real_truth_reads=0,updates=0,remote_calls=0)))
