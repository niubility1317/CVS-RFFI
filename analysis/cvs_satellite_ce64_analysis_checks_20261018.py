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

MODULE=Path(__file__).with_name('cvs_satellite_ce64_analysis_20261018.py')
spec=importlib.util.spec_from_file_location('satellite_ce64_analysis',MODULE)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')

def decision(status='SOURCE_READOUT_DECIDED',qualified=False):
 return dict(status=status,formal_scores_read=False,target_feedback=False,model_seed_count=1,NN=0,source_qualified=qualified,checks=dict(clean=True,midlow=qualified,low=True,temperature_vs_half=True))

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

def ce_record(arm='CE_T2'):
 temperature,weight=(1.,.5) if arm=='CE_HALF' else (2.,1.)
 old=.8;new=.4 if arm=='CE_HALF' else .6;correction=.25*(new-old)
 p=dict(step=1,cumulative_step=4441,arm=arm,temperature=temperature,weight=weight,label_smoothing=.01,T_squared=False,old_sat_CE0=old,old_sat_CE1=old,old_sat_CE=old,new_sat_CE0=new,new_sat_CE1=new,new_sat_CE=new,satellite_CE_delta_unscaled=new-old,satellite_CE_correction=correction,mixed_group_weight=.5,satellite_fraction=.5,satellite_objective_coefficient=.25,per_pass_coefficient=.125,satellite_slots=64,mixed_slots=128,passes=2,two_separate_CE_mean=True,extra_NN=0,extra_parameters=0,base_L_before_ce=1.25,total_L=1.25+correction,gradient_alive=True,forward_count=5,channel_calls=1,all_satellite_CE_retained=True)
 return p,dict(total_L=p['total_L'],satellite_CE_original=old,satellite_CE_modified=new,satellite_CE_correction=correction)

def ce_summary(records):
 p=records[0];n=len(records);keys=(*a.CE_SUMS,'old_sat_CE0','old_sat_CE1','new_sat_CE0','new_sat_CE1','satellite_slots','mixed_slots')
 sums={k:sum(r[k] for r in records) for k in keys}
 s={k+'_sum':v for k,v in sums.items()};s.update({k+'_mean':v/n for k,v in sums.items()});s.update(steps=n,arm=p['arm'],temperature=p['temperature'],weight=p['weight'],label_smoothing=.01,T_squared=False,extra_NN=0,extra_parameters=0,gradient_alive_steps=n,correction_nonzero_steps=sum(r['satellite_CE_correction']!=0 for r in records),forward_reuse=True,all_labeled_satellite_CE_retained=True,two_separate_CE_mean=True,clean_CE_unchanged=True,ordinary_CE_unchanged=True,mixed_RDrop_KL_unchanged=True,sample_quality_weighting=False,confidence_gate=False)
 return s

class Checks(unittest.TestCase):
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

 def test_truth_last_once_after_four_rows_seven_views(self):
  with tempfile.TemporaryDirectory() as t:
   specs,ids,truth,checker=fixture(t);seen=[];read=a.read
   def counted(path):
    if Path(path)==truth:seen.append('truth')
    else:seen.append(str(path))
    return read(path)
   with patch.object(a,'read',side_effect=counted):data,meta=a.truth_last_arrays(specs,ids,truth,t,count=6,checker=checker)
   self.assertEqual(len(data),4);self.assertEqual(seen.count('truth'),1);self.assertEqual(seen[-1],'truth');self.assertEqual(meta['y'].tolist(),list(range(6)))

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

 def test_fixed_ce_record_and_negative_decomposition(self):
  for arm in a.NEW_ARMS:
   p,r=ce_record(arm);a.validate_ce_record(p,1,r,1.25,arm)
   for change in [dict(temperature=3.),dict(weight=.75),dict(T_squared=True),dict(satellite_slots=128),dict(extra_NN=128),dict(two_separate_CE_mean=False),dict(satellite_CE_correction=0.),dict(cumulative_step=1),dict(total_L=1.25),dict(gradient_alive=False),dict(all_satellite_CE_retained=False),dict(new_sat_CE=float('nan'))]:
    bad=dict(p,**change)
    with self.subTest(arm=arm,change=change),self.assertRaises(ValueError):a.validate_ce_record(bad,1,r,1.25,arm)
  p,r=ce_record('CE_HALF');p.update(new_sat_CE0=.5,new_sat_CE1=.5,new_sat_CE=.5,satellite_CE_delta_unscaled=-.3,satellite_CE_correction=-.075,total_L=1.175);r.update(satellite_CE_modified=.5,satellite_CE_correction=-.075,total_L=1.175)
  with self.assertRaises(ValueError):a.validate_ce_record(p,1,r,1.25,'CE_HALF')

 def test_complete_ce_summary(self):
  for arm in a.NEW_ARMS:
   p,_=ce_record(arm);rows=[dict(p) for _ in range(2220)];s=ce_summary(rows);a.check_ce_summary(s,rows,arm)
   s['new_sat_CE_sum']-=1
   with self.assertRaises(ValueError):a.check_ce_summary(s,rows,arm)

 def test_entire_2220_step_tail_uses_post_replacement_total_once(self):
  for arm in a.NEW_ARMS:
   with tempfile.TemporaryDirectory() as t:
    p=Path(t);ce,r0=ce_record(arm);ce_rows=[]
    details=[dict(mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,phase=phase,level=level) for phase in ('first','second') for level in ('time_down','t2.pool')]
    files={n:gzip.open(p/(n+'.jsonl.gz'),'wt',encoding='utf-8') for n in ('reference_steps','shape_steps','step_metrics','unit_steps','satellite_ce_steps')}
    for i in range(1,2221):
     e=(i-1)//222+1;clock=dict(step=i,cumulative_step=4440+i)
     r=dict(clock,L_forwards=6,As_packets=64,Ag_packets=0,Ag_mode='NONE',extra_control_packets=128,ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,common_CE=.8,common_KL=.2,extra_CE=.8,extra_KL=.2,SupCon=0.,SupCon_weight=.1,base_L_before_shape=1.,base_L_before_ce=1.25,ordinary_raw_MSE=.01,crossview_mode='ORD128',new_losses=False,extra_U_NN=0,SupCon_details=details,**r0)
     shape=dict(clock,mode='NONE',NN=256,packets=128,applied=0,weight=.25,same_IQ_each_pair=True,new_boundary_samples=False,relative_noise_RMS=None,shape_CE=.8,shape_KL=.2,shape_RD=1.)
     native=dict(step=i,epoch=e,metrics={'train/loss':1.4,'train/pseudo_total':124 if i%222==0 else 256,'train/pseudo_selected':0,'train/grad_norm':1.})
     unit=dict(clock,physical_clip_calls=i);cr=dict(ce,**clock);ce_rows.append(cr)
     for name,value in zip(files,(r,shape,native,unit,cr)):files[name].write(json.dumps(value)+chr(10))
    for f in files.values():f.close()
    summary=ce_summary(ce_rows)
    dump(p/'result.json',dict(L_NN=1704960,L_As=142080,optimizer_steps=2220,Ag_packets=0,Ag_control_NN=568320,extra_U_NN=0,U_As=0,satellite_ce_summary=summary,student_train_NN=1,student_eval_NN=2,teacher_train_NN=0,teacher_eval_NN=3,wall_seconds=1.))
    dump(p/'satellite_ce_progress.json',summary);dump(p/'completion.json',{})
    for name in ('metrics_epoch.jsonl','epoch_metrics.jsonl'):
     with (p/name).open('w',encoding='utf-8') as f:
      for e in range(1,11):f.write(json.dumps(dict(epoch=e,lr=.0002-(e-1)*(.0002-.000001)/9,val_tx_acc=98.,train_grad_backbone=1.,epoch_time_s=1.))+chr(10))
    x=a.parse_tail(p,arm);self.assertEqual(x['old_U_total'],567000);self.assertEqual(len(x['ce_rows']),2220);self.assertEqual(x['gradient_telemetry_field_counts']['train/grad_norm'],2220);self.assertLess(x['maximum_L_decomposition_error'],1e-12)
    self.assertAlmostEqual(x['curve'][0]['total_L'],ce['total_L'])
 
 def test_cost_deduplicates_shared_parent_and_actual_smoke(self):
  with tempfile.TemporaryDirectory() as t:
   specs=a.row_specs(t)
   for s in specs:
    for stage in ('stage20','source') if s['reused'] else ('source',):dump(s['root']/stage/'result.json',dict(student_train_NN=100,student_eval_NN=20,teacher_train_NN=0,teacher_eval_NN=30,wall_seconds=1.))
    dump(s['root']/'source_qualification/complete.json',dict(status='SOURCE_READOUT_COMPLETE',NN=12600,wall_seconds=2.))
    dump(s['root']/'prediction/complete.json',dict(actual_NN=1176002,wall_seconds=3.))
   dump(Path(t)/'runs'/a.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2))
   c=a.cost_analysis(t,specs,None)
   self.assertEqual(c['new_total_remote_NN_known'],2*(150+12600+1176002)+2);self.assertEqual(c['unique_total_remote_NN_known'],c['new_total_remote_NN_known']+2*(300+12600+1176002));self.assertEqual(c['candidate_lifetime_NN_each']['CE_T2'],150+12600+1176002+150)
   self.assertEqual(c['unique_total_updates_including_reused'],17760);self.assertEqual(c['new_source_readout_wall_seconds'],4.);self.assertEqual(c['new_extra_CE_NN'],0)
 
 def test_output_immutability_before_any_formal_access(self):
  with tempfile.TemporaryDirectory() as t:
   dump(Path(t)/'runs'/a.RUN/'source_matrix_decision.json',decision());out=Path(t)/'existing';out.mkdir()
   with patch.object(a,'formal_analysis',side_effect=AssertionError('No formal read')),self.assertRaises(ValueError):a.run(t,out)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(1)
 print(json.dumps(dict(status='PASS',tests=result.testsRun,NN=0,IQ_reads=0,checkpoint_reads=0,real_truth_reads=0,updates=0,remote_calls=0)))
