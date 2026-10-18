"""Focused tests for seed identity, paired inference and truth-last ordering."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_As_seed_replication49_results_20261018 as a

class Checks(unittest.TestCase):
 def rates(self):
  return {s:{'B00':dict(clean=.9,rx={str(r):.9 for r in (1,3,4,6,8)},urban=[[.8,.5,.4],[.8,.5,.4]]),'B01':dict(clean=.9,rx={str(r):.9 for r in (1,3,4,6,8)},urban=[[.8,.6,.5],[.8,.6,.5]])} for s in a.SEEDS}
 def test_source_qualification_all_seeds_and_negative(self):
  r=self.rates();self.assertTrue(a.qualify_source(r)['source_As_replicated_all3']);r[a.SEEDS[-1]]['B01']['clean']=.89;self.assertFalse(a.qualify_source(r)['source_As_replicated_all3'])
 def test_source_missing_seed_and_nonfinite(self):
  r=self.rates();r.pop(a.SEEDS[-1])
  with self.assertRaises(ValueError):a.qualify_source(r)
  r=self.rates();r[a.SEEDS[-1]]['B01']['urban'][0][1]=float('nan')
  with self.assertRaises(ValueError):a.qualify_source(r)
 def test_fixed_clean_cohort_and_error_reallocation(self):
  r=a.clean_cohort_effect(np.array([0,1,2,3]),np.array([0,0,2,3]),np.array([0,1,0,3]),np.array([1,1,2,0]))
  self.assertEqual((r['repairs'],r['new_errors'],r['net']),(2,1,1));self.assertEqual(r['accuracy_delta_pp'],25);self.assertEqual((r['fixed_B00_clean_correct_count'],r['B00_retained'],r['B01_retained']),(3,1,2))
 def test_empty_cohort_is_missing_not_zero_accuracy(self):
  r=a.clean_cohort_effect(np.array([0,1]),np.array([1,0]),np.array([0,1]),np.array([1,0]));self.assertIsNone(r['B01_retention']);self.assertEqual(r['fixed_B00_clean_correct_count'],0)
 def test_model_seed_stamp_covers_all_metrics(self):
  r=dict(metrics=[dict(model_seed=a.SEEDS[0])],views={'clean':dict(model_seed=a.SEEDS[0])});row=a.rows()[-1];a.stamp_row_identity(r,row);self.assertEqual(r['model_seed'],a.SEEDS[-1]);self.assertEqual(r['metrics'][0]['model_seed'],a.SEEDS[-1]);self.assertEqual(r['views']['clean']['model_seed'],a.SEEDS[-1]);self.assertFalse(r['reused_seed01'])
 def test_paired_mean_sample_SD_not_channel_draw_count(self):
  rows=[dict(model_seed=s,view=v,accuracy_delta_pp=g,macro_F1_delta_pp=g/2) for s,g in zip(a.SEEDS,[1,3,5]) for v in a.VIEWS];r=a.paired_summary(rows)[0];self.assertEqual(r['models'],3);self.assertEqual(r['paired_accuracy_mean_pp'],3);self.assertEqual(r['paired_accuracy_SD_pp'],2)
  with self.assertRaises(ValueError):a.paired_summary(rows[:-1])
 def test_macroF1_counts_absent_registered_classes(self):
  r=a.base.metric([0,1,2,3],[0,1,0,3]);self.assertEqual(r['accuracy'],.75);self.assertAlmostEqual(r['macro_f1'],4/9)
 def test_truth_never_opens_on_last_incomplete_row(self):
  seen=[]
  def guard(r):
   seen.append(r['key'])
   if r==a.rows()[-1]:raise ValueError('last prediction incomplete')
  with patch.object(a.base,'read',return_value=dict(status='SOURCE_ANALYZED',formal_scores_read=False,model_seed_count=3)),patch.object(a,'formal_complete',side_effect=guard),patch.object(a.base,'collect_row') as collect:
   with self.assertRaises(ValueError):a.final_analysis(Path('/unused-output'),Path('/source-complete'),Path('/must-stay-closed-truth'))
   collect.assert_not_called();self.assertEqual(len(seen),6)
 def test_full_synthetic_aggregate_and_cost_separation(self):
  checked=[];private_y=np.array([0,1,2,3]);ids=np.array(['p0','p1','p2','p3']);records=a.rows()
  def guard(r):checked.append(r['key']);return {},{},{}
  def read(path):
   path=Path(path)
   if path.name=='complete.json':return dict(status='SOURCE_ANALYZED',formal_scores_read=False,model_seed_count=3,source_confusions=6)
   if path.name.endswith('_training.json'):return dict(cost=dict(source_identity_NN=100,prediction_identity_NN=200),curve=[])
   if path.name=='synthetic_smoke.json':return dict(NN=3072,different_seed_head_and_pool=True,matched_initial_state_per_seed=True)
   if path.name=='real_checkpoint_smoke.json':return dict(NN=2)
   if path.name=='source_decision.json':return dict(source_As_replicated_all3=True,formal_scores_read=False)
   raise AssertionError(str(path))
  def collect(key,truth):
   self.assertEqual(len(checked),6);row=next(r for r in records if r['key']==key);pred={v:private_y.copy() if row['arm']=='B01' else np.array([0,1,2,0]) for v in a.VIEWS};views={}
   for v in a.VIEWS:views[v]=dict(a.base.metric(private_y,pred[v]),worst_RX_accuracy=.5,mean_true_margin_over30=.1,model_seed=2026092701)
   return dict(views=views,metrics=[dict(views[v],view=v,model_seed=2026092701) for v in a.VIEWS],source=dict(cost=dict(source_identity_NN=100,prediction_identity_NN=200),curve=[])),dict(ids=ids,y=private_y,pred=pred)
  with tempfile.TemporaryDirectory() as td,patch.object(a,'formal_complete',side_effect=guard),patch.object(a.base,'read',side_effect=read),patch.object(a.base,'collect_row',side_effect=collect),patch.object(a,'plot_final'):
   result=a.final_analysis(Path(td),Path('/synthetic-source'),Path('/synthetic-truth'))
   self.assertEqual(result['new_total_NN'],4*300+3074);self.assertEqual(result['reused_total_NN'],2*300);self.assertEqual(result['model_seed_count'],3);self.assertFalse(result['Ag_proved']);self.assertFalse(result['goal_complete']);self.assertTrue(result['factorial_interaction_not_applicable']);self.assertEqual({r['model_seed'] for r in result['rows'].values()},set(a.SEEDS));self.assertEqual(len(result['paired_summary']),7)
 def test_sequence_recount_detects_drift_without_U_label_decode(self):
  import gzip,json
  with tempfile.TemporaryDirectory() as td:
   left=Path(td)/'l';right=Path(td)/'r'
   for root in (left,right):
    for stage in ('stage20','source'):
     (root/stage).mkdir(parents=True)
     with gzip.open(root/stage/'training_sequence.jsonl.gz','wt',encoding='utf8') as f:f.write(json.dumps(dict(role='U',ids=['opaque1']))+'\n')
   self.assertTrue(a.compare_training_streams(left,right)['all_identical'])
   with gzip.open(right/'source/training_sequence.jsonl.gz','wt',encoding='utf8') as f:f.write(json.dumps(dict(role='U',ids=['opaque2']))+'\n')
   self.assertFalse(a.compare_training_streams(left,right)['all_identical'])
 def test_coverage_uses_explicit_row_root_before_reading_metadata(self):
  import types
  import cvs_As_seed_replication49_coverage_20261018 as coverage
  class ExpectedStop(Exception):pass
  for row in a.rows():
   seen=[]
   def read(path):seen.append(Path(path));raise ExpectedStop()
   with patch.object(coverage.old,'channel_seeds',return_value={}),patch.object(coverage,'read',side_effect=read):
    with self.assertRaises(ExpectedStop):coverage.replay(row,types.SimpleNamespace(b=None),None,None,[],set(),{})
   self.assertEqual(seen,[row['root']/'stage20/result.json'])
 def test_coverage_incomplete_source_never_bootstraps_runtime(self):
  import cvs_As_seed_replication49_coverage_20261018 as coverage
  def guard(row):
   if row==a.rows()[-1]:raise ValueError('last source row unfinished')
  with patch.object(coverage,'read',return_value=dict(status='SOURCE_ANALYZED',formal_scores_read=False,model_seed_count=3)),patch.object(a,'source_complete',side_effect=guard),patch.object(coverage.importlib.util,'spec_from_file_location') as load:
   with self.assertRaises(ValueError):coverage.run(Path('/unused-output'),Path('/synthetic-source'))
   load.assert_not_called()
if __name__=='__main__':unittest.main()
