"""Synthetic recovery routing/protocol checks; no real data or model objects."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_continuous53_results_20261018 as a
import cvs_continuous53_pipeline_20261018 as p
import cvs_continuous53_publish_analysis_20261018 as publish

class Checks(unittest.TestCase):
 def test_exact_three_reused_one_new_routes(self):
  r={x['key']:x for x in a.rows()}
  self.assertEqual(len(r),4);self.assertEqual({x['seed'] for x in r.values()},{2026092701})
  self.assertEqual(r['CONT_B00']['run'],a.ORIGINAL_RUN);self.assertEqual(r['CONT_B01']['run'],a.RUN)
  self.assertEqual({r[k]['run'] for k in ['STAGED_B00','STAGED_B01']},{a.REF})
  self.assertEqual(sum(not x['execution_reused'] for x in r.values()),1)
  self.assertFalse(r['CONT_B00']['reused']) # Parsing shape, not execution cost.
  self.assertTrue(r['CONT_B00']['execution_reused'])
  self.assertNotEqual(r['CONT_B00']['root'].parent,r['CONT_B01']['root'].parent)

 def fixture(self,row):
  c=dict(epochs=30,label_epochs=13,pseudo_epochs=17,updates=6660,run_id=row['run'],row_id=row['root'].name,arm=row['arm'],model_seed=a.SEED,training_mode='single_continuous_E30',epoch_offset=0,step_offset=0,parent_source=None,checkpoint_sources=[],new_Ag=False,Ag_mode='NONE')
  done=dict(epoch=30,optimizer_steps=6660,config=c,target_access=False,head_fit_updates=0)
  init=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],target_contact=False,source_roles='EXACT_MATCH')
  docs={'completion.json':done,'initialization.json':init,'source_frozen.json':dict(config=c,selection='final_only',target_access=False),'result.json':dict(config=c,head_fit_updates=0,U_truth_read=False,target_access=False),'source_contract.json':dict(role_ids={'L_s':['opaque']}),'actual_native_args.json':dict(epochs=30,label_epochs=13,pseudo_epochs=17,seed=a.SEED,baseline_ckpt='',teacher_ckpt=''),'seed_binding.json':dict(actual_F_factory_seed=a.SEED)}
  return c,done,docs

 def test_native30_and_distinct_run_provenance(self):
  for row in [r for r in a.rows() if r['style']=='CONT']:
   c,done,docs=self.fixture(row)
   with patch.object(a,'read',side_effect=lambda path:docs[path.name]):
    self.assertIs(a.completed_training(row['root']),done)
    for field,value in [('run_id',a.REF),('updates',2220),('parent_source','forbidden-parent')]:
     previous=c[field];c[field]=value
     with self.assertRaises(ValueError):a.completed_training(row['root'])
     c[field]=previous
    done['epoch']=10
    with self.assertRaises(ValueError):a.completed_training(row['root'])

 def test_default_head_and_hidden_U_required(self):
  row=next(r for r in a.rows() if r['key']=='CONT_B01');c,done,docs=self.fixture(row)
  with patch.object(a,'read',side_effect=lambda path:docs[path.name]):
   done['head_fit_updates']=1
   with self.assertRaises(ValueError):a.completed_training(row['root'])
   done['head_fit_updates']=0;docs['result.json']['U_truth_read']=True
   with self.assertRaises(ValueError):a.completed_training(row['root'])
  with patch.object(a,'read',side_effect=AssertionError('wrongroot must reject before read')):
   with self.assertRaises(ValueError):a.completed_training(a.ROOT/'B00-s2026092701')

 def test_truth_closed_until_every_formal_row_complete(self):
  seen=[]
  def guard(row):
   seen.append(row['key'])
   if row==a.rows()[-1]:raise ValueError('unfinished recovered row')
  with patch.object(a.core,'read',return_value=dict(status='SOURCE_ANALYZED',formal_scores_read=False,model_seed_count=1)),patch.object(a.core,'formal_complete',side_effect=guard),patch.object(a.core.base,'collect_row') as collect:
   with self.assertRaises(ValueError):a.final_analysis(Path('/unused'),Path('/source'),Path('/closedtruth'))
   collect.assert_not_called();self.assertEqual(len(seen),4)

 def test_point_one_absolute_pp_and_one_seed_not_reproducible(self):
  r={x['key']:dict(views={v:dict(accuracy=.8) for v in a.VIEWS}) for x in a.rows()}
  for v in a.VIEWS[-3:]:r['CONT_B01']['views'][v]['accuracy']=.9
  r['CONT_B01']['views']['clean']['accuracy']=.799
  q=a.stage_assessment(dict(rows=r));self.assertAlmostEqual(q['clean_delta_pp'],-.1);self.assertTrue(q['clean_cost_within_stage_limit']);self.assertAlmostEqual(q['urban_mean_delta_pp'],10);self.assertFalse(q['reproducible_dual_gain'])
  r['CONT_B01']['views']['clean']['accuracy']=.798999
  self.assertFalse(a.stage_assessment(dict(rows=r))['clean_cost_within_stage_limit'])
  r['CONT_B01']['views']['clean']['accuracy']=.801
  q=a.stage_assessment(dict(rows=r));self.assertTrue(q['observed_dual_gain']);self.assertFalse(q['crossseed_confirmed'])

 def test_recovery_cost_excludes_reused_control_from_new_training(self):
  r={x['key']:dict(views={v:dict(accuracy=.8) for v in a.VIEWS}) for x in a.rows()}
  q=dict(rows=r,cost={k:dict(total=300,reused=False) for k in r},shared_new_smoke_NN=1538)
  failed=dict(known_NN_lower_bound=3409920)
  def read(path):
   if path.name=='failed_original_attempt.json':return failed
   if path.name=='synthetic_smoke.json':return dict(NN=1536)
   if path.name=='real_checkpoint_smoke.json':return dict(NN=2)
   if path.name=='complete.json':return dict(status='ANALYZED')
   raise AssertionError(path)
  with tempfile.TemporaryDirectory() as td,patch.object(a.core,'final_analysis',return_value=q),patch.object(a,'read',side_effect=read),patch.object(a,'write'),patch.object(a,'matched_cohort_retention',return_value=[dict(synthetic=True)]):
   z=a.final_analysis(Path(td),Path('/source'),Path('/truth'))
  self.assertEqual((z['new_models'],z['reused_models']),(1,3));self.assertEqual((z['new_model_epochs'],z['reused_model_epochs']),(30,90));self.assertEqual(z['failed_original_model_epochs'],20);self.assertEqual(z['new_total_NN'],1838);self.assertEqual(z['reused_total_NN'],900);self.assertIsNone(z['total_NN_including_failed']);self.assertEqual(z['known_total_NN_lower_bound_including_failed'],1838+900+1538+3409920)

 def test_matched_C0_cohort_is_true_intersection_not_all_correct(self):
  y=np.array([0,1,2,3]);ids=np.array(['a','b','c','d'])
  data={r['key']:dict(ids=ids.copy(),y=y.copy(),pred={v:y.copy() for v in a.VIEWS}) for r in a.rows()}
  data['CONT_B00']['pred']['clean']=np.array([0,0,2,3])
  data['CONT_B01']['pred']['practical_low_urban']=np.array([0,1,0,3])
  records=a.matched_cohort_retention(data);q=next(r for r in records if (r['row_id'],r['view'])==('CONT_B01','practical_low_urban'))
  self.assertEqual(q['denominator'],3);self.assertEqual(q['retained'],2);self.assertAlmostEqual(q['retention'],2/3);self.assertFalse(q['training_parent_cohort'])
  data['CONT_B00']['pred']['clean']=np.array([1,0,0,0]);self.assertTrue(all(q['retention'] is None for q in a.matched_cohort_retention(data)))

 def test_four_row_readiness_and_no_truth_in_source_commands(self):
  source=p.source_metadata();formal=p.formal_metadata()
  self.assertEqual(len(source),9);self.assertEqual(len(formal),12)
  self.assertIn(a.ORIGINAL_ROOT/'B00-s2026092701/source/completion.json',source)
  self.assertNotIn(a.ROOT/'B00-s2026092701/source/completion.json',source)
  for stage in ['source','coverage']:self.assertNotIn('--truth',p.command(stage,Path('/out'),Path('/closedtruth')))
  self.assertIn('--truth',p.command('final',Path('/out'),Path('/closedtruth')))
  self.assertIn('coverage',p.command('coverage',Path('/out'),Path('/closedtruth')))

 def test_publisher_output_exactly_matches_pipeline_root(self):
  output=Path(publish.OUTPUT)
  self.assertEqual(output.parent,p.ROOT);self.assertEqual(output.name,'analysis_auto_r01')
  self.assertEqual(Path(publish.RELEASE).name,'cvs_FcrossRX_continuous_recovery_analysis_20261018_r01')
  self.assertEqual(Path(publish.OUTPUT),a.ROOT/'analysis_auto_r01')
  self.assertIn('cvs_continuous53_results_20261018.py',publish.FILES)
  self.assertIn('cvs_continuous52_results_20261018.py',publish.FILES)
  self.assertIn('cvs_continuous53_pipeline_20261018.py',publish.FILES)

 def test_wait_checks_owned_handles_without_mutating_processes(self):
  class EndWait(Exception):pass
  with tempfile.TemporaryDirectory() as td:
   p.wait_until('SOURCE',[],Path(td),ready=lambda _:True,probe=lambda:(_ for _ in ()).throw(AssertionError('completed metadata needs no process')))
   for handles in [([123],False),([],True)]:
    with self.assertRaises(EndWait):p.wait_until('SOURCE',[],Path(td),ready=lambda _:False,probe=lambda:handles,failure=lambda:None,sleep=lambda _:(_ for _ in ()).throw(EndWait()))
   with self.assertRaises(p.IncompleteRun):p.wait_until('SOURCE',[],Path(td),ready=lambda _:False,probe=lambda:([],False),failure=lambda:None,sleep=lambda _:None)

 def test_historical_failure_does_not_stop_healthy_recovery(self):
  expected=[a.ROOT/'failure.json']+[r['root']/'source_qualification_failure.json' for r in a.rows() if not r['reused']]
  self.assertNotIn(a.ORIGINAL_ROOT/'failure.json',expected)
  with patch.object(Path,'exists',lambda x:x==a.ORIGINAL_ROOT/'failure.json'):self.assertIsNone(p.technical_failure())

if __name__=='__main__':unittest.main()
