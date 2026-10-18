"""Synthetic scoring/absolute-pp boundary/truth-last negative checks;no real artifacts."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_satellite_dropout_results_20261018 as a

class Checks(unittest.TestCase):
 def test_absolute_pp_integer_boundary(self):
  self.assertTrue(a.stage_clean_pass(-168,168000));self.assertFalse(a.stage_clean_pass(-169,168000));self.assertTrue(a.stage_clean_pass(-27,27000));self.assertFalse(a.stage_clean_pass(-28,27000))
 def test_handcomputed_confusion_F1(self):
  m=a.metric(np.array([0,0,1,2]),np.array([0,1,1,2]));self.assertEqual(m['correct'],3);self.assertEqual(m['confusion'][0][1],1);self.assertAlmostEqual(m['macro_f1'],7/18)
 def test_repairs_and_fixed_cohort(self):
  y=np.array([0,0,1,1]);p=np.array([0,1,1,0]);n=np.array([1,0,1,1]);r=a.paired(y,p,n,np.array([1,1,0,0],bool));self.assertEqual((r['repairs'],r['new_errors'],r['net_repairs']),(2,1,1));self.assertEqual((r['cohort_repairs'],r['cohort_new_errors']),(1,1));self.assertEqual(r['accuracy_delta_pp'],25)
 def test_source_decision_limits(self):
  s={k:dict(full_V_clean=.98,mid_low_mean_pp=60.) for k in ('NOAS','AS_RD','AS_CE','AS_DET')};s['NOAS']['mid_low_mean_pp']=20.;s['AS_DET'].update(full_V_clean=.981,mid_low_mean_pp=61.);self.assertTrue(all(a.source_conditions(s).values()));s['AS_DET']['full_V_clean']=.978;self.assertFalse(a.source_conditions(s)['stage_clean'])
 def fixture(self,project,n=12):
  ids=np.array(['SYNTHETIC'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.array(['A' if i%2 else 'B' for i in range(n)]);day=np.array(['d1' if i<6 else 'd2' for i in range(n)]);truth={str(ids[i]):dict(label=int(y[i]),receiver=rx[i],day=day[i]) for i in range(n)};truth_path=Path(project)/'synthetic_truth.json';a.write(truth_path,truth)
  for arm,root in a.rows(project):
   for segment,epoch,updates in [('stage20',20,4440),('source',10,2220)]:
    (root/segment).mkdir(parents=True);a.write(root/segment/'completion.json',dict(epoch=epoch,optimizer_steps=updates,target_access=False,config=dict(model_seed=a.SEED,row_id=root.name,run_id=a.REFERENCE_RUN if arm in ('NOAS','AS_RD') else a.RUN)))
   a.write(root/'source/source_frozen.json',dict(selection='final_only',target_access=False));(root/'source_qualification').mkdir();a.write(root/'source_qualification/complete.json',dict(target_access=False,NN=12600));(root/'prediction').mkdir();(root/'scoring').mkdir();pred={};scores=[]
   for view in a.VIEWS:
    p=y.copy()
    if arm=='NOAS':p[0]=1
    if arm=='AS_CE':p[1]=2
    if arm=='AS_DET' and view=='clean':p[0]=1
    logits=np.full((n,6),-1.,np.float32);logits[np.arange(n),p]=1.;np.savez(root/'prediction'/(view+'_scores.npz'),ids=ids,logits=logits);pred[view]=p
    for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
     for value in values:
      mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;cm=[[0]*6 for _ in range(6)]
      for true,guess in zip(y[mask],p[mask]):cm[int(true)][int(guess)]+=1
      den=[sum(cm[j][i] for j in range(6))+sum(cm[i]) for i in range(6)];f1=sum(2*cm[i][i]/den[i] if den[i] else 0. for i in range(6))/6;count=int(mask.sum());correct=sum(cm[i][i] for i in range(6));scores.append(dict(view=view,dimension=dim,stratum=str(value),accuracy=correct/count,macro_f1=f1,query_count=count,confusion=cm))
   np.savez(root/'prediction/predictions.npz',ids=ids,**pred);a.write(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(a.VIEWS),count=n,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=n*7+2,wall_seconds=1.));a.write(root/'scoring/complete.json',dict(status='SCORED_COMPLETE'));a.write(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=scores));a.write(root/'complete.json',dict(status='TRAINED_AND_SCORED'))
  return truth_path
 def test_full_small_formal_recount(self):
  with tempfile.TemporaryDirectory() as td,patch.object(a,'COUNT',12):
   truth=self.fixture(td);s=dict(status='SOURCE_ANALYZED',run_id=a.RUN,target_scores_read=False,decision={});r,p=a.formal_analysis(td,truth,s);self.assertEqual(r['independent_confusion_matrices'],4*7*11);self.assertEqual(r['partition_reconciliations'],84);self.assertTrue(r['stage']['AS_DET']['clean_absolute_0p1pp_guard_met']);self.assertFalse(r['repeatable_clean_positive_from_As_proven']);self.assertEqual(r['paired']['AS_DET__NOAS']['practical_low_urban']['net_repairs'],1)
 def test_incomplete_last_row_keeps_truth_closed(self):
  with tempfile.TemporaryDirectory() as td,patch.object(a,'COUNT',12):
   truth=self.fixture(td);dict(a.rows(td))['AS_DET'].joinpath('prediction/complete.json').unlink();orig=a.read;opened=[]
   def watched(path):
    if Path(path)==truth:opened.append(True)
    return orig(path)
   with patch.object(a,'read',side_effect=watched),self.assertRaises(a.AwaitingArtifacts):a.formal_analysis(td,truth,dict(status='SOURCE_ANALYZED',run_id=a.RUN,target_scores_read=False))
   self.assertEqual(opened,[])
 def test_bad_saved_metric_rejected(self):
  with tempfile.TemporaryDirectory() as td,patch.object(a,'COUNT',12):
   truth=self.fixture(td);p=dict(a.rows(td))['AS_DET']/'scoring/scores.json';r=a.read(p);r['results'][0]['accuracy']=.123;a.write(p,r)
   with self.assertRaisesRegex(ValueError,'Independent formal metric differs'):a.formal_analysis(td,truth,dict(status='SOURCE_ANALYZED',run_id=a.RUN,target_scores_read=False,decision={}))
 def test_missing_source_analysis_rejected(self):
  with patch.object(a,'ensure_metadata',return_value={}):
   with self.assertRaisesRegex(ValueError,'source analysis first'):a.formal_analysis('unused','never_open',dict(status='WAIT',run_id=a.RUN,target_scores_read=False))
if __name__=='__main__':unittest.main()
