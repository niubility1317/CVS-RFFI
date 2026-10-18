"""Focused synthetic P59 truth-boundary and metric checks; no real inputs."""
import itertools,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_urban_prior59_analysis_20261018 as a

class Checks(unittest.TestCase):
 def make_rows(self,root):
  n=12;ids=np.asarray(['toy'+str(i) for i in range(n)]);y=np.arange(n)%6;z=np.full((n,6),-1.,np.float32);z[np.arange(n),y]=2.;specs=[]
  for key in ('NOAS30','AS30','URBAN'):
   folder=root/key;pred=folder/'prediction';score=folder/'scoring';pred.mkdir(parents=True);score.mkdir();cfg={'toy_row':key};specs.append(dict(key=key,root=folder))
   def save(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
   save(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=key,count=n,views=list(a.VIEWS),config=cfg,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=n*7+2))
   save(pred/'resolved_config.json',dict(config=cfg,views=list(a.VIEWS),model_frozen=True,separate_head_fit=False,truth_read=False,query_fit=False,checkpoint=str(folder/'source/final_ssdg.pth')))
   save(score/'complete.json',dict(status='SCORED_COMPLETE',row_id=key,views=list(a.VIEWS),truth_last=True));save(folder/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=key))
   np.savez_compressed(pred/'predictions.npz',ids=ids,**{v:y for v in a.VIEWS})
   for v in a.VIEWS:np.savez_compressed(pred/(v+'_scores.npz'),ids=ids,logits=z)
  return specs,ids,y,z
 def checker(self,spec,project):return {'config':{'toy_row':spec['key']}}
 def test_exact_three_logical_rows_and_no_model_import(self):
  r=a.row_specs(Path('/toy'));self.assertEqual([x['key'] for x in r],['NOAS30','AS30','URBAN']);self.assertEqual(sum(not x['reused'] for x in r),1)
  self.assertEqual(r[2]['root'].parent.name,a.RUN);self.assertEqual(r[0]['root'].parent.name,a.REF)
 def gate_fixture(self,mutate):
  with tempfile.TemporaryDirectory() as td:
   specs,ids,y,z=self.make_rows(Path(td));mutate(specs[-1]['root'],ids,y,z);truth=Path(td)/'sentinel_truth.json';opened=[];read=a.read
   def watching(path):
    if Path(path)==truth:opened.append(path);raise AssertionError('Truth was opened before failed gate')
    return read(path)
   with patch.object(a,'read',side_effect=watching):
    with self.assertRaises((ValueError,FileNotFoundError)):a.truth_last_arrays(specs,ids,truth,Path(td),count=12,checker=self.checker)
   self.assertEqual(opened,[])
 def test_incomplete_final_row_never_opens_truth(self):
  self.gate_fixture(lambda root,*_: (root/'prediction/complete.json').unlink())
 def test_unfrozen_final_row_never_opens_truth(self):
  def mutate(root,*_):
   p=root/'prediction/complete.json';d=a.read(p);d['all_parameters_frozen']=False;p.write_text(json.dumps(d))
  self.gate_fixture(mutate)
 def test_incomplete_independent_scoring_never_opens_truth(self):
  def mutate(root,*_):
   p=root/'scoring/complete.json';d=a.read(p);d['status']='RUNNING';p.write_text(json.dumps(d))
  self.gate_fixture(mutate)
 def test_mismatched_ids_never_open_truth(self):
  def mutate(root,ids,y,z):np.savez_compressed(root/'prediction'/('practical_low_urban_scores.npz'),ids=ids[::-1],logits=z)
  self.gate_fixture(mutate)
 def test_nan_logits_never_open_truth(self):
  def mutate(root,ids,y,z):
   z=z.copy();z[0,0]=np.nan;np.savez_compressed(root/'prediction'/('practical_low_urban_scores.npz'),ids=ids,logits=z)
  self.gate_fixture(mutate)
 def test_argmax_mismatch_never_opens_truth(self):
  def mutate(root,ids,y,z):np.savez_compressed(root/'prediction'/('practical_low_urban_scores.npz'),ids=ids,logits=z[:,::-1])
  self.gate_fixture(mutate)
 def test_valid_all_rows_read_truth_once_after_all7(self):
  with tempfile.TemporaryDirectory() as td:
   specs,ids,y,z=self.make_rows(Path(td));truth=Path(td)/'truth.json';truth.write_text(json.dumps({s:dict(label=int(v),receiver='toyRX',day='toyDay') for s,v in zip(ids,y)}));opened=[];read=a.read
   def watching(path):
    if Path(path)==truth:opened.append(path)
    return read(path)
   with patch.object(a,'read',side_effect=watching):data,meta=a.truth_last_arrays(specs,ids,truth,Path(td),count=12,checker=self.checker)
   self.assertEqual(len(opened),1);self.assertEqual(set(data),{'NOAS30','AS30','URBAN'});np.testing.assert_array_equal(meta['y'],y)
 def test_all_sixteen_correctness_and_cohort_combinations(self):
  # 4 correctness pairs x cohort false/true x repeated label false/true =16.
  for oldok,newok,inside,labelbit in itertools.product((False,True),repeat=4):
   y=np.asarray([int(labelbit)]);old=np.where(oldok,y,(y+1)%6);new=np.where(newok,y,(y+1)%6);q=a.paired(y,old,new,np.asarray([inside]))
   self.assertEqual(q['repairs'],int(not oldok and newok));self.assertEqual(q['new_errors'],int(oldok and not newok));self.assertEqual(q['net_repairs'],int(newok)-int(oldok));self.assertEqual(q['cohort_repairs'],int(not oldok and newok and inside));self.assertEqual(q['new_retention'],float(newok) if inside else None)
 def test_macro_f1_and_partition_reconciliation(self):
  y=np.arange(6);p=np.asarray([0,1,2,3,4,0]);m=a.metric(y,p);self.assertEqual(m['correct'],5);self.assertAlmostEqual(m['accuracy'],5/6);self.assertAlmostEqual(m['macro_f1'],(2/3+4)/6)
  self.assertTrue(a.reconcile_partitions(m,[a.metric(y[:3],p[:3]),a.metric(y[3:],p[3:])]))
  with self.assertRaises(ValueError):a.reconcile_partitions(m,[a.metric(y[:3],p[:3])])
 def test_empty_small_cells_and_scale30_margin(self):
  q=a.logit_metric(np.asarray([],int),np.empty((0,6)),np.asarray([],bool));self.assertTrue(q['empty_cell']);self.assertTrue(q['small_cell']);self.assertIsNone(q['accuracy']);self.assertIsNone(q['satellite_retention'])
  z=np.zeros((1,6));z[0,0]=15.;q=a.logit_metric(np.asarray([0]),z,np.asarray([True]));self.assertEqual(q['mean_margin_over30'],.5)
 def test_json_numpy_scalars_roundtrip_without_losing_boolean_type(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'scalar.json';a.write(p,{'flag':np.bool_(True),'integer':np.int64(3),'single':np.float32(.25),'double':np.float64(.5),'nested':[np.bool_(False)]});q=a.read(p)
   self.assertIs(q['flag'],True);self.assertIs(q['nested'][0],False);self.assertIs(type(q['integer']),int);self.assertEqual(q['integer'],3);self.assertEqual(q['single'],.25);self.assertEqual(q['double'],.5)
 def test_json_nan_is_rejected_after_numpy_scalar_conversion(self):
  for value in (float('nan'),np.float32('nan'),np.float64('nan')):
   with tempfile.TemporaryDirectory() as td:
    with self.assertRaises(ValueError):a.write(Path(td)/'nan.json',{'value':value})
 def test_json_unknown_types_remain_rejected(self):
  class Unknown:pass
  for value in (Unknown(),np.asarray([1,2]),{1,2}):
   with tempfile.TemporaryDirectory() as td:
    with self.assertRaises(TypeError):a.write(Path(td)/'unknown.json',{'value':value})
 def recovery_fixture(self):
  decision=dict(status='SOURCE_READOUT_DECIDED',source_qualified=False,checks={'scientific_threshold':False},formal_scores_read=False,target_feedback=False,model_seed_count=1,NN=0)
  failure={'status':'SOURCE_READOUT_FAILED','message':'historical scalar serialization'}
  recovery=dict(status='VERIFIED',criteria_unchanged=True,source_decision=decision.copy(),original_failure_preserved=True,formal_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,source_qualification_rerun=False,training_rerun=False,formal_rerun=False)
  return decision,failure,recovery
 def test_verified_source_report_recovery_preserves_scientific_failure(self):
  decision,failure,recovery=self.recovery_fixture();q=a.source_recovery_state(decision,failure,recovery)
  self.assertTrue(q['verified']);self.assertFalse(q['unresolved']);self.assertIs(q['historical_failure'],failure);self.assertFalse(decision['source_qualified']);self.assertEqual(decision,recovery['source_decision'])
 def test_invalid_source_recovery_cannot_clear_original_failure(self):
  for key,value in [('status','UNKNOWN'),('criteria_unchanged',False),('source_decision',{'retuned':True}),('NN',1),('original_failure_preserved',False),('formal_scores_read',True),('source_qualification_rerun',True),('training_rerun',True),('formal_rerun',True)]:
   decision,failure,recovery=self.recovery_fixture();recovery[key]=value;q=a.source_recovery_state(decision,failure,recovery)
   self.assertFalse(q['verified']);self.assertTrue(q['unresolved']);self.assertIs(q['historical_failure'],failure)
  decision,failure,recovery=self.recovery_fixture();self.assertTrue(a.source_recovery_state(decision,failure,None)['unresolved'])
if __name__=='__main__':unittest.main()
