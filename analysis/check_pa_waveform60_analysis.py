"""Focused synthetic P60 truth-boundary and metric checks; no real inputs."""
import itertools,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_pa_waveform60_analysis_20261018 as a

class Checks(unittest.TestCase):
 def make_rows(self,root):
  n=12;ids=np.asarray(['toy'+str(i) for i in range(n)]);y=np.arange(n)%6;z=np.full((n,6),-1.,np.float32);z[np.arange(n),y]=2.;specs=[]
  for key in ('NOAS30','AS30','NOAS_PAOFF','AS_PAOFF'):
   folder=root/key;pred=folder/'prediction';score=folder/'scoring';pred.mkdir(parents=True);score.mkdir();cfg={'toy_row':key};specs.append(dict(key=key,root=folder))
   def save(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
   save(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=key,count=n,views=list(a.VIEWS),config=cfg,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=n*7+2))
   save(pred/'resolved_config.json',dict(config=cfg,views=list(a.VIEWS),model_frozen=True,separate_head_fit=False,truth_read=False,query_fit=False,checkpoint=str(folder/'source/final_ssdg.pth')))
   save(score/'complete.json',dict(status='SCORED_COMPLETE',row_id=key,views=list(a.VIEWS),truth_last=True));save(folder/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=key))
   np.savez_compressed(pred/'predictions.npz',ids=ids,**{v:y for v in a.VIEWS})
   for v in a.VIEWS:np.savez_compressed(pred/(v+'_scores.npz'),ids=ids,logits=z)
  return specs,ids,y,z
 def checker(self,spec,project):return {'config':{'toy_row':spec['key']}}
 def test_exact_four_logical_rows_and_no_model_import(self):
  r=a.row_specs(Path('/toy'));self.assertEqual([x['key'] for x in r],['NOAS30','AS30','NOAS_PAOFF','AS_PAOFF']);self.assertEqual(sum(not x['reused'] for x in r),2)
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
   self.assertEqual(len(opened),1);self.assertEqual(set(data),{'NOAS30','AS30','NOAS_PAOFF','AS_PAOFF'});np.testing.assert_array_equal(meta['y'],y)
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
 def test_paired_interaction_direction_matches_integer_error_changes(self):
  y=np.arange(6);pred={k:y.copy() for k in a.ARMS}
  pred['NOAS30'][:3]=(pred['NOAS30'][:3]+1)%6;pred['AS30'][:2]=(pred['AS30'][:2]+1)%6;pred['NOAS_PAOFF'][:2]=(pred['NOAS_PAOFF'][:2]+1)%6
  metrics={k:a.metric(y,p)['accuracy'] for k,p in pred.items()};q=a.interaction_value(metrics)
  matched=a.paired(y,pred['NOAS_PAOFF'],pred['AS_PAOFF']);original=a.paired(y,pred['NOAS30'],pred['AS30'])
  self.assertAlmostEqual(q,1/6);self.assertEqual(matched['net_repairs']-original['net_repairs'],1);self.assertAlmostEqual(100*q,matched['accuracy_delta_pp']-original['accuracy_delta_pp'])
  with self.assertRaises(ValueError):a.interaction_value({'AS_PAOFF':.9})
 def test_actual_metadata_contract_for_both_new_own20_tails(self):
  with tempfile.TemporaryDirectory() as td:
   project=Path(td);specs=a.row_specs(project);roles={'L_s':['opaque-L'],'U_s':['opaque-U'],'V':['opaque-V']};contract=project/'runs'/a.CONTRACT_RUN/'source_contract.json';contract.parent.mkdir(parents=True);contract.write_text(json.dumps({'role_ids':roles}))
   for spec in specs:
    parent=project/'runs'/a.REF/(spec['reference_arm']+'-s'+str(a.SEED))/'stage20'
    for folder,e,l,u,steps in [(parent,20,13,7,4440),(spec['root']/'source',10,0,10,2220)]:
     folder.mkdir(parents=True,exist_ok=True);isp=folder==parent;cfg=dict(run_id=a.REF if isp else spec['run'],row_id=spec['reference_arm']+'-s'+str(a.SEED) if isp else spec['row_id'],model_seed=a.SEED,epochs=e,label_epochs=l,pseudo_epochs=u,updates=steps,Ag_mode='NONE',new_Ag=False,L_NN_per_step=768,total_L_loss_scale=1.25)
     if not isp:
      cfg.update(parent_source=str(parent),epoch_offset=20,step_offset=4440)
      if not spec['reused']:cfg.update(arm=spec['key'],As=spec['key']=='AS_PAOFF',PA_waveform=dict(injected_PA_packets=0,parent_E1_E20_PA=True,native_p_pa_unchanged=True,original_simulate_pa_runs_on_clone=True,U_changed=False,As_changed=False),scene_prior=dict(probabilities=[1/3]*3))
     docs={'completion.json':dict(epoch=e,optimizer_steps=steps,config=cfg,target_access=False,head_fit_updates=0,native_fitted_probes_disabled=True,final_source_metrics={'source_val_count':27000}),'source_frozen.json':dict(status='SOURCE_FROZEN',config=cfg,selection='final_only',target_access=False),'initialization.json':dict(checkpoint_sources=[] if isp else [str(parent/'final_ssdg.pth')],scratch_only=isp,target_contact=False,source_roles='EXACT_MATCH'),'result.json':dict(config=cfg,head_fit_updates=0,U_truth_read=False,target_access=False,source_RX=list(a.RX),source_RX_holdout=None),'source_contract.json':{'role_ids':roles}}
     for name,obj in docs.items():(folder/name).write_text(json.dumps(obj))
   for spec in specs:a.training_complete(spec,project)
   spec=specs[-1];p=spec['root']/'source/completion.json';d=a.read(p);d['config']['parent_source']='wrong-parent';p.write_text(json.dumps(d))
   with self.assertRaises(ValueError):a.training_complete(spec,project)
 def test_actual_partial_source_schema_has_no_invented_seed_requirement(self):
  partial=dict(status='SOURCE_READOUT_PARTIAL',missing=['source metadata'],source_qualified=False,formal_evaluation_always_required=True,formal_scores_read=False,no_automatic_budget_or_seed_extension=True,NN=0,target_feedback=False)
  self.assertEqual(a.stored_source_status(partial),'SOURCE_ANALYZED_WITH_STORED_DECISION_PARTIAL')
  decided=dict(status='SOURCE_READOUT_DECIDED',source_qualified=False,formal_scores_read=False,target_feedback=False,NN=0,model_seed_count=1)
  self.assertEqual(a.stored_source_status(decided),'SOURCE_ANALYZED');self.assertFalse(decided['source_qualified'])
  decided['formal_scores_read']=True
  with self.assertRaises(ValueError):a.stored_source_status(decided)
if __name__=='__main__':unittest.main()
