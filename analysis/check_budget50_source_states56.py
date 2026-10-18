"""Fixtures only: no real IQ, NPZ, checkpoint, source label or formal artifact."""
import copy,gzip,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('source_states56',Path(__file__).with_name('cvs_budget50_source_states56_20261018.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,allow_nan=False),encoding='utf8')

def logits(correct,y):
 z=np.zeros((len(y),6),np.float32)
 for i,label in enumerate(y):z[i,int(label) if i in correct else (int(label)+1)%6]=30.
 return z

def fixture():
 n=6;y=np.arange(6);ids=np.asarray(['fixture-'+str(i) for i in y]);rx=np.asarray([1,3,4,6,8,1]);day=np.asarray([1,2,3,1,2,3]);data={}
 for arm,good,clean in [('NOAS30',{0,1},{0,1,2,3}),('AS30',{0,1,2,3},{0,1,2}),('NOAS50',{0,2},{0,1,2,3}),('AS50',{0,2,3,4,5},{0,1,2,3,4})]:
  z=logits(good,y);data[arm]=dict(ids=ids.copy(),TX=y.copy(),RX=rx.copy(),day=day.copy(),clean=logits(clean,y),satellite=np.tile(z,(2,3,1,1)))
 shape=(2,3,11,n);physics={key:np.ones(shape) for key in a.PHYSICS_FIELDS}
 values={'state':[0,1,1,2,2,2],'lock':[0,0,1,1,2,2],'signal_to_added_noise_db':[2.99,3.,7.99,8.,14.99,15.],'actual_waveform_quality_db':[2.,4.,9.,16.,6.,15.],'white_proxy_quality_db':[3.,3.,3.,8.,8.,15.],'output_frequency_hz':[0.,100.,-1000.,10000.,-20.,50.],'agc_limited':[0.,0.,1.,0.,1.,1.],'noise_coloration_dynamic_range_db':[0.,0.,0.,0.,0.,0.],'residual_kernel_energy':[1e-4,.01,.1,1.,2.,0.],'interior_count':[0.,127.,128.,192.,224.,256.],'kernel_length':[257.,130.,129.,65.,33.,1.],'numerical_delay':[12.]*n,'agc_gain_db':[20.]*n,'residual_scale':[1.]*n}
 values['waveform_minus_white_power_db']=(np.asarray(values['actual_waveform_quality_db'])-np.asarray(values['white_proxy_quality_db'])).tolist()
 for key,value in values.items():physics[key][:]=value
 roles=dict(role='V',counts=dict(L_s=6300,U_s=56700,V=27000),V_panel=n,held_RX=None,extra_Ldev=False,U_truth_used=False,target_access=False,updates=0,physical_ids=ids.tolist())
 completion=dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=n,draws=2,scenes=list(a.SCENES),conditions=list(a.CONDITIONS),target_access=False,training_updates=0)
 records=[]
 for draw in range(2):
  for si,scene in enumerate(a.SCENES):
   for i in range(n):
    full={key:float(value[draw,si,0,i]) for key,value in physics.items() if key not in ('state','lock')};full.update(state=a.STATES[int(physics['state'][draw,si,0,i])],receiver_lock_status=a.LOCKS[int(physics['lock'][draw,si,0,i])],true_total_SNR_known=False,channel_equalization_applied=False,phase_tracking_applied=True,boundary='reflect')
    records.append(dict(draw=draw,scene=scene,index=i,physical_id=str(ids[i]),TX=int(y[i]),RX=int(rx[i]),day=int(day[i]),session='rx'+str(rx[i])+':day'+str(day[i])+':eq1',seed=1000+draw*100+si*10+i,conditions={key:copy.deepcopy(full) for key in a.CONDITIONS}))
 return data,physics,roles,completion,records

def write_fixture(root):
 data,physics,roles,completion,records=fixture();panel=root/'runs'/a.PANEL_RUN/'diagnostic';dump(panel/'data_roles.json',roles);dump(panel/'observations/completion.json',completion)
 np.savez_compressed(panel/'observations/physics.npz',**physics)
 with gzip.open(panel/'observations/physical_records.jsonl.gz','wt',encoding='utf8') as f:
  for q in records:f.write(json.dumps(q)+'\n')
 for arm,path in a.rows(root).items():
  out=path/'source_qualification';out.mkdir(parents=True);x=data[arm];np.savez_compressed(out/'predictions.npz',**x)
  dump(out/'complete.json',dict(status='SOURCE_READOUT_COMPLETE',config=dict(model_seed=a.SEED,row_id=path.name,run_id=a.RUN50 if arm.endswith('50') else a.RUN30),target_access=False,formal_scores_read=False,training_updates=0,head_fit_updates=0,NN=42,clean_accuracy=float((x['clean'].argmax(-1)==x['TX']).mean()),source_accuracies=(x['satellite'].argmax(-1)==x['TX']).mean(-1).tolist()))
 return data,physics,roles,completion,records

class Checks(unittest.TestCase):
 def test_bins_exact_boundaries_and_missing(self):
  values=np.array([-np.inf,2.999,3.,7.999,8.,14.999,15.,np.inf,np.nan]);np.testing.assert_array_equal(a.bin_codes(values,a.QUALITY_EDGES),[0,0,1,1,2,2,3,3,-1])
 def test_predictions_roles_and_order_rejected(self):
  data,physics,roles,_,_=fixture();ids=a.check_roles(roles,6);a.validate_predictions(data['NOAS30'],ids,n=6)
  wrong=copy.deepcopy(data['AS50']);wrong['ids']=wrong['ids'][::-1]
  with self.assertRaisesRegex(ValueError,'order differs'):a.validate_predictions(wrong,ids,data['NOAS30'],6)
  wrong=copy.deepcopy(data['AS50']);wrong['TX'][0]=1
  with self.assertRaisesRegex(ValueError,'TX alignment'):a.validate_predictions(wrong,ids,data['NOAS30'],6)
  wrong=copy.deepcopy(roles);wrong['U_truth_used']=True
  with self.assertRaisesRegex(ValueError,'legal V'):a.check_roles(wrong,6)
 def test_condition_and_scene_order_rejected(self):
  data,physics,roles,completion,records=fixture();a.check_completion(completion,6)
  wrong=copy.deepcopy(completion);wrong['conditions'][0],wrong['conditions'][1]=wrong['conditions'][1],wrong['conditions'][0]
  with self.assertRaisesRegex(ValueError,'condition order'):a.check_completion(wrong,6)
  wrong=copy.deepcopy(completion);wrong['scenes'].reverse()
  with self.assertRaisesRegex(ValueError,'scene/condition'):a.check_completion(wrong,6)
  a.check_records(iter(records),physics,data['NOAS30'],6)
  wrong=copy.deepcopy(records);keys=list(wrong[0]['conditions']);keys[0],keys[1]=keys[1],keys[0];wrong[0]['conditions']={k:wrong[0]['conditions'][k] for k in keys}
  with self.assertRaisesRegex(ValueError,'condition order'):a.check_records(iter(wrong),physics,data['NOAS30'],6)
 def test_draw_and_index_and_metadata_rejected(self):
  data,physics,roles,completion,records=fixture()
  for key,value in [('draw',1),('index',1),('TX',1),('physical_id','other')]:
   wrong=copy.deepcopy(records);wrong[0][key]=value
   with self.assertRaises(ValueError):a.check_records(iter(wrong),physics,data['NOAS30'],6)
  wrong=copy.deepcopy(records);wrong[0]['conditions']['full']['actual_waveform_quality_db']+=1
  with self.assertRaisesRegex(ValueError,'numeric metadata'):a.check_records(iter(wrong),physics,data['NOAS30'],6)
  with self.assertRaisesRegex(ValueError,'incomplete'):a.check_records(iter(records[:-1]),physics,data['NOAS30'],6)
 def test_full_input_fixture_recounts(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);data,physics,*_=write_fixture(root);loaded,p,proof=a.load_inputs(root,6)
   self.assertEqual(proof['record_alignment']['records'],36);self.assertEqual(proof['missing_physics_fields'],[]);np.testing.assert_array_equal(loaded['AS50']['satellite'],data['AS50']['satellite'])
   # Only an existing source NPZ is changed, and its own source completion must catch it.
   path=a.rows(root)['AS50']/'source_qualification/predictions.npz';bad=copy.deepcopy(data['AS50']);bad['satellite'][0,0,0]=logits(set(),bad['TX'])[0];np.savez_compressed(path,**bad)
   with self.assertRaisesRegex(ValueError,'source readout'):a.load_inputs(root,6)
 def test_paired_and_parent_cohort(self):
  data,physics,*_=fixture();strata,effects,ints,overall=a.analyze(data,physics)
  found=lambda seq,**kw:next(row for row in seq if all(row[k]==v for k,v in kw.items()))
  context=dict(scene=a.SCENES[-1],draw=-1,dimension='overall',stratum='ALL')
  m=found(strata,row_id='AS50',**context);self.assertEqual(m['count'],12);self.assertEqual(m['correct'],10);self.assertEqual(m['fixed_NOAS30_clean_correct_count'],8);self.assertEqual(m['fixed_NOAS30_clean_correct_retained'],6);self.assertEqual(m['satellite_retention'],.75)
  q=found(effects,new='AS50',control='AS30',**context);self.assertEqual((q['repairs'],q['new_errors'],q['net_repairs']),(4,2,2));self.assertEqual((q['cohort_repairs'],q['cohort_new_errors']),(0,2));self.assertAlmostEqual(q['accuracy_delta_pp'],100/6)
  inter=found(ints,**context);self.assertEqual(inter['budget_x_As_net_correct'],2);self.assertAlmostEqual(inter['budget_x_As_accuracy_pp'],100/6)
  raw=found(strata,row_id='AS50',scene=a.SCENES[-1],draw=0,dimension='state',stratum='LOS');self.assertEqual(raw['mean_logit_margin'],30.);self.assertEqual(raw['mean_margin_over30'],1.)
 def test_empty_cells_fixed_bins_and_cm(self):
  data,physics,*_=fixture();strata,effects,ints,overall=a.analyze(data,physics)
  rows=[r for r in strata if r['row_id']=='AS50' and r['scene']==a.SCENES[-1] and r['draw']==0]
  self.assertEqual(len([r for r in rows if r['dimension']=='state_lock']),9)
  empty=next(r for r in rows if r['dimension']=='state_lock' and r['stratum']=='LOS/unlocked');self.assertTrue(empty['empty_cell']);self.assertIsNone(empty['accuracy']);self.assertEqual(np.asarray(empty['confusion']).sum(),0)
  quality=[r for r in rows if r['dimension']=='signal_to_added_noise_db'];self.assertEqual([r['count'] for r in quality],[1,2,2,1,0]);self.assertEqual(sum(r['count'] for r in quality),6)
  for r in rows:self.assertEqual(np.asarray(r['confusion']).sum(),r['count'])
 def test_waveform_white_groups_and_optional_missing(self):
  data,physics,*_=fixture();sub={k:v[0,0,0] for k,v in physics.items()};groups=a.masks(sub);quality={label:int(mask.sum()) for dim,label,mask in groups if dim=='waveform_vs_white_quality_bin'};self.assertEqual(quality,dict(waveform_lower=2,same_bin=2,waveform_higher=2,MISSING=0))
  reduced={k:v for k,v in sub.items() if k!='noise_coloration_dynamic_range_db'};self.assertFalse(any(dim=='noise_coloration_dynamic_range_db' for dim,_,_ in a.masks(reduced)))
 def test_source_summary_reconciliation(self):
  data,physics,*_=fixture();_,_,_,overall=a.analyze(data,physics);summary=dict(status='SOURCE_ANALYZED',target_scores_read=False,summary={})
  for arm in a.ARMS:summary['summary'][arm]=dict(panel={scene:dict(accuracy=overall[arm][scene+'__-1']['accuracy'],macro_F1=overall[arm][scene+'__-1']['macro_f1']) for scene in ('clean',*a.SCENES)})
  self.assertEqual(a.verify_source_summary(summary,overall)['accuracy_and_macro_f1_reconciliations'],32);summary['summary']['AS50']['panel']['clean']['accuracy']+=.01
  with self.assertRaisesRegex(ValueError,'recount differs'):a.verify_source_summary(summary,overall)
 def test_output_root_immutable(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaisesRegex(FileExistsError,'Immutable'):a.run(Path(td),Path(td))

if __name__=='__main__':unittest.main(verbosity=2)
