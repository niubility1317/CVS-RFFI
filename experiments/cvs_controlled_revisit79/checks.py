"""Focused stdlib-only protocol fixtures; zero real CP/IQ/NN or remote calls."""
import ast,copy,json,math,sys,unittest
from pathlib import Path
from . import design as d,protocol as p

def panel_fixture():
 cells=[(tx,rx,day) for unused in range(20) for tx in range(6) for rx in d.RX for day in d.DAYS];ids=['source-'+str(i) for i in range(d.N)]
 roles=dict(role='V',counts=dict(L_s=6300,U_s=56700,V=27000),V_panel=d.N,held_RX=None,extra_Ldev=False,U_truth_used=False,target_access=False,updates=0,physical_ids=ids)
 return roles,ids,[c[0] for c in cells],[c[1] for c in cells],[c[2] for c in cells]

def metadata_fixture(arm='B01'):
 project=Path('/original-project');source=d.source(project,arm);parent=source.parent/'stage20';c=dict(run_id=d.REFERENCE_RUN,row_id=arm+'-s'+str(d.SEED),model_seed=d.SEED,segment=30,epochs=10,updates=2220,epoch_offset=20,step_offset=4440,As=arm=='B01',output_root=str(source),parent_source=str(parent),checkpoint_sources=[str(parent/'final_ssdg.pth')]);done=dict(epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0,config=c);frozen=dict(selection='final_only',target_access=False,config=c);init=dict(scratch_only=False,target_contact=False,source_roles='EXACT_MATCH',checkpoint_sources=c['checkpoint_sources'],ancestors=[str(parent)]);canonical=dict(role_ids=dict(L_s=['l0'],U_s=['u0'],V=['v0']))
 return project,arm,c,done,frozen,init,copy.deepcopy(canonical),canonical

def observations_fixture():return dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=1800,draws=2,scenes=list(d.SCENES),conditions=list(d.CONDITIONS),physical_draws=10800,training_updates=0,target_access=False,formal_test_changed=False)

def gradients_fixture():
 group=dict(clean_norm=0.,satellite_norm=0.,dot=0.,cosine=None,finite=True)
 return [dict(draw=draw,scene=scene,count=90,clean_CE=1.,satellite_CE=2.,groups={name:dict(group) for name in d.GRADIENT_GROUPS},scope='frozen weights, deterministic eval-mode smoothedCE, not stochastic training gradients') for draw in d.DRAWS for scene in d.SCENES]

class ProtocolChecks(unittest.TestCase):
 def test_fixed_P43_student_scope_and_no_automatic_expansion(self):
  c=d.config('/original-project');self.assertIs(p.check_config(c),c);self.assertEqual([r['label'] for r in c['rows']],list(d.LABELS));self.assertTrue(all(d.REFERENCE_RUN in r['checkpoint'] for r in c['rows']))
  for key,value in [('checkpoint_state','ema_model'),('EMA_loaded',True),('model_seed',2026092702),('reference_commit','other'),('reference_run','P78'),('query_access',True),('formal_scores_read',True),('U_truth_read',True),('U_iteration',True),('training_epochs',1),('optimizer_updates',1),('EMA_updates',1),('head_fit_updates',1),('new_IQ_generated',1),('new_channel_draws',1),('source_promotion',True),('automatic_training',True),('automatic_formal',True),('automatic_budget_extension',True),('formal_test_changed',True)]:
   wrong=copy.deepcopy(c);wrong[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):p.check_config(wrong)
 def test_exact_own20_final30_metadata_and_role_handles(self):
  for arm in d.ARMS:p.check_source_metadata(*metadata_fixture(arm))
  for section,key,value in [(2,'epochs',20),(2,'segment',20),(2,'updates',4440),(2,'row_id','P78-B01'),(2,'run_id','other'),(2,'parent_source','different'),(3,'epoch',20),(3,'optimizer_steps',6660),(3,'target_access',True),(3,'head_fit_updates',1),(4,'selection','target_best'),(4,'target_access',True),(5,'scratch_only',True),(5,'target_contact',True),(5,'ancestors',[])]:
   values=list(metadata_fixture());values[section]=copy.deepcopy(values[section]);values[section][key]=value
   with self.subTest(section=section,key=key),self.assertRaises(ValueError):p.check_source_metadata(*values)
  values=list(metadata_fixture());values[6]=dict(role_ids=dict(L_s=['other'],U_s=['u0'],V=['v0']))
  with self.assertRaises(ValueError):p.check_source_metadata(*values)
 def test_full_default_head_shape_scale_parameters_and_state(self):
  p.check_head((6,160),30.,900722,206)
  for args in [((6,160),1.,900722,206),((6,161),30.,900722,206),((6,160),30.,900723,206),((6,160),30.,900722,205)]:
   with self.subTest(args=args),self.assertRaises(ValueError):p.check_head(*args)
 def test_existing_V_roles_cells_and_order(self):
  roles,ids,y,rx,day=panel_fixture();p.check_panel(roles,ids,y,rx,day,(1800,2,256))
  for key,value in [('role','U_s'),('V_panel',1799),('held_RX',1),('extra_Ldev',True),('U_truth_used',True),('target_access',True),('updates',1)]:
   wrong=copy.deepcopy(roles);wrong[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):p.check_panel(wrong,ids,y,rx,day,(1800,2,256))
  with self.assertRaises(ValueError):p.check_panel(roles,ids[::-1],y,rx,day,(1800,2,256))
  wrong=list(rx);wrong[0]=2
  with self.assertRaises(ValueError):p.check_panel(roles,ids,y,wrong,day,(1800,2,256))
 def test_full_eleven_condition_cache_schema_and_no_query(self):
  meta=observations_fixture();p.check_observations(meta,d.IQ_SHAPE,'float32')
  for key,value in [('conditions',list(d.CONDITIONS)[::-1]),('scenes',list(d.SCENES)[::-1]),('draws',1),('physical_draws',118800),('target_access',True),('training_updates',1),('formal_test_changed',True)]:
   wrong=copy.deepcopy(meta);wrong[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):p.check_observations(wrong,d.IQ_SHAPE,'float32')
  for shape,dtype,corder in [((2,3,1,1800,2,256),'float32',True),(d.IQ_SHAPE,'float64',True),(d.IQ_SHAPE,'float32',False)]:
   with self.assertRaises(ValueError):p.check_observations(meta,shape,dtype,corder)
 def test_physics_headers_do_not_replace_original_missing_scalars(self):
  headers={key:dict(shape=d.PHYSICS_SHAPE,kind='i' if key in ('state','lock') else 'f') for key in d.PHYSICS_FIELDS};p.check_physics_headers(headers)
  wrong=copy.deepcopy(headers);wrong['state']['shape']=(2,3,1,1800)
  with self.assertRaises(ValueError):p.check_physics_headers(wrong)
  wrong=copy.deepcopy(headers);wrong.pop('signal_to_added_noise_db')
  with self.assertRaises(ValueError):p.check_physics_headers(wrong)
 def test_actual_five_branch_predictions_must_be_finite(self):
  args=((1800,6),(2,3,11,1800,6),(2,3,1800,5,3),d.BRANCHES,d.BRANCH_METRICS,True);p.check_prediction_shapes(*args)
  for index,value in [(0,(1800,7)),(1,(2,3,1,1800,6)),(2,(2,3,1800,4,3)),(3,('time',)),(4,('unknown',)),(5,False)]:
   wrong=list(args);wrong[index]=value
   with self.subTest(index=index),self.assertRaises(ValueError):p.check_prediction_shapes(*wrong)
 def test_gradients_are_six_eval_readouts_not_updates(self):
  rows=gradients_fixture();p.check_gradients(rows);json.dumps(rows,allow_nan=False)
  for mutate in ('drop','order','count','scope','group','nonfinite'):
   wrong=copy.deepcopy(rows)
   if mutate=='drop':wrong=wrong[:-1]
   elif mutate=='order':wrong[0]['scene']='other'
   elif mutate=='count':wrong[0]['count']=1800
   elif mutate=='scope':wrong[0]['scope']='training gradients'
   elif mutate=='group':wrong[0]['groups'].pop('default_head')
   else:wrong[0]['groups']['default_head']['dot']=math.nan
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):p.check_gradients(wrong)
 def test_exact_forward_budget_and_no_train_packets(self):
  self.assertEqual((d.PREDICTION_NN,d.GRADIENT_NN,d.ROW_NN,d.TOTAL_NN),(120600,630,121232,242464));p.check_budget(121232,0)
  for values in ((121230,0),(121232,1),(242464,0)):
   with self.assertRaises(ValueError):p.check_budget(*values)
 def test_probe_delegates_original_observers_and_student_loader(self):
  root=Path(__file__).resolve().parent;tree=ast.parse((root/'probe.py').read_text(encoding='utf8'));calls=[ast.unparse(q.func) for q in ast.walk(tree) if isinstance(q,ast.Call)];self.assertIn('api.own_model',calls);self.assertIn('original_panel',calls);self.assertIn('observe.predict',calls);self.assertIn('observe.gradients',calls);self.assertIn('paths.gates',calls);self.assertNotIn('paths.load',calls);self.assertNotIn('observations',calls);self.assertFalse(any(name.endswith(('.backward','.step','.configurations')) for name in calls));text=(root/'runner.py').read_text(encoding='utf8');self.assertIn(d.REFERENCE_COMMIT,text);self.assertIn('experiments.cvs_fullsource_residual_diag24',text)
 def test_checks_and_protocol_do_not_import_numeric_or_model_runtime(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
