import unittest
from types import SimpleNamespace
from .adapter import adapt,apply_args,rows
class Checks(unittest.TestCase):
 def base(self,arm='B01'):return dict(segment=20,epochs=20,label_epochs=13,pseudo_epochs=7,updates=4440,new_Ag=False,Ag_mode='NONE',As=arm=='B01',L_NN_per_step=768,model_seed=2026092701,training_mechanism={},channel={'train_seed':42},crossview={'mode':'ORD128'},U_transfer={'mode':'ORIGINAL_ONLY'})
 def test_single_budget_and_no_parent(self):
  c=adapt(self.base(),rows()[1],'run','/out','owner');self.assertEqual((c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates']),(30,13,17,6660));self.assertEqual(c['checkpoint_sources'],[]);self.assertIsNone(c['parent_source']);self.assertEqual(c['epoch_offset'],0);self.assertEqual(c['step_offset'],0)
 def test_recipe_and_channel_unchanged(self):
  b=self.base();c=adapt(b,rows()[1],'r','/o','owner')
  for k in ['new_Ag','Ag_mode','As','L_NN_per_step','channel','crossview','U_transfer']:self.assertEqual(c[k],b[k])
  self.assertEqual(b['epochs'],20);self.assertEqual(c['training_mechanism']['expected_L_forward_packets'],5114880)
 def test_actual_args(self):
  c=adapt(self.base(),rows()[1],'r','/o','owner');a=SimpleNamespace(epochs=20,label_epochs=13,pseudo_epochs=7,teacher_ckpt='',baseline_ckpt='');a=apply_args(a,c);self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs),(30,13,17));self.assertTrue(a.from_scratch);self.assertEqual(a.baseline_ckpt,'')
 def test_bad_reference_rejected(self):
  for k,v in [('segment',30),('epochs',200),('label_epochs',20),('pseudo_epochs',17),('updates',6660),('new_Ag',True),('Ag_mode','BG'),('L_NN_per_step',512),('As',False)]:
   b=self.base();b[k]=v
   with self.assertRaises(ValueError):adapt(b,rows()[1],'r','/o','owner')
 def test_bad_rows_and_initializers_rejected(self):
  with self.assertRaises(ValueError):adapt(self.base(),dict(arm='B11',row_id='bad',model_seed=2026092703),'r','/o','owner')
  c=adapt(self.base(),rows()[1],'r','/o','owner')
  for name in ['teacher_ckpt','baseline_ckpt']:
   a=SimpleNamespace(epochs=20,label_epochs=13,pseudo_epochs=7,teacher_ckpt='',baseline_ckpt='');setattr(a,name,'forbidden')
   with self.assertRaises(ValueError):apply_args(a,c)
 def test_real_publish_archive_preserves_namespace(self):
  import tempfile,sys,types,tarfile,ast
  from pathlib import Path
  from unittest.mock import patch
  from . import publish
  with tempfile.TemporaryDirectory() as td:
   seen=[]
   def fake(args,**kw):
    if 'input' in kw:ast.parse(kw['input'].decode());seen.append(kw['input']);return types.SimpleNamespace(returncode=0,stdout=b'{"status":"VERIFIED"}',stderr=b'')
    return types.SimpleNamespace(returncode=0)
   for stage in ['transfer','launch']:
    a=['p','--stage',stage,'--commit','synthetic','--source-root',str(Path(__file__).resolve().parents[2]),'--artifact-root',td,'--project-root','/mock/project','--ssh-config','mock']
    with patch.object(sys,'argv',a),patch.object(publish.subprocess,'run',side_effect=fake):publish.main()
   with tarfile.open(next(Path(td).glob('*.tar.gz'))) as t:
    self.assertNotIn('experiments/__init__.py',t.getnames());expected={'experiments/cvs_continuous52/'+p.name for p in Path(__file__).parent.glob('*.py')}|{'release_commit.txt'};self.assertEqual(set(t.getnames()),expected)
   self.assertEqual(len(seen),2)
 def test_source_metrics_cannot_change_dispatch(self):
  import ast
  from pathlib import Path
  s=(Path(__file__).parent/'dispatch.py').read_text();ast.parse(s);self.assertNotIn('for segment in (20,30)',s);self.assertIn("execute('predict',row)",s);self.assertIn('formal_evaluation_still_required=True',s)
 def test_continuous_channel_budget_boundaries(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'channel.py').read_text())
  klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Practical');apply=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='apply');guard=compile(ast.Expression(apply.body[0].test),'<continuous channel guard>','eval')
  x=SimpleNamespace(ndim=3,shape=(64,2,256));torch=SimpleNamespace(isfinite=lambda _:SimpleNamespace(all=lambda:True))
  for epoch,step in [(1,1),(20,4440),(21,4441),(30,6660)]:self.assertFalse(eval(guard,{'epoch':epoch,'step':step,'x':x,'torch':torch}))
  for epoch,step in [(0,1),(31,6660),(30,0),(30,6661)]:self.assertTrue(eval(guard,{'epoch':epoch,'step':step,'x':x,'torch':torch}))
 def test_owned_channel_installed_for_training_and_prediction(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'model.py').read_text())
  for name in ('install','build'):
   fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
   found=[x for x in ast.walk(fn) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_full_channel' for t in x.targets)]
   self.assertEqual(len(found),1);self.assertEqual(ast.unparse(found[0].value),'Practical()')
   guarded=[x for x in fn.body if isinstance(x,ast.If) and ast.unparse(x.test)=="c['As']" and found[0] in x.body];self.assertEqual(len(guarded),1)
 def test_original_prediction_array_bridge_remains_exported(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'channel.py').read_text())
  imports=[x for x in tree.body if isinstance(x,ast.ImportFrom) and x.module=='experiments.cvs_crossview_supcon41.channel']
  self.assertEqual([a.name for x in imports for a in x.names],['tensor_iq'])
 def test_smoke_cost_follows_actual_rows(self):
  import ast
  from pathlib import Path
  tree=ast.parse((Path(__file__).parent/'preflight.py').read_text())
  fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='smoke_counts');ns={}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual smoke counts>','exec'),ns)
  for row_count,expected in [(1,770),(2,1538)]:
   counts=ns['smoke_counts']([dict(NN=768) for _ in range(row_count)],2)
   self.assertEqual(counts,dict(rows=row_count,NN=768*row_count,real_checkpoint_NN=2,total_NN=expected))
 def isolated_design(self):
  import ast
  from pathlib import Path
  from .adapter import SEED,ARMS
  path=Path(__file__).parent/'design.py'
  old=SimpleNamespace(PROJECT='/mock',SOURCE='/source',RX=[1,3,4,6,8],FULL_FP32_POLICY='fixed',CLASSES=6,SCENES=[],VIEWS=[],FORMAL_VIEWS='/formal',TRUTH='/truth',PANEL='/panel',SOURCE_CACHE='/cache',BASE=Path('/staged'),read=None,write=None,rows=rows,config=lambda row,segment:self.base(row['arm']))
  tree=ast.parse(path.read_text(encoding='utf8'));tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  ns=dict(Path=Path,old=old,rows=rows,adapt=adapt,apply_args=apply_args,SEED=SEED,ARMS=ARMS,__file__=str(path))
  exec(compile(tree,'<actual recovery design>','exec'),ns)
  return ns
 def test_recovery_preserves_control_identity_and_As_recipe(self):
  from pathlib import Path
  d=SimpleNamespace(**self.isolated_design());b00,b01=d.rows();control=d.config(b00);new=d.config(b01)
  self.assertEqual(control,adapt(self.base('B00'),b00,d.CONTROL_RUN,d.CONTROL_BASE.as_posix(),d.CONTROL_OWNER))
  prior=adapt(self.base('B01'),b01,d.CONTROL_RUN,d.CONTROL_BASE.as_posix(),d.CONTROL_OWNER)
  identity={'run_id','launch_owner','output_root'}
  self.assertEqual({k:v for k,v in new.items() if k not in identity},{k:v for k,v in prior.items() if k not in identity})
  self.assertEqual({k for k in new if new[k]!=prior[k]},identity)
  self.assertEqual(d.train_rows(),[b01])
  for row,c in [(b00,control),(b01,new)]:
   self.assertEqual(Path(c['output_root']).parent,d.row_root(row));self.assertEqual(d.row_root(row),d.row_root(row['row_id']));self.assertEqual(d.validate(c),c)
  self.assertEqual(d.row_root(b00),d.CONTROL_BASE/b00['row_id']);self.assertEqual(d.row_root(b01),d.BASE/b01['row_id'])
 def test_recovery_rejects_control_execution_in_all_modes(self):
  import ast,argparse,sys,os
  from pathlib import Path
  from unittest.mock import patch
  tree=ast.parse((Path(__file__).parent/'dispatch.py').read_text(encoding='utf8'));d=SimpleNamespace(**self.isolated_design());ns=dict(d=d,argparse=argparse,sys=sys,os=os)
  exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('main','execute')],type_ignores=[]),'<actual recovery entrypoints>','exec'),ns)
  for mode in ('worker','predict','score'):
   with patch.object(sys,'argv',['test','--mode',mode,'--row',d.rows()[0]['row_id']]):
    with self.assertRaises(ValueError):ns['main']()
   with self.assertRaises(ValueError):ns['execute'](mode,d.rows()[0]['row_id'])
 def test_recovery_report_reuses_control_and_keeps_stage_condition_report_only(self):
  import ast,tempfile
  from copy import deepcopy
  from pathlib import Path
  with tempfile.TemporaryDirectory() as td:
   globals_=self.isolated_design();globals_['BASE']=Path(td)/'new';globals_['CONTROL_BASE']=Path(td)/'control';d=SimpleNamespace(**globals_);d.BASE.mkdir()
   d.VIEWS=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];store={};writes=[]
   decision=dict(source_qualified=False,criteria={'fixed_source_check':False},formal_scores_read=False)
   fixed=deepcopy(decision);decision_path=d.BASE/'source_matrix_decision.json';decision_path.touch();store[decision_path]=decision
   d.read=lambda p:store[p]
   def write(p,v):store[p]=v;writes.append(p)
   d.write=write
   for row in d.rows():
    p=d.row_root(row);store[p/'scoring/complete.json']=dict(status='SCORED_COMPLETE');store[p/'prediction/complete.json']=dict(views=d.VIEWS);store[p/'source/result.json']=dict(actual='cost_'+row['arm'])
    store[p/'scoring/scores.json']=dict(results=[dict(row_id=row['row_id'],dimension='overall',view=v,accuracy=.85 if row['arm']=='B00' else .849 if v=='clean' else .9) for v in d.VIEWS])
   tree=ast.parse((Path(__file__).parent/'report.py').read_text(encoding='utf8'));ns=dict(d=d)
   exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'<actual recovery report>','exec'),ns)
   for clean,passes in [(.849,True),(.8485,False)]:
    store[d.row_root(d.rows()[1])/'scoring/scores.json']['results'][0]['accuracy']=clean
    result=ns['collect']();self.assertEqual(result['status'],'MATRIX_SCORED');self.assertFalse(result['source_qualified'])
    report=store[d.BASE/'final_scores.json'];self.assertEqual(report['source_decision'],fixed);self.assertEqual(decision,fixed)
    self.assertEqual(report['stage_assessment']['clean_cost_within_stage_limit'],passes);self.assertTrue(report['stage_assessment']['report_only']);self.assertFalse(report['stage_assessment']['changes_source_criteria'])
    self.assertEqual((report['new_model_epochs'],report['new_updates'],report['reused_model_epochs'],report['reused_updates'],report['failed_original_As_epochs'],report['failed_original_As_updates']),(30,6660,30,6660,20,4440))
    self.assertEqual(report['cost'],[{'actual':'cost_B00'},{'actual':'cost_B01'}]);self.assertEqual(len(report['results']),14)
   self.assertEqual(writes,[d.BASE/'final_scores.json']*2)
if __name__=='__main__':unittest.main()
