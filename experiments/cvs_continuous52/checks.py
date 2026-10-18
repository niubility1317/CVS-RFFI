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
if __name__=='__main__':unittest.main()
