import unittest
from types import SimpleNamespace
from .adapter import adapt,apply_args,SEEDS,ARMS
class Checks(unittest.TestCase):
 def ref(self,seg=20,arm='B01'):return dict(segment=seg,new_Ag=False,Ag_mode='NONE',As=arm=='B01',L_NN_per_step=768,model_seed=2026092701,epochs=20 if seg==20 else 10,channel={'seed':42},crossview={'mode':'ORD128'},U_transfer={'mode':'ORIGINAL_ONLY'})
 def row(self,seed=2026092702,arm='B01'):return dict(arm=arm,row_id=arm+'-s'+str(seed),model_seed=seed)
 def test_only_seed_run_bindings(self):
  base=self.ref();x=adapt(base,self.row(),20,'run','/out','owner');self.assertEqual(base['model_seed'],2026092701)
  for k in ['channel','crossview','U_transfer','epochs','new_Ag','As','L_NN_per_step']:self.assertEqual(x[k],base[k])
 def test_own_parent_only(self):
  x=adapt(self.ref(30),self.row(),30,'run','/out','owner');self.assertEqual(x['checkpoint_sources'],['/out/B01-s2026092702/stage20/final_ssdg.pth'])
 def test_scratch_no_ancestors(self):
  x=adapt(self.ref(),self.row(),20,'run','/out','owner');self.assertIsNone(x['parent_source']);self.assertEqual(x['checkpoint_sources'],[])
 def test_invalid_rows_and_budget(self):
  for row,seg in [(self.row(2026092701),20),(self.row(2026092702,'B11_BG'),20),(self.row(),50)]:
   with self.assertRaises(ValueError):adapt(self.ref(),row,seg,'r','/out','o')
 def test_no_newAg_or_unmatched_cost(self):
  for k,v in [('new_Ag',True),('L_NN_per_step',512),('Ag_mode','BG')]:
   b=self.ref();b[k]=v
   with self.assertRaises(ValueError):adapt(b,self.row(),20,'r','/out','o')
 def test_actual_arg_seed_and_E30(self):
  for seg in (20,30):
   c=adapt(self.ref(seg),self.row(),seg,'r','/out','o');a=SimpleNamespace(seed=1,epochs=20 if seg==20 else 10,label_epochs=13 if seg==20 else 0,pseudo_epochs=7 if seg==20 else 10,teacher_ckpt='');apply_args(a,c);self.assertEqual(a.seed,2026092702)
   a.epochs=200
   with self.assertRaises(ValueError):apply_args(a,c)
 def test_namespace_bootstrap_regression(self):
  import tempfile,subprocess,sys,os
  from pathlib import Path
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);old=p/'old';(old/'experiments/legacy').mkdir(parents=True);(old/'experiments/legacy/design.py').write_text('VALUE=7')
   for marker in (True,False):
    root=p/('regular' if marker else 'namespace');(root/'experiments').mkdir(parents=True)
    if marker:(root/'experiments/__init__.py').write_text('')
    code="import experiments,sys;sys.path.insert(0,sys.argv[1]);from experiments.legacy.design import VALUE;print(VALUE)"
    r=subprocess.run([sys.executable,'-c',code,str(old)],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True)
    if marker:self.assertNotEqual(r.returncode,0);self.assertIn(b'ModuleNotFoundError',r.stderr)
    else:self.assertEqual(r.returncode,0);self.assertEqual(r.stdout.strip(),b'7')
 def test_real_publish_archive_preserves_namespace(self):
  import tempfile,sys,types,tarfile,ast
  from pathlib import Path
  from unittest.mock import patch
  from . import publish
  with tempfile.TemporaryDirectory() as td:
   def fake(args,**kw):
    if 'input' in kw:ast.parse(kw['input'].decode());return types.SimpleNamespace(returncode=0,stdout=b'{"status":"VERIFIED"}',stderr=b'')
    return types.SimpleNamespace(returncode=0)
   args=['p','--stage','transfer','--commit','synthetic','--source-root',str(Path(__file__).resolve().parents[2]),'--artifact-root',td,'--project-root','/mock/project','--ssh-config','mock']
   with patch.object(sys,'argv',args),patch.object(publish.subprocess,'run',side_effect=fake):publish.main()
   archives=list(Path(td).glob('*.tar.gz'));self.assertEqual(len(archives),1)
   with tarfile.open(archives[0]) as t:self.assertNotIn('experiments/__init__.py',t.getnames())
if __name__=='__main__':unittest.main()
