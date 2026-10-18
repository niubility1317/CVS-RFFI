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
if __name__=='__main__':unittest.main()
