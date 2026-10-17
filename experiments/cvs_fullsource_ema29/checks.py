"""Focused source-role, numerical, and fixed-rule negatives; no real IQ/CP."""
import copy,unittest
import numpy as np
from .stats import metric,verdict,SCENES
from .probe import valid_roles,N,NN
class Checks(unittest.TestCase):
 def test_metrics(self):
  y=np.arange(6);z=np.eye(6)*30;r=metric(y,z)
  self.assertEqual(r['accuracy'],1.);self.assertEqual(r['macro_F1'],1.);self.assertEqual(r['margin_scale30'],1.)
  for yy,zz in [(y.astype(float),z),(y,z[:5]),(np.array([6]*6),z),(y,z*np.nan),(np.array([],int),z[:0])]:
   with self.assertRaises(ValueError):metric(yy,zz)
 def test_verdict(self):
  self.assertEqual(verdict(0,[0]*5,[[1,1],[1,1]])['status'],'PASS_SOURCE_READOUT_ONLY')
  exact=100*(1242/1800-1224/1800)
  boundary=verdict(0,[0]*5,[[exact]*2]*2)
  self.assertEqual(boundary['status'],'PASS_SOURCE_READOUT_ONLY');self.assertEqual(boundary['cells_passed'],4)
  self.assertEqual(verdict(0,[0]*5,[[1.-1e-8,1],[1,1]])['status'],'FAIL_FIXED_FINAL30_EMA_READOUT')
  for c,r,u in [(-.0001,[0]*5,[[1,1],[1,1]]),(0,[-.5001]+[0]*4,[[1,1],[1,1]]),(0,[0]*5,[[.9999,1],[1,1]])]:self.assertEqual(verdict(c,r,u)['status'],'FAIL_FIXED_FINAL30_EMA_READOUT')
  for c,r,u in [(0,[0]*4,[[1,1],[1,1]]),(0,[0]*5,[[1,1]]),(float('nan'),[0]*5,[[1,1],[1,1]])]:
   with self.assertRaises(ValueError):verdict(c,r,u)
 def test_source_negatives(self):
  r=dict(role='V',counts={'L_s':6300,'U_s':56700,'V':27000},V_panel=N,held_RX=None,extra_Ldev=False,U_truth_used=False,target_access=False,updates=0)
  o=dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=N,draws=2,scenes=list(SCENES),physical_draws=10800,target_access=False,training_updates=0,formal_test_changed=False,conditions=['full']+['control']*10)
  valid_roles(r,o)
  for k,v in [('role','L_s'),('held_RX',1),('extra_Ldev',True),('U_truth_used',True),('target_access',True),('updates',1),('V_panel',720),('counts',{'L_s':5580,'U_s':56700,'V':27000})]:
   x=copy.deepcopy(r);x[k]=v
   with self.assertRaises(ValueError):valid_roles(x,o)
  for k,v in [('draws',1),('scenes',['clean']),('target_access',True),('training_updates',1),('formal_test_changed',True),('conditions',['edge']*11)]:
   x=copy.deepcopy(o);x[k]=v
   with self.assertRaises(ValueError):valid_roles(r,x)
  self.assertEqual(NN,2+1800+2*3*1800)
if __name__=='__main__':unittest.main()
