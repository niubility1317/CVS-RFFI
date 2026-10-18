import unittest
import numpy as np
from .core import cells,evaluate,BASIS,COEFFS,load_observations
class Checks(unittest.TestCase):
 def fixture(self):
  gamma=np.array([-.1,.2,-.3,.4,-.2]);raw=np.sin(np.arange(1,128)*.3);raw-=raw.mean();raw-=raw.dot(BASIS)/BASIS.dot(BASIS)*BASIS
  V=np.broadcast_to(gamma[None,:,None,None]*BASIS,(6,5,3,127)).copy();L=V+np.arange(1,7)[:,None,None,None]*np.array([1,-2,3,-1,-1])[None,:,None,None]*raw[None,None,None,:]*.02
  return L,V,L.copy(),L.copy()
 def test_shared_law_and_sign(self):
  L,V,L0,L1=self.fixture();r,a=evaluate(L,V,L0,L1)
  self.assertTrue(r['source_qualified']);self.assertLess(r['SSE']['shared'],1e-20);np.testing.assert_allclose(COEFFS[a['fit_indices'][0,0]],np.array([0,.3,-.2,.5,-.1]),atol=1e-14)
 def test_no_effect_fails(self):
  z=np.zeros((6,5,3,127));r,_=evaluate(z,z,z,z);self.assertFalse(r['source_qualified'])
 def test_boundary_fails(self):
  L,V,L0,L1=self.fixture();L[:,4]+=BASIS*2;r,_=evaluate(L,V,L,L);self.assertGreater(r['boundary_fits'],0);self.assertFalse(r['criteria']['bounded_interior_fits'])
 def test_unrepeatable_split_fails(self):
  L,V,L0,L1=self.fixture();r,_=evaluate(L,V,L,-L);self.assertFalse(r['criteria']['split_RX_contrast_power'])
 def test_cell_log_and_constant_gain_removal(self):
  y=np.repeat(np.arange(6),5*3*70);rx=np.tile(np.repeat([1,3,4,6,8],3*70),6);day=np.tile(np.repeat([1,2,3],70),30);R=np.broadcast_to(.2*BASIS,(len(y),127));f=np.zeros((len(y),4,127));f[:,0]=10**((R+3)/20);f[:,1]=10**(-(R+3)/20);f[:,2]=1
  c=cells(f,y,rx,day,70);np.testing.assert_allclose(c,np.broadcast_to(.2*BASIS,c.shape),atol=1e-12)
  for bad in (f[:1],f.copy()):
   if len(bad)>1:bad[0,0,1]=np.nan
   with self.assertRaises(ValueError):cells(bad,y,rx,day,70)
  f[:,0,0]=0
  with self.assertRaises(ValueError):cells(f,y,rx,day,70)
 def test_metadata_mismatch_rejected(self):
  import tempfile,json
  from pathlib import Path
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'observations.npz';cfg=Path(t)/'resolved_config.json';cfg.write_text(json.dumps({'source_roles':'EXACT_MATCH','config':{'L':1,'V':27000,'U_used':False}}))
   with self.assertRaises(ValueError):load_observations(p,cfg)
 def test_fixed_support(self):
  from .design import config
  c=config();self.assertEqual(c['source_RX'],[1,3,4,6,8]);self.assertEqual(c['positive_rfft_bins'],list(range(1,128)));self.assertFalse(c['U_used']);self.assertEqual(c['training_updates'],0);self.assertEqual(len(COEFFS),101);self.assertAlmostEqual(BASIS.mean(),0)
if __name__=='__main__':unittest.main()
