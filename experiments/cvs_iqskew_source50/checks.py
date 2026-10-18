import sys,types,unittest
from unittest.mock import patch
import numpy as np
from . import core as c,design as d

def fixture():
 base=np.empty((6,3,127),complex)
 for t in range(6):
  for day in range(3):base[t,day]=(.2+.015*t+.01*day)*np.exp(1j*(.3*t+.2*day+.1*t*c.OMEGA))
 delays=c.DELAYS[[50,60,42,66,30]]
 return base[:,None,:,:]*np.exp(1j*delays[None,:,None,None]*c.OMEGA)

class Checks(unittest.TestCase):
 def test_periodic_signature_is_only_math_fixture(self):
  rng=np.random.default_rng(2050);x=rng.normal(size=(8,2,256));x/=np.sqrt(np.mean(x[:,0]**2+x[:,1]**2,axis=1))[:,None,None]
  a=c.features(x);q=x.copy();q[:,1]=np.roll(q[:,1],1,axis=1);b=c.features(q)
  np.testing.assert_allclose(b[:,:2],a[:,:2],rtol=1e-12,atol=1e-10)
  np.testing.assert_allclose(b[:,2]+1j*b[:,3],(a[:,2]+1j*a[:,3])*np.exp(1j*c.OMEGA),rtol=1e-11,atol=1e-10)
  both=c.features(np.roll(x,3,axis=2));np.testing.assert_allclose(both,a,rtol=1e-10,atol=1e-9)
 def test_shared_law_recovers_interior_and_predicts(self):
  x=fixture();r,z=c.evaluate(x,x,x,x);self.assertTrue(r['source_qualified']);self.assertLess(r['SSE']['shared'],1e-20);self.assertEqual(r['boundary_fits'],0)
  np.testing.assert_array_equal(z['fit_indices'][0,0],[50,60,42,66,30])
 def test_null_RX_effect_is_not_a_pass(self):
  x=fixture()[:,0:1].repeat(5,axis=1);r,_=c.evaluate(x,x,x,x);self.assertFalse(r['source_qualified']);self.assertFalse(r['criteria']['nonzero_RX_difference'])
 def test_split_noise_cannot_be_called_coherence(self):
  x=fixture();r,_=c.evaluate(x,x,x,-x);self.assertFalse(r['criteria']['split_coherent_power']);self.assertFalse(r['source_qualified'])
 def test_TX_specific_clock_does_not_prove_shared_RX(self):
  x=fixture();x[3:]*=np.exp(1j*np.asarray([0,.5,-.4,.6,-.5])[None,:,None,None]*c.OMEGA);r,_=c.evaluate(x,x,x,x);self.assertFalse(r['source_qualified'])
 def test_boundary_is_reported_without_widening(self):
  x=fixture()[:,0:1].repeat(5,axis=1);x[:,1:]*=np.exp(.5j*c.OMEGA);r,_=c.evaluate(x,x,x,x);self.assertEqual(r['boundary_fits'],72);self.assertFalse(r['criteria']['bounded_interior_fits'])
 def test_bad_observable_and_missing_cell_rejected(self):
  for x in (np.zeros((2,2,255)),np.full((2,2,256),np.nan),np.zeros((2,2,256))):
   with self.assertRaises(ValueError):c.features(x)
  with self.assertRaises(ValueError):c.cells(np.ones((1,4,127)),np.array([0]),np.array([1]),np.array([1]),70)
  x=fixture();x[0,0,0,0]=np.nan
  with self.assertRaises(ValueError):c.evaluate(x,fixture(),fixture(),fixture())
 def test_folds_and_protocol_are_fixed(self):
  x=d.config();self.assertEqual(x['positive_rfft_bins'],list(range(1,128)));self.assertFalse(x['U_used']);self.assertFalse(x['target_access']);self.assertIsNone(x['source_RX_holdout']);self.assertEqual(x['head_fits'],0);self.assertEqual(x['model_forwards'],0)
 def test_source_loader_never_accesses_U(self):
  from .source import load_source
  expected=dict(role_ids={'L_s':[],'V':[],'U_s':['opaque_hidden']},native_role_comparison='EXACT_MATCH')
  packed={}
  for role,key,count in [('L_s','labeled',70),('V','validation',300)]:
   y=[];rx=[];day=[]
   for t in range(6):
    for r in d.RX:
     for k in (1,2,3):y.extend([t]*count);rx.extend([r]*count);day.extend([k]*count)
   n=len(y);ids=[role+'_'+str(j) for j in range(n)];expected['role_ids'][role]=ids;packed[key]=dict(x=np.broadcast_to(np.zeros((1,2,256)),(n,2,256)),y=np.array(y),rx=np.array(rx),day=np.array(day),ids=ids)
  packed['contract']=expected
  class Guard(dict):
   def __getitem__(self,key):
    if key=='unlabeled':raise AssertionError('U forbidden')
    return super().__getitem__(key)
  with patch.dict(sys.modules,torch=types.SimpleNamespace(load=lambda *a,**k:Guard(packed))),patch.object(d,'read',return_value=expected):
   roles=load_source();self.assertEqual(set(roles),{'L_s','V'})
   packed['validation']['ids'][0]=packed['labeled']['ids'][0]
   with self.assertRaises(ValueError):load_source()
if __name__=='__main__':unittest.main()
