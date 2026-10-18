import unittest
import numpy as np
from .core import curves,iq_complex,bound_from_L,cell_means,evaluate,U
class Checks(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(4801);self.z=rng.normal(size=(7,256))+1j*rng.normal(size=(7,256));self.z/=np.sqrt(np.mean(abs(self.z)**2,1,keepdims=True))
 def test_exact_nonlinear_translation(self):
  b=.013;p=abs(self.z)**2;zt=self.z*np.exp(1j*b*(p-1))
  np.testing.assert_allclose(curves(zt,np.array([0.])),curves(self.z,np.array([b])),atol=8e-16,rtol=3e-14)
  np.testing.assert_allclose(abs(zt),abs(self.z),atol=1e-15)
 def test_constant_and_linear_phase_invariance(self):
  n=np.arange(256);z=self.z*np.exp(1j*(.7+.003*n))
  np.testing.assert_allclose(curves(z,np.array([0.,.02])),curves(self.z,np.array([0.,.02])),atol=4e-16,rtol=1e-12)
 def test_input_negatives(self):
  for x in (np.zeros((2,2,256)),np.ones((2,2,255)),np.full((1,2,256),np.nan)):
   with self.assertRaises(ValueError):iq_complex(x)
 def test_missing_cell_rejected(self):
  with self.assertRaises(ValueError):cell_means(np.zeros((1,21,3)),np.array([0]),np.array([1]),np.array([1]),70)
 def test_exact_shared_relation(self):
  beta=np.linspace(-.02,.02,21);L=np.empty((6,5,3,21,3));V=np.empty_like(L);shifts=np.array([0,-.012,-.006,.006,.012])
  for t in range(6):
   for r in range(5):
    for d in range(3):
     v=U[None,:]+beta[:,None]+shifts[r];L[t,r,d]=(.3+.035*t+.017*d)*np.cos((.4+.13*t)*v)+.05*np.sin((1+.21*d)*v)
     V[t,r,d]=L[t,r,d]
  res,a=evaluate(L,V,beta);self.assertTrue(res['source_qualified']);self.assertLess(res['SSE']['shared'],1e-25)
  V2=V.copy();V2[0]*=.7;_,a2=evaluate(L,V2,beta);np.testing.assert_array_equal(a['fit_indices'],a2['fit_indices'])
 def test_additive_rejected(self):
  beta=np.linspace(-.02,.02,21);L=np.empty((6,5,3,21,3))
  for t in range(6):
   for r in range(5):
    for d in range(3):L[t,r,d]=.3+.02*t+.01*d+np.array([.1,.2,.3])[None,:]*beta[:,None]+.004*r
  res,_=evaluate(L,L.copy(),beta);self.assertFalse(res['source_qualified'])
 def test_no_RX_effect_rejected(self):
  beta=np.linspace(-.02,.02,21);L=np.ones((6,5,3,21,3))*.2;res,_=evaluate(L,L.copy(),beta);self.assertFalse(res['source_qualified'])
if __name__=='__main__':unittest.main()
