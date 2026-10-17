"""Focused synthetic negatives for physical-L routing; zero query/model forwards."""
import unittest,random
import torch
from .route import CycleRoute
class Cases(unittest.TestCase):
 def meta(self):return {k:torch.full((128,),v,dtype=torch.long) for k,v in [('eq_i',1),('rx_i',3),('day_i',2)]}|{'sig_i':torch.arange(128),'base_index':torch.arange(128)}
 def test_factorial_and_refresh(self):
  g=torch.Generator().manual_seed(71);raw=torch.randn(128,2,256,generator=g);ordinary=raw+1.;meta=self.meta();rng=random.getstate();trng=torch.random.get_rng_state().clone();a=CycleRoute(True);b=CycleRoute(True);r=CycleRoute(False)
  x,log=a.apply(raw,ordinary,meta,1);other,_=b.apply(raw,ordinary,meta,1);same,_=r.apply(raw,ordinary,meta,1)
  self.assertTrue(torch.equal(x,other));self.assertTrue(torch.equal(same,raw));self.assertFalse(torch.equal(x,ordinary));self.assertTrue(torch.equal(x[:,:,:80],raw[:,:,:80]));self.assertTrue(torch.equal(x[:,:,160:],raw[:,:,160:]));self.assertEqual(rng,random.getstate());self.assertTrue(torch.equal(trng,torch.random.get_rng_state()))
  xx,log2=a.apply(raw,ordinary,meta,2);self.assertNotEqual(log['view'],log2['view']);self.assertNotEqual(log['meta']['permutation'],log2['meta']['permutation']);self.assertEqual(a.calls,2);self.assertEqual(a.packets,256)
 def test_rejects_illegal_or_stale(self):
  x=torch.ones(128,2,256);m=self.meta()
  for raw,ordinary,meta,step in [(x[:64],x,m,1),(x,x.double(),m,1),(x,x,{**m,'rx_i':torch.zeros(128,dtype=torch.long)},1),(x,x,{k:v[:127] for k,v in m.items()},1),(x,x,m,2)]:
   with self.assertRaises(ValueError):CycleRoute(True).apply(raw,ordinary,meta,step)
  c=CycleRoute(True);c.apply(x,x,m,1)
  with self.assertRaises(ValueError):c.apply(x,x,m,1)
 def test_repeated_physical_observations_are_retained(self):
  x=torch.ones(128,2,256);m=self.meta();m={k:v*0+v[0] for k,v in m.items()};c=CycleRoute(True);z,log=c.apply(x,x,m,1);self.assertEqual(z.shape,x.shape);self.assertEqual(len(set(log['ids'])),128);self.assertEqual(len(set(log['physical_ids'])),1);self.assertEqual(c.packets,128)
 def test_no_label_dependence(self):
  x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(9));m=self.meta();other=dict(m,TX=torch.arange(128)%6)
  a,_=CycleRoute(True).apply(x,x,m,1);b,_=CycleRoute(True).apply(x,x,other,1);self.assertTrue(torch.equal(a,b))
if __name__=='__main__':unittest.main()
