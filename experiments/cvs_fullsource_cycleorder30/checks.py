"""Focused operator/role negatives; synthetic only, no real IQ or CP."""
import itertools,random,unittest
import numpy as np
import torch
from .iq_order import apply_parameters,transform
from .stats import verdict
from .probe import valid_roles,N,NN,SCENES
class Checks(unittest.TestCase):
 def test_all24_and_roundtrip(self):
  g=torch.Generator().manual_seed(30);x=torch.randn(24,2,256,generator=g);p=torch.tensor(list(itertools.permutations(range(4))));phase=torch.linspace(-3,3,24)
  yy=apply_parameters(x,p,phase);inverse=p.argsort(1);zz=apply_parameters(yy,inverse,phase)
  self.assertTrue(torch.allclose(zz,x,atol=2e-6,rtol=2e-6));self.assertTrue(torch.equal(yy[:,:,:80],x[:,:,:80]));self.assertTrue(torch.equal(yy[:,:,160:],x[:,:,160:]));self.assertTrue(torch.allclose(yy.square().sum((1,2)),x.square().sum((1,2)),atol=1e-4,rtol=2e-6))
  for badp in [p[:,:3],torch.zeros(24,4),torch.full((24,4),4)]:
   with self.assertRaises(ValueError):apply_parameters(x,badp,phase)
  for badx in [x.double(),x*float('nan'),x[:,:,:255],x[:0]]:
   with self.assertRaises(ValueError):apply_parameters(badx,p,phase)
 def test_ideal_periodic_and_private_rng(self):
  g=torch.Generator().manual_seed(302);base=torch.randn(1,20,generator=g)+1j*torch.randn(1,20,generator=g);n=torch.arange(256);z=base[:,n%20]*torch.exp(1j*n[None,:]*.017);x=torch.stack((z.real,z.imag),1).repeat(24,1,1);p=torch.tensor(list(itertools.permutations(range(4))));y=apply_parameters(x,p,torch.full((24,),.34));self.assertTrue(torch.allclose(x,y,atol=2e-6,rtol=2e-6))
  ids=['id'+str(i) for i in range(24)];a=random.getstate();b=torch.get_rng_state().clone();first,meta=transform(x,ids,'synthetic/clean');again,other=transform(x,ids,'synthetic/clean');self.assertTrue(torch.equal(first,again));self.assertEqual(meta,other);self.assertEqual(a,random.getstate());self.assertTrue(torch.equal(b,torch.get_rng_state()));self.assertLess(float((first-x).abs().max()),3e-6)
  zero,m=transform(torch.zeros(24,2,256),ids,'synthetic/zero');self.assertTrue(torch.equal(zero,torch.zeros_like(zero)));self.assertFalse(any(m['phase_defined']))
  for badids,tag in [(ids[:-1],'x'),(['same']*24,'x'),(ids,''),(list(range(24)),'x')]:
   with self.assertRaises(ValueError):transform(x,badids,tag)
 def test_phase_covariance(self):
  g=torch.Generator().manual_seed(303);x=torch.randn(32,2,256,generator=g);ids=['p'+str(i) for i in range(32)];a,meta=transform(x,ids,'synthetic/random');z=torch.complex(x[:,0],x[:,1]);phase=torch.exp(torch.tensor(.27j));rot=z*phase;b,_=transform(torch.stack((rot.real,rot.imag),1),ids,'synthetic/random');az=torch.complex(a[:,0],a[:,1])*phase;self.assertTrue(torch.allclose(torch.complex(b[:,0],b[:,1]),az,atol=4e-6,rtol=4e-6))
  gain,_=transform(x*1.05,ids,'synthetic/random');self.assertTrue(torch.allclose(gain,a*1.05,atol=4e-6,rtol=4e-6))
 def test_rule_and_roles(self):
  args=[-.5,[-1]*5,[-1]*6,.99,[[-.5]*2]*2,.8,.01,0];self.assertEqual(verdict(*args)['status'],'PASS_SOURCE_FALSIFICATION_ONLY')
  for i,bad in [(0,-.50001),(1,[-1.01]+[0]*4),(2,[-1.01]+[0]*5),(3,.98999),(4,[[-.50001,0],[0,0]]),(5,.79999),(6,.00999),(7,1)]:
   v=list(args);v[i]=bad;self.assertEqual(verdict(*v)['status'],'FAIL_FIXED_CYCLE_ORDER')
  self.assertEqual(verdict(-.5,[-1]*5,[-1]*6,.99,[[-5]*2]*2,.8,.01,0,require_urban=False)['status'],'PASS_SOURCE_FALSIFICATION_ONLY')
  for bad in [[[0,0]],[[float('nan'),0],[0,0]]]:
   with self.assertRaises(ValueError):verdict(0,[0]*5,[0]*6,1,bad,1,.1,0)
  r=dict(role='V',counts={'L_s':6300,'U_s':56700,'V':27000},V_panel=N,held_RX=None,extra_Ldev=False,U_truth_used=False,target_access=False,updates=0);o=dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=N,draws=2,scenes=list(SCENES),physical_draws=10800,target_access=False,training_updates=0,formal_test_changed=False,conditions=['full']+['control']*10);valid_roles(r,o)
  for key,value in [('held_RX',1),('extra_Ldev',True),('U_truth_used',True),('target_access',True),('updates',1),('V_panel',720)]:
   bad=dict(r);bad[key]=value
   with self.assertRaises(ValueError):valid_roles(bad,o)
  self.assertEqual(NN,12602)
if __name__=='__main__':unittest.main()
