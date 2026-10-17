import importlib.util,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).parent
for name in ('noise','criteria','protocol'):
 s=importlib.util.spec_from_file_location('n43_'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);globals()[name]=m
class Checks(unittest.TestCase):
 def x(self):return torch.randn((128,2,256),generator=torch.Generator().manual_seed(43))
 def test_energy_both_shapes(self):
  x=self.x();before=x.clone();outs=[]
  for mode in ('WHITE','BG'):
   y,r=noise.apply(x,1,mode);delta=y-x;ratio=delta.square().sum((1,2))/x.square().sum((1,2));self.assertTrue(torch.allclose(ratio,torch.full((128,),.001),atol=2e-8));self.assertEqual(r['new_boundary_samples'],0);outs.append(r)
  self.assertTrue(torch.equal(before,x));self.assertGreater(outs[1]['noise_peak_fraction_mean'],3*outs[0]['noise_peak_fraction_mean'])
 def test_private_rng_refresh_and_control(self):
  x=self.x();state=torch.random.get_rng_state().clone();a,_=noise.apply(x,2,'BG');b,_=noise.apply(x,2,'BG');c,_=noise.apply(x,3,'BG');raw,_=noise.apply(x,2,'NONE');self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,c));self.assertTrue(torch.equal(x,raw));self.assertNotEqual(x.data_ptr(),raw.data_ptr());self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
 def test_zeroIQ(self):
  for mode in noise.MODES:self.assertTrue(torch.equal(noise.apply(torch.zeros(2,2,256),1,mode)[0],torch.zeros(2,2,256)))
 def test_invalid_noise(self):
  cases=[(torch.zeros(2,3,256),1,'BG'),(torch.zeros(2,2,255),1,'BG'),(torch.zeros(2,2,256,dtype=torch.float64),1,'BG'),(torch.full((2,2,256),float('nan')),1,'BG'),(self.x(),0,'BG'),(self.x(),True,'BG'),(self.x(),1,'UNKNOWN')]
  for args in cases:
   with self.assertRaises(ValueError):noise.apply(*args)
 def metrics(self):
  c={a:.8 for a in criteria.ARMS};c.update(B10_BG=.804,B11_BG=.806);u={a:[[.8,.5,.4],[.8,.5,.4]] for a in criteria.ARMS};u['B01']=[[.8,.6,.5],[.8,.6,.5]];u['B11_WHITE']=[[.8,.61,.51],[.8,.61,.51]];u['B11_BG']=[[.8,.63,.53],[.8,.63,.53]];rx={a:{str(r):.8 for r in (1,3,4,6,8)} for a in criteria.ARMS};return c,u,rx
 def test_criterion_pass_notproof(self):
  r=criteria.decide(*self.metrics());self.assertTrue(r['source_qualified']);self.assertFalse(r['Ag_proved']);self.assertFalse(r['goal_complete'])
 def test_criterion_science_fail(self):
  c,u,r=self.metrics();c['B11_BG']=.799;v=criteria.decide(c,u,r);self.assertFalse(v['source_qualified']);self.assertFalse(v['checks']['joint_clean_net_gain'])
 def test_incomplete_or_nonfinite(self):
  c,u,r=self.metrics();c.pop('B00')
  with self.assertRaises(ValueError):criteria.decide(c,u,r)
  c,u,r=self.metrics();u['B11_BG'][0][1]=float('nan')
  with self.assertRaises(ValueError):criteria.decide(c,u,r)
  c,u,r=self.metrics();r['B00'].pop('1')
  with self.assertRaises(ValueError):criteria.decide(c,u,r)
 def test_checkpoint_boundary(self):
  self.assertEqual(protocol.loader_kind('/x/final','/x/parent','/x/final'),'final');self.assertEqual(protocol.loader_kind('/x/parent','/x/parent','/x/final'),'parent')
  with self.assertRaises(ValueError):protocol.loader_kind('/other','/x/parent','/x/final')
if __name__=='__main__':unittest.main()
