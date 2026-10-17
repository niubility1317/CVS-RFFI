"""Protocol negatives and native-renderer intervention checks, synthetic IQ only."""
import copy,json,unittest
import numpy as np
import torch
from . import design as d
from .controls import compare,render
from .engine import imports
from .model import gradient,compare_grad
from .summarize import metric
class Checks(unittest.TestCase):
 def test_protocol_negatives(self):
  c=d.config();d.validate(c)
  for patch in [{'source_role':'target'},{'extra_held_RX':6},{'extra_Ldev':True},{'training_updates':1},{'source_RX':[1,3,4,8]},{'conditions':['full']}]:
   bad=copy.deepcopy(c);bad.update(patch)
   with self.subTest(patch=patch),self.assertRaises(ValueError):d.validate(bad)
  meta=dict(role='V',counts=dict(L_s=6300,U_s=56700,V=27000),held_RX=None,U_truth_used=False,target_access=False,updates=0);d.validate_input(meta)
  for patch in [dict(role='U_s'),dict(held_RX=1),dict(U_truth_used=True),dict(target_access=True),dict(updates=1)]:
   bad=dict(meta,**patch)
   with self.subTest(patch=patch),self.assertRaises(ValueError):d.validate_input(bad)
 def test_native_counterfactuals(self):
  from dataclasses import replace
  ch,residual=imports();rng=np.random.default_rng(2401);x=rng.normal(size=256)+1j*rng.normal(size=256);x/=np.sqrt(np.mean(abs(x)**2))
  for scene in d.SCENES:
   cfg=ch.Config(fs_hz=25000000.,fc_hz=2450000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,'synthetic RX/day, no TX')
   for seed in (2401,2402):
    values,records,meta=compare(x,cfg,seed,receiver)
    self.assertEqual(tuple(values),d.CONDITIONS)
    base,bm,tr=residual.ResidualChannel(cfg,seed,receiver).process(x,return_traces=True)
    np.testing.assert_allclose(values['full'],base,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(values['no_noise/fixed'],tr['pre_agc_signal']*10**(bm['agc_gain_db']/20),atol=2e-6,rtol=2e-6)
    for name,r in records.items():
     self.assertTrue(np.isfinite(values[name]).all())
     if name.endswith('/fixed') or name=='edge_boundary':
      for k in ['receiver_lock_status','residual_scale','estimated_frequency_hz','phase_tracking_applied','channel_equalization_applied','agc_gain_db']:self.assertEqual(r[k],records['full'][k])
    self.assertEqual(records['no_carrier/fixed']['output_frequency_hz'],0)
    self.assertLessEqual(records['full']['boundary_interior_relative_error'],2e-6)
    self.assertFalse(meta['tx_impairment_correction_applied']);self.assertFalse(meta['channel_equalization_applied'])
   with self.assertRaises(ValueError):compare(x,replace(cfg,equalization_enabled=True),2401,receiver)
 def test_readonly_gradient_units(self):
  a=torch.nn.Parameter(torch.tensor([2.]));b=torch.nn.Parameter(torch.tensor([3.]));unused=torch.nn.Parameter(torch.tensor([1.]))
  named=[('core.sinc.low_coordinate',a),('core.cls_head.weight',b),('unused',unused)]
  g=gradient(a.square().sum()+b.square().sum(),named)
  self.assertAlmostEqual(float(g['Sinc_Hz'][0]),4/25000,places=12)
  stats=compare_grad(g,{k:-v for k,v in g.items()});self.assertAlmostEqual(stats['whole']['cosine'],-1.,places=12)
  self.assertTrue(all(p.grad is None for n,p in named))
 def test_hand_confusion(self):
  r=metric(np.asarray([0,0,1,1,2,3,4,5]),np.asarray([0,1,1,1,2,3,4,5]))
  self.assertEqual(r['correct'],7);self.assertAlmostEqual(r['accuracy'],7/8)
  self.assertAlmostEqual(r['macro_f1'],(2/3+4/5+4)/6)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(1)
 print(json.dumps(dict(status='PASS',tests=result.testsRun,protocol_negatives=11,native_cases=6,synthetic_physical_calls=84,NN=0,real_IQ=0,checkpoint=0,optimizer_updates=0,target_access=False)))
