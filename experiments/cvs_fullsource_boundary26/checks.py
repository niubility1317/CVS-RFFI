"""Focused boundary/permission/native-coupling checks; synthetic only."""
import copy,json,sys,unittest
from pathlib import Path
from contextvars import ContextVar
from dataclasses import asdict,replace
from types import SimpleNamespace
import numpy as np
from . import design as d
from .boundary import boundary_for_call,training_boundary
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/adv3b02_xuc/code'))
from leo_practical.channel import Config,receiver_for_session
from leo_practical.residual import ResidualChannel
class Checks(unittest.TestCase):
 def test_budget_and_scope(self):
  c=d.config(d.rows()[0]);d.validate(c)
  self.assertEqual(len(d.rows()),1);self.assertEqual((c['epochs'],c['label_epochs'],c['pseudo_epochs'],c['updates']),(30,20,10,6660))
  self.assertTrue(c['As']);self.assertFalse(c['Ag'])
  for patch in [dict(epochs=20),dict(updates=4440),dict(held_RX=1),dict(source_RX=[3,4,6,8]),dict(source_roles=dict(L_s=5040,U_s=45360,V=21600)),dict(Ag=True),dict(checkpoint_sources=['unverified']),dict(checkpoint_selection='target_best')]:
   with self.subTest(patch=patch),self.assertRaises(ValueError):d.validate(dict(c,**patch))
 def test_schedule(self):
  for e in range(30):self.assertEqual([boundary_for_call(e*222+k) for k in range(1,223)].count('edge'),111)
  for bad in (0,-1,6661,1.,True,None):
   with self.subTest(call=bad),self.assertRaises(ValueError):boundary_for_call(bad)
 def test_config_only_boundary_and_restore(self):
  cfg=Config(fs_hz=25000000.,fc_hz=2450000000.,scenario='practical_low_urban',processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
  context=ContextVar('test_context',default=None);pa=SimpleNamespace(_context=context,practical_config=lambda *a:cfg);orig=pa.practical_config
  token=context.set((['synthetic'],['RX1day1'],'source_dynamic_E1_L'))
  with training_boundary(pa,'edge'):
   q=pa.practical_config('practical_low_urban',None)
   self.assertEqual({k for k,v in asdict(cfg).items() if asdict(q)[k]!=v},{'boundary'})
  self.assertIs(pa.practical_config,orig);self.assertEqual(cfg.boundary,'reflect')
  with self.assertRaises(RuntimeError):
   with training_boundary(pa,'edge'):raise RuntimeError('synthetic inner failure')
  self.assertIs(pa.practical_config,orig);context.reset(token)
  for namespace in [None,'target_fixed_v3','source_validation_fixed_v3','source_dynamic_E21_U']:
   token=context.set(None if namespace is None else (['x'],['r'],namespace))
   with self.subTest(namespace=namespace),self.assertRaises(ValueError):
    with training_boundary(pa,'edge'):pa.practical_config('practical_low_urban',None)
   self.assertIs(pa.practical_config,orig);context.reset(token)
  token=context.set((['x'],['r'],'source_dynamic_E1_L'))
  for patch in [dict(equalization_enabled=True),dict(processing_route='full'),dict(reference_snr_db=31.),dict(boundary='edge'),dict(quality_adaptation_enabled=False),dict(phase_tracking_enabled=False),dict(iq_compensation_enabled=False)]:
   pa.practical_config=lambda *a,patch=patch:replace(cfg,**patch);before=pa.practical_config
   with self.subTest(patch=patch),self.assertRaises(ValueError):
    with training_boundary(pa,'edge'):pa.practical_config('practical_low_urban',None)
   self.assertIs(pa.practical_config,before)
  context.reset(token)
 def test_native_coupled_receiver(self):
  rng=np.random.default_rng(202610172601);t=np.arange(256)
  samples=[rng.normal(size=256)+1j*rng.normal(size=256),np.exp(2j*np.pi*t/20),np.r_[np.ones(61),np.ones(195)*.2]*(rng.normal(size=256)+1j*rng.normal(size=256)),np.r_[np.zeros(70),rng.normal(size=160)+1j*rng.normal(size=160),np.zeros(26)]]
  for scene in d.TRAIN_SCENES:
   cfg=Config(fs_hz=25000000.,fc_hz=2450000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')
   receiver=receiver_for_session(cfg,2026092701,'synthetic RX/day; independent of TX')
   for seed in (2601,2602):
    for x in samples:
     x=x/np.sqrt(np.mean(abs(x)**2))
     a,ma,ta=ResidualChannel(cfg,seed,receiver).process(x,return_traces=True)
     b,mb,tb=ResidualChannel(replace(cfg,boundary='edge'),seed,receiver).process(x,return_traces=True)
     self.assertEqual(ma['boundary'],'synthetic_reflect');self.assertEqual(mb['boundary'],'synthetic_edge')
     for k in ['state_end','quality_snr_db','receiver_lock_status','residual_scale','estimated_frequency_hz','output_frequency_hz','phase_tracking_applied','channel_equalization_applied','residual_kernel_length','numerical_filter_latency_samples']:self.assertEqual(ma[k],mb[k],k)
     self.assertFalse(ma['channel_equalization_applied']);self.assertFalse(mb['channel_equalization_applied'])
     np.testing.assert_allclose(ta['pre_agc_signal'][61:],tb['pre_agc_signal'][61:],atol=2e-6,rtol=2e-6)
     np.testing.assert_allclose(ta['pre_agc_noise'],tb['pre_agc_noise'],atol=2e-6,rtol=2e-6)
     for y,m,tr in [(a,ma,ta),(b,mb,tb)]:
      self.assertTrue(np.isfinite(y).all());np.testing.assert_allclose(y,(tr['pre_agc_signal']+tr['pre_agc_noise'])*10**(m['agc_gain_db']/20),atol=2e-6,rtol=2e-6)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise SystemExit(1)
 print(json.dumps(dict(status='PASS',tests=result.testsRun,protocol_negatives=8,schedule_negatives=6,namespace_negatives=4,physics_config_negatives=7,native_physical_calls=48,NN=0,real_IQ=0,checkpoint=0,updates=0,target_access=False)))
