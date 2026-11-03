import unittest
from types import SimpleNamespace
from unittest.mock import patch
import experiments
from experiments.phase1_iterative_multiscale_fullsource.replay import model_for_schema
from experiments.phase1_multiscale_token.model import MultiScaleToken
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq

class FactoryTest(unittest.TestCase):
 def test_registered_factories_survive_path_reset(self):
  factories=dict(first_factory=lambda:SimpleNamespace(kind='multi'),second_factory=lambda:SimpleNamespace(kind='iterative'))
  with patch.object(experiments,'__path__',[]):
   self.assertEqual(model_for_schema('phase1_multiscale_token_v1',**factories).kind,'multi')
   self.assertEqual(model_for_schema('phase1_iterative_timefreq_v1',**factories).kind,'iterative')
  with self.assertRaises(ValueError):model_for_schema('invalid',**factories)
 def test_actual_factories_create_registered_models(self):
  self.assertIs(type(model_for_schema('phase1_multiscale_token_v1')),MultiScaleToken)
  self.assertIs(type(model_for_schema('phase1_iterative_timefreq_v1')),IterativeTimeFreq)

if __name__=='__main__':unittest.main()
