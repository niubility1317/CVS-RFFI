import unittest
from types import SimpleNamespace
from unittest.mock import patch
import experiments
import torch
from torch import nn
from experiments.phase1_projector_objective_fullsource.replay import model_for_schema,verify_projector_reader

class FactoryTest(unittest.TestCase):
 def test_registered_factories_survive_path_reset(self):
  factories=dict(first_factory=lambda:SimpleNamespace(kind='multi'),second_factory=lambda:SimpleNamespace(kind='projector'))
  with patch.object(experiments,'__path__',[]):
   self.assertEqual(model_for_schema('phase1_frozen_feature_projector_v1',**factories).kind,'multi')
   self.assertEqual(model_for_schema('phase1_projector_aligned_v1',**factories).kind,'projector')
  with self.assertRaises(ValueError):model_for_schema('invalid',**factories)
 def test_rejects_changed_or_missing_frozen_reader(self):
  class Model(nn.Module):
   def __init__(self):super().__init__();self.reader=nn.Linear(3,2);self.trainable=nn.Linear(2,1)
  model=Model();state={k:v.clone() for k,v in model.state_dict().items()}
  verify_projector_reader(model,state)
  changed={k:v.clone() for k,v in state.items()};changed['reader.weight']+=1
  with self.assertRaises(ValueError):verify_projector_reader(model,changed)
  del state['reader.bias']
  with self.assertRaises(ValueError):verify_projector_reader(model,state)

if __name__=='__main__':unittest.main()
