import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('confirm_plan',Path(__file__).with_name('plan.py'))
plan=importlib.util.module_from_spec(s);s.loader.exec_module(plan)

class PlanTests(unittest.TestCase):
 def test_exact_two_new_seeds(self):
  self.assertEqual(plan.SEEDS,(2026092701,2026092702,2026092703))
  self.assertEqual(plan.NEW_SEEDS,(2026092702,2026092703))
  for seed in plan.NEW_SEEDS:
   p=plan.profile(seed)
   self.assertEqual((p['model_seed'],p['epochs'],p['label_epochs'],p['pseudo_epochs'],p['updates']),(seed,20,0,20,4440))
   self.assertEqual((p['epoch_offset'],p['step_offset'],p['cumulative_epochs'],p['cumulative_updates']),(30,6660,50,11100))
 def test_reused_or_unregistered_training_rejected(self):
  for seed in [2026092701,2026092704,None,392005]:
   with self.assertRaises(ValueError):plan.profile(seed)
 def test_mutated_seed_budget_or_truth_rejected(self):
  for seed in plan.NEW_SEEDS:
   for key,value in [('model_seed',2026092701),('epochs',21),('As_KL',1.),('target_training',True)]:
    p=plan.profile(seed);p[key]=value
    with self.assertRaises(ValueError):plan.validate(p)

if __name__=='__main__':unittest.main()
