import importlib.util
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).parent

def module(name):
    path = HERE / (name + '.py')
    assert path.exists(), 'Missing production boundary: ' + name
    spec = importlib.util.spec_from_file_location('sard_dura_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

class BoundaryTests(unittest.TestCase):
    def test_inference_output_becomes_owned_training_input(self):
        boundary = module('boundary')
        @torch.inference_mode()
        def engine(x):
            return (x * .75,)
        x = torch.randn(4, 2, 256, requires_grad=True)
        out = boundary.training_tensor(engine, x)
        self.assertFalse(torch.is_inference(out))
        self.assertFalse(out.requires_grad)
        linear = torch.nn.Linear(256, 6)
        linear(out).square().mean().backward()
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNone(x.grad)
        self.assertTrue(torch.equal(out, x.detach() * .75))

    def test_nonfinite_restoration_fails_instead_of_raw_fallback(self):
        boundary = module('boundary')
        @torch.inference_mode()
        def engine(x):
            return (x * float('nan'),)
        with self.assertRaises(FloatingPointError):
            boundary.training_tensor(engine, torch.ones(4, 2, 256))

    def test_source_only_roles(self):
        boundary = module('boundary')
        boundary.require_training_role('L_s')
        for role in ('U_s', 'V', 'target', 'query'):
            with self.assertRaises(ValueError):
                boundary.require_training_role(role)

    def test_fixed_full50_calendar_and_truth_closure(self):
        plan = module('plan')
        for arm in plan.ARMS:
            c = plan.profile(arm)
            self.assertEqual((c['epochs'], c['label_epochs'], c['pseudo_epochs']), (50, 13, 37))
            self.assertEqual(c['updates'], 11100)
            self.assertEqual(c['As_KL'], 0.)
            self.assertFalse(c['target_training'])
        with self.assertRaises(ValueError):
            plan.require_complete_predictions(['raw__clean'])
        plan.require_complete_predictions([m+'__'+v for m in ('raw', 'dura') for v in plan.VIEWS])

if __name__ == '__main__':
    unittest.main()
