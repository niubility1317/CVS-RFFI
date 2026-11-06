import unittest

import torch

from experiments.phase1_dura_orthogonal.train_source import action_response_loss


class WidelyLinearAction:
    def __init__(self, main, image):
        self.main, self.image = main, image

    def __call__(self, x):
        return self.main*x + self.image*x.conj()


class SimulatorActionTrainingTest(unittest.TestCase):
    def test_matches_response_and_backpropagates_without_reference_gradient(self):
        probe = torch.complex(torch.ones(2, 256), torch.ones(2, 256)*.5)
        main = torch.tensor(1., requires_grad=True)
        image = torch.tensor(0., requires_grad=True)
        teacher_main = torch.tensor(1., requires_grad=True)
        reference = WidelyLinearAction(teacher_main, torch.tensor(0.))
        same = action_response_loss(WidelyLinearAction(main, image), reference, probe)
        torch.testing.assert_close(same, torch.tensor(0.))
        mismatched = action_response_loss(WidelyLinearAction(main*1.2, image+.1), reference, probe)
        self.assertGreater(float(mismatched.detach()), 0.)
        mismatched.backward()
        self.assertTrue(torch.isfinite(main.grad))
        self.assertTrue(torch.isfinite(image.grad))
        self.assertIsNone(teacher_main.grad)


if __name__ == '__main__':
    unittest.main()
