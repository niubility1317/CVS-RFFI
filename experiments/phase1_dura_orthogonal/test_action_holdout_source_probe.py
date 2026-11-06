import unittest

import torch

from experiments.phase1_dura_orthogonal.action_holdout_source_probe import per_record_probe_error
from experiments.phase1_dura_orthogonal.train_source import action_response_loss


class Action:
    def __init__(self, main, image):
        self.main, self.image = main, image

    def __call__(self, x):
        return self.main*x+self.image*x.conj()


class ActionHoldoutTest(unittest.TestCase):
    def test_per_record_metric_matches_training_mean(self):
        probe = torch.complex(torch.randn(3, 256), torch.randn(3, 256))
        estimated = Action(torch.tensor(1.2), torch.tensor(.1))
        reference = Action(torch.tensor(1.), torch.tensor(0.))
        actual = per_record_probe_error(estimated, reference, probe)
        self.assertEqual(tuple(actual.shape), (3,))
        torch.testing.assert_close(actual.mean(), action_response_loss(estimated, reference, probe))


if __name__ == '__main__':
    unittest.main()
