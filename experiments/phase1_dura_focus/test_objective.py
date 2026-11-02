import unittest
import torch

from experiments.phase1_dura_focus.objective import difficult_weights, group_weighted_mean


class IdentityReader:
    def features(self, waveform):
        return waveform

    def classify_features(self, features):
        return features


class DifficultSourceTest(unittest.TestCase):
    def test_only_high_confidence_disagreements_receive_extra_weight(self):
        clean = torch.tensor([[12., 0, 0, 0, 0, 0],
                              [0., 0, 0, 0, 0, 0],
                              [0., 12, 0, 0, 0, 0]])
        anchor = torch.tensor([[0., 12, 0, 0, 0, 0],
                               [0., 12, 0, 0, 0, 0],
                               [0., 12, 0, 0, 0, 0]], requires_grad=True)
        weights = difficult_weights(IdentityReader(), anchor, clean)
        torch.testing.assert_close(weights, torch.tensor([8., 1., 1.]))
        self.assertFalse(weights.requires_grad)

    def test_weights_do_not_depend_on_class_number(self):
        clean = torch.tensor([[12., 0, 0, 0, 0, 0], [0., 12, 0, 0, 0, 0]])
        anchor = torch.tensor([[0., 12, 0, 0, 0, 0], [0., 12, 0, 0, 0, 0]])
        permutation = torch.tensor([5, 2, 0, 1, 4, 3])
        a = difficult_weights(IdentityReader(), anchor, clean)
        b = difficult_weights(IdentityReader(), anchor[:, permutation], clean[:, permutation])
        torch.testing.assert_close(a, b)

    def test_weighted_means_balance_receivers(self):
        per = torch.tensor([1., 2., 5.], requires_grad=True)
        weights = torch.tensor([1., 8., 1.])
        groups = torch.tensor([4, 4, 8])
        value = group_weighted_mean(per, weights, groups)
        self.assertAlmostEqual(float(value), (17 / 9 + 5) / 2, places=6)
        value.backward()
        self.assertTrue(torch.isfinite(per.grad).all())


if __name__ == '__main__': unittest.main()
