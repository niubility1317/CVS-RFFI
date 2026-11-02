import unittest
import torch
from experiments.phase1_spectral_margin.test_objective import Reader
from experiments.phase1_spectral_crossview.objective import source_loss
from experiments.phase1_dynamic_volterra.train_source import training_indices


class CrossviewObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(717); torch.set_num_threads(2)
        self.reader = Reader().eval().requires_grad_(False)
        self.single = torch.randn(2, 2, 256)
        self.clean = self.single.repeat(3, 1, 1)
        self.features = self.reader.features(self.clean).detach()
        self.groups = torch.tensor([4, 8]).repeat(3)

    def test_identical_views_have_zero_consistency_error(self):
        loss, terms = source_loss(self.reader, self.clean, self.clean, self.features, self.groups)
        self.assertLess(abs(float(loss)), 1e-6)
        self.assertLess(float(terms['view_distribution'].abs().max()), 1e-6)
        self.assertLess(float(terms['view_feature'].abs().max()), 1e-6)

    def test_different_views_backpropagate_without_reader_updates(self):
        out = self.clean.clone(); out[2:4] *= -1; out.requires_grad_(True)
        loss, terms = source_loss(self.reader, out, self.clean, self.features, self.groups)
        self.assertGreater(float(terms['view_distribution'].detach().sum()), 0.)
        loss.backward(); self.assertTrue(torch.isfinite(out.grad).all())
        self.assertGreater(float(out.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.reader.parameters()))

    def test_mismatched_pairing_is_rejected(self):
        bad = self.clean.clone(); bad[2] = bad[3]
        with self.assertRaisesRegex(ValueError, 'same ordered source records'):
            source_loss(self.reader, bad, bad, self.features, self.groups)

    def test_batch_indices_preserve_record_pairing(self):
        selected = torch.tensor([3, 1])
        self.assertTrue(torch.equal(training_indices(selected, 10, True), torch.tensor([3, 1, 13, 11, 23, 21])))
        self.assertTrue(torch.equal(training_indices(selected, 10, False), selected))


if __name__ == '__main__': unittest.main()
