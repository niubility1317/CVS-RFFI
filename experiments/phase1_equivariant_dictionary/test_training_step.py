import unittest

import torch
from torch.nn import functional as F

from experiments.phase1_equivariant_dictionary.model import EquivariantPatchDictionary
from experiments.phase1_equivariant_dictionary.objective import source_loss
from experiments.phase1_equivariant_dictionary.test_objective import Reader


class TrainingStepTest(unittest.TestCase):
    def test_gate_then_dictionary_learn_from_frozen_reader_objective(self):
        torch.manual_seed(2031)
        model = EquivariantPatchDictionary(atoms=8, width=12)
        reader = Reader().eval().requires_grad_(False)
        received, initial, clean = (torch.randn(4, 2, 256) for _ in range(3))
        logvar = torch.zeros(4)
        features = reader.features(clean)
        soft = F.softmax(reader.classify_features(features)/2, dim=1)
        labels = torch.tensor([1, -1, 2, -1])
        groups = torch.tensor([4, 4, 6, 6])
        before = model.atom_real.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        gradients = []
        for _ in range(2):
            output = model(received, initial, logvar)
            loss, _ = source_loss(reader, output, clean, features, soft, labels, groups)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients.append(float(model.atom_real.grad.abs().sum()))
            optimizer.step()
        self.assertEqual(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)
        self.assertFalse(torch.equal(model.atom_real, before))
        self.assertTrue(all(p.grad is None for p in reader.parameters()))


if __name__ == '__main__':
    unittest.main()
