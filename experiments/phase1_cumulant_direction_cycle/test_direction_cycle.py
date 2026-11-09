import unittest
from collections import Counter
import torch
from experiments.phase1_cumulant_direction_cycle.train import direction_offset, SCHEDULE, scaled_perturbation, EPSILON
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import perturbations

class DirectionCycleTest(unittest.TestCase):
    def test_each_record_covers_all_directions_at_fixed_budget(self):
        for record in (0,1,2,3,32767):
            directions=[direction_offset((epoch-1)%3*32768+record,epoch)%4
                        for group in SCHEDULE for epoch in group]
            counts=Counter(directions)
            self.assertEqual(len(directions),60)
            self.assertEqual(sorted(counts.values()),[14,14,16,16])

    def test_applied_source_direction_follows_epoch(self):
        torch.manual_seed(5);x=torch.randn(8,2,256)
        for epoch in (1,2,3,4,30):
            out=scaled_perturbation(x,direction_offset(0,epoch))
            directions=perturbations(x)
            expected=torch.stack([directions[(i+epoch-1)%4][i] for i in range(len(x))])*10
            torch.testing.assert_close(out,expected,rtol=0,atol=0)
        self.assertAlmostEqual(EPSILON,.1)

    def test_invalid_epoch_rejected(self):
        for value in (0,31,-1):
            with self.assertRaises(ValueError):direction_offset(0,value)

if __name__ == '__main__': unittest.main()
