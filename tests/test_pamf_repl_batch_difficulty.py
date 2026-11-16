import unittest
from experiments.pamf_repl_dual_ce.analyze_batch_difficulty import quintiles


class BatchDifficultyTests(unittest.TestCase):
    def test_epoch_rank_counts_defined_and_undefined_separately(self):
        rows=[dict(CE=float(i),L_norm=1.,full=None if i==0 else (-.2 if i>=177 else .1),
                   ce=-.3 if i>=177 else .2) for i in range(222)]
        bins=quintiles(rows)
        self.assertEqual(sum(b['steps'] for b in bins),222)
        self.assertEqual(bins[0]['full_undefined'],1)
        self.assertEqual(bins[0]['full_conflict'],0)
        self.assertEqual(bins[-1]['full_conflict'],45)
        self.assertEqual(bins[-1]['ce_conflict'],45)
        self.assertEqual(bins[-1]['full_defined'],45)
        with self.assertRaisesRegex(ValueError,'complete'):quintiles(rows[:-1])


if __name__=='__main__':unittest.main()
