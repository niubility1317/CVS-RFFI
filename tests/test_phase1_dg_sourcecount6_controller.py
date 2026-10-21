import unittest
from experiments.phase1_dg_sourcecount6.rebalance import may_launch,worker_identity,active_rows


class ResourcePolicyTests(unittest.TestCase):
    def test_five_source_exclusion_preserves_every_other_recipe(self):
        from experiments.phase1_dg_sourcecount6.plan import make_plan
        original=make_plan()['rows']
        selected=active_rows(original,[5])
        self.assertEqual(len(selected),360)
        self.assertEqual(set(r['source_count'] for r in selected),{1,2,3,4,6})
        self.assertTrue(all(not r['reuse_parent_row_id'] for r in selected))
        self.assertEqual(selected,[r for r in original if r['source_count']!=5])
        self.assertEqual(active_rows(original,[]),original)
        with self.assertRaises(ValueError):active_rows(original,[6])

    def test_only_two_newly_authorized_shared_cards(self):
        for gpu in range(8):
            self.assertTrue(may_launch(gpu,0,22000))
            self.assertEqual(may_launch(gpu,1,22000),gpu in (2,3))
            self.assertFalse(may_launch(gpu,2,22000))
            self.assertFalse(may_launch(gpu,1,11000))

    def test_takeover_discovers_only_exact_owned_workers(self):
        rt=dict(python='/env/python',release_root='/release/original',run_root='/runs/this')
        args=['/env/python','-u','-m','experiments.phase1_dg_sourcecount6.driver','worker','--config','/runs/this/row.json']
        self.assertEqual(worker_identity(args,'/release/original',rt),'row')
        self.assertIsNone(worker_identity(args,'/release/other',rt))
        self.assertIsNone(worker_identity(args[:-1]+['/runs/other/row.json'],'/release/original',rt))
        wrong=list(args);wrong[4]='dispatch'
        self.assertIsNone(worker_identity(wrong,'/release/original',rt))


if __name__=='__main__':unittest.main()
