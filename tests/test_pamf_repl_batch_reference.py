import unittest


class BatchReferenceTests(unittest.TestCase):
    def test_within_cross_and_same_objective_references_are_distinct(self):
        import torch
        from experiments.pamf_repl_attribution_factorial.source_batch_reference import summarize
        named = [('x.cls_head.weight', torch.nn.Parameter(torch.zeros(2)))]
        records = [dict(setting='ten_class', offset=0, named=named,
                        L=(torch.tensor([1., 0.]),), U=(torch.tensor([0., 1.]),)),
                   dict(setting='ten_class', offset=1, named=named,
                        L=(torch.tensor([-1., 0.]),), U=(torch.tensor([0., -1.]),))]
        values = summarize(records)['ten_class']['cls_head']
        self.assertEqual([r['cosine'] for r in values['L_vs_L']], [-1.])
        self.assertEqual([r['cosine'] for r in values['U_vs_U']], [-1.])
        self.assertEqual(len(values['L_vs_U_same_batch']), 2)
        self.assertEqual(len(values['L_vs_U_cross_batch']), 2)
        self.assertTrue(all(r['cosine'] == 0. for r in values['L_vs_U_same_batch']))


if __name__ == '__main__':
    unittest.main()
