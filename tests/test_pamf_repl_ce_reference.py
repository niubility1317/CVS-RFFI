import unittest
from unittest.mock import patch


class CEReferenceTests(unittest.TestCase):
    def test_matrix_and_run_scope_match_full_reference_factorial(self):
        from experiments.pamf_repl_ce_reference import plan
        from experiments.pamf_repl_step_guard import plan as original
        from experiments.pamf_repl_component_guard import plan as component
        before=original.RUN,original.RECIPES,component.apply_recipe
        with plan.scoped():
            rows=original.candidate_rows()
            self.assertEqual(len(rows),8)
            self.assertEqual({(v['weight'],v['reference_cap']) for v in plan.RECIPES.values()},
                {(w,c) for w in (3.,6.) for c in (.25,.5)})
            for row in rows:
                with self.assertRaises(FileNotFoundError):
                    original.configuration({},row,'/missing','commit')
        self.assertEqual(original.RUN,before[0])
        self.assertIs(original.RECIPES,before[1])
        self.assertIs(component.apply_recipe,before[2])

    def test_reference_uses_ce_gradient_not_full_supervised_gradient(self):
        import torch
        from experiments.pamf_repl_ce_reference.objective import CEReferenceGuard
        from experiments.pamf_repl_reference_dominance.objective import ReferenceDominanceGuard
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        named=[('x.cls_head.weight',parameter)]
        full=(torch.tensor([2.]),);ce=(torch.tensor([3.]),);empty=(None,)
        guard=CEReferenceGuard(None,.25)
        guard.pending=(object(),named,(full,ce,empty))
        with patch.object(ReferenceDominanceGuard,'_reference_step',return_value=('parameters',{})) as method:
            result=guard._reference_step(object(),named,object(),full,clip_impl=object())
        self.assertIs(method.call_args.args[3],ce)
        self.assertEqual(result[1]['reference_loss'],'ordinary_labeled_CE')

    def test_ce_reference_requires_a_connected_ce_gradient(self):
        import torch
        from experiments.pamf_repl_ce_reference.objective import CEReferenceGuard
        p=torch.nn.Parameter(torch.tensor([1.]));named=[('x.cls_head.weight',p)];empty=(None,)
        guard=CEReferenceGuard(None,.25);guard.pending=(object(),named,(empty,empty,empty))
        with self.assertRaises(ValueError):
            guard._reference_step(object(),named,object(),empty,clip_impl=object())


if __name__=='__main__':
    unittest.main()
