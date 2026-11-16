import unittest


class RepairedReferenceTests(unittest.TestCase):
    def test_real_adamw_step_preserves_reference_cap_and_anchors(self):
        from test_pamf_optimizer_reference import make,grad
        from experiments.cric_dura_comparison import f_optimizer
        from experiments.pamf_ssl_research.optimizer_step_guard import installed
        from experiments.pamf_repl_reference_repair.objective import RepairedReferenceGuard
        model=make();opt=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=.01)
        for p,g in zip(model.parameters(),grad(model,2.)):p.grad=g.clone()
        f_optimizer.clip_physical(opt,model.parameters(),5.);opt.step()
        named=list(model.named_parameters());full=grad(model,1.);guard=RepairedReferenceGuard(None,.25)
        with installed():
            guard.capture(model,named,full,full,tuple(None for _ in named))
            for p,g in zip(model.parameters(),grad(model,-2.)):p.grad=g.clone()
            f_optimizer.clip_physical(opt,model.parameters(),5.);opt.step()
            state=guard.state_dict();self.assertEqual(state['calls'],1)
            for d in guard.last_details.values():
                if d.get('reference_extra_ratio') is not None:
                    self.assertLessEqual(d['reference_extra_ratio'],.25*(1+1e-6))
                for value in d['anchor_cosines_after'].values():
                    self.assertTrue(value is None or value >= -1.000001e-6)

    def test_matrix_uses_matched_source_and_fixed_cap(self):
        from experiments.pamf_repl_reference_repair import plan
        from experiments.pamf_repl_step_guard import plan as original
        old=original.RUN,original.RECIPES
        with plan.scoped():
            rows=original.candidate_rows();self.assertEqual(len(rows),4)
            self.assertEqual({(v['weight'],v['reference_cap']) for v in plan.RECIPES.values()},
                             {(3.,.25),(6.,.25)})
            for row in rows:
                with self.assertRaises(FileNotFoundError):original.configuration({},row,'/missing','commit')
        self.assertEqual(original.RUN,old[0]);self.assertIs(original.RECIPES,old[1])


if __name__=='__main__':unittest.main()
