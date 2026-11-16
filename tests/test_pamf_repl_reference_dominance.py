import unittest


class ReferenceDominanceTests(unittest.TestCase):
    def test_small_nonzero_reference_never_exceeds_relative_cap(self):
        import torch
        from experiments.pamf_repl_reference_dominance.projection import cap_extra_to_reference
        p = torch.nn.Parameter(torch.tensor([1e-6], dtype=torch.float32))
        ulp = float(torch.nextafter(p.detach(), torch.tensor([float('inf')]))-p.detach())
        before = (p.detach().clone(),)
        baseline = ((p.detach().double()-19*ulp).float(),)
        proposed = ((baseline[0].double()-19*ulp).float(),)
        details = {'cls_head': {'raw_increment_norm': 19*ulp, 'quantization_fallback': False,
                    'anchor_cosines_after': {}, 'constraint_dots_after': {},
                    'near_zero_component_anchors_omitted': ['L_As']}}
        values, facts = cap_extra_to_reference([('x.cls_head.weight', p)], before, baseline, proposed,
            (torch.tensor([1.]),), (torch.tensor([1.]),), (None,), details, .25)
        actual = float((baseline[0].double()-values[0].double()).norm())
        reference = float((before[0].double()-baseline[0].double()).norm())
        self.assertLessEqual(actual, .25*reference*(1+1e-6))
        self.assertTrue(facts['cls_head']['reference_rounding_fallback'])

    def test_real_adamw_step_respects_reference_bound_with_history(self):
        import torch
        from test_pamf_optimizer_reference import make, grad
        from experiments.cric_dura_comparison import f_optimizer
        from experiments.pamf_ssl_research.optimizer_step_guard import installed
        from experiments.pamf_repl_reference_dominance.objective import ReferenceDominanceGuard
        model = make()
        opt = f_optimizer.build(model, model.parameters(), 'CENTER_UNITS', lr=2e-4, weight_decay=.01)
        for index in range(2):
            for parameter, value in zip(model.parameters(), grad(model, index+2)):
                parameter.grad = value.clone()
            f_optimizer.clip_physical(opt, model.parameters(), 5.)
            opt.step()
        named = list(model.named_parameters())
        full = grad(model, 1.)
        guard = ReferenceDominanceGuard(None, .25)
        with installed():
            guard.capture(model, named, full, full, tuple(None for _ in named))
            for parameter, value in zip(model.parameters(), grad(model, -2.)):
                parameter.grad = value.clone()
            f_optimizer.clip_physical(opt, model.parameters(), 5.)
            opt.step()
            state = guard.state_dict()
            self.assertEqual(state['calls'], 1)
            self.assertGreater(state['reference_cosines_defined'], 0)
            for detail in guard.last_details.values():
                ratio = detail.get('reference_extra_ratio')
                cosine = detail.get('reference_dominance_cosine')
                if ratio is not None:
                    self.assertLessEqual(ratio, .25+1e-6)
                if cosine is not None:
                    self.assertGreaterEqual(cosine, (1-.25**2)**.5-1e-6)
            self.assertTrue(all(value['step'].item() == 3 for value in opt.state.values()))

    def test_matrix_is_fixed_and_restores_prior_scope(self):
        from experiments.pamf_repl_reference_dominance import plan
        from experiments.pamf_repl_step_guard import plan as original
        from experiments.pamf_repl_component_guard import plan as component
        old = original.RUN, original.RECIPES, component.apply_recipe
        with plan.scoped():
            rows = original.candidate_rows()
            self.assertEqual(len(rows), 8)
            self.assertEqual({(v['weight'],v['reference_cap']) for v in plan.RECIPES.values()},
                             {(w,c) for w in (3.,6.) for c in (.25,.5)})
            base = dict(hard_pl={'weight':6.}, hard_pl_gradient_guard={'weight':6.})
            for row in rows:
                value = component.apply_recipe(base,row['recipe'])
                self.assertEqual(value['hard_pl']['weight'],plan.RECIPES[row['recipe']]['weight'])
                with self.assertRaises(FileNotFoundError):
                    original.configuration({},row,'/missing','commit')
        self.assertEqual(original.RUN, old[0])
        self.assertIs(original.RECIPES, old[1])
        self.assertIs(component.apply_recipe, old[2])

    def test_extra_is_bounded_and_preserves_supervised_anchors(self):
        import torch
        from experiments.pamf_repl_reference_dominance.projection import cap_extra_to_reference
        p = torch.nn.Parameter(torch.tensor([1., 1.]))
        named = [('x.cls_head.weight', p)]
        before = (torch.tensor([1., 1.]),)
        baseline = (torch.tensor([.9, 1.]),)
        proposed = (torch.tensor([.7, .8]),)
        full = (torch.tensor([1., 0.]),)
        ce = (torch.tensor([0., 1.]),)
        details = {'cls_head': {'raw_increment_norm': float(torch.tensor([.2,.2]).norm()),
                    'quantization_fallback': False, 'anchor_cosines_after': {},
                    'constraint_dots_after': {}, 'near_zero_component_anchors_omitted': ['L_As']}}
        values, out = cap_extra_to_reference(named, before, baseline, proposed, full, ce, (None,), details, .25)
        r, u = before[0]-baseline[0], baseline[0]-values[0]
        self.assertLessEqual(float(u.norm()), .25*float(r.norm())+1e-7)
        self.assertGreaterEqual(float(torch.dot(full[0],u)), -1e-7)
        self.assertGreaterEqual(float(torch.dot(ce[0],u)), -1e-7)
        self.assertGreaterEqual(float(torch.cosine_similarity(r[None],(r+u)[None])), (1-.25**2)**.5-1e-6)
        self.assertTrue(out['cls_head']['reference_dominance_checked'])

    def test_zero_reference_falls_back_to_labeled_step(self):
        import torch
        from experiments.pamf_repl_reference_dominance.projection import cap_extra_to_reference
        p = torch.nn.Parameter(torch.tensor([1.]))
        details = {'cls_head': {'raw_increment_norm': .1, 'quantization_fallback': False,
                    'anchor_cosines_after': {}, 'constraint_dots_after': {},
                    'near_zero_component_anchors_omitted': ['L_As']}}
        out, facts = cap_extra_to_reference([('x.cls_head.weight', p)], (p.detach().clone(),),
            (p.detach().clone(),), (torch.tensor([.9]),), (torch.tensor([1.]),),
            (torch.tensor([1.]),), (None,), details, .25)
        torch.testing.assert_close(out[0], p.detach())
        self.assertTrue(facts['cls_head']['reference_zero_fallback'])
        self.assertIsNone(facts['cls_head']['reference_dominance_cosine'])


if __name__ == '__main__':
    unittest.main()
