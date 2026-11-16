import importlib.util
import unittest
from unittest.mock import patch


class DualCEProtectionTests(unittest.TestCase):
    def test_fixed_matrix_scope_and_target_rejection(self):
        from experiments.pamf_repl_dual_ce import plan
        from experiments.pamf_repl_step_guard import plan as original
        from experiments.pamf_repl_component_guard import plan as component
        before = original.RUN, original.RECIPES, component.apply_recipe
        configuration = dict(hard_pl={'weight': 6.0},
                             hard_pl_gradient_guard={'weight': 6.0, 'U_to_L_norm_cap': 1.0})
        with plan.scoped():
            rows = original.candidate_rows()
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['setting'] for r in rows}, {'ten_class', 'source_rx_6'})
            self.assertEqual({r['recipe'] for r in rows}, {'dual_ce6', 'dual_ce3'})
            for row in rows:
                recipe = component.apply_recipe(configuration, row['recipe'])
                self.assertEqual(recipe['hard_pl']['weight'], plan.RECIPES[row['recipe']]['weight'])
                self.assertEqual(recipe['hard_pl_gradient_guard']['U_to_L_norm_cap'], 1.0)
                cfg = dict(run_id=plan.RUN, row_id=row['row_id'], tuning_row=row,
                    tuning_recipe=plan.RECIPES[row['recipe']],
                    output_root='/project/runs/'+plan.RUN+'/F/'+row['row_id']+'/source',
                    model_seed=row['model_seed'], protocol_id=row['protocol_id'],
                    algorithm_id='pamf_repl_adamw_full_L_guard_v1')
                self.assertEqual(original.validate(cfg), row)
                with self.assertRaises(ValueError):
                    original.validate(dict(cfg, target_truth='/forbidden'))
        self.assertEqual(original.RUN, before[0])
        self.assertIs(original.RECIPES, before[1])
        self.assertIs(component.apply_recipe, before[2])

    def test_actual_extra_update_satisfies_full_and_CE_anchors(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.pamf_repl_dual_ce'))
        import torch
        from experiments.pamf_ssl_research.parameter_step_projection import project_parameters
        parameter = torch.nn.Parameter(torch.zeros(2))
        named = [('x.cls_head.weight', parameter)]
        baseline = (torch.tensor([0., 0.]),)
        joint = (torch.tensor([-1., 1.]),)
        full = (torch.tensor([1., 0.]),)
        ce = (torch.tensor([0., 1.]),)
        projected, details = project_parameters(named, baseline, joint, full, ce, (None,))
        delta = baseline[0]-projected[0]
        self.assertGreaterEqual(float(torch.dot(full[0], delta)), -1e-6)
        self.assertGreaterEqual(float(torch.dot(ce[0], delta)), -1e-6)
        self.assertTrue(details['cls_head']['actual_parameter_check'])
        self.assertTrue(details['cls_head']['norm_nonexpansion'])

    def test_guide_captures_real_CE_derivative_without_gradient_writes(self):
        if importlib.util.find_spec('experiments.pamf_repl_dual_ce') is None:
            self.skipTest('Dual-anchor objective not implemented yet')
        import torch
        from experiments.pamf_repl_dual_ce.objective import DualCEObjective
        from experiments.pamf_repl_step_guard.objective import StepProtectedREPL
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.tensor([1., 2.]))
        class Guard:
            def capture(self, model, named, full, clean, physical):
                self.anchors = (full, clean, physical)
        student = Student()
        student._diagnostic_identity_loss = student.x.square().sum()
        guard = Guard()
        objective = DualCEObjective.__new__(DualCEObjective)
        objective.parameter_guard = guard
        objective.ce_derivative_calls = 0
        def native_guide(self, labeled, model):
            self.parameter_guard.capture(model, list(model.named_parameters()),
                                         (torch.tensor([1., 0.]),), (None,), (None,))
            return labeled
        with patch.object(StepProtectedREPL, 'guide', new=native_guide):
            self.assertEqual(objective.guide(3., student), 3.)
        torch.testing.assert_close(guard.anchors[1][0], torch.tensor([2., 4.]))
        self.assertIsNone(guard.anchors[2][0])
        self.assertIsNone(student.x.grad)
        self.assertEqual(objective.ce_derivative_calls, 1)
        self.assertNotIn('capture', guard.__dict__)


if __name__ == '__main__':
    unittest.main()
