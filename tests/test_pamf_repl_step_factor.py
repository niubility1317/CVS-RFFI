"""Check immutable factor identities and temporary reuse of the existing runner."""
import unittest
import importlib.util


class StepFactorPlanTests(unittest.TestCase):
    def test_matrix_and_recipe_scope_do_not_change_original_method(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.pamf_repl_step_factor'))
        from experiments.pamf_repl_step_factor import plan
        from experiments.pamf_repl_step_guard import plan as original
        from experiments.pamf_repl_component_guard import plan as component

        run, recipes, apply = original.RUN, original.RECIPES, component.apply_recipe
        configuration = dict(hard_pl={'weight': 6.0},
                             hard_pl_gradient_guard={'weight': 6.0, 'U_to_L_norm_cap': 1.0})
        with plan.scoped():
            rows = original.candidate_rows()
            self.assertEqual(len(rows), 4)
            self.assertEqual({row['setting'] for row in rows}, {'ten_class', 'source_rx_6'})
            self.assertEqual({row['recipe'] for row in rows}, {'step_weight3', 'step_guard05'})
            weight = component.apply_recipe(configuration, 'step_weight3')
            cap = component.apply_recipe(configuration, 'step_guard05')
            self.assertEqual(weight['hard_pl']['weight'], 3.0)
            self.assertEqual(weight['hard_pl_gradient_guard'], {'weight': 3.0, 'U_to_L_norm_cap': 1.0})
            self.assertEqual(cap['hard_pl']['weight'], 6.0)
            self.assertEqual(cap['hard_pl_gradient_guard'], {'weight': 6.0, 'U_to_L_norm_cap': 0.5})
            self.assertEqual(configuration['hard_pl']['weight'], 6.0)
            self.assertEqual(configuration['hard_pl_gradient_guard']['U_to_L_norm_cap'], 1.0)
        self.assertEqual(original.RUN, run)
        self.assertIs(original.RECIPES, recipes)
        self.assertIs(component.apply_recipe, apply)

    def test_recipe_scope_restores_original_on_error(self):
        if importlib.util.find_spec('experiments.pamf_repl_step_factor') is None:
            self.skipTest('Matrix module not implemented yet')
        from experiments.pamf_repl_step_factor import plan
        from experiments.pamf_repl_step_guard import plan as original
        from experiments.pamf_repl_component_guard import plan as component

        run, recipes, apply = original.RUN, original.RECIPES, component.apply_recipe
        with self.assertRaisesRegex(RuntimeError, 'scope exit'):
            with plan.scoped():
                raise RuntimeError('scope exit')
        self.assertEqual(original.RUN, run)
        self.assertIs(original.RECIPES, recipes)
        self.assertIs(component.apply_recipe, apply)

    def test_registered_rows_pass_identity_gate_and_unknown_rows_fail(self):
        from experiments.pamf_repl_step_factor import plan
        from experiments.pamf_repl_step_guard import plan as original

        with plan.scoped():
            for row in original.candidate_rows():
                with self.assertRaises(FileNotFoundError):
                    original.configuration({}, row, '/missing', 'commit')
                bad = dict(row, recipe='unregistered')
                with self.assertRaises(ValueError):
                    original.configuration({}, bad, '/missing', 'commit')

    def test_training_rejects_target_access_and_changed_recipe(self):
        from experiments.pamf_repl_step_factor import plan
        from experiments.pamf_repl_step_guard import plan as original

        with plan.scoped():
            row = original.candidate_rows()[0]
            config = dict(run_id=plan.RUN, row_id=row['row_id'], tuning_row=row,
                          tuning_recipe=plan.RECIPES[row['recipe']],
                          output_root='/project/runs/'+plan.RUN+'/F/'+row['row_id']+'/source',
                          model_seed=row['model_seed'], protocol_id=row['protocol_id'],
                          algorithm_id='pamf_repl_adamw_full_L_guard_v1')
            self.assertEqual(original.validate(config), row)
            with self.assertRaises(ValueError):
                original.validate(dict(config, target_truth='/forbidden'))
            with self.assertRaises(ValueError):
                original.validate(dict(config, tuning_recipe={'weight': 6.0}))


if __name__ == '__main__':
    unittest.main()
