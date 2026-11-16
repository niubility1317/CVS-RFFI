import unittest


class AttributionFactorialTests(unittest.TestCase):
    def test_fixed_2_by_2_by_2_matrix_and_scope(self):
        from experiments.pamf_repl_attribution_factorial import plan
        from experiments.pamf_repl_component_guard import plan as original
        before = original.RUN, original.RECIPES, original.configuration
        with plan.scoped():
            rows = original.candidate_rows()
            self.assertEqual(len(rows), 16)
            self.assertEqual({r['setting'] for r in rows}, {'ten_class', 'source_rx_6'})
            combinations = {(v['weight'], v['shallow_factor'], v['extrap_coefficient'])
                            for v in plan.RECIPES.values()}
            self.assertEqual(combinations, {(w, s, e) for w in (3., 6.)
                                           for s in (0., 1.) for e in (0., .25)})
            base = dict(hard_pl={'weight': 6.}, hard_pl_gradient_guard={'weight': 6.},
                        hard_pl_receiver_extrapolation={})
            for recipe, spec in plan.RECIPES.items():
                out = original.apply_recipe(base, recipe)
                self.assertEqual(out['hard_pl']['weight'], spec['weight'])
                self.assertEqual(out['hard_pl_gradient_guard']['weight'], spec['weight'])
                self.assertEqual(out['hard_pl_receiver_extrapolation']['extrapolated_CE_coefficient'], spec['extrap_coefficient'])
                self.assertEqual(out['hard_pl_receiver_extrapolation']['ordinary_CE_coefficient'], .5-spec['extrap_coefficient'])
            row = rows[0]
            # This must reach the source-file check using the current run ID,
            # rather than the reused function's definition-time default run.
            with self.assertRaises(FileNotFoundError):
                original.configuration({}, row, '/missing', 'commit')
            config = dict(run_id=plan.RUN, row_id=row['row_id'], tuning_row=row,
                          tuning_recipe=plan.RECIPES[row['recipe']],
                          output_root='/project/runs/'+plan.RUN+'/F/'+row['row_id']+'/source',
                          model_seed=row['model_seed'], protocol_id=row['protocol_id'])
            self.assertEqual(original.validate(config), row)
            with self.assertRaises(ValueError):
                original.validate(dict(config, target_truth='/forbidden'))
        self.assertEqual(original.RUN, before[0])
        self.assertIs(original.RECIPES, before[1])
        self.assertIs(original.configuration, before[2])

    def test_observation_and_mechanism_objective_are_composed(self):
        from experiments.pamf_repl_attribution_factorial.objective import AttributionREPL
        from experiments.pamf_repl_weight.objective import ObservedREPL
        from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationMixin
        self.assertTrue(issubclass(AttributionREPL, ObservedREPL))
        self.assertTrue(issubclass(AttributionREPL, PseudoObservationMixin))


if __name__ == '__main__':
    unittest.main()
