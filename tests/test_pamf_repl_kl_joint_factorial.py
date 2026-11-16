import unittest


class KLJointFactorialTests(unittest.TestCase):
    def test_full_matrix_and_configuration_scope(self):
        from experiments.pamf_repl_kl_joint_factorial import plan
        from experiments.pamf_repl_component_guard import plan as original
        old = original.RUN, original.RECIPES, original.configuration
        with plan.scoped():
            rows = original.candidate_rows()
            self.assertEqual(len(rows), 16)
            self.assertEqual({(v['weight'],v['joint_weight'],v['kl_factor']) for v in plan.RECIPES.values()},
                {(w,j,k) for w in (3.,6.) for j in (0.,.05) for k in (0.,1.)})
            config = dict(hard_pl={'weight':6.}, hard_pl_gradient_guard={'weight':6.},
                          hard_pl_joint_contrast={'weight':.05})
            for row in rows:
                result = original.apply_recipe(config,row['recipe'])
                recipe = plan.RECIPES[row['recipe']]
                self.assertEqual(result['hard_pl']['weight'], recipe['weight'])
                self.assertEqual(result['hard_pl_joint_contrast']['weight'],recipe['joint_weight'])
                with self.assertRaises(FileNotFoundError):
                    original.configuration({},row,'/missing','commit')
        self.assertEqual(original.RUN, old[0])
        self.assertIs(original.RECIPES, old[1])
        self.assertIs(original.configuration, old[2])

    def test_kl_factor_changes_kl_only(self):
        import torch
        from experiments.pamf_ssl_research.hard_pl_objective import masked_pair
        from experiments.pamf_repl_kl_joint_factorial.objective import factor_masked_pair
        a=torch.tensor([[2.,0.],[0.,2.]],requires_grad=True)
        b=torch.tensor([[1.,1.],[1.,1.]],requires_grad=True)
        y=torch.tensor([0,1]);accepted=torch.tensor([True,True])
        zero = factor_masked_pair(masked_pair,0.)(a,b,y,accepted,1.)
        one = factor_masked_pair(masked_pair,1.)(a,b,y,accepted,1.)
        torch.testing.assert_close(zero[1],one[1])
        torch.testing.assert_close(zero[0],zero[1])
        torch.testing.assert_close(one[0],one[1]+one[2])
        self.assertGreater(float(one[0]),float(zero[0]))


if __name__ == '__main__':
    unittest.main()
