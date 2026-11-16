import unittest


class CrossLossDiagnosticTests(unittest.TestCase):
    def test_supervised_ce_kl_and_residual_sum_to_original(self):
        import torch
        from experiments.pamf_repl_attribution_factorial.source_cross_loss_diagnostic import labeled_components
        x = torch.tensor(2., requires_grad=True)
        pairs = [(3*x, x, 2*x), (5*x, 2*x, 3*x), (7*x, 7*x, 9*x), (11*x, 3*x, 8*x)]
        total = .5*pairs[0][0]+.25*pairs[1][0]+.25*pairs[2][0]+.25*pairs[3][0]+4*x
        parts = labeled_components(pairs, total)
        self.assertEqual(set(parts), {'L_all_CE', 'L_pair_KL', 'L_residual'})
        torch.testing.assert_close(parts['L_all_CE'], .5*pairs[0][1]+.25*pairs[1][1]+.25*pairs[2][1]+.25*pairs[3][1])
        torch.testing.assert_close(parts['L_pair_KL'], .5*pairs[0][2]+.25*pairs[1][2]+.25*pairs[3][2])
        torch.testing.assert_close(sum(parts.values()), total)
        torch.testing.assert_close(torch.autograd.grad(sum(parts.values()), x, retain_graph=True)[0],
                                   torch.autograd.grad(total, x)[0])


if __name__ == '__main__':
    unittest.main()
