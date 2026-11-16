import unittest


class SourceComponentDiagnosticTests(unittest.TestCase):
    def test_closure_and_loss_values_are_saved_per_source_batch(self):
        from experiments.pamf_repl_attribution_factorial.source_component_diagnostic import attach_facts
        rows = [{}, {}]
        values = [{'component_gradient_closure': {'max_absolute_error': 0.}, 'component_losses': {'pseudo_CE': 1.}},
                  {'component_gradient_closure': {'max_absolute_error': 1e-8}, 'component_losses': {'pseudo_CE': 2.}}]
        attach_facts(rows, values)
        self.assertEqual(rows[0]['component_losses']['pseudo_CE'], 1.)
        self.assertEqual(rows[1]['component_gradient_closure']['max_absolute_error'], 1e-8)
        with self.assertRaises(ValueError):
            attach_facts(rows, values[:1])

    def test_decomposition_reconstructs_original_mechanism_and_gradients(self):
        import torch
        from experiments.pamf_ssl_research.hard_pl_objective import mechanism_loss
        from experiments.pamf_repl_attribution_factorial.source_component_diagnostic import decompose_mechanism
        torch.manual_seed(7)
        logits = [torch.randn(8, 3, requires_grad=True) for _ in range(6)]
        labels = torch.tensor([0,1,2,0,1,2,0,1])
        accepted = torch.tensor([1,1,0,1,1,1,0,1], dtype=torch.bool)
        physical = torch.tensor([0,1,0,1,0,1,0,1], dtype=torch.bool)
        aux = logits[0].square().mean()
        parts = decompose_mechanism(*logits, labels, accepted, physical, aux, weight=6.)
        original, _ = mechanism_loss(*logits, labels, accepted, physical, aux)
        total = sum(parts.values())
        torch.testing.assert_close(total, 6.*original)
        expected = torch.autograd.grad(6.*original, logits, retain_graph=True, allow_unused=True)
        actual = torch.autograd.grad(total, logits, retain_graph=True, allow_unused=True)
        for a,b in zip(expected, actual):
            torch.testing.assert_close(a,b)


if __name__ == '__main__':
    unittest.main()
