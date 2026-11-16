"""Focused legality and scalar/gradient reconstruction checks."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from experiments.tri_source_transfer.diagnostic import choose_indices, check_partition, decompose, validate_config, ARMS
from experiments.pamf_ssl_research.hard_pl_objective import mechanism_loss


class SourceTransferChecks(unittest.TestCase):
    def test_decomposition_reconstructs_value_and_derivatives(self):
        torch.manual_seed(17)
        tensors = [torch.randn(8, 3, requires_grad=True) for _ in range(6)]
        aux = torch.tensor(.4, requires_grad=True)
        y = torch.arange(8).remainder(3)
        accepted = torch.tensor([True, False, True, True, False, True, True, False])
        physical = torch.arange(8).remainder(2).bool()
        full = 6*mechanism_loss(*tensors, y, accepted, physical, aux)[0]
        terms = decompose(*tensors, y, accepted, physical, aux, 6)
        torch.testing.assert_close(sum(terms.values()), full)
        params = tensors+[aux]
        expected = torch.autograd.grad(full, params, retain_graph=True)
        actual = torch.autograd.grad(sum(terms.values()), params)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_rx_exclusion_and_physical_overlap(self):
        meta = dict(rx_i=np.array([1]*12+[3]*12), y=np.tile([0,1],12), ids=np.array([f'L{i}' for i in range(24)]))
        u = dict(rx_i=meta['rx_i'].copy(), ids=np.array([f'U{i}' for i in range(24)]))
        lds, uds = SimpleNamespace(metadata=meta), SimpleNamespace(metadata=u)
        ri = choose_indices(meta, allowed_rx=[3], n=4, seed=1, classes=2)
        li = choose_indices(meta, allowed_rx=[1], n=4, seed=2, classes=2, forbidden=ri)
        ui = choose_indices(u, allowed_rx=[1], n=4, seed=3)
        check_partition(lds, uds, li, ui, ri, 3, 'rx_held_out')
        with self.assertRaises(ValueError):
            check_partition(lds, uds, ri, ui, ri, 3, 'sample_disjoint_two_rx')
        with self.assertRaises(ValueError):
            check_partition(lds, uds, li, np.array([12,13]), ri, 3, 'rx_held_out')

    def test_target_handle_rejected_before_config_use(self):
        for key in ('truth','target_truth','target_inputs','views_root','tests','dataset_path','teacher_checkpoint'):
            with self.assertRaisesRegex(ValueError, 'source handles'):
                validate_config({key:'forbidden'})

    def test_bound_matrix(self):
        self.assertEqual(len(ARMS), 8)
        self.assertEqual(len(set(ARMS)), 8)


if __name__ == '__main__':
    unittest.main()
