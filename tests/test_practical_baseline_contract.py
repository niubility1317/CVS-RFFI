import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from baselines.common.practical_source import (
    ContractRoleDataset, PracticalResidualAugment, match_roles, physical_id,
)


class ContractTests(unittest.TestCase):
    def test_roles_must_match_all_physical_ids_and_be_disjoint(self):
        items = [SimpleNamespace(tx_i=0, rx_i=1, day_i=1, eq_i=1, sig_i=i) for i in range(3)]
        base = SimpleNamespace(index=items)
        ids = [physical_id(x) for x in items]
        contract = {'role_ids': dict(zip(('L_s', 'U_s', 'V'), [[s] for s in ids]))}
        self.assertEqual(match_roles(base, contract), {'L_s': [0], 'U_s': [1], 'V': [2]})
        contract['role_ids']['U_s'] = [ids[0]]
        with self.assertRaises(ValueError):
            match_roles(base, contract)

    def test_unlabeled_payload_contains_no_hidden_class(self):
        class Base:
            index = [SimpleNamespace(tx_i=4, rx_i=1, day_i=1, eq_i=1, sig_i=2)]
            def __getitem__(self, k):
                return torch.ones(2, 256), 4, 1, dict(tx='secret', true_tx_i=4,
                    rx_i=1, day_i=1, sig_i=2, rx='R1', day='D1')
        item = ContractRoleDataset(Base(), [0], 'U_s')[0]
        self.assertEqual(item['label'], -1)
        self.assertNotIn('true_label', item)
        self.assertEqual(set(item['meta']), {'sample_id', 'session_id', 'rx_i', 'day_i'})
        self.assertNotIn('secret', str(item['meta']))

    def test_residual_is_reproducible_independent_of_batch_order(self):
        aug = PracticalResidualAugment(fs_hz=25e6)
        aug.set_epoch(100)
        x = torch.randn(4, 2, 256)
        meta = [dict(sample_id=f'opaque-{i}', session_id='rx:A/day:1') for i in range(4)]
        first = aug(x, metadata=meta)
        order = [3, 1, 0, 2]
        second = aug(x[order], metadata=[meta[i] for i in order])
        self.assertTrue(torch.equal(first[order], second))
        self.assertFalse(torch.equal(first, x))
        self.assertTrue(all(c.processing_route == 'residual' and not c.equalization_enabled
                            for c in aug.configs.values()))
        with self.assertRaises(ValueError):
            aug(x)


if __name__ == '__main__':
    unittest.main()
