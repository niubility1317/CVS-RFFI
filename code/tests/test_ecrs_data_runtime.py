from types import SimpleNamespace
import unittest

import torch

from cvsrffi.ecrs_data_runtime import MatchedLabeledLoader, make_revision_l_loader


class IndexedDataset:
    tx_label_visible = True

    def __init__(self):
        self.index = [SimpleNamespace(tx_i=tx, rx_i=rx, day_i=day)
                      for tx in range(6) for rx in range(5) for day in range(2) for _ in range(4)]
        self.reads = 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        self.reads += 1
        row = self.index[i]
        return torch.tensor([float(i)]), row.tx_i, row.rx_i, {'receiver_id': row.rx_i, 'day_id': row.day_i, 'physical_sample_id': str(i)}


class DataRuntimeTests(unittest.TestCase):
    def test_default_collate_no_prefetch_and_exact_restore(self):
        ds = IndexedDataset()
        loader = MatchedLabeledLoader(ds, seed=3)
        self.assertEqual(ds.reads, 0)
        self.assertEqual(len(loader), 2)
        self.assertEqual(loader.num_workers, 0)
        iterator = iter(loader)
        self.assertEqual(ds.reads, 0)
        first = next(iterator)
        self.assertEqual(ds.reads, 120)
        self.assertEqual(first[0].shape, (120, 1))
        self.assertEqual(set(first[3]['day_id'].tolist()), {0})
        state = loader.state_dict()
        expected = next(iterator)
        restored = MatchedLabeledLoader(ds, seed=999)
        restored.load_state_dict(state)
        actual = next(iter(restored))
        torch.testing.assert_close(actual[0], expected[0])
        self.assertEqual(actual[3]['physical_sample_id'], expected[3]['physical_sample_id'])
        self.assertEqual(restored.last_counts, loader.last_counts)
        self.assertEqual(set(actual[3]['day_id'].tolist()), {1})

    def test_factory_preserves_legacy_and_rejects_hidden_tx(self):
        ds = IndexedDataset()
        args = SimpleNamespace(use_ecrs=True, ecrs_version='v2', ecrs_sampler_mode='legacy')
        self.assertIsNone(make_revision_l_loader(ds, args))
        args.ecrs_sampler_mode = 'balanced_tx_rx'
        self.assertIsInstance(make_revision_l_loader(ds, args), MatchedLabeledLoader)
        ds.tx_label_visible = False
        with self.assertRaisesRegex(ValueError, 'TX-hidden'):
            make_revision_l_loader(ds, args)
        self.assertEqual(ds.reads, 0)

    def test_missing_cells_are_retained_and_reported(self):
        ds = IndexedDataset()
        ds.index = [row for row in ds.index if not (row.tx_i == 0 and row.rx_i == 0)]
        loader = MatchedLabeledLoader(ds)
        actual = next(iter(loader))
        self.assertEqual(actual[0].shape[0], 116)
        self.assertEqual(loader.last_counts['missing_cells'], 1)
        self.assertEqual(loader.last_counts['missing_samples'], 4)


if __name__ == '__main__':
    unittest.main()
