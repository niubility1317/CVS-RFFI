import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    from experiments.pamf_phase3_open_world import source_fold
except ImportError:
    source_fold = None


SOURCE_RX = [1, 3, 4, 6, 8]
SOURCE_DAYS = [1, 2, 3]
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
FOLD = dict(unknown_source_labels=[0, 1], heldout_source_rx=8)


def metadata(role, per_cell):
    n = 6 * 5 * 3 * per_cell
    return dict(
        y=np.repeat(np.arange(6, dtype=np.int64), 5 * 3 * per_cell),
        rx_i=np.tile(np.repeat(np.asarray(SOURCE_RX), 3 * per_cell), 6),
        day_i=np.tile(np.repeat(np.asarray(SOURCE_DAYS), per_cell), 6 * 5),
        eq_i=np.ones(n, dtype=np.int64),
        sig_i=np.tile(np.arange(per_cell, dtype=np.int64) + (700 if role == 'V' else 0), 6 * 5 * 3),
        base_index=np.arange(n, dtype=np.int64),
        domain=np.tile(np.repeat(np.arange(15, dtype=np.int64), per_cell), 6),
        ids=np.asarray([f'{role}-{i:06d}' for i in range(n)], dtype='U32'),
    )


def source_fixture():
    labeled, validation = metadata('L_s', 70), metadata('V', 300)
    contract = dict(schema='core90_game_source_roles_v1', source_rxs=SOURCE_RX,
        source_days=SOURCE_DAYS, num_classes=6, classes=CLASSES,
        counts=dict(L_s=6300, U_s=56700, V=27000), split_seed=392005,
        checkpoint_init='scratch_only', target_access_before_freeze=False,
        role_ids=dict(L_s=labeled['ids'].tolist(), V=validation['ids'].tolist(),
                      U_s=[f'U_s-{i:06d}' for i in range(56700)]))
    return labeled, validation, contract


class SourceFoldTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(source_fold, 'Source-only fold selector is not implemented')
        self.labeled, self.validation, self.contract = source_fixture()

    def select(self, **kwargs):
        return source_fold.select_fold(kwargs.get('labeled', self.labeled),
            kwargs.get('validation', self.validation), kwargs.get('contract', self.contract),
            kwargs.get('fold', FOLD))

    def test_complete_validation_and_excluded_tx_rx_training(self):
        result = self.select()
        indices = result['train_indices']
        self.assertEqual(len(indices), 3360)
        self.assertEqual(result['known_source_labels'], [2, 3, 4, 5])
        self.assertEqual(result['unknown_source_labels'], [0, 1])
        self.assertEqual(result['training_receivers'], [1, 3, 4, 6])
        self.assertEqual(result['heldout_receiver'], 8)
        self.assertFalse(np.isin(self.labeled['y'][indices], [0, 1]).any())
        self.assertFalse((self.labeled['rx_i'][indices] == 8).any())
        np.testing.assert_array_equal(result['train_ids'], self.labeled['ids'][indices])
        np.testing.assert_array_equal(result['validation_ids'], self.validation['ids'])
        np.testing.assert_array_equal(result['validation_receivers'], self.validation['rx_i'])
        for old, new in [(2, 0), (3, 1), (4, 2), (5, 3), (0, 4), (1, 5)]:
            self.assertTrue((result['validation_y'][self.validation['y'] == old] == new).all())
        self.assertEqual(result['counts']['train'], 3360)
        self.assertEqual(result['counts']['validation'], 27000)
        self.assertEqual(result['counts']['validation_heldout_receiver'], 5400)
        self.assertEqual(result['counts']['validation_known'], 18000)
        self.assertEqual(result['counts']['validation_unknown'], 9000)
        self.assertTrue(result['manifest']['source_only_development'])
        self.assertFalse(result['manifest']['formal_target_unknown_evidence'])
        self.assertFalse(result['manifest']['U_s_opened'])
        self.assertEqual(result['manifest']['validation_policy'], 'complete_V_readonly_no_updates')
        self.assertEqual(result['manifest']['unknown_source_labels'], [0, 1])
        self.assertEqual(result['manifest']['heldout_source_rx'], 8)

    def test_fold_bool_duplicates_extra_fields_or_other_fold_are_rejected(self):
        bad = [dict(FOLD, unknown_source_labels=[False, 1]),
               dict(FOLD, unknown_source_labels=[0, 0]),
               dict(FOLD, heldout_source_rx=True),
               dict(FOLD, target_inputs='forbidden'),
               dict(FOLD, unknown_source_labels=[0, 2]),
               dict(FOLD, heldout_source_rx=6)]
        for fold in bad:
            with self.subTest(fold=fold), self.assertRaises(ValueError):
                self.select(fold=fold)

    def test_exact_role_id_order_uniqueness_and_cross_role_disjointness(self):
        for role in ('L_s', 'V'):
            current = copy.deepcopy(self.labeled if role == 'L_s' else self.validation)
            current['ids'][0], current['ids'][1] = current['ids'][1], current['ids'][0]
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'ID|id|order'):
                self.select(**{'labeled' if role == 'L_s' else 'validation': current})
        changed = copy.deepcopy(self.validation)
        changed['ids'][0] = self.labeled['ids'][0]
        contract = copy.deepcopy(self.contract)
        contract['role_ids']['V'] = changed['ids'].tolist()
        with self.assertRaisesRegex(ValueError, 'overlap|disjoint|role'):
            self.select(validation=changed, contract=contract)
        changed = copy.deepcopy(self.labeled)
        changed['ids'][1] = changed['ids'][0]
        contract = copy.deepcopy(self.contract)
        contract['role_ids']['L_s'] = changed['ids'].tolist()
        with self.assertRaisesRegex(ValueError, 'duplicate|unique|ID|id'):
            self.select(labeled=changed, contract=contract)

    def test_target_receiver_bool_metadata_and_extra_metadata_are_rejected(self):
        for field, value in [('rx_i', 0), ('day_i', 0), ('y', 6), ('domain', 19)]:
            changed = copy.deepcopy(self.labeled)
            changed[field][0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.select(labeled=changed)
        changed = copy.deepcopy(self.validation)
        changed['y'] = changed['y'].astype(bool)
        with self.assertRaises(ValueError):
            self.select(validation=changed)
        changed = dict(self.labeled, target_role=np.asarray(['query'] * len(self.labeled['ids'])))
        with self.assertRaisesRegex(ValueError, 'metadata|field'):
            self.select(labeled=changed)

    def test_contract_target_access_or_wrong_source_scope_is_rejected(self):
        for key, value in [('source_rxs', [0, 3, 4, 6, 8]),
                           ('source_days', [0, 1, 2]),
                           ('num_classes', True),
                           ('target_access_before_freeze', True),
                           ('query_ids', ['target'])]:
            contract = dict(self.contract, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.select(contract=contract)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'verified Torch environment required')
    def test_load_fold_uses_only_original_l_and_v_loaders_and_selected_training_rows(self):
        import torch
        calls = []
        originals = {}
        labeled, validation = self.labeled, self.validation
        class FakeSourceDataset:
            def __init__(self, root, role, contract):
                calls.append(role)
                if role not in ('L_s', 'V'):
                    raise AssertionError('U_s must never be opened')
                self.metadata = labeled if role == 'L_s' else validation
                self.x = torch.arange(len(self.metadata['ids']), dtype=torch.float32).view(-1, 1, 1).expand(-1, 2, 256)
                originals[role] = self.x
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / 'L_s_metadata.npz', **self.labeled)
            np.savez(root / 'V_metadata.npz', **self.validation)
            with patch('experiments.cric_dura_comparison.f_data.SourceDataset', FakeSourceDataset):
                result = source_fold.load_fold(root, self.contract, FOLD)
            self.assertEqual(calls, ['L_s', 'V'])
            self.assertEqual(tuple(result['train_x'].shape), (3360, 2, 256))
            self.assertIs(result['validation_x'], originals['V'])
            np.testing.assert_array_equal(result['train_x'][:, 0, 0].numpy(), result['train_indices'])

    def test_class_coverage_and_original_role_counts_are_required(self):
        changed = copy.deepcopy(self.validation)
        changed['y'][changed['y'] == 1] = 0
        with self.assertRaisesRegex(ValueError, 'class|label|coverage'):
            self.select(validation=changed)
        contract = copy.deepcopy(self.contract)
        contract['counts']['L_s'] = 6301
        with self.assertRaisesRegex(ValueError, 'count|contract'):
            self.select(contract=contract)


if __name__ == '__main__':
    unittest.main()
