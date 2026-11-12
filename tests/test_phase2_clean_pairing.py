import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.pamf_phase2_practical.clean_pairing import map_ids, normalize_batch
from experiments.pamf_phase2_practical.episodic_io import load_task


class CleanPairingTests(unittest.TestCase):
    def test_mapping_preserves_order_and_rejects_duplicate_and_missing_ids(self):
        bank = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(map_ids(bank, np.array(['c', 'a'])), [2, 0])
        with self.assertRaises(ValueError):
            map_ids(bank, np.array(['a', 'a']))
        with self.assertRaises(ValueError):
            map_ids(bank, np.array(['x']))
        with self.assertRaises(ValueError):
            map_ids(np.array(['a', 'a']), np.array(['a']))

    def test_matches_original_complex128_normalization(self):
        rng = np.random.default_rng(20261112)
        raw = rng.standard_normal((32, 2, 256)).astype(np.float32)
        observed = normalize_batch(raw)
        for i, item in enumerate(raw):
            z = item[0].astype(np.float64) + 1j * item[1].astype(np.float64)
            z = z / np.sqrt(np.mean(np.abs(z) ** 2))
            expected = np.stack((z.real, z.imag)).astype(np.float32)
            np.testing.assert_array_equal(observed[i], expected)

    def test_rejects_zero_and_nonfinite_input(self):
        raw = np.ones((2, 2, 256), np.float32)
        raw[0] = 0
        with self.assertRaises(ValueError):
            normalize_batch(raw)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            normalize_batch(raw)

    def test_reader_uses_only_mapped_rows_and_checks_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = root/'parent'; clean = root/'clean'
            cap = parent/'capsules'/'P01'/'MIX01'/'1-1'
            for d in ('splits', 'stage_rows', 'support_labels'):
                (cap/d).mkdir(parents=True, exist_ok=True)
            clean_cap = clean/'capsules'/'P01'/'MIX01'/'1-1'
            clean_cap.mkdir(parents=True)
            (clean/'banks'/'1-1').mkdir(parents=True)
            split_id = 'a'*24
            common = {'protocol_schema':'p2_min_v1', 'phase2_data_status':'VALIDATED_ONCE', 'capsule_id':'cap'}
            (cap/'manifest.json').write_text(json.dumps(dict(common, storage='episodic-pool-complement-v1',
                                                              received_count=12, receiver='1-1')))
            split = dict(common, split_id=split_id, new_count=0, k=1, registered_classes=list(range(6)),
                         support_indices=list(range(6)), query_count=6)
            (cap/'splits'/(split_id+'.json')).write_text(json.dumps(split))
            np.save(cap/'ids.npy', np.array([str(i) for i in range(12)]))
            np.save(cap/'stage_rows'/'N00.npy', np.arange(12, dtype=np.int32))
            np.save(cap/'support_labels'/(split_id+'.npy'), np.arange(6))
            (clean/'completion.json').write_text(json.dumps({'status':'CLEAN_PAIRED_COMPLETE', 'parent_task_root':str(parent)}))
            (clean_cap/'manifest.json').write_text(json.dumps({'capsule_id':'cap', 'receiver':'1-1',
                'input_condition':'clean_ground_pre_satellite_normalized'}))
            np.save(clean_cap/'row_to_clean.npy', np.arange(11, -1, -1, dtype=np.int32))
            bank = np.broadcast_to(np.arange(12, dtype=np.float32)[:, None, None], (12, 2, 256)).copy()
            np.save(clean/'banks'/'1-1'/'clean.npy', bank)
            task = load_task(cap, split_id, stage='B', clean_root=clean, task_root=parent)
            np.testing.assert_array_equal(task['support_x'][:, 0, 0], [11, 10, 9, 8, 7, 6])
            np.testing.assert_array_equal(task['query_x'][:, 0, 0], [5, 4, 3, 2, 1, 0])
            self.assertNotIn('support_x', load_task(cap, split_id, stage='A', clean_root=clean, task_root=parent))
            np.save(cap/'received.npy', bank)
            legacy = load_task(cap, split_id, stage='B')
            np.testing.assert_array_equal(legacy['support_x'][:, 0, 0], [0, 1, 2, 3, 4, 5])
            assert legacy['input_condition'] == 'satellite_received'
            with self.assertRaises((ValueError, KeyError)):
                load_task(cap, split_id, stage='B', clean_root=clean, task_root=root/'wrong')


if __name__ == '__main__':
    unittest.main()
