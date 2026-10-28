from collections import Counter
import importlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


KNOWN = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
RESERVED = ['11-4', '3-13', '12-20', '5-20']
TRAIN = ['11-14', '14-19', '17-20', '18-19']
NOVEL = ['19-15', '19-16', '7-1']
RX = ['1-19', '18-2', '19-2', '2-19', '3-19']
DATES = ['2021_03_08', '2021_03_15', '2021_03_23']


class SourceOEDataTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_oe_data')

    def fixture(self, root, *, short=False, bad_shape=False, nan=False):
        full, validation = root / 'full', root / 'validation'
        full.mkdir(); validation.mkdir()
        compact = KNOWN + RESERVED + ['compact-%03d' % i for i in range(140)]
        memberships = dict(compact={name: dict(count=count, tx_list=compact[:count])
            for name, count in [('ManySig', 6), ('ManyTx', 150), ('ManyRx', 10), ('SingleDay', 28)]},
            compact_union_count=150, compact_union_not_in_full=[], full_tx_count=174,
            full_tx_absent_all_compact=TRAIN + NOVEL + ['unused-full-%02d' % i for i in range(17)],
            old_six=KNOWN, formal_unknown_excluded=RESERVED, source_receivers=RX, source_dates=DATES,
            candidate_capacity={tx: {day: {rx: 80 for rx in RX} for day in DATES} for tx in TRAIN + NOVEL})
        membership_path = root / 'compact_tx_memberships.json'
        membership_path.write_text(json.dumps(memberships), encoding='utf-8')
        summary = dict(status='VERIFIED', dataset_directory=str(full), members_checked=158,
            expected_nonempty_RX_date_members=158, count_mismatch_cells=0, errors=[])
        (validation / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        rows = []
        for day in DATES:
            for receiver in RX:
                archive = 'pkl_wifi_eq_' + day + '-synthetic-' + receiver + '.zip'
                member = 'pkl_wifi_eq_' + day + '/dataset_' + day + '_node' + receiver + '.pkl'
                arrays = []
                for j, tx in enumerate(TRAIN + NOVEL):
                    values = np.ones((80, 256, 2), dtype=np.float64)
                    values[:, :, 1] = .1 + j * .05
                    values[:, 0, 0] = np.arange(80) + 1.
                    arrays.append(values)
                if short:
                    arrays[0] = arrays[0][:69]
                if bad_shape:
                    arrays[0] = arrays[0][:, :128]
                if nan:
                    arrays[0][:] = np.nan
                with zipfile.ZipFile(full / archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
                    output.writestr(member, pickle.dumps(dict(data=arrays, node_list=TRAIN + NOVEL), protocol=4))
                rows.append(dict(archive=archive, member=member, receiver=receiver, date=day, status='PASS'))
        (validation / 'members.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return dict(full_root=str(full), validation_root=str(validation), compact_memberships=str(membership_path),
            known_classes=KNOWN, reserved_target_classes=RESERVED, receivers=RX, dates=DATES,
            exposure_train_classes=TRAIN, novel_validation_classes=NOVEL, per_cell=70, selection_seed=2026102905)

    def test_public_domains_private_tx_and_unique_physical_roles(self):
        module = self.module()
        from experiments.phase1_external_dg import data as iq_data
        from experiments.pamf_phase3_open_world import full_pool
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root)
            with patch.object(iq_data, 'iq_record', wraps=iq_data.iq_record) as normalized, \
                 patch.object(full_pool, '_channel_api', side_effect=AssertionError('OE must not simulate channels')):
                result = module.build(config, root / 'output')
            self.assertEqual(normalized.call_count, 7350)
            self.assertEqual(result['status'], 'SOURCE_OE_DATA_VALIDATED')
            self.assertEqual(result['data_status'], 'VALIDATED_ONCE')
            self.assertEqual(result['counts'], {'exposure_train': 4200, 'novel_validation': 3150})
            all_ids, tx_roles = set(), []
            for role, count, expected in [('exposure_train', 4200, TRAIN), ('novel_validation', 3150, NOVEL)]:
                public = root / 'output' / role / 'public'
                self.assertEqual({path.name for path in public.iterdir()}, {'iq.npy', 'ids.npy', 'metadata.npz'})
                iq, ids = np.load(public / 'iq.npy', allow_pickle=False), np.load(public / 'ids.npy', allow_pickle=False)
                self.assertEqual(iq.shape, (count, 2, 256)); self.assertEqual(iq.dtype, np.dtype('float32'))
                self.assertTrue(np.allclose(np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1), 1., atol=2e-6))
                self.assertEqual(ids.dtype.kind, 'U'); self.assertEqual(len(set(ids.tolist())), count)
                self.assertTrue(np.array_equal(ids, np.sort(ids)))
                self.assertTrue(all(value.startswith('p3oe-') and len(value) == 29 for value in ids))
                self.assertFalse(all_ids & set(ids.tolist())); all_ids.update(ids.tolist())
                with np.load(public / 'metadata.npz', allow_pickle=False) as domain:
                    self.assertEqual(set(domain.files), {'rx_i', 'day_i', 'domain'})
                    for key in domain.files:
                        self.assertEqual(domain[key].dtype, np.dtype('int64'))
                    self.assertEqual(set(domain['rx_i'].tolist()), {1, 3, 4, 6, 8})
                    self.assertEqual(set(domain['day_i'].tolist()), {1, 2, 3})
                    np.testing.assert_array_equal(domain['domain'], np.searchsorted([1, 3, 4, 6, 8], domain['rx_i']) * 3 + domain['day_i'] - 1)
                    rx, day = domain['rx_i'].copy(), domain['day_i'].copy()
                with np.load(root / 'output' / role / 'truth/truth.npz', allow_pickle=False) as truth:
                    self.assertEqual(set(truth.files), {'ids', 'tx'})
                    np.testing.assert_array_equal(truth['ids'], ids)
                    self.assertEqual(set(truth['tx'].tolist()), set(expected))
                    self.assertEqual(set(Counter(zip(truth['tx'].tolist(), rx.tolist(), day.tolist())).values()), {70})
                    tx_roles.append(set(truth['tx'].tolist()))
            self.assertFalse(tx_roles[0] & tx_roles[1])
            report = json.loads((root / 'output/builder_report.json').read_text())
            canonical = [tuple(item['canonical_record']) for role in report['mapping'].values() for item in role]
            self.assertEqual(len(canonical), 7350); self.assertEqual(len(set(canonical)), 7350)
            self.assertTrue(all(item[0] == 'FullWiSigEq' and item[4] == 1 and 0 <= item[5] < 80 for item in canonical))
            self.assertEqual(report['original_U_s_reads'], 0); self.assertEqual(report['model_reads'], 0)
            self.assertFalse(report['whole_project_never_exposed_claim'])
            self.assertTrue(report['source_OE_membership_is_additional_supervision'])

    def test_only_fifteen_indexed_members_are_read_and_marker_is_last(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root)
            (root / 'full/unrelated-invalid.zip').write_bytes(b'invalid')
            original_zip, original_write = zipfile.ZipFile, module._write
            opened, writes = [], []
            def selected(path, *args, **kwargs):
                if isinstance(path, (str, Path)):
                    opened.append(Path(path).name)
                return original_zip(path, *args, **kwargs)
            def writing(path, value):
                self.assertFalse((root / 'output/complete.json').exists())
                if Path(path).name == 'complete.json':
                    self.assertTrue((root / 'output/novel_validation/truth/truth.npz').is_file())
                    self.assertTrue((root / 'output/builder_report.json').is_file())
                writes.append(Path(path).name)
                return original_write(path, value)
            with patch.object(module.zipfile, 'ZipFile', side_effect=selected), \
                 patch.object(Path, 'glob', side_effect=AssertionError('No full archive rescan')), \
                 patch.object(module, '_write', side_effect=writing):
                module.build(config, root / 'output')
            self.assertEqual(len(opened), 15); self.assertEqual(len(set(opened)), 15)
            self.assertEqual(writes[-1], 'complete.json')

    def test_compact_known_reserved_or_overlapping_tx_are_rejected_before_iq(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root)
            for key, value in [('exposure_train_classes', KNOWN[:4]), ('novel_validation_classes', RESERVED[:3]),
                               ('source_cache', 'forbidden'), ('per_cell', 69), ('selection_seed', True)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: value}))
            memberships = json.loads((root / 'compact_tx_memberships.json').read_text())
            memberships['compact']['ManyTx']['tx_list'][20] = TRAIN[0]
            (root / 'compact_tx_memberships.json').write_text(json.dumps(memberships), encoding='utf-8')
            with patch.object(module.zipfile, 'ZipFile', side_effect=AssertionError('No IQ on compact TX overlap')), self.assertRaises(ValueError):
                module.build(config, root / 'output')
            self.assertFalse((root / 'output/complete.json').exists())

    def test_short_nan_or_wrong_shape_cannot_complete(self):
        module = self.module()
        for options in (dict(short=True), dict(nan=True), dict(bad_shape=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config = self.fixture(root, **options)
                with self.assertRaises(ValueError):
                    module.build(config, root / 'output')
                self.assertFalse((root / 'output/complete.json').exists())

    def test_wrong_member_index_and_cached_count_are_not_revalidated(self):
        module = self.module()
        for failure in ('member', 'count'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config = self.fixture(root)
                if failure == 'member':
                    path = root / 'validation/members.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]['member'] = 'wrong-member.pkl'
                    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                else:
                    path = root / 'validation/summary.json'; summary = json.loads(path.read_text())
                    summary['count_mismatch_cells'] = 1; path.write_text(json.dumps(summary), encoding='utf-8')
                with patch.object(module.zipfile, 'ZipFile', side_effect=AssertionError('Invalid locator must not open archives')), self.assertRaises(ValueError):
                    module.build(config, root / 'output')
                self.assertFalse((root / 'output/complete.json').exists())

    def test_output_collision_precedes_any_reads_and_selection_is_stable(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = self.fixture(root); (root / 'output').mkdir()
            with patch.object(module, '_read', side_effect=AssertionError('No reads after collision')), self.assertRaises(FileExistsError):
                module.build(config, root / 'output')
            a = module._indices(80, 2026102905, RX[0], DATES[0], TRAIN[0], 70)
            b = module._indices(80, 2026102905, RX[0], DATES[0], TRAIN[0], 70)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(np.unique(a)), 70)
            self.assertFalse(np.array_equal(a, np.arange(70)))
            c = module._indices(80, 2026102905, RX[0], DATES[0], TRAIN[1], 70)
            self.assertFalse(np.array_equal(a, c))


if __name__ == '__main__':
    unittest.main()
