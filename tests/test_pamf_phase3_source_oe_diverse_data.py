from collections import Counter
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


CLASSES = ['1-20', '1-9', '10-20', '11-11', '11-14', '13-8', '14-19', '15-15', '17-20', '17-3',
    '18-19', '18-6', '19-5', '2-18', '2-2', '20-10', '20-11', '20-17', '20-6', '3-3', '6-20']
OLD4 = ['11-14', '14-19', '17-20', '18-19']
NOVEL = ['19-15', '19-16', '7-1']
RX = ['1-19', '18-2', '19-2', '2-19', '3-19']
DATES = ['2021_03_08', '2021_03_15', '2021_03_23']
KNOWN = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
RESERVED = ['11-4', '3-13', '12-20', '5-20']


class SourceOEDiverseDataTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_oe_diverse_data')

    def fixture(self, root, *, raw_mismatch=False):
        full, validation = root / 'full', root / 'validation'; full.mkdir(); validation.mkdir()
        compact = KNOWN + RESERVED + ['compact-%03d' % i for i in range(140)]
        capacity = {tx: {day: {} for day in DATES} for tx in CLASSES}
        small = np.array([0]*95 + [4]*32 + [22] + [33]*127)
        np.random.default_rng(7).shuffle(small); position = 0
        for tx in CLASSES:
            for day in DATES:
                for receiver in RX:
                    if tx in OLD4: count = 33
                    else: count = int(small[position]); position += 1
                    capacity[tx][day][receiver] = count
        evidence = dict(compact={name: dict(count=n, tx_list=compact[:n]) for name, n in [('ManySig', 6), ('ManyTx', 150), ('ManyRx', 10), ('SingleDay', 28)]},
            compact_union_count=150, compact_union_not_in_full=[], full_tx_count=174,
            full_tx_absent_all_compact=sorted(CLASSES + NOVEL), candidates=sorted(CLASSES + NOVEL),
            old_six=KNOWN, formal_unknown_excluded=RESERVED, source_receivers=RX, source_dates=DATES, candidate_capacity=capacity)
        evidence_path = root / 'compact.json'; evidence_path.write_text(json.dumps(evidence), encoding='utf-8')
        (validation / 'summary.json').write_text(json.dumps(dict(status='VERIFIED', dataset_directory=str(full),
            members_checked=158, expected_nonempty_RX_date_members=158, count_mismatch_cells=0, errors=[])), encoding='utf-8')
        rows = []
        for day in DATES:
            for receiver in RX:
                names, arrays = [], []
                for tx in CLASSES:
                    count = capacity[tx][day][receiver]
                    if not count: continue  # Zero cells may be absent from original node_list.
                    raw = np.ones((count, 256, 2), dtype=np.float64); raw[:, :, 1] = .5
                    raw[:, 0, 0] = np.arange(count) + 1.
                    if raw_mismatch and tx == '11-14': raw = raw[:-1]
                    names.append(tx); arrays.append(raw)
                archive = 'pkl_wifi_eq_' + day + '-synthetic-' + receiver + '.zip'
                member = 'pkl_wifi_eq_' + day + '/dataset_' + day + '_node' + receiver + '.pkl'
                with zipfile.ZipFile(full / archive, 'w', compression=zipfile.ZIP_DEFLATED) as stored:
                    stored.writestr(member, pickle.dumps(dict(data=arrays, node_list=names), protocol=4))
                rows.append(dict(archive=archive, member=member, receiver=receiver, date=day, status='PASS'))
        (validation / 'members.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return dict(full_root=str(full), validation_root=str(validation), compact_memberships=str(evidence_path),
            known_classes=KNOWN, reserved_target_classes=RESERVED, receivers=RX, dates=DATES,
            exposure_train_classes=CLASSES, excluded_novel_classes=NOVEL, per_cell=32, selection_seed=2026102906), capacity

    def test_real_cap_normalization_ids_private_tx_and_no_replication(self):
        module = self.module()
        from experiments.phase1_external_dg import data as iq_data
        from experiments.pamf_phase3_open_world import full_pool
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, capacity = self.fixture(root)
            with patch.object(iq_data, 'iq_record', wraps=iq_data.iq_record) as normalizations, \
                 patch.object(full_pool, '_channel_api', side_effect=AssertionError('No channel generation')):
                result = module.run(config, root / 'output')
            self.assertEqual(normalizations.call_count, 6134)
            self.assertEqual(result['status'], 'SOURCE_OE_DIVERSE_DATA_VALIDATED'); self.assertEqual(result['count'], 6134)
            public = root / 'output/exposure_train/public'
            self.assertEqual({path.name for path in public.iterdir()}, {'iq.npy', 'ids.npy', 'metadata.npz'})
            iq, ids = np.load(public / 'iq.npy'), np.load(public / 'ids.npy')
            self.assertEqual(iq.shape, (6134, 2, 256)); self.assertEqual(iq.dtype, np.dtype('float32'))
            self.assertTrue(np.allclose(np.mean(iq[:, 0]**2 + iq[:, 1]**2, axis=1), 1., atol=2e-6))
            self.assertEqual(len(set(ids.tolist())), 6134); np.testing.assert_array_equal(ids, np.sort(ids))
            with np.load(public / 'metadata.npz') as meta:
                self.assertEqual(set(meta.files), {'rx_i', 'day_i', 'domain'})
                self.assertTrue(all(meta[name].dtype == np.int64 for name in meta.files))
                np.testing.assert_array_equal(meta['domain'], np.searchsorted([1, 3, 4, 6, 8], meta['rx_i']) * 3 + meta['day_i'] - 1)
            with np.load(root / 'output/exposure_train/truth/truth.npz') as truth:
                self.assertEqual(set(truth.files), {'ids', 'tx'}); np.testing.assert_array_equal(truth['ids'], ids)
                self.assertEqual(set(truth['tx'].tolist()), set(CLASSES))
                self.assertFalse(set(truth['tx'].tolist()) & set(KNOWN + RESERVED + NOVEL))
            report = json.loads((root / 'output/builder_report.json').read_text())
            self.assertEqual(report['allocation']['nonempty_cells'], 220); self.assertEqual(report['allocation']['zero_cells'], 95)
            self.assertEqual(report['allocation']['original4_count'], 1920); self.assertEqual(report['allocation']['new17_count'], 4214)
            records = [tuple(item['canonical_record']) for item in report['mapping']]
            self.assertEqual(len(records), len(set(records)))
            cells = Counter((record[1], record[3], record[2]) for record in records)
            for tx in CLASSES:
                for day in DATES:
                    for receiver in RX: self.assertEqual(cells[tx, day, receiver], min(capacity[tx][day][receiver], 32))
            self.assertFalse(report['samples_replicated']); self.assertFalse(report['sample_weights_assigned'])
            self.assertNotIn('original4_count', result); self.assertNotIn('new17_count', result)

    def test_fifteen_member_reads_and_marker_last(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, _ = self.fixture(root); (root / 'full/unrelated.zip').write_bytes(b'ignored')
            original_zip, original_write = zipfile.ZipFile, module._write
            opened, writes = [], []
            def opening(path, *args, **kwargs):
                if isinstance(path, (str, Path)): opened.append(str(path))
                return original_zip(path, *args, **kwargs)
            def writing(path, value):
                self.assertFalse((root / 'output/complete.json').exists())
                if Path(path).name == 'complete.json': self.assertTrue((root / 'output/exposure_train/truth/truth.npz').is_file())
                writes.append(Path(path).name); return original_write(path, value)
            with patch.object(module.zipfile, 'ZipFile', side_effect=opening), patch.object(module, '_write', side_effect=writing), \
                 patch.object(Path, 'glob', side_effect=AssertionError('No full archive rescan')):
                module.run(config, root / 'output')
            self.assertEqual(len(opened), 15); self.assertEqual(writes[-1], 'complete.json')

    def test_metadata_or_raw_capacity_error_cannot_complete(self):
        module = self.module()
        for error in ['metadata', 'raw']:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, _ = self.fixture(root, raw_mismatch=error == 'raw')
                if error == 'metadata':
                    path = root / 'compact.json'; value = json.loads(path.read_text())
                    value['candidate_capacity']['11-14'][DATES[0]][RX[0]] = 31
                    path.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError): module.run(config, root / 'output')
                self.assertFalse((root / 'output/complete.json').exists())

    def test_output_protection_and_forbidden_tx_or_budget_inputs(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, _ = self.fixture(root); (root / 'output').mkdir()
            with patch.object(module, '_read', side_effect=AssertionError('No read after collision')), self.assertRaises(FileExistsError):
                module.run(config, root / 'output')
            for tx in KNOWN[:1] + RESERVED[:1] + NOVEL[:1]:
                changed = list(CLASSES); changed[0] = tx
                with self.subTest(tx=tx), self.assertRaises(ValueError): module.validate_config(dict(config, exposure_train_classes=changed))
            with self.assertRaises(ValueError): module.validate_config(dict(config, source_cache='forbidden'))
            with self.assertRaises(ValueError): module.validate_config(dict(config, selection_seed=2026102905))

    def test_cli_prints_actual_runtime_without_process_dependency(self):
        module = self.module(); argv = ['source_oe_diverse_data.py', '--config', 'config.json', '--output', 'new-root']
        output = io.StringIO()
        with patch.object(module.sys, 'argv', argv), patch.object(module, '_read', return_value={}), \
             patch.object(module, 'run', return_value={'status': 'SOURCE_OE_DIVERSE_DATA_VALIDATED', 'count': 6134}), \
             contextlib.redirect_stdout(output):
            self.assertEqual(module.main(argv[1:]), 0)
        runtime = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(runtime, {'event': 'SOURCE_OE_DIVERSE_RUNTIME', 'pid': os.getpid(), 'cwd': os.getcwd(), 'argv': argv})


if __name__ == '__main__': unittest.main()
