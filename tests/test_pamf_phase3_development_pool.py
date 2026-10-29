from collections import Counter
import importlib
import io
import json
from pathlib import Path
import pickle
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


OLD = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
FOUND = ['19-15', '19-16']
FUTURE = ['7-1']
SOURCE_RX = ['1-19', '18-2', '19-2', '2-19', '3-19']
PHASE2_RX = ['1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8']
SCENES = ['practical_low_suburban', 'practical_mid_urban', 'practical_low_urban']
MEMBERS = {
    '2021_03_08': ('pkl_wifi_eq_2021_03_08-20261028T143859Z-1-002.zip', 'pkl_wifi_eq_2021_03_08/dataset_2021_03_08_node24-1.pkl'),
    '2021_03_15': ('pkl_wifi_eq_2021_03_15-20261028T144306Z-1-004.zip', 'pkl_wifi_eq_2021_03_15/dataset_2021_03_15_node24-1.pkl'),
}


class DevelopmentPoolTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/development_pool.py'
        self.assertTrue(path.is_file(), 'Independent development RX builder is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.development_pool')

    def fixture(self, root, *, count=96, bad_shape=False, nan=False):
        full, validation = root / 'full', root / 'validation'
        full.mkdir()
        validation.mkdir()
        config = dict(validation_root=str(validation), full_root=str(full),
            compact_receiver_evidence=str(root / 'compact.json'), source_receivers=SOURCE_RX,
            formal_receiver='24-18', receiver='24-1', receiver_index=26,
            registered_classes=OLD, discovery_classes=FOUND, future_unknown_classes=FUTURE,
            days=dict(discovery='2021_03_08', evaluation='2021_03_15'),
            per_scene=dict(discovery=16, evaluation=32), scenes=SCENES,
            seeds=dict(selection=2026102911, receiver=2026102912, channel=2026102913, shuffle=2026102914))
        summary = dict(status='VERIFIED', dataset_directory=str(full), members_checked=158,
            expected_nonempty_RX_date_members=158, count_mismatch_cells=0, errors=[])
        (validation / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        receivers = [f'other-rx-{i}' for i in range(41)]
        receivers[26] = '24-1'
        receivers[30] = '24-18'
        for i, name in enumerate(SOURCE_RX + PHASE2_RX):
            receivers[i] = name
        transmitters = [f'other-tx-{i}' for i in range(174)]
        transmitters[:9] = list(reversed(OLD + FOUND + FUTURE))
        dates = ['2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23']
        counts = np.zeros((4, 41, 174), dtype=np.int64)
        counts[1:3, 26, :9] = 192
        (validation / 'counts.json').write_text(json.dumps(dict(dates=dates, receivers=receivers,
            transmitters=transmitters, records=counts.tolist())), encoding='utf-8')
        evidence = dict(status='VERIFIED', rows=[dict(file=name, receiver_names=SOURCE_RX + PHASE2_RX,
            receiver_count=12) for name in ('ManyRx.pkl', 'ManySig.pkl', 'ManyTx.pkl', 'SingleDay.pkl')])
        (root / 'compact.json').write_text(json.dumps(evidence), encoding='utf-8')
        locator = []
        names = list(reversed(OLD + FOUND + FUTURE))
        for day, (archive, member) in MEMBERS.items():
            arrays = []
            for tx in names:
                value = np.ones((count, 256, 2), dtype=np.float64)
                value[:, :, 0] = (OLD + FOUND + FUTURE).index(tx) + 1.
                arrays.append(value)
            if bad_shape:
                arrays[names.index(OLD[0])] = arrays[names.index(OLD[0])][:, :128]
            if nan:
                arrays[names.index(OLD[0])][:] = np.nan
            with zipfile.ZipFile(full / archive, 'w', compression=zipfile.ZIP_DEFLATED) as stored:
                stored.writestr(member, pickle.dumps(dict(data=arrays, node_list=names), protocol=4))
                stored.writestr(member.replace('24-1.pkl', '24-18.pkl'), b'FORMAL TARGET MUST NOT OPEN')
            locator.append(dict(archive=archive, member=member, date=day, receiver='24-1', status='PASS'))
        locator.append(dict(archive='formal-only.zip', member='formal-only.pkl', receiver='24-18',
                            date='2021_03_08', status='PASS'))
        (validation / 'members.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in locator), encoding='utf-8')
        return config

    def fake_channel(self, normalized, cfg, seed, receiver):
        self.assertEqual(normalized.dtype, np.dtype('float32'))
        self.assertEqual(normalized.shape, (2, 256))
        self.assertAlmostEqual(float(np.mean(normalized[0] ** 2 + normalized[1] ** 2)), 1., places=6)
        return (normalized * 2).astype(np.float32), dict(scenario=cfg.scenario, seed=int(seed))

    def test_complete_fake_build_has_private_truth_and_exact_public_contract(self):
        module = self.module()
        from experiments.pamf_phase3_open_world.data import load_public, PUBLIC_FIELDS, PUBLIC_FILES
        from experiments.phase1_external_dg import data as source_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            with patch.object(module, '_apply_channel', side_effect=self.fake_channel) as calls, \
                 patch.object(source_data, 'iq_record', wraps=source_data.iq_record) as rms:
                result = module.build(config, root / 'pool')
            self.assertEqual((calls.call_count, rms.call_count), (1248, 1248))
            self.assertEqual(result['status'], 'PHASE3_DEVELOPMENT_POOL_VALIDATED')
            self.assertEqual(result['data_status'], 'VALIDATED_ONCE')
            self.assertTrue(result['development_only'])
            for key in ('formal_target_reads', 'model_reads', 'model_predictions', 'optimizer_updates'):
                self.assertEqual(result[key], 0)
            seen = set()
            for role, count, nclass in [('discovery', 384, 8), ('evaluation', 864, 9)]:
                public = root / 'pool' / role / 'public'
                self.assertEqual({p.name for p in public.iterdir()}, PUBLIC_FILES)
                manifest, ids, received = load_public(public)
                self.assertEqual(set(manifest), PUBLIC_FIELDS)
                self.assertEqual(manifest['registered_classes'], OLD)
                self.assertEqual(received.shape, (count, 2, 256))
                self.assertTrue(received.flags.writeable is False)
                self.assertTrue(np.allclose(np.mean(received[:, 0] ** 2 + received[:, 1] ** 2, axis=1), 4.))
                self.assertTrue(all(re.fullmatch(r'p3dev-[0-9a-f]{24}', item) for item in ids))
                self.assertFalse(seen & set(ids.tolist()))
                seen.update(ids.tolist())
                for secret in FOUND + FUTURE + SCENES + ['24-1', str(root), 'unknown_count', 'quota']:
                    self.assertNotIn(secret, (public / 'manifest.json').read_text())
                    self.assertFalse(any(secret in item for item in ids))
                with np.load(root / 'pool' / role / 'truth/truth.npz', allow_pickle=False) as truth:
                    self.assertEqual(set(truth.files), {'ids', 'y', 'receiver', 'scenario'})
                    self.assertTrue(np.array_equal(ids, truth['ids']))
                    self.assertEqual(set(truth['y'].tolist()), set(range(nclass)))
                    self.assertTrue(bool((truth['receiver'] == 26).all()))
                    self.assertEqual(set(Counter(zip(truth['y'], truth['scenario'])).values()), {config['per_scene'][role]})
                    self.assertTrue(np.allclose(received[:, 0, 0] / received[:, 1, 0], truth['y'] + 1.))
                received._mmap.close()
            report = json.loads((root / 'pool/builder_report.json').read_text())
            records = [record for rows in report['mapping'].values() for record in rows]
            canonical = [tuple(record['canonical_record']) for record in records]
            self.assertEqual((len(canonical), len(set(canonical))), (1248, 1248))
            self.assertEqual(set(row[2] for row in canonical), {'24-1'})
            self.assertTrue(all(row[0] == 'FullWiSigEq' and row[4] == 1 for row in canonical))
            self.assertNotIn('7-1', {row['canonical_record'][1] for row in report['mapping']['discovery']})
            self.assertTrue(report['source_development_formal_receivers_disjoint'])
            self.assertTrue(report['scene_and_role_canonical_disjoint'])
            self.assertFalse(report['clean_reference_written'])
            self.assertEqual(report['input_RMS_normalizations_per_record'], 1)

    def test_only_two_indexed_members_open_without_formal_read_or_zip_rescan(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            (root / 'full/unrelated-invalid.zip').write_bytes(b'NO READ')
            original_open = zipfile.ZipFile.open
            opened = []
            def selected(stored, name, *args, **kwargs):
                if stored.mode == 'r':
                    opened.append((Path(stored.filename).name, name))
                    self.assertNotIn('24-18', name)
                return original_open(stored, name, *args, **kwargs)
            with patch.object(module, '_apply_channel', side_effect=self.fake_channel), \
                 patch.object(zipfile.ZipFile, 'open', selected), \
                 patch.object(zipfile.ZipFile, 'namelist', side_effect=AssertionError('No ZIP directory scan')), \
                 patch.object(Path, 'glob', side_effect=AssertionError('No Full archive rescan')):
                module.build(config, root / 'pool')
            self.assertEqual(opened, list(MEMBERS.values()))

    def test_config_rejects_changed_fixed_roles_and_target_or_model_inputs(self):
        module = self.module()
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_development_pool_20261029.json').read_text())
        self.assertIs(module.validate_config(config), config)
        mutations = [('receiver', '24-18'), ('receiver', SOURCE_RX[0]), ('receiver_index', 30),
            ('formal_receiver', '24-1'), ('registered_classes', list(reversed(OLD))),
            ('days', dict(discovery='2021_03_08', evaluation='2021_03_08')),
            ('scenes', list(reversed(SCENES))), ('seeds', dict(config['seeds'], channel=1)),
            ('per_scene', dict(discovery=1, evaluation=1)), ('model', 'forbidden'), ('target_root', 'forbidden')]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                module.validate_config(dict(config, **{key: value}))

    def test_cached_member_identity_status_and_missing_member_fail_before_iq(self):
        module = self.module()
        for key, value in [('receiver', '24-18'), ('date', '2021_03_23'), ('status', 'FAIL'), ('member', 'wrong.pkl')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.fixture(root)
                path = root / 'validation/members.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0][key] = value
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                with patch.object(module.zipfile, 'ZipFile', side_effect=AssertionError('No IQ read')), self.assertRaises(ValueError):
                    module.build(config, root / 'pool')
                self.assertFalse((root / 'pool/complete.json').exists())

    def test_cached_counts_shape_rx_namespace_and_capacity_rejected_before_iq(self):
        module = self.module()
        for problem in ('shape', 'receiver_index', 'count', 'transmitter'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.fixture(root)
                path = root / 'validation/counts.json'
                value = json.loads(path.read_text())
                if problem == 'shape':
                    value['records'].pop()
                elif problem == 'receiver_index':
                    value['receivers'][26] = 'wrong-rx'
                elif problem == 'count':
                    value['records'][2][26][0] = 95
                else:
                    value['transmitters'][0] = 'wrong-tx'
                path.write_text(json.dumps(value), encoding='utf-8')
                with patch.object(module.zipfile, 'ZipFile', side_effect=AssertionError('No IQ read')), self.assertRaises(ValueError):
                    module.build(config, root / 'pool')

    def test_compact_rx_evidence_must_cover_all_four_and_exclude_development(self):
        module = self.module()
        for problem in ('included', 'missing', 'unverified', 'phase2_missing'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.fixture(root)
                path = root / 'compact.json'
                value = json.loads(path.read_text())
                if problem == 'included':
                    value['rows'][0]['receiver_names'].append('24-1')
                    value['rows'][0]['receiver_count'] += 1
                elif problem == 'missing':
                    value['rows'].pop()
                elif problem == 'unverified':
                    value['status'] = 'UNKNOWN'
                else:
                    for row in value['rows']:
                        row['receiver_names'].remove('1-1')
                        row['receiver_count'] -= 1
                path.write_text(json.dumps(value), encoding='utf-8')
                with patch.object(module.zipfile, 'ZipFile', side_effect=AssertionError('No IQ read')), self.assertRaises(ValueError):
                    module.build(config, root / 'pool')

    def test_raw_shape_nonfinite_and_capacity_fail_without_completion(self):
        module = self.module()
        for kwargs in (dict(bad_shape=True), dict(nan=True), dict(count=95)):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.fixture(root, **kwargs)
                with patch.object(module, '_apply_channel', side_effect=AssertionError('Raw validation first')), self.assertRaises(ValueError):
                    module.build(config, root / 'pool')
                self.assertFalse((root / 'pool/complete.json').exists())

    def test_existing_output_prevents_any_input_reads_and_channel_failure_stays_incomplete(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            (root / 'existing').mkdir()
            with patch.object(module, '_validated_members', side_effect=AssertionError('No input read')), self.assertRaises(FileExistsError):
                module.build(config, root / 'existing')
            with patch.object(module, '_apply_channel', side_effect=RuntimeError('channel failure')), self.assertRaises(RuntimeError):
                module.build(config, root / 'failed')
            self.assertFalse((root / 'failed/complete.json').exists())

    def test_scene_and_seed_allocation_is_role_blind_and_original_rows_unique(self):
        module = self.module()
        _, _, stable_seed, _ = module._channel_api()
        rows = np.ones((1300, 256, 2), dtype=np.float64)
        for tx in (OLD[0], FOUND[0], FUTURE[0]):
            selected = module._select_rows(rows, tx, '24-1', '2021_03_08', 16, SCENES, 2026102911)
            indices = [record['canonical_record'][-1] for record in selected]
            self.assertEqual(len(set(indices)), 48)
            self.assertEqual([record['scene'] for record in selected], SCENES * 16)
            for record in selected:
                self.assertEqual(module._channel_seed(2026102913, record['scene'], record['canonical_record']),
                                 stable_seed(2026102913, record['scene'], *record['canonical_record']))

    def test_duplicate_canonical_roles_fail_before_channel_and_output(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            original = module._selected_member
            first = []
            def duplicate(*args, **kwargs):
                rows = original(*args, **kwargs)
                if not first:
                    first.append(rows[0]['canonical_record'])
                else:
                    rows[0]['canonical_record'] = first[0]
                return rows
            with patch.object(module, '_selected_member', side_effect=duplicate), \
                 patch.object(module, '_apply_channel', side_effect=AssertionError('Identity validation first')), self.assertRaises(ValueError):
                module.build(config, root / 'pool')
            self.assertFalse((root / 'pool').exists())

    def test_restricted_pickle_and_real_residual_channel_three_scenes(self):
        module = self.module()
        from experiments.phase1_external_dg.data import iq_record
        with self.assertRaises(pickle.UnpicklingError):
            module._read_numpy_pickle(io.BytesIO(pickle.dumps(eval, protocol=4)))
        _, receiver_for_session, stable_seed, _ = module._channel_api()
        n = np.arange(256)
        normalized = iq_record(np.column_stack((np.cos(.11 * n), np.sin(.11 * n))).astype(np.float64))
        for scene in SCENES:
            cfg = module._channel_config(scene)
            self.assertEqual((cfg.fs_hz, cfg.fc_hz, cfg.processing_route, cfg.mode, cfg.equalization_enabled),
                             (25e6, 2462e6, 'residual', 'post_sync', False))
            self.assertEqual((cfg.reference_snr_db, cfg.boundary), (30., 'reflect'))
            receiver = receiver_for_session(cfg, 2026102912, '24-1/2021_03_08/1')
            received, metadata = module._apply_channel(normalized, cfg, stable_seed(2026102913, scene, 'synthetic'), receiver)
            self.assertEqual((received.dtype, received.shape), (np.dtype('float32'), (2, 256)))
            self.assertTrue(np.isfinite(received).all())
            self.assertFalse(metadata['channel_equalization_applied'])
            self.assertFalse(metadata['full_channel_waveform_synthesized'])

    def test_cli_first_line_identifies_actual_process(self):
        module = self.module()
        import contextlib
        import os
        with io.StringIO() as output, contextlib.redirect_stdout(output), patch.object(module, 'build', return_value={
                'status': 'PHASE3_DEVELOPMENT_POOL_VALIDATED', 'discovery_count': 384, 'evaluation_count': 864}), \
             patch.object(module, '_read_json', return_value={}):
            module.main(['--config', 'fake.json', '--output', 'fake-output'])
            first = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(first['pid'], os.getpid())
        self.assertEqual(Path(first['cwd']), Path.cwd())
        self.assertEqual(first['argv'], ['--config', 'fake.json', '--output', 'fake-output'])


if __name__ == '__main__':
    unittest.main()
