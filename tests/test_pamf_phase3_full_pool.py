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
FOUND = ['11-4', '3-13']
FUTURE = ['12-20', '5-20']
SCENES = ['practical_low_suburban', 'practical_mid_urban', 'practical_low_urban']
MEMBERS = {
    '2021_03_08': ('pkl_wifi_eq_2021_03_08-20261028T143859Z-1-002.zip', 'pkl_wifi_eq_2021_03_08/dataset_2021_03_08_node24-18.pkl'),
    '2021_03_23': ('pkl_wifi_eq_2021_03_23-20261028T144307Z-1-005.zip', 'pkl_wifi_eq_2021_03_23/dataset_2021_03_23_node24-18.pkl'),
}


class FullPoolTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/full_pool.py'
        self.assertTrue(path.is_file(), 'Independent Full eq Phase3 pool builder is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.full_pool')

    def fixture(self, root, *, count=192, bad_shape=False, nan=False):
        full, validation = root / 'full', root / 'validation'
        full.mkdir()
        validation.mkdir()
        config = dict(validation_root=str(validation), full_root=str(full),
            source_receivers=['1-19', '18-2', '19-2', '2-19', '3-19'], receiver='24-18', receiver_index=30,
            registered_classes=OLD, discovery_classes=FOUND, future_unknown_classes=FUTURE,
            days=dict(discovery='2021_03_08', evaluation='2021_03_23'),
            per_scene=dict(discovery=32, evaluation=64), scenes=SCENES,
            seeds=dict(selection=2026102901, receiver=2026102902, channel=2026102903, shuffle=2026102904))
        summary = dict(status='VERIFIED', dataset_directory=str(full), members_checked=158,
                       expected_nonempty_RX_date_members=158, count_mismatch_cells=0, errors=[])
        (validation / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        locator = []
        names = OLD + FOUND + FUTURE
        for day, (archive, member) in MEMBERS.items():
            arrays = [np.full((count, 256, 2), i + 1., dtype=np.float64) for i in range(len(names))]
            if bad_shape:
                arrays[0] = arrays[0][:, :128]
            if nan:
                arrays[0][:] = np.nan
            with zipfile.ZipFile(full / archive, 'w', compression=zipfile.ZIP_DEFLATED) as stored:
                stored.writestr(member, pickle.dumps(dict(data=arrays, node_list=names), protocol=4))
            locator.append(dict(archive=archive, member=member, date=day, receiver='24-18', status='PASS'))
        (validation / 'members.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in locator), encoding='utf-8')
        return config

    def fake_channel(self, normalized, cfg, seed, receiver):
        self.assertEqual(normalized.dtype, np.dtype('float32'))
        self.assertAlmostEqual(float(np.mean(np.square(normalized[0]) + np.square(normalized[1]))), 1., places=6)
        return (normalized * 2).astype(np.float32), dict(scenario=cfg.scenario, seed=int(seed), processing_route='residual')

    def test_counts_private_truth_and_opaque_public_contract(self):
        module = self.module()
        from experiments.pamf_phase3_open_world.data import load_public, PUBLIC_FIELDS, PUBLIC_FILES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            with patch.object(module, '_apply_channel', side_effect=self.fake_channel) as calls:
                result = module.build(config, root / 'pool')
            self.assertEqual(calls.call_count, 768 + 1920)
            self.assertEqual(result['status'], 'PHASE3_FULL_POOL_VALIDATED')
            seen = set()
            for role, count, classes in [('discovery', 768, 8), ('evaluation', 1920, 10)]:
                public = root / 'pool' / role / 'public'
                self.assertEqual({p.name for p in public.iterdir()}, PUBLIC_FILES)
                manifest, ids, received = load_public(public)
                self.assertEqual(set(manifest), PUBLIC_FIELDS)
                self.assertEqual(received.shape, (count, 2, 256))
                self.assertEqual(received.dtype, np.dtype('float32'))
                self.assertEqual(manifest['registered_classes'], OLD)
                self.assertFalse(seen & set(ids.tolist()))
                seen.update(ids.tolist())
                self.assertTrue(all(re.fullmatch(r'p3f-[0-9a-f]{24}', value) for value in ids.tolist()))
                self.assertTrue(np.allclose(np.mean(received[:, 0] ** 2 + received[:, 1] ** 2, axis=1), 4.))
                with np.load(root / 'pool' / role / 'truth/truth.npz', allow_pickle=False) as truth:
                    self.assertEqual(set(truth.files), {'ids', 'y', 'receiver', 'scenario'})
                    self.assertTrue(np.array_equal(ids, truth['ids']))
                    self.assertEqual(set(truth['y'].tolist()), set(range(classes)))
                    self.assertTrue(bool((truth['receiver'] == 30).all()))
                    self.assertEqual(set(truth['scenario'].tolist()), set(SCENES))
                    per_cell = Counter(zip(truth['y'].tolist(), truth['scenario'].tolist()))
                    self.assertEqual(set(per_cell.values()), {config['per_scene'][role]})
                received._mmap.close()
            report = json.loads((root / 'pool/builder_report.json').read_text())
            records = [record for role in ('discovery', 'evaluation') for record in report['mapping'][role]]
            canonical = [tuple(record['canonical_record']) for record in records]
            self.assertEqual(len(canonical), len(set(canonical)))
            self.assertTrue(all(row[0] == 'FullWiSigEq' and row[2] == '24-18' and row[4] == 1 for row in canonical))
            self.assertIn('Full', report['receiver_index_namespace'])
            self.assertEqual(report['model_reads'], 0)
            self.assertTrue((root / 'pool/complete.json').is_file())

    def test_only_two_indexed_archives_are_opened_without_full_rescan(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            (root / 'full/unrelated-invalid.zip').write_bytes(b'not a zip')
            original = zipfile.ZipFile
            opened = []
            def selected(path, *args, **kwargs):
                if isinstance(path, (str, Path)):
                    opened.append(Path(path).name)
                return original(path, *args, **kwargs)
            with patch.object(module.zipfile, 'ZipFile', side_effect=selected), patch.object(module, '_apply_channel', side_effect=self.fake_channel), \
                 patch.object(Path, 'glob', side_effect=AssertionError('No archive directory rescan')):
                module.build(config, root / 'pool')
            self.assertEqual(opened, [MEMBERS[day][0] for day in ('2021_03_08', '2021_03_23')])

    def test_selection_and_channel_seeds_are_role_blind_and_use_original_indices(self):
        module = self.module()
        Config, receiver_for_session, stable_seed, _ = module._channel_api()
        rows = np.ones((1200, 256, 2), dtype=np.float64)
        selected = module._select_rows(rows, '11-4', '24-18', '2021_03_08', 32, SCENES, 2026102901)
        indices = [record['canonical_record'][-1] for record in selected]
        self.assertEqual(len(indices), 96)
        self.assertEqual(len(set(indices)), 96)
        self.assertTrue(any(index < 1000 for index in indices))
        self.assertTrue(any(index >= 1000 for index in indices))
        self.assertEqual([record['scene'] for record in selected], [SCENES[j % 3] for j in range(96)])
        for record in selected:
            expected = stable_seed(2026102903, record['scene'], *record['canonical_record'])
            self.assertEqual(module._channel_seed(2026102903, record['scene'], record['canonical_record']), expected)

    def test_identical_config_replays_ids_iq_and_private_mapping(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            with patch.object(module, '_apply_channel', side_effect=self.fake_channel):
                module.build(config, root / 'a')
                module.build(config, root / 'b')
            for role in ('discovery', 'evaluation'):
                for filename in ('ids.npy', 'received.npy'):
                    a = np.load(root / 'a' / role / 'public' / filename, allow_pickle=False)
                    b = np.load(root / 'b' / role / 'public' / filename, allow_pickle=False)
                    self.assertTrue(np.array_equal(a, b))
            a = json.loads((root / 'a/builder_report.json').read_text())
            b = json.loads((root / 'b/builder_report.json').read_text())
            self.assertEqual(a['mapping'], b['mapping'])

    def test_bad_shape_nan_and_insufficient_rows_never_complete(self):
        module = self.module()
        for kwargs in (dict(bad_shape=True), dict(nan=True), dict(count=95)):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.fixture(root, **kwargs)
                with patch.object(module, '_apply_channel', side_effect=AssertionError('Invalid raw data must not enter channel')), self.assertRaises(ValueError):
                    module.build(config, root / 'pool')
                self.assertFalse((root / 'pool/complete.json').exists())

    def test_wrong_indexed_member_and_output_collision_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            rows = [json.loads(line) for line in (root / 'validation/members.jsonl').read_text().splitlines()]
            rows[0]['member'] = 'wrong-member.pkl'
            (root / 'validation/members.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                module.build(config, root / 'pool')
            self.assertFalse((root / 'pool/complete.json').exists())
            (root / 'pool').mkdir(exist_ok=True)
            with patch.object(module, '_validated_members', side_effect=AssertionError('No input reads after collision')), self.assertRaises(FileExistsError):
                module.build(config, root / 'pool')

    def test_invalid_validation_or_config_is_not_revalidated_or_used(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            for key, value in [('receiver_index', 0), ('receiver', '1-19'), ('model', 'forbidden')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: value}))
            summary = json.loads((root / 'validation/summary.json').read_text())
            summary['count_mismatch_cells'] = 1
            (root / 'validation/summary.json').write_text(json.dumps(summary), encoding='utf-8')
            with patch.object(module.zipfile, 'ZipFile', side_effect=AssertionError('Invalid validation must not open archives')), self.assertRaises(ValueError):
                module.build(config, root / 'pool')

    def test_restricted_numpy_pickle_rejects_other_globals(self):
        module = self.module()
        with self.assertRaises((ValueError, pickle.UnpicklingError)):
            module._read_numpy_pickle(io.BytesIO(pickle.dumps(eval, protocol=4)))
        payload = dict(data=[np.ones((2, 256, 2), dtype=np.float64)], node_list=['14-10'])
        restored = module._read_numpy_pickle(io.BytesIO(pickle.dumps(payload, protocol=4)))
        self.assertEqual(restored['data'][0].dtype, np.dtype('float64'))

    def test_real_residual_channel_small_synthetic_iq(self):
        module = self.module()
        from experiments.phase1_external_dg.data import iq_record
        Config, receiver_for_session, stable_seed, _ = module._channel_api()
        n = np.arange(256)
        raw = np.column_stack((np.cos(.11 * n), np.sin(.11 * n))).astype(np.float64)
        normalized = iq_record(raw)
        for scene in SCENES:
            cfg = module._channel_config(scene)
            self.assertEqual(cfg.input_processing_state, 'WiSig_equalized1_center256_unit_rms')
            self.assertEqual((cfg.fs_hz, cfg.fc_hz, cfg.processing_route, cfg.mode, cfg.equalization_enabled),
                             (25e6, 2462e6, 'residual', 'post_sync', False))
            receiver = receiver_for_session(cfg, 2026102902, '24-18/2021_03_08/1')
            received, metadata = module._apply_channel(normalized, cfg, stable_seed(2026102903, scene, 'synthetic'), receiver)
            self.assertEqual(received.shape, (2, 256))
            self.assertEqual(received.dtype, np.dtype('float32'))
            self.assertTrue(np.isfinite(received).all())
            self.assertFalse(metadata['full_channel_waveform_synthesized'])
            self.assertFalse(metadata['channel_equalization_applied'])
            self.assertEqual(metadata['processing_route'], 'residual')

    def test_channel_failure_keeps_completion_closed(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            with patch.object(module, '_apply_channel', side_effect=RuntimeError('Synthetic channel failure')), self.assertRaises(RuntimeError):
                module.build(config, root / 'pool')
            self.assertFalse((root / 'pool/complete.json').exists())

    def test_completion_marker_is_last_and_iq_record_is_called_once(self):
        module = self.module()
        from experiments.phase1_external_dg import data as source_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.fixture(root)
            written = []
            original = module._write_json
            def record_write(path, value):
                self.assertFalse((root / 'pool/complete.json').exists())
                if Path(path).name == 'complete.json':
                    for role in ('discovery', 'evaluation'):
                        self.assertTrue((root / 'pool' / role / 'truth/truth.npz').is_file())
                        self.assertTrue((root / 'pool' / role / 'public/received.npy').is_file())
                    self.assertTrue((root / 'pool/builder_report.json').is_file())
                written.append(Path(path).name)
                return original(path, value)
            with patch.object(module, '_apply_channel', side_effect=self.fake_channel), \
                 patch.object(source_data, 'iq_record', wraps=source_data.iq_record) as normalization, \
                 patch.object(module, '_write_json', side_effect=record_write):
                module.build(config, root / 'pool')
            self.assertEqual(normalization.call_count, 2688)
            self.assertEqual(written[-1], 'complete.json')


if __name__ == '__main__':
    unittest.main()
