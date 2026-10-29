"""Synthetic-only checks for the fixed paired clean control boundary."""
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


OLD = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
NOVEL = ['19-15', '19-16', '7-1']
MEMBERS = {
    '2021_03_08': ('pkl_wifi_eq_2021_03_08-20261028T143859Z-1-002.zip',
                  'pkl_wifi_eq_2021_03_08/dataset_2021_03_08_node24-1.pkl'),
    '2021_03_15': ('pkl_wifi_eq_2021_03_15-20261028T144306Z-1-004.zip',
                  'pkl_wifi_eq_2021_03_15/dataset_2021_03_15_node24-1.pkl'),
}


class DevelopmentCleanTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/development_clean.py'
        self.assertTrue(path.is_file(), 'Paired clean development builder is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.development_clean')

    def fixture(self, root):
        full, paired = root / 'full', root / 'paired'
        full.mkdir(); paired.mkdir()
        cfg_path = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_development_pool_20261029.json'
        parent_config = json.loads(cfg_path.read_text(encoding='utf-8'))
        parent_config['full_root'] = str(full)
        parent_config['validation_root'] = str(root / 'already-validated')
        parent_config['compact_receiver_evidence'] = str(root / 'already-checked-compact.json')
        report = dict(schema='pamf_phase3_development_builder_v1', config=parent_config,
            development_only=True, validation_reused=dict(status='VERIFIED', members_checked=158,
            count_mismatch_cells=0), archive_reads=2, pickle_reads=2, formal_target_reads=0,
            canonical_physical_unique=True, scene_and_role_canonical_disjoint=True, mapping={})
        complete = dict(status='PHASE3_DEVELOPMENT_POOL_VALIDATED', data_status='VALIDATED_ONCE',
            development_only=True, discovery_count=384, evaluation_count=864,
            unique_canonical_count=1248, canonical_physical_unique=True, single_observation=True,
            source_development_formal_receivers_disjoint=True, phase2_development_receivers_disjoint=True,
            formal_target_reads=0, model_reads=0, model_predictions=0, optimizer_updates=0)
        for role, day, per_tx, txs in [('discovery', '2021_03_08', 48, OLD + NOVEL[:2]),
                                       ('evaluation', '2021_03_15', 96, OLD + NOVEL)]:
            archive, member = MEMBERS[day]
            arrays = []
            for tx in reversed(OLD + NOVEL):
                value = np.ones((194, 256, 2), dtype=np.float64)
                value[:, :, 0] = (OLD + NOVEL).index(tx) + 1 + np.arange(194)[:, None] / 1000
                value[0] = np.nan  # Unselected record must not be inspected or substituted.
                arrays.append(value)
            with zipfile.ZipFile(full / archive, 'w') as stored:
                stored.writestr(member, pickle.dumps(dict(data=arrays, node_list=list(reversed(OLD + NOVEL))), protocol=4))
                stored.writestr(member.replace('24-1.pkl', '24-18.pkl'), b'FORMAL MUST NOT OPEN')
            records = [dict(opaque_id=f'opaque-{role}-{j}-{i}',
                canonical_record=['FullWiSigEq', tx, '24-1', day, 1, 2 * i + 1],
                source_archive=archive, source_member=member, scene='practical_low_urban')
                for j, tx in enumerate(txs) for i in range(per_tx)]
            # Parent public order is deliberately unrelated to class/index order.
            records = list(reversed(records[::2])) + records[1::2]
            report['mapping'][role] = records
            ids = np.asarray([row['opaque_id'] for row in records])
            public, truth = paired / role / 'public', paired / role / 'truth'
            public.mkdir(parents=True); truth.mkdir()
            manifest = dict(schema='phase3_min_v1', data_status='VALIDATED_ONCE', capsule_id='star-' + role,
                split_id='star-split-' + role, count=len(ids), iq_shape=[len(ids), 2, 256],
                registered_classes=OLD, processing_route='residual', mode='post_sync',
                equalization_enabled=False, fs_hz=25000000, single_observation=True)
            (public / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            np.save(public / 'ids.npy', ids)
            (public / 'received.npy').write_bytes(b'EXISTING STAR IQ MUST NOT BE OPENED')
            np.savez(truth / 'truth.npz', ids=ids,
                y=np.asarray([(OLD + NOVEL).index(row['canonical_record'][1]) for row in records], dtype=np.int64),
                receiver=np.full(len(ids), 26, dtype=np.int64), scenario=np.full(len(ids), 'practical_low_urban'))
            complete[role] = dict(count=len(ids), capsule_id=manifest['capsule_id'], split_id=manifest['split_id'])
        (paired / 'complete.json').write_text(json.dumps(complete), encoding='utf-8')
        (paired / 'builder_report.json').write_text(json.dumps(report), encoding='utf-8')
        return dict(full_root=str(full), paired_pool=str(paired)), report

    def test_same_mapping_ids_order_and_real_iq_record_without_channel(self):
        module = self.module()
        from experiments.phase1_external_dg import data as source_data
        from experiments.pamf_phase3_open_world import development_pool, data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, report = self.fixture(root)
            with patch.object(source_data, 'iq_record', wraps=source_data.iq_record) as normalize, \
                 patch.object(development_pool, '_channel_api', side_effect=AssertionError('No channel')), \
                 patch.object(development_pool, '_apply_channel', side_effect=AssertionError('No channel')), \
                 patch.object(development_pool, '_select_rows', side_effect=AssertionError('No selection')):
                result = module.build(config, root / 'clean')
            self.assertEqual(normalize.call_count, 1248)
            self.assertEqual(result['status'], 'PHASE3_DEVELOPMENT_CLEAN_VALIDATED')
            self.assertEqual(result['unique_canonical_count'], 1248)
            for name in ('formal_target_reads', 'model_reads', 'model_predictions', 'optimizer_updates', 'channel_records'):
                self.assertEqual(result[name], 0)
            for role, count in [('discovery', 384), ('evaluation', 864)]:
                public = root / 'clean' / role / 'public'
                manifest, ids, iq = module.load_clean_public(public)
                self.assertEqual(set(p.name for p in public.iterdir()), data.PUBLIC_FILES)
                self.assertEqual(set(manifest), data.PUBLIC_FIELDS)
                self.assertEqual((manifest['schema'], manifest['processing_route'], manifest['mode']),
                                 ('phase3_clean_min_v1', 'clean', 'clean'))
                self.assertEqual(manifest['registered_classes'], OLD)
                self.assertEqual(iq.shape, (count, 2, 256)); self.assertFalse(iq.flags.writeable)
                self.assertTrue(np.array_equal(ids, np.load(Path(config['paired_pool']) / role / 'public/ids.npy')))
                self.assertNotEqual(manifest['capsule_id'], 'star-' + role)
                self.assertNotEqual(manifest['split_id'], 'star-split-' + role)
                self.assertTrue(np.allclose(np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1), 1., atol=2e-7))
                want_ratio = [(OLD + NOVEL).index(row['canonical_record'][1]) + 1 + row['canonical_record'][5] / 1000
                              for row in report['mapping'][role]]
                np.testing.assert_allclose(iq[:, 0, 0] / iq[:, 1, 0], want_ratio, rtol=2e-7)
                with np.load(root / 'clean' / role / 'truth/truth.npz', allow_pickle=False) as truth, \
                     np.load(Path(config['paired_pool']) / role / 'truth/truth.npz', allow_pickle=False) as parent:
                    self.assertEqual(set(truth.files), {'ids', 'y', 'receiver', 'scenario'})
                    for key in ('ids', 'y', 'receiver'): np.testing.assert_array_equal(truth[key], parent[key])
                    self.assertEqual(set(truth['scenario'].tolist()), {'clean'})
                self.assertNotIn('canonical', (public / 'manifest.json').read_text())
                self.assertFalse(any(tx in (public / 'manifest.json').read_text() for tx in NOVEL))
                iq._mmap.close()
                with self.assertRaises(ValueError): data.load_public(public)
            built = json.loads((root / 'clean/builder_report.json').read_text())
            self.assertEqual((built['archive_reads'], built['pickle_reads']), (2, 2))
            self.assertTrue(built['paired_original_not_new_records'])
            self.assertEqual(built['input_RMS_normalizations_per_record'], 1)
            self.assertFalse(built['full_archive_rescan'])

    def test_only_two_mapped_members_and_no_full_rescan(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, _ = self.fixture(root)
            original_open = zipfile.ZipFile.open; opened = []
            def record(stored, name, *args, **kwargs):
                if stored.mode == 'r' and str(stored.filename).endswith('.zip'):
                    opened.append((Path(stored.filename).name, name))
                return original_open(stored, name, *args, **kwargs)
            original_names = zipfile.ZipFile.namelist
            def no_archive_scan(stored):
                if str(stored.filename).endswith('.zip'):
                    raise AssertionError('No Full archive rescan')
                return original_names(stored)  # Private NPZ truth is an ordinary builder input.
            with patch.object(zipfile.ZipFile, 'open', record), \
                 patch.object(zipfile.ZipFile, 'namelist', no_archive_scan), \
                 patch.object(Path, 'glob', side_effect=AssertionError('No rescan')):
                module.build(config, root / 'clean')
            self.assertEqual(opened, list(MEMBERS.values()))

    def test_changed_parent_ids_or_canonical_rejected_before_archive_reads(self):
        module = self.module()
        for mutation in ('status', 'receiver', 'manifest_capsule', 'ids_order', 'duplicate_canonical',
                         'formal_canonical', 'bad_index', 'wrong_archive', 'wrong_member'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, report = self.fixture(root); paired = Path(config['paired_pool'])
                complete = json.loads((paired / 'complete.json').read_text())
                if mutation == 'status': complete['status'] = 'RUNNING'
                elif mutation == 'receiver': report['config']['receiver'] = '24-18'
                elif mutation == 'manifest_capsule': complete['discovery']['capsule_id'] = 'wrong'
                elif mutation == 'ids_order':
                    f = paired / 'discovery/public/ids.npy'; np.save(f, np.load(f)[::-1])
                elif mutation == 'duplicate_canonical':
                    report['mapping']['evaluation'][0]['canonical_record'] = report['mapping']['discovery'][0]['canonical_record']
                elif mutation == 'formal_canonical': report['mapping']['discovery'][0]['canonical_record'][2] = '24-18'
                elif mutation == 'bad_index': report['mapping']['discovery'][0]['canonical_record'][5] = -1
                elif mutation == 'wrong_archive': report['mapping']['discovery'][0]['source_archive'] = 'unrelated.zip'
                elif mutation == 'wrong_member': report['mapping']['discovery'][0]['source_member'] = 'formal.pkl'
                (paired / 'complete.json').write_text(json.dumps(complete), encoding='utf-8')
                (paired / 'builder_report.json').write_text(json.dumps(report), encoding='utf-8')
                original_open = zipfile.ZipFile.open
                def forbid_full(stored, name, *args, **kwargs):
                    if str(stored.filename).endswith('.zip'):
                        raise AssertionError('Validate parent before Full IQ')
                    return original_open(stored, name, *args, **kwargs)
                with patch.object(zipfile.ZipFile, 'open', forbid_full), \
                     self.assertRaises(ValueError): module.build(config, root / 'clean')
                self.assertFalse((root / 'clean').exists())

    def test_missing_selected_member_or_out_of_range_record_has_no_complete(self):
        module = self.module()
        for mutation in ('archive_missing', 'member_missing', 'out_of_range'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, report = self.fixture(root)
                archive, _ = MEMBERS['2021_03_15']; archive = Path(config['full_root']) / archive
                if mutation == 'archive_missing': archive.unlink()
                elif mutation == 'member_missing':
                    with zipfile.ZipFile(archive, 'w') as stored: stored.writestr('other', b'not selected')
                else:
                    report['mapping']['evaluation'][0]['canonical_record'][5] = 10000
                    (Path(config['paired_pool']) / 'builder_report.json').write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaises(ValueError): module.build(config, root / 'clean')
                self.assertFalse((root / 'clean/complete.json').exists())

    def test_config_does_not_allow_reselection_or_model_inputs(self):
        module = self.module()
        for key in ('registered_classes', 'seed', 'receiver', 'checkpoint', 'channel', 'validation_root'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_config(dict(full_root='full', paired_pool='paired', **{key: 'forbidden'}))

    def test_output_root_cannot_be_overwritten(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, _ = self.fixture(root); out = root / 'clean'; out.mkdir()
            (out / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError): module.build(config, out)
            self.assertEqual((out / 'keep.txt').read_text(), 'keep')

    def public_fixture(self, root):
        root.mkdir()
        manifest = dict(schema='phase3_clean_min_v1', data_status='VALIDATED_ONCE', capsule_id='clean-capsule',
            split_id='clean-split', count=2, iq_shape=[2, 2, 256], registered_classes=OLD,
            processing_route='clean', mode='clean', equalization_enabled=False, fs_hz=25000000,
            single_observation=True)
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        np.save(root / 'ids.npy', np.asarray(['opaque-a', 'opaque-b']))
        np.save(root / 'received.npy', np.ones((2, 2, 256), dtype=np.float32))
        return manifest

    def test_clean_loader_rejects_private_files_wrong_contract_and_bad_arrays(self):
        module = self.module()
        for mutation in ('truth_file', 'source_path', 'schema', 'route', 'mode', 'classes', 'dtype',
                         'shape', 'nonfinite', 'duplicate_ids', 'ids_count'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'public'; manifest = self.public_fixture(root)
                if mutation == 'truth_file': (root / 'truth.npz').write_bytes(b'private')
                elif mutation == 'source_path': manifest['source_root'] = 'private'
                elif mutation == 'schema': manifest['schema'] = 'phase3_min_v1'
                elif mutation == 'route': manifest['processing_route'] = 'residual'
                elif mutation == 'mode': manifest['mode'] = 'post_sync'
                elif mutation == 'classes': manifest['registered_classes'] = OLD[:-1] + NOVEL[:1]
                elif mutation == 'dtype': np.save(root / 'received.npy', np.ones((2, 2, 256), dtype=np.float64))
                elif mutation == 'shape': np.save(root / 'received.npy', np.ones((2, 2, 128), dtype=np.float32))
                elif mutation == 'nonfinite': np.save(root / 'received.npy', np.full((2, 2, 256), np.nan, dtype=np.float32))
                elif mutation == 'duplicate_ids': np.save(root / 'ids.npy', np.asarray(['same', 'same']))
                elif mutation == 'ids_count': np.save(root / 'ids.npy', np.asarray(['one']))
                (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError): module.load_clean_public(root)

    def test_cli_first_line_identifies_actual_process(self):
        module = self.module()
        with io.StringIO() as output, contextlib.redirect_stdout(output), \
             patch.object(module, '_read_json', return_value={}), patch.object(module, 'build', return_value=dict(
                 status='PHASE3_DEVELOPMENT_CLEAN_VALIDATED', discovery_count=384, evaluation_count=864)):
            module.main(['--config', 'synthetic.json', '--output', 'synthetic-output'])
            first = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(first['pid'], os.getpid()); self.assertEqual(Path(first['cwd']), Path.cwd())
        self.assertEqual(first['argv'], ['--config', 'synthetic.json', '--output', 'synthetic-output'])


if __name__ == '__main__':
    unittest.main()
