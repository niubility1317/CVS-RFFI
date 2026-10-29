"""Synthetic-only tests for the immutable G10 source-heldout builder."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class StarSourceNovelTests(unittest.TestCase):
    def module(self):
        self.assertTrue((ROOT/'experiments/pamf_phase3_open_world/star_source_novel.py').is_file(),
                        'G10 source novel builder is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.star_source_novel')

    def fixture(self, root):
        from experiments.pamf_phase3_open_world import source_oe_data as old
        from experiments.pamf_phase3_open_world.data import _identity
        full, parent = root/'full', root/'parent'
        full.mkdir(); parent.mkdir()
        old_config = dict(full_root=str(full), validation_root='already-validated', compact_memberships='already-verified',
            known_classes=old.KNOWN, reserved_target_classes=old.RESERVED, receivers=old.RECEIVERS, dates=old.DATES,
            exposure_train_classes=old.TRAIN, novel_validation_classes=old.NOVEL, per_cell=70, selection_seed=2026102905)
        members, mapping, payloads = [], [], []
        for day in old.DATES:
            for receiver in old.RECEIVERS:
                archive = f'pkl_wifi_eq_{day}-{receiver}.zip'
                member = f'pkl_wifi_eq_{day}/dataset_{day}_node{receiver}.pkl'
                members.append(dict(archive=archive, member=member, receiver=receiver, date=day))
                with zipfile.ZipFile(full/archive, 'w') as stored:
                    stored.writestr(member, b'SYNTHETIC PATCHED NUMPY PAYLOAD')
                    stored.writestr('not-a-permitted-member.pkl', b'DO NOT OPEN')
                arrays = []
                for j, tx in enumerate(old.NOVEL):
                    raw = np.ones((72, 256, 2), np.float64)
                    raw[:, :, 0] = j + 2 + np.arange(72)[:, None]/1000
                    raw[0] = np.nan  # Unselected rows are not scanned or reselected.
                    arrays.append(raw)
                    for index in range(1, 71):
                        canonical = ['FullWiSigEq', tx, receiver, day, 1, index]
                        opaque = 'p3oe-' + _identity(['source_oe_physical', canonical])
                        mapping.append(dict(opaque_id=opaque, canonical_record=canonical,
                                            source_archive=archive, source_member=member))
                # Reverse association order to require lookup by node_list, not class index.
                payloads.append(dict(data=arrays[::-1], node_list=old.NOVEL[::-1]))
        mapping.sort(key=lambda row: row['opaque_id'])
        ids = np.asarray([row['opaque_id'] for row in mapping])
        rx = np.asarray([old.MAIN_RX[old.RECEIVERS.index(row['canonical_record'][2])] for row in mapping], np.int64)
        day = np.asarray([old.DATES.index(row['canonical_record'][3])+1 for row in mapping], np.int64)
        domain = np.searchsorted(old.MAIN_RX, rx)*3 + day - 1
        public = parent/'novel_validation/public'; public.mkdir(parents=True)
        np.save(public/'ids.npy', ids)
        np.savez(public/'metadata.npz', rx_i=rx, day_i=day, domain=domain)
        (public/'iq.npy').write_bytes(b'OLD NORMALIZED IQ MUST NOT OPEN')
        truth = parent/'novel_validation/truth'; truth.mkdir()
        (truth/'truth.npz').write_bytes(b'OLD TRUTH MUST NOT OPEN')
        handle = dict(public_root=str(public), truth_path=str(truth/'truth.npz'), count=3150,
                      capsule_id='old-novel-capsule', split_id='old-novel-split',
                      data_status='VALIDATED_ONCE', iq_shape=[3150, 2, 256], iq_dtype='float32')
        complete = dict(status='SOURCE_OE_DATA_VALIDATED', data_status='VALIDATED_ONCE', source_only=True,
            counts=dict(exposure_train=4200, novel_validation=3150), novel_validation_count=3150,
            canonical_physical_unique=True, novel_validation=handle, config=old_config,
            target_reads=0, original_L_s_reads=0, original_U_s_reads=0, original_V_reads=0,
            model_reads=0, model_predictions=0, optimizer_updates=0)
        report = dict(schema='pamf_phase3_source_oe_data_v1', config=old_config, canonical_physical_unique=True,
                      members=members, mapping=dict(novel_validation=mapping))
        (parent/'complete.json').write_text(json.dumps(complete), encoding='utf-8')
        (parent/'builder_report.json').write_text(json.dumps(report), encoding='utf-8')
        mixture = root/'mixture.json'
        mixture.write_text((ROOT/'configs/pamf_phase2_practical_mixture_20261028.json').read_text(encoding='utf-8'), encoding='utf-8')
        config = dict(run_id='20261029-phase3-star-source-novel-g10-r01', parent_root=str(parent),
            full_root=str(full), mixture_config_path=str(mixture),
            episode=dict(id='MIX01', scene_seed=2026103001, channel_seed=2026104001), code_commit='a'*40)
        return config, report, payloads

    @staticmethod
    def observer(raw, **kwargs):
        assert raw.dtype == np.float64 and raw.shape == (2, 256)
        assert set(kwargs) == {'physical_id', 'receiver', 'day', 'episode', 'mixture'}
        assert np.sqrt(np.mean(np.sum(raw**2, axis=0))) > 2  # No earlier RMS.
        value = raw.astype(np.float32)
        value /= np.sqrt(np.mean(np.sum(value.astype(np.float64)**2, axis=0)))
        return value*.7, dict(scenario='practical_low_urban', overlay_seed=7, input_rms=2.5, geometry={})

    def test_build_preserves_exact_ids_order_domains_and_raw_once_observation(self):
        m = self.module()
        from experiments.pamf_phase3_open_world.star_channel import identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, parent, payloads = self.fixture(root)
            calls = []; original = np.load
            def guarded(path, *args, **kwargs):
                if Path(path).is_relative_to(Path(config['parent_root'])):
                    self.assertIn(Path(path).name, {'ids.npy', 'metadata.npz'})
                return original(path, *args, **kwargs)
            def observe(raw, **kwargs):
                calls.append((raw.copy(), kwargs)); return self.observer(raw, **kwargs)
            with patch.object(m, '_read_numpy_pickle', side_effect=payloads), patch.object(m, '_observe', side_effect=observe), \
                 patch.object(np, 'load', side_effect=guarded):
                done = m.build(config, root/'out')
            self.assertEqual(done['status'], 'STAR_SOURCE_NOVEL_VALIDATED')
            self.assertEqual((done['count'], done['new_observation_count']), (3150, 3150))
            self.assertEqual(len(calls), 3150)
            manifest, ids, received, meta = m.load_public(root/'out/public', done['capsule_id'], done['split_id'])
            expected = parent['mapping']['novel_validation']
            np.testing.assert_array_equal(ids, [row['opaque_id'] for row in expected])
            self.assertEqual(received.dtype, np.float32); self.assertFalse(received.flags.writeable)
            self.assertEqual(set(meta), {'rx_i', 'day_i', 'domain'})
            self.assertEqual(set((root/'out/public').iterdir()), {root/'out/public'/name for name in m.PUBLIC_FILES})
            self.assertEqual(manifest['role'], 'source_heldout_validation'); self.assertTrue(manifest['source_only'])
            self.assertEqual(set(manifest), m.PUBLIC_FIELDS)
            np.testing.assert_allclose(np.mean(np.sum(received.astype(np.float64)**2, axis=1), axis=1), .49, rtol=3e-7)
            received._mmap.close()
            report = json.loads((root/'out/builder_report.json').read_text())
            self.assertEqual([row['canonical_record'] for row in report['mapping']], [row['canonical_record'] for row in expected])
            self.assertEqual((report['archive_reads'], report['pickle_reads']), (15, 15))
            physical = {identity(['WiSig/Full/eq1', r['canonical_record'][1], r['canonical_record'][2],
                                 r['canonical_record'][3], r['canonical_record'][5]]): r['canonical_record'] for r in expected}
            for raw, kwargs in calls:
                self.assertEqual(set(kwargs), {'physical_id', 'receiver', 'day', 'episode', 'mixture'})
                _, tx, rx, day, _, index = physical[kwargs['physical_id']]
                self.assertEqual((kwargs['receiver'], kwargs['day']), (rx, day))
                self.assertEqual(raw[0, 0], ['19-15', '19-16', '7-1'].index(tx)+2+index/1000)
            self.assertEqual({kw['physical_id'] for _, kw in calls},
                {identity(['WiSig/Full/eq1', r['canonical_record'][1], r['canonical_record'][2], r['canonical_record'][3], r['canonical_record'][5]]) for r in expected})
            for key in ('target_reads', 'formal_target_reads', 'exposure_train_reads', 'original_L_s_reads',
                        'original_U_s_reads', 'original_V_reads', 'model_reads', 'model_predictions', 'optimizer_updates'):
                self.assertEqual(done[key], 0)
            self.assertFalse(report['old_normalized_IQ_read']); self.assertFalse(report['old_truth_read'])
            with np.load(root/'out/truth/truth.npz', allow_pickle=False) as truth:
                self.assertEqual(set(truth.files), {'ids', 'tx', 'rx_i', 'day_i', 'scenario'})
                np.testing.assert_array_equal(truth['ids'], ids)
                np.testing.assert_array_equal(truth['tx'], [r['canonical_record'][1] for r in expected])
                np.testing.assert_array_equal(truth['rx_i'], meta['rx_i'])
                self.assertTrue(np.all(truth['scenario'] == 'practical_low_urban'))

    def test_rejects_controls_parent_mapping_and_metadata_before_raw_open(self):
        m = self.module()
        for mutation in ('config_label', 'episode', 'commit', 'status', 'full_root', 'ids_order', 'canonical', 'locator', 'metadata_label', 'metadata_domain'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, report, _ = self.fixture(root); parent = Path(config['parent_root'])
                if mutation == 'config_label': config['true_unknown_count'] = 3
                elif mutation == 'episode': config['episode']['channel_seed'] += 1
                elif mutation == 'commit': config['code_commit'] = 'unknown'
                elif mutation == 'status':
                    path = parent/'complete.json'; value=json.loads(path.read_text()); value['status']='PARTIAL'; path.write_text(json.dumps(value))
                elif mutation == 'full_root': config['full_root'] = str(root/'other')
                elif mutation == 'ids_order':
                    path=parent/'novel_validation/public/ids.npy'; np.save(path,np.load(path)[::-1])
                elif mutation == 'canonical': report['mapping']['novel_validation'][0]['canonical_record'][2]='24-18'
                elif mutation == 'locator': report['mapping']['novel_validation'][0]['source_member']='wrong.pkl'
                else:
                    path=parent/'novel_validation/public/metadata.npz'
                    with np.load(path) as stored: metadata={k:stored[k] for k in stored.files}
                    if mutation == 'metadata_label': metadata['tx']=np.full(3150,'not-public')
                    else: metadata['domain'][0]=99
                    np.savez(path, **metadata)
                (parent/'builder_report.json').write_text(json.dumps(report))
                with patch.object(m,'_read_numpy_pickle',side_effect=AssertionError('No raw before handles')), self.assertRaises(ValueError):
                    m.build(config, root/'out')
                self.assertFalse((root/'out').exists())

    def test_only_fifteen_exact_members_no_rescan_selection_or_oe_four(self):
        m = self.module()
        from experiments.pamf_phase3_open_world import source_oe_data as old
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config, report, payloads=self.fixture(root); opened=[]
            original, names = zipfile.ZipFile.open, zipfile.ZipFile.namelist
            def read(stored, name, *args, **kwargs):
                if str(stored.filename).endswith('.zip'):
                    opened.append((Path(stored.filename).name,name))
                return original(stored,name,*args,**kwargs)
            def no_full_scan(stored):
                if str(stored.filename).endswith('.zip'):
                    raise AssertionError('No Full ZIP scan')
                return names(stored)
            with patch.object(zipfile.ZipFile,'open',read), patch.object(zipfile.ZipFile,'namelist',no_full_scan), \
                 patch.object(old,'_indices',side_effect=AssertionError('No reselection')), patch.object(old,'_selected',side_effect=AssertionError('No OE role reads')), \
                 patch.object(m,'_read_numpy_pickle',side_effect=payloads), patch.object(m,'_observe',side_effect=self.observer):
                m.build(config, root/'out')
            self.assertEqual(opened, [(r['archive'],r['member']) for r in report['members']])

    def test_bad_selected_packet_and_existing_output_never_complete_or_overwrite(self):
        m = self.module()
        for mutation in ('index', 'dtype', 'nonfinite', 'output'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); config, _, payloads=self.fixture(root); output=root/'out'
                if mutation == 'index': payloads[0]['data'][0]=payloads[0]['data'][0][:60]
                elif mutation == 'dtype': payloads[0]['data'][0]=payloads[0]['data'][0].astype(np.float32)
                elif mutation == 'nonfinite': payloads[0]['data'][0][1]=np.nan
                else: output.mkdir(); (output/'keep').write_text('history')
                with patch.object(m,'_read_numpy_pickle',side_effect=payloads), patch.object(m,'_observe',side_effect=self.observer), \
                     self.assertRaises((ValueError, FileExistsError)):
                    m.build(config, output)
                self.assertFalse((output/'complete.json').exists())
                if mutation == 'output': self.assertEqual((output/'keep').read_text(),'history')

    def test_public_loader_rejects_truth_leaks_wrong_handles_and_partial_fields(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config, _, payloads=self.fixture(root)
            with patch.object(m,'_read_numpy_pickle',side_effect=payloads), patch.object(m,'_observe',side_effect=self.observer): done=m.build(config,root/'out')
            public=root/'out/public'; cap, split=done['capsule_id'], done['split_id']
            with self.assertRaises(ValueError): m.load_public(public, 'wrong-capsule', split)
            with self.assertRaises(ValueError): m.load_public(public, cap, 'wrong-split')
            leak=public/'truth.npz'; leak.write_bytes(b'not public')
            with self.assertRaises(ValueError): m.load_public(public, cap, split)
            leak.unlink(); path=public/'manifest.json'; manifest=json.loads(path.read_text()); manifest['tx']=['secret']
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): m.load_public(public, cap, split)

    def test_checked_in_configuration_and_cli_use_fixed_source_only_contract(self):
        m = self.module()
        path=ROOT/'configs/pamf_phase3_star_source_novel_20261029.json'
        self.assertTrue(path.is_file())
        m.validate_config(json.loads(path.read_text(encoding='utf-8')))
        with io.StringIO() as text, contextlib.redirect_stdout(text), patch.object(m,'_read_json',return_value={}), \
             patch.object(m,'build',return_value=dict(status='STAR_SOURCE_NOVEL_VALIDATED',count=3150)):
            self.assertEqual(m.main(['--config','synthetic.json','--output','synthetic-out']),0)
            first=json.loads(text.getvalue().splitlines()[0])
        import os
        self.assertEqual(first['pid'],os.getpid()); self.assertEqual(first['argv'],['--config','synthetic.json','--output','synthetic-out'])


if __name__ == '__main__': unittest.main()
