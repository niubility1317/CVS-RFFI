import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from experiments.pamf_phase3_open_world import data
except ImportError:
    data = None


KNOWN = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
UNKNOWN = ['11-4', '3-13']
SCENES = ['practical_high', 'practical_mid', 'practical_low_urban']
PUBLIC_FIELDS = {
    'schema', 'data_status', 'capsule_id', 'split_id', 'count', 'iq_shape',
    'registered_classes', 'processing_route', 'mode', 'equalization_enabled',
    'fs_hz', 'single_observation',
}


def historical_fixture(root):
    capsule = root / 'legacy' / 'capsule'
    capsule.mkdir(parents=True)
    (capsule.parent / 'score_only').mkdir()
    rows, arrays = [], []
    for receiver in ['1-1', '14-7']:
        for tx in KNOWN + UNKNOWN:
            serial = 0
            for scene in SCENES:
                for role, count in [('support_pool', 2), ('query', 3)]:
                    for _ in range(count):
                        sid = f'physical-{len(rows):05d}'
                        rows.append(dict(index=len(rows), physical_id=sid,
                            transmitter=tx, receiver=receiver, day='D0',
                            signal_index=serial, scene=scene, pool_role=role,
                            old=tx in KNOWN, overlay_seed=2027 + serial))
                        arrays.append(np.full((2, 256), len(rows) / 1000, np.float32))
                        serial += 1
    np.savez(capsule / 'received.npz', iq=np.stack(arrays),
             ids=np.asarray([r['physical_id'] for r in rows]))
    manifest = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
        capsule_id='fixture-legacy', received_count=len(rows), signal_shape=[2, 256],
        scenarios=SCENES, channel=dict(route='residual', mode='post_sync',
            equalization_enabled=False, fs_hz=25000000),
        checks=dict(unique_physical_observation=True, scenes_physically_disjoint=True,
            support_query_physically_disjoint=True, query_labels_absent=True,
            source_paths_absent=True, finite_received=True))
    (capsule / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    ledger = {r['physical_id']: r for r in rows}
    (capsule.parent / 'score_only' / 'truth.json').write_text(json.dumps(ledger), encoding='utf-8')
    (capsule.parent / 'builder_report.json').write_text(json.dumps(dict(
        config=dict(old_classes=KNOWN, new_classes=UNKNOWN,
            source_receivers=['1-19', '18-2', '19-2', '2-19', '3-19'],
            target_receivers=['1-1', '14-7']), manifest=manifest)), encoding='utf-8')
    return capsule, ledger


class Phase3DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, 'Phase3 data conversion module is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.ledger = historical_fixture(self.root)

    def build(self, name='converted', **kwargs):
        return data.build_from_practical_capsule(self.source, self.root / name,
            KNOWN, UNKNOWN, [0, 2], **kwargs)

    def test_query_only_conversion_preserves_iq_and_exposes_only_public_inputs(self):
        result = self.build()
        info = result['evaluation']
        public = Path(info['public_root'])
        manifest, ids, iq = data.load_public(public)
        self.addCleanup(iq._mmap.close)
        self.assertEqual(set(manifest), PUBLIC_FIELDS)
        self.assertEqual(set(p.name for p in public.iterdir()), {'manifest.json', 'ids.npy', 'received.npy'})
        self.assertEqual(manifest['registered_classes'], KNOWN)
        self.assertEqual(manifest['count'], 2 * 8 * 3 * 3)
        self.assertIsInstance(iq, np.memmap)
        self.assertEqual(iq.dtype, np.dtype('float32'))
        self.assertEqual(iq.shape, (manifest['count'], 2, 256))
        self.assertTrue(ids.dtype.kind == 'U')
        self.assertEqual(len(set(ids.tolist())), len(ids))
        self.assertFalse(set(ids.tolist()) & set(self.ledger))
        with np.load(info['truth_path'], allow_pickle=False) as truth:
            np.testing.assert_array_equal(ids, truth['ids'])
            self.assertEqual(set(truth['y'].tolist()), set(range(8)))
            self.assertEqual(set(truth['receiver'].tolist()), {0, 2})
            self.assertIn(truth['receiver'].dtype.kind, 'iu')
            self.assertEqual(set(truth['scenario'].tolist()), set(SCENES))
        report = json.loads(Path(result['builder_report']).read_text(encoding='utf-8'))
        self.assertTrue(report['historical_exposed'])
        self.assertTrue(report['non_blind'])
        self.assertTrue(report['canonical_physical_unique'])
        self.assertEqual({r['source_role'] for r in report['evaluation_mapping']}, {'query'})
        original = np.load(self.source / 'received.npz', allow_pickle=False)
        by_id = {str(sid): original['iq'][i] for i, sid in enumerate(original['ids'])}
        for i, entry in enumerate(report['evaluation_mapping']):
            self.assertEqual(str(ids[i]), entry['opaque_id'])
            np.testing.assert_array_equal(iq[i], by_id[entry['source_physical_id']])
        original.close()

    def test_seed_is_repeatable_and_storage_is_not_label_blocked(self):
        first, second = self.build('first'), self.build('second')
        _, a, _ = data.load_public(first['evaluation']['public_root'])
        _, b, _ = data.load_public(second['evaluation']['public_root'])
        np.testing.assert_array_equal(a, b)
        with np.load(first['evaluation']['truth_path'], allow_pickle=False) as truth:
            self.assertFalse(np.array_equal(truth['y'], np.sort(truth['y'])))

    def test_loader_ignores_external_truth_and_rejects_extra_manifest_fields(self):
        result = self.build()
        public = Path(result['evaluation']['public_root'])
        Path(result['evaluation']['truth_path']).write_bytes(b'not a valid NPZ; must never be read by loader')
        data.load_public(public)
        path = public / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['unknown_classes'] = UNKNOWN
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest|field|public'):
            data.load_public(public)

    def test_loader_rejects_duplicate_ids_nonfinite_and_wrong_shape(self):
        result = self.build()
        public = Path(result['evaluation']['public_root'])
        ids = np.load(public / 'ids.npy', allow_pickle=False)
        ids[1] = ids[0]
        np.save(public / 'ids.npy', ids)
        with self.assertRaisesRegex(ValueError, 'ID|id|duplicate'):
            data.load_public(public)
        second = self.build('nonfinite')
        public = Path(second['evaluation']['public_root'])
        iq = np.load(public / 'received.npy', mmap_mode='r+')
        iq[0, 0, 0] = np.nan
        iq.flush()
        del iq
        with self.assertRaisesRegex(ValueError, 'finite'):
            data.load_public(public)
        third = self.build('shape')
        public = Path(third['evaluation']['public_root'])
        np.save(public / 'received.npy', np.zeros((1, 2, 255), np.float32))
        with self.assertRaisesRegex(ValueError, 'shape|count'):
            data.load_public(public)

    def test_source_role_or_scene_alias_is_rejected_before_output(self):
        rows = list(self.ledger.values())
        rows[1]['signal_index'] = rows[0]['signal_index']
        path = self.source.parent / 'score_only' / 'truth.json'
        path.write_text(json.dumps(self.ledger), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'physical|canonical'):
            self.build()
        self.assertFalse((self.root / 'converted').exists())

    def test_invalid_class_lists_or_absent_unknown_fail_without_output(self):
        for known, unknown in [(KNOWN, [KNOWN[0]]), (KNOWN[:-1], UNKNOWN),
                               (KNOWN, ['missing-device'])]:
            with self.subTest(known=known, unknown=unknown), self.assertRaises(ValueError):
                data.build_from_practical_capsule(self.source, self.root / 'bad', known, unknown, [0])
        self.assertFalse((self.root / 'bad').exists())

    def test_existing_output_is_never_overwritten(self):
        self.build()
        marker = self.root / 'converted' / 'marker'
        marker.write_text('preserve', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve')

    def test_unverified_historical_discovery_is_refused_but_evaluation_remains_valid(self):
        result = self.build(include_discovery=True)
        self.assertEqual(result['discovery_status'], 'REFUSED_HISTORICAL_QUERY_PROVENANCE_UNVERIFIED')
        self.assertIsNone(result['discovery'])
        self.assertFalse((self.root / 'converted' / 'discovery').exists())
        data.load_public(result['evaluation']['public_root'])


if __name__ == '__main__':
    unittest.main()
