import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


OLD = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
METHODS = ['msp', 'maxlogit', 'margin', 'energy', 'cosine', 'linear_oe', 'mlp16_oe']
READOUT_RUN = '20261029-phase3-source-embedding-readout-s2026092701-r01'
RISK_RUN = '20261029-phase3-source-baseline-risk-s2026092701-r01'
ENCODER_RUN = '20261029-phase3-source-embedding-increment-s2026092701-r02'


class FakeReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.register_buffer('fixed', torch.zeros(1, dtype=torch.float32))
        self.seen = []
        self.eval().requires_grad_(False)

    def features(self, x):
        self.seen.extend(x[:, 0, 0].tolist())
        out = torch.zeros((len(x), 160), dtype=torch.float32, device=x.device)
        out[:, 0] = x[:, 0, 0]
        return out

    def classify_features(self, features):
        labels = features[:, 0].long() % 9
        logits = torch.zeros((len(features), 6), dtype=torch.float32, device=features.device)
        logits[torch.arange(len(logits)), labels % 6] = torch.where(labels < 6, 3., .5)
        return logits


class FakeDura(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('offset', torch.tensor(.25, dtype=torch.float32))
        self.eval().requires_grad_(False)

    def forward(self, x):
        return x + self.offset


class DevelopmentTransferTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/development_transfer.py'
        self.assertTrue(path.is_file(), 'Frozen development transfer is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.development_transfer')

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def fixture(self, root):
        from experiments.pamf_phase3_open_world import source_open_readout as core
        from experiments.pamf_phase3_open_world import source_embedding_readout_run as readout
        from experiments.pamf_phase3_open_world.data import PUBLIC_FIELDS
        row = root / READOUT_RUN / 'rows/uniform_oe'
        risk = root / RISK_RUN / 'rows/uniform_oe/predict'
        identity = dict(arm='uniform_oe', run_id=ENCODER_RUN, row_id='uniform_oe-s2026092701',
            code_commit='28d53ea9b38c19f642c59e31f178a3f65f40894d', updates=222, cumulative_updates=11322,
            checkpoint=str(root / ENCODER_RUN / 'rows/uniform_oe/train/final.pt'))
        export = dict(status='SOURCE_EMBEDDING_BRANCH_FEATURES_EXPORTED', source_only=True,
            arm='uniform_oe', counts=readout.parent.COUNTS, feature_refs=readout.parent.FEATURE_REFS,
            model_classes=6, parameters=232993, head_changed=False, default_head_replaced=False,
            encoder_state_identity=identity, parent_export_identity={'fixed': 'original F6'},
            source_OE_handles={'fixed': 'original four OE'}, config={'row_id': 'uniform_oe-s2026092701'})
        export.update({key: 0 for key in ('target_reads', 'original_U_s_reads', 'OE_truth_reads',
            'optimizer_updates', 'new_encoder_updates', 'new_identity_head_updates')})
        self.write(row / 'export/complete.json', export)
        fit = dict(status='SOURCE_EMBEDDING_READOUTS_FROZEN', export_identity=readout._identity(export),
            encoder_state_identity=identity, descriptor_schema=readout.consensus.DESCRIPTOR_SCHEMA,
            readout_training_regime=readout.REGIME, geometry_fitted=True, source_only=True,
            readout_steps_per_model=200, method_names=METHODS)
        fit.update({key: 0 for key in ('target_reads', 'new_encoder_updates', 'new_identity_head_updates',
            'source_validation_reads', 'novel_validation_reads', 'OE_truth_reads')})
        self.write(row / 'fit/complete.json', fit)
        geometry = dict(schema=core.GEOMETRY_SCHEMA, class_count=6, branches={})
        for name, dim in core.DIMENSIONS.items():
            geometry['branches'][name] = dict(mean=np.zeros(dim), scale=np.ones(dim),
                prototypes=np.zeros((6, dim)), whitener=np.eye(dim))
        state = dict(schema=core.READOUT_SCHEMA, descriptor_dim=100, method_names=list(core.METHODS),
            mean=np.zeros(100), scale=np.ones(100), training={}, models={
                'linear_oe': dict(weight=np.zeros((1, 100), np.float32), bias=np.zeros(1, np.float32)),
                'mlp16_oe': dict(weight1=np.zeros((16, 100), np.float32), bias1=np.zeros(16, np.float32),
                    weight2=np.zeros((1, 16), np.float32), bias2=np.zeros(1, np.float32))})
        state['models']['linear_oe']['weight'][0, 97] = 1
        state['models']['mlp16_oe']['weight1'][0, 97] = 1
        state['models']['mlp16_oe']['weight2'][0, 0] = 1
        readout._save(row / 'fit', geometry, state)
        risk_marker = dict(status='SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE', source_only=True,
            arm='uniform_oe', target_reads=0, unknown_truth_read=False, truth_read=False, query_fit=False,
            default_head_replaced=False, method_names=METHODS, source_V_labels_read=True,
            source_known_label_calibration=True, encoder_state_identity=identity, encoder_updates=222,
            identity_head_updates=222, new_encoder_updates=0, new_identity_head_updates=0, new_readout_updates=0,
            readout_steps_per_model=200, known_count=27000, novel_count=3150, count=30150,
            run_id=RISK_RUN, parent_row_root=str(row), calibration_schema='phase3_source_original_baseline_risk_v1',
            descriptor_schema=readout.consensus.DESCRIPTOR_SCHEMA, readout_training_regime=readout.REGIME,
            thresholds_ref='calibration/thresholds.npy')
        self.write(risk / 'complete.json', risk_marker)
        calibration = dict(status='SOURCE_ORIGINAL_BASELINE_RISK_THRESHOLDS_FROZEN',
            schema='phase3_source_original_baseline_risk_v1', calibration_schema='phase3_source_original_baseline_risk_v1',
            source_only=True, method_names=METHODS, known_count=27000, class_count=6,
            source_V_labels_read=True, source_known_label_calibration=True, thresholds_ref='thresholds.npy')
        calibration.update({key: 0 for key in ('target_reads', 'unknown_calibration_scores_read',
            'unknown_calibration_labels_read', 'novel_validation_reads', 'new_encoder_updates',
            'new_identity_head_updates', 'new_readout_updates')})
        self.write(risk / 'calibration/complete.json', calibration)
        np.save(risk / 'calibration/thresholds.npy', np.asarray([.5, 1.5, 1.5, 2., -1., 1.5, 1.5]), allow_pickle=False)
        public = root / 'development/evaluation/public'
        public.mkdir(parents=True)
        n = 18
        ids = np.asarray([f'p3dev-{index:024x}' for index in range(n)])
        iq = np.zeros((n, 2, 256), dtype=np.float32)
        iq[:, 0, :] = np.arange(n)[:, None]
        manifest = dict(schema='phase3_min_v1', data_status='VALIDATED_ONCE', capsule_id='p3dev-capsule',
            split_id='dev-split', count=n, iq_shape=[n, 2, 256], registered_classes=OLD,
            processing_route='residual', mode='post_sync', equalization_enabled=False,
            fs_hz=25000000, single_observation=True)
        self.assertEqual(set(manifest), PUBLIC_FIELDS)
        self.write(public / 'manifest.json', manifest)
        np.save(public / 'ids.npy', ids, allow_pickle=False)
        np.save(public / 'received.npy', iq, allow_pickle=False)
        truth = root / 'private/truth.npz'
        truth.parent.mkdir()
        scenes = np.asarray(['practical_low_suburban', 'practical_mid_urban', 'practical_low_urban'] * 6)
        np.savez(truth, ids=ids[::-1], y=(np.arange(n) % 9)[::-1],
                 receiver=np.full(n, 26, np.int64), scenario=scenes[::-1])
        config = dict(run_id='development-transfer-test', code_commit='a' * 40,
            public_root=str(public), capsule_id=manifest['capsule_id'], split_id=manifest['split_id'],
            expected_count=n, source_row_root=str(row), risk_prediction_root=str(risk),
            output_root=str(root / 'predictions'), input_paths=['P0', 'P1'], batch_size=128,
            view_family='practical_cross_receiver', dura_checkpoint=str(root / 'original-r69/final.pt'),
            dura_release=str(root / 'frozen-G-release'))
        return config, truth

    def fake_encode(self, reader, x):
        from experiments.pamf_phase3_open_world.source_open_readout import DIMENSIONS
        z = reader.features(x)
        logits = reader.classify_features(z)
        data = {'features__' + name: np.zeros((len(x), dim), np.float32) for name, dim in DIMENSIONS.items()}
        data['features__fused160'] = np.asarray(z.tolist(), np.float32)
        data['logits'] = np.asarray(logits.tolist(), np.float32)
        return data

    def clean_fixture(self, root):
        config, _ = self.fixture(root)
        star_public = Path(config['public_root'])
        public = root / 'clean-development/evaluation/public'
        public.mkdir(parents=True)
        count = 36
        ids = np.asarray([f'p3dev-{index:024x}' for index in range(count)])
        iq = np.zeros((count, 2, 256), np.float32)
        iq[:, 0, :] = np.arange(count)[:, None]
        manifest = json.loads((star_public / 'manifest.json').read_text())
        manifest.update(schema='phase3_clean_min_v1', capsule_id='p3dev-clean-capsule',
            split_id='clean-dev-split', count=count, iq_shape=[count, 2, 256],
            processing_route='clean', mode='clean')
        self.write(public / 'manifest.json', manifest)
        np.save(public / 'ids.npy', ids, allow_pickle=False)
        np.save(public / 'received.npy', iq, allow_pickle=False)
        truth = root / 'private-clean/truth.npz'
        truth.parent.mkdir()
        np.savez(truth, ids=ids[::-1], y=(np.arange(count) % 9)[::-1],
            receiver=np.full(count, 26, np.int64), scenario=np.full(count, 'clean'))
        config.update(public_root=str(public), capsule_id=manifest['capsule_id'], split_id=manifest['split_id'],
            expected_count=count, view_family='clean_cross_receiver', input_paths=['C0'],
            dura_checkpoint=None, dura_release=None)
        return config, truth, star_public

    def test_clean_config_accepts_only_isolated_C0_without_DURA_or_paired_reference(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, _, star_public = self.clean_fixture(Path(tmp))
            self.assertIs(module.validate_config(config), config)
            for key, value in [('input_paths', ['P0', 'P1']), ('input_paths', ['C0', 'P1']),
                    ('input_paths', ['P0']), ('dura_checkpoint', 'a-frozen-G.pt'),
                    ('dura_release', 'frozen-G-code'), ('paired_reference', str(star_public)),
                    ('clean_reference', str(star_public)), ('star_public_root', str(star_public))]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: value}))

    def test_clean_public_uses_independent_loader_without_opening_star(self):
        module = self.module()
        from experiments.pamf_phase3_open_world import development_clean
        with tempfile.TemporaryDirectory() as tmp:
            config, _, star_public = self.clean_fixture(Path(tmp))
            numpy_load = np.load
            def only_clean(path, *args, **kwargs):
                if isinstance(path, (str, Path)):
                    self.assertFalse(Path(path).resolve().is_relative_to(star_public.resolve()),
                                     'C0 must never open a star public artifact')
                return numpy_load(path, *args, **kwargs)
            with patch.object(module, 'load_public', side_effect=AssertionError('No residual public loader for C0')), \
                 patch.object(development_clean, 'load_clean_public', wraps=development_clean.load_clean_public) as clean_load, \
                 patch.object(development_clean, '_parent', side_effect=AssertionError('No paired builder access')), \
                 patch.object(np, 'load', side_effect=only_clean):
                manifest, ids, values = module._public(config)
            try:
                clean_load.assert_called_once_with(config['public_root'])
                self.assertEqual(manifest['schema'], 'phase3_clean_min_v1')
                self.assertEqual((manifest['processing_route'], manifest['mode']), ('clean', 'clean'))
                self.assertEqual(values.shape, (36, 2, 256))
                self.assertFalse(values.flags.writeable)
                self.assertEqual(len(ids), 36)
            finally:
                values._mmap.close()

    def test_clean_frozen_C0_prediction_and_score_use_95_and_clean_scene_only(self):
        module = self.module()
        from experiments.pamf_phase3_open_world import development_clean
        from experiments.pamf_phase3_open_world import source_open_readout as core, source_baseline_risk
        targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth, _ = self.clean_fixture(root)
            with self.models(module) as (encoder, original, _), \
                 patch.object(module, '_load_dura', side_effect=AssertionError('C0 has no DURA')), \
                 patch.object(module, 'load_public', side_effect=AssertionError('C0 never reads star')), \
                 patch.object(development_clean, 'load_clean_public', wraps=development_clean.load_clean_public) as clean_load, \
                 patch.object(core, 'fit_geometry', side_effect=AssertionError('No target fit')), \
                 patch.object(core, 'fit_readouts', side_effect=AssertionError('No target fit')), \
                 patch.object(source_baseline_risk, 'calibrate_known', side_effect=AssertionError('No recalibration')):
                predicted = module.predict(config, 'cpu')
            clean_load.assert_called_once_with(config['public_root'])
            self.assertEqual(predicted['status'], 'CLEAN_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE')
            self.assertEqual(predicted['input_paths'], ['C0'])
            self.assertTrue(predicted['clean_condition_complete'])
            self.assertFalse(predicted['paired_complete'])
            self.assertIsNone(predicted['dura_version'])
            self.assertEqual(encoder.seen, original.seen)
            self.assertEqual(encoder.seen[2:], list(range(36)))
            self.assertTrue(predicted['frozen_parameters_buffers_unchanged'])
            self.assertTrue(predicted['frozen_source_aggregates_unchanged'])
            self.assertEqual(predicted['new_threshold_calibrations'], 0)
            # Keep one of twelve unknown rows accepted. The resulting 11/12
            # rejection distinguishes the current clean95 target from star90.
            path = root / 'predictions/predictions.npz'
            with np.load(path, allow_pickle=False) as data:
                values = {key: data[key].copy() for key in data.files}
            values['knownness'][0, 6, -1] = values['thresholds'][-1]
            values['accepted'][0, 6, -1] = True
            values['prediction'][0, 6, -1] = values['closed_prediction'][0, 6]
            np.savez(path, **values)
            result = module.score(root / 'predictions', truth, root / 'score', targets)
            self.assertEqual(result['condition_unknown_rejection_min'], .95)
            self.assertEqual(set(result['input_paths']), {'C0'})
            self.assertIsNone(result['P1_minus_P0'])
            self.assertEqual(result['primary_method'], 'mlp16_oe')
            self.assertFalse(result['paired_complete'])
            c0 = result['input_paths']['C0']
            overall = c0['methods']['mlp16_oe']['overall']
            self.assertAlmostEqual(overall['unknown_macro_rejection'], 11 / 12)
            self.assertGreaterEqual(overall['unknown_macro_rejection'], .90)
            self.assertLess(overall['unknown_macro_rejection'], .95)
            self.assertEqual(overall['known_macro_acceptance'], 1.)
            self.assertEqual(overall['original_closed_minus_new_open'], 0.)
            self.assertEqual(set(c0['methods']['mlp16_oe']['by_scenario']), {'clean'})
            self.assertTrue(c0['primary_goals']['checks']['main_scene_coverage'])
            self.assertFalse(c0['primary_goals']['checks']['overall_condition_targets'])
            self.assertFalse(c0['primary_goals']['pass'])
            self.assertFalse(result['final_phase3_complete'])

    @contextlib.contextmanager
    def models(self, module, *, encoder=None, original=None, dura=None):
        encoder = FakeReader() if encoder is None else encoder
        original = FakeReader() if original is None else original
        dura = FakeDura() if dura is None else dura
        with patch.object(module, '_load_readers', return_value=(encoder, original)), \
             patch.object(module, '_load_dura', return_value=(dura, lambda x: (dura(x),))), \
             patch.object(module, '_encode_new', side_effect=self.fake_encode):
            yield encoder, original, dura

    def test_frozen_paired_predictions_and_truth_last_aligned_scoring(self):
        module = self.module()
        from experiments.pamf_phase3_open_world import source_open_readout as core
        from experiments.pamf_phase3_open_world import source_baseline_risk
        targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth = self.fixture(root)
            with self.models(module) as (encoder, baseline, dura), \
                 patch.object(core, 'fit_geometry', side_effect=AssertionError('No query fit')), \
                 patch.object(core, 'fit_readouts', side_effect=AssertionError('No query fit')), \
                 patch.object(source_baseline_risk, 'calibrate_known', side_effect=AssertionError('No recalibration')):
                result = module.predict(config, 'cpu')
            self.assertEqual(result['status'], 'ALL_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE')
            self.assertTrue(result['paired_complete'])
            self.assertEqual(result['primary_method'], 'mlp16_oe')
            self.assertEqual(encoder.seen, baseline.seen)
            self.assertEqual(encoder.seen[4:22], list(range(18)))
            self.assertEqual(encoder.seen[22:], [index + .25 for index in range(18)])
            self.assertTrue(result['real_checkpoint_synthetic_smoke_passed'])
            self.assertTrue(torch.equal(dura.offset, torch.tensor(.25)))
            with np.load(root / 'predictions/predictions.npz', allow_pickle=False) as p:
                self.assertEqual(p['knownness'].shape, (2, 18, 7))
                self.assertEqual(p['baseline_closed_prediction'].shape, (2, 18))
                self.assertTrue(np.array_equal(p['accepted'], p['knownness'] >= p['thresholds']))
                self.assertNotIn('y', p.files)
                self.assertNotIn('scenario', p.files)
            scored = module.score(root / 'predictions', truth, root / 'score', targets)
            self.assertEqual(scored['primary_method'], 'mlp16_oe')
            primary = scored['input_paths']['P0']['methods']['mlp16_oe']
            self.assertEqual(primary['overall']['unknown_macro_rejection'], 1.)
            self.assertEqual(primary['overall']['known_macro_frr'], 0.)
            self.assertEqual(primary['overall']['original_closed_minus_new_open'], 0.)
            self.assertEqual(len(primary['by_transmitter']), 9)
            self.assertEqual(len(primary['by_scenario']), 3)
            self.assertEqual(scored['condition_unknown_rejection_min'], .90)
            self.assertFalse(scored['final_phase3_complete'])

    def test_predictor_rejects_truth_U_formal_ids_and_extra_public_files(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth = self.fixture(root)
            for key in ('truth', 'unknown_count', 'scenario', 'source_features', 'targets'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: str(truth)}))
            with self.assertRaises(ValueError):
                module.validate_config(dict(config, public_root=str(root / 'development/discovery/public')))
            np.save(Path(config['public_root']) / 'ids.npy', np.asarray([f'p3f-{i:024x}' for i in range(18)]))
            with self.models(module), self.assertRaises(ValueError):
                module.predict(config, 'cpu')
            np.save(Path(config['public_root']) / 'ids.npy', np.asarray([f'p3dev-{i:024x}' for i in range(18)]))
            (Path(config['public_root']) / 'truth.npz').write_bytes(b'private')
            with self.models(module), self.assertRaises(ValueError):
                module.predict(config, 'cpu')

    def test_source_encoder_risk_identity_and_thresholds_must_match(self):
        module = self.module()
        for problem in ('encoder', 'risk_parent', 'risk_schema', 'threshold_shape'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, truth = self.fixture(root)
                risk = Path(config['risk_prediction_root'])
                if problem == 'encoder':
                    p = Path(config['source_row_root']) / 'export/complete.json'
                    value = json.loads(p.read_text()); value['encoder_state_identity']['arm'] = 'supervised_only'
                    self.write(p, value)
                elif problem == 'risk_parent':
                    p = risk / 'complete.json'
                    value = json.loads(p.read_text()); value['parent_row_root'] = 'a-different-readout'
                    self.write(p, value)
                elif problem == 'risk_schema':
                    p = risk / 'calibration/complete.json'
                    value = json.loads(p.read_text()); value['calibration_schema'] = 'old-self-risk'
                    self.write(p, value)
                else:
                    np.save(risk / 'calibration/thresholds.npy', np.zeros(6))
                with patch.object(module, '_load_readers', side_effect=AssertionError('No model before source validation')), self.assertRaises(ValueError):
                    module.predict(config, 'cpu')

    def test_missing_DURA_is_explicit_P0_only_and_never_passes_paired_score_gate(self):
        module = self.module()
        targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth = self.fixture(root)
            with self.assertRaises(ValueError):
                module.validate_config(dict(config, dura_checkpoint=None))
            config.update(input_paths=['P0'], dura_checkpoint=None, dura_release=None)
            with self.models(module), patch.object(module, '_load_dura', side_effect=AssertionError('No DURA in P0-only')):
                result = module.predict(config, 'cpu')
            self.assertEqual(result['status'], 'P0_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE_UNPAIRED')
            self.assertFalse(result['paired_complete'])
            with patch.object(module, '_truth', side_effect=AssertionError('Truth must stay closed')), self.assertRaises(ValueError):
                module.score(root / 'predictions', truth, root / 'score', targets)

    def test_channel_failure_and_mutated_frozen_state_cannot_complete(self):
        module = self.module()
        for problem in ('DURA_failure', 'encoder_buffer', 'readout_array'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, truth = self.fixture(root)
                encoder = FakeReader()
                with self.models(module, encoder=encoder):
                    if problem == 'DURA_failure':
                        extra = patch.object(module, '_load_dura', return_value=(FakeDura(), lambda x: (_ for _ in ()).throw(RuntimeError('failure'))))
                    elif problem == 'encoder_buffer':
                        def mutate(reader, x):
                            reader.fixed.add_(1)
                            return self.fake_encode(reader, x)
                        extra = patch.object(module, '_encode_new', side_effect=mutate)
                    else:
                        original_score = module._scores
                        def mutate(data, source):
                            source['state']['mean'][0] += 1
                            return original_score(data, source)
                        extra = patch.object(module, '_scores', side_effect=mutate)
                    with extra, self.assertRaises((ValueError, RuntimeError)):
                        module.predict(config, 'cpu')
                self.assertFalse((root / 'predictions/complete.json').exists())

    def test_score_gate_rejects_incomplete_bad_alignment_and_bad_decisions_before_truth(self):
        module = self.module()
        targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
        for problem in ('incomplete', 'shape', 'decision', 'duplicate_ids'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, truth = self.fixture(root)
                with self.models(module): module.predict(config, 'cpu')
                if problem == 'incomplete':
                    p = root / 'predictions/complete.json'
                    value = json.loads(p.read_text()); value['paired_complete'] = False
                    self.write(p, value)
                else:
                    p = root / 'predictions/predictions.npz'
                    with np.load(p, allow_pickle=False) as data: value = {key: data[key] for key in data.files}
                    if problem == 'shape': value['knownness'] = value['knownness'][:, :-1]
                    elif problem == 'decision': value['prediction'][0, 0, 0] = -1 if value['accepted'][0, 0, 0] else 0
                    else: value['ids'][1] = value['ids'][0]
                    np.savez(p, **value)
                with patch.object(module, '_truth', side_effect=AssertionError('Truth must stay closed')), self.assertRaises(ValueError):
                    module.score(root / 'predictions', truth, root / 'score', targets)

    def test_collision_and_public_identity_mismatch_stop_before_models(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth = self.fixture(root)
            (root / 'predictions').mkdir()
            with patch.object(module, '_source_bundle', side_effect=AssertionError('No reads after collision')), self.assertRaises(FileExistsError):
                module.predict(config, 'cpu')
            config['output_root'] = str(root / 'other-predictions')
            config['capsule_id'] = 'wrong'
            with self.models(module), self.assertRaises(ValueError):
                module.predict(config, 'cpu')

    def test_real_checkpoint_synthetic_smoke_precedes_public_IQ_and_failure_keeps_it_closed(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth = self.fixture(root)
            with self.models(module), patch.object(module, '_public', side_effect=AssertionError('Public remains closed')), \
                 patch.object(module, '_encode_new', side_effect=RuntimeError('real-checkpoint failure')), self.assertRaises(RuntimeError):
                module.predict(config, 'cpu')
            original = module._public
            with self.models(module) as (reader, baseline, dura):
                def after_smoke(c):
                    self.assertEqual(len(reader.seen), 4)
                    self.assertEqual(len(baseline.seen), 4)
                    return original(c)
                with patch.object(module, '_public', side_effect=after_smoke):
                    module.predict(config, 'cpu')

    def test_original_baseline_measures_same_input_damage_and_primary_never_reselected(self):
        module = self.module()
        targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, truth = self.fixture(root)
            with self.models(module): module.predict(config, 'cpu')
            p = root / 'predictions/predictions.npz'
            with np.load(p, allow_pickle=False) as data: value = {key: data[key].copy() for key in data.files}
            # Force MLP to reject every row while static methods keep their
            # original decisions. The scorer must retain the prescribed MLP.
            value['knownness'][:, :, -1] = value['thresholds'][-1] - 1
            value['accepted'][:, :, -1] = False
            value['prediction'][:, :, -1] = -1
            np.savez(p, **value)
            result = module.score(root / 'predictions', truth, root / 'score', targets)
            self.assertEqual(result['primary_method'], 'mlp16_oe')
            primary = result['input_paths']['P1']['methods']['mlp16_oe']['overall']
            self.assertEqual(primary['original_closed_macro_accuracy'], 1.)
            self.assertEqual(primary['known_macro_open_accuracy'], 0.)
            self.assertEqual(primary['original_closed_minus_new_open'], 1.)
            self.assertFalse(result['input_paths']['P1']['primary_goals']['pass'])

    def test_cli_first_line_has_actual_process_and_score_requires_targets(self):
        module = self.module()
        import os
        with io.StringIO() as output, contextlib.redirect_stdout(output), \
             patch.object(module, 'predict', return_value={'status': 'stub'}), patch.object(module, '_read', return_value={}):
            module.main(['predict', '--config', 'synthetic.json', '--device', 'cpu'])
            first = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(first['pid'], os.getpid())
        self.assertEqual(Path(first['cwd']), Path.cwd())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main(['score', '--predictions-root', 'p', '--truth', 't', '--output', 'o'])


if __name__ == '__main__':
    unittest.main()
