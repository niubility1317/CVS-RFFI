"""Synthetic backends only: U fitting and future prediction remain separated."""
import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


METHODS = ['msp', 'maxlogit', 'margin', 'energy', 'cosine', 'linear_oe', 'mlp16_oe']
OLD = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
DIMS = dict(fused160=160, time160=160, frequency160=160, pa160=160,
            reference29=29, signature336=336, spectrum128=128, pa_moments48=48)
RUN = '20261029-phase3-development-alignment-s2026092701-r01'
READOUT = '20261029-phase3-source-embedding-readout-s2026092701-r01'
RISK = '20261029-phase3-source-baseline-risk-s2026092701-r01'
OLD_RUN = dict(C0='20261029-phase3-development-clean-transfer-s2026092701-r01',
              P1='20261029-phase3-development-transfer-s2026092701-r01')


class FakeReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('fixed', torch.zeros(1, dtype=torch.float32))
        self.eval().requires_grad_(False)
        self.seen = []; self.head_features = []

    def features(self, x):
        self.seen.extend(x[:, 0, 0].tolist())
        z = torch.zeros((len(x), 160), dtype=torch.float32, device=x.device)
        z[:, 0] = x[:, 0, 0]
        return z

    def classify_features(self, z):
        self.head_features.extend(z[:, 0].tolist())
        labels = z[:, 0].long() % 9
        logits = torch.zeros((len(z), 6), dtype=torch.float32, device=z.device)
        logits[torch.arange(len(z)), labels % 6] = torch.where(labels < 6, 3., .5)
        return logits


class FakeDura(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('fixed', torch.zeros(1, dtype=torch.float32))
        self.eval().requires_grad_(False)

    def forward(self, x):
        return x.clone()


class AlignmentPipelineTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/development_alignment.py'
        self.assertTrue(path.is_file(), 'U-only alignment pipeline is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.development_alignment')

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def fixture(self, root, condition='C0'):
        family = 'clean_cross_receiver' if condition == 'C0' else 'practical_cross_receiver'
        identity = dict(run_id='20261029-phase3-source-embedding-increment-s2026092701-r02',
            row_id='uniform_oe-s2026092701', arm='uniform_oe', updates=222, cumulative_updates=11322)
        source = dict(marker=dict(encoder_state_identity=identity), geometry=dict(branches={
            name: dict(prototypes=np.zeros((6, dim)), mean=np.zeros(dim), scale=np.ones(dim), whitener=np.eye(dim))
            for name, dim in DIMS.items()}), state=dict(fixed=np.ones(2)), thresholds=np.full(7, 1.5, np.float64))
        config = dict(run_id=RUN, code_commit='a' * 40, condition=condition, view_family=family,
            source_row_root=str(root / READOUT / 'rows/uniform_oe'),
            risk_prediction_root=str(root / RISK / 'rows/uniform_oe/predict'),
            d0_prediction_root=str(root / OLD_RUN[condition] / 'predict'),
            output_root=str(root / 'new/predict'), batch_size=128,
            dura_checkpoint=None if condition == 'C0' else str(root / 'G/final.pt'),
            dura_release=None if condition == 'C0' else str(root / 'G/release'))
        for prefix, role, count, offset in [('U', 'discovery', 384, 10000), ('Q', 'evaluation', 864, 0)]:
            public = root / 'public-pool' / role / 'public'; public.mkdir(parents=True)
            ids = np.asarray([f'p3dev-{index+offset:024x}' for index in range(count)])
            cap = ('p3dev-clean-' if condition == 'C0' else 'p3dev-') + role
            manifest = dict(schema='phase3_clean_min_v1' if condition == 'C0' else 'phase3_min_v1',
                data_status='VALIDATED_ONCE', capsule_id=cap, split_id='split-' + role,
                count=count, iq_shape=[count, 2, 256], registered_classes=OLD,
                processing_route='clean' if condition == 'C0' else 'residual',
                mode='clean' if condition == 'C0' else 'post_sync', equalization_enabled=False,
                fs_hz=25000000, single_observation=True)
            self.write(public / 'manifest.json', manifest); np.save(public / 'ids.npy', ids)
            iq = np.zeros((count, 2, 256), np.float32); iq[:, 0, :] = (np.arange(count) + offset)[:, None]
            np.save(public / 'received.npy', iq)
            config.update({prefix + '_public_root': str(public), prefix + '_capsule_id': cap,
                           prefix + '_split_id': manifest['split_id'], prefix + '_expected_count': count})
        old = Path(config['d0_prediction_root']); old.mkdir(parents=True)
        paths = ['C0'] if condition == 'C0' else ['P0', 'P1']; count = 864
        y = np.arange(count) % 9; closed = np.tile(y % 6, (len(paths), 1))
        values = np.tile(np.where(y < 6, 3., .5)[None, :, None], (len(paths), 1, 7))
        accepted = values >= source['thresholds']
        np.savez(old / 'predictions.npz', ids=np.load(Path(config['Q_public_root']) / 'ids.npy'),
            method_names=np.asarray(METHODS), input_paths=np.asarray(paths), thresholds=source['thresholds'],
            closed_prediction=closed, baseline_closed_prediction=closed.copy(), knownness=values,
            accepted=accepted, prediction=np.where(accepted, closed[:, :, None], -1))
        marker = dict(status='CLEAN_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE' if condition == 'C0'
                else 'ALL_DEVELOPMENT_TRANSFER_PREDICTIONS_COMPLETE',
            schema='pamf_phase3_development_transfer_predictions_v1', input_paths=paths,
            paired_complete=condition == 'P1', clean_condition_complete=condition == 'C0',
            view_family=family, primary_method='mlp16_oe', method_names=METHODS, count=count,
            public_capsule_id=config['Q_capsule_id'], public_split_id=config['Q_split_id'],
            encoder_state_identity=identity, source_row_root=config['source_row_root'],
            risk_prediction_root=config['risk_prediction_root'], same_input_original_F6_baseline=True,
            frozen_parameters_buffers_unchanged=True, frozen_source_aggregates_unchanged=True,
            truth_read=False, query_fit=False, run_id=OLD_RUN[condition], code_commit='b' * 40)
        marker.update({key: 0 for key in ['U_dev_reads', 'formal_target_reads', 'source_IQ_reads',
            'optimizer_updates', 'new_encoder_updates', 'new_readout_updates', 'new_threshold_calibrations']})
        marker.update(dura_version='dura_v1' if condition == 'P1' else None,
            dura_release=config['dura_release'], dura_execution=(dict(reuse=True, graph=True, native=False,
            FP32=True, TF32=False) if condition == 'P1' else None))
        self.write(old / 'complete.json', marker)
        truth = root / 'private/truth.npz'; truth.parent.mkdir()
        scenario = np.full(count, 'clean') if condition == 'C0' else np.asarray(
            ['practical_low_suburban'] * 288 + ['practical_mid_urban'] * 288 + ['practical_low_urban'] * 288)
        np.savez(truth, ids=np.load(Path(config['Q_public_root']) / 'ids.npy'), y=y.astype(np.int64),
                 receiver=np.full(count, 26, np.int64), scenario=scenario)
        return config, source, truth

    def state(self, applied=True):
        return dict(schema='target_orthogonal_alignment_v1', parameters={}, applied=applied,
            reason='applied' if applied else 'insufficient_reliable_core', sample_count=384,
            pseudo_label_counts=np.asarray([8, 8, 8, 0, 0, 0]),
            selected_class_counts=np.asarray([8, 8, 8, 0, 0, 0]), class_weights=np.ones(6) / 6,
            rotations={name: np.eye(dim) for name, dim in DIMS.items()}, diagnostics=dict(core_count=24))

    @contextlib.contextmanager
    def backend(self, module, source, applied=True, events=None):
        events = [] if events is None else events
        reader, original, dura = FakeReader(), FakeReader(), FakeDura()
        def encode(model, x):
            z = model.features(x); events.append('encode-U' if float(x[0, 0, 0]) >= 10000 else 'encode-Q-or-smoke')
            result = {'features__' + name: np.zeros((len(x), dim), np.float32) for name, dim in DIMS.items()}
            result['features__fused160'] = z.cpu().numpy().copy()
            result['logits'] = model.classify_features(z).cpu().numpy().copy(); return result
        def scores(data, bundle):
            return np.tile(data['logits'].max(1)[:, None], (1, 7)).astype(np.float64)
        def fit(branches, logits, values, thresholds, geometry):
            events.append('fit-U'); self.assertEqual(len(logits), 384)
            self.assertEqual(set(branches), set(DIMS)); return self.state(applied)
        def apply(branches, state, geometry):
            events.append('apply-Q'); out = {key: value.copy() for key, value in branches.items()}
            out['fused160'][:, 0] += 1; return out
        with patch.object(module, '_source_bundle', return_value=source), \
             patch.object(module, '_load_readers', return_value=(reader, original)), \
             patch.object(module, '_load_dura', return_value=(dura, lambda x: (dura(x),))), \
             patch.object(module, '_encode_new', side_effect=encode), patch.object(module, '_scores', side_effect=scores), \
             patch.object(module.transfer, '_encode_new', side_effect=encode), \
             patch.object(module.transfer, '_scores', side_effect=scores), \
             patch.object(module, '_fit_alignment', side_effect=fit), patch.object(module, '_apply_alignment', side_effect=apply):
            yield reader, original, events

    def test_U_fit_freeze_then_D0_and_future_native_head_recomputed(self):
        module = self.module()
        for condition in ['C0', 'P1']:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, source, truth = self.fixture(root, condition); events = []
                load_ref, opened = module._reference, module._open_public
                def reference(c, s, ids):
                    self.assertTrue((Path(c['output_root']) / 'adapter/complete.json').is_file())
                    self.assertIn('fit-U', events); events.append('D0-after-freeze'); return load_ref(c, s, ids)
                def public(c, role):
                    if role == 'Q': self.assertIn('D0-after-freeze', events)
                    events.append('open-' + role); return opened(c, role)
                with self.backend(module, source, events=events) as (reader, original, _), \
                     patch.object(module, '_reference', side_effect=reference), patch.object(module, '_open_public', side_effect=public):
                    done = module.predict(config, 'cpu')
                self.assertLess(events.index('open-U'), events.index('fit-U'))
                self.assertLess(events.index('fit-U'), events.index('D0-after-freeze'))
                self.assertLess(events.index('D0-after-freeze'), events.index('open-Q'))
                self.assertEqual(len(original.seen), 2, 'Original F runs only synthetic smoke, never Q again')
                self.assertTrue(done['adapter_applied']); self.assertTrue(done['frozen_alignment_unchanged'])
                with np.load(root / 'new/predict/predictions.npz') as saved:
                    self.assertEqual(saved['states'].tolist(), ['D0', 'D1'])
                    self.assertFalse(np.array_equal(saved['closed_prediction'][0], saved['closed_prediction'][1]))
                    np.testing.assert_array_equal(saved['baseline_closed_prediction'][0], saved['baseline_closed_prediction'][1])
                    self.assertNotIn('y', saved.files)
                with np.load(root / 'new/predict/adapter/alignment.npz') as saved:
                    self.assertFalse(any('core_mask' in key or 'selected_indices' in key or 'features' in key for key in saved.files))

    def test_no_core_fallback_reuses_D0_without_opening_query_IQ(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, source, truth = self.fixture(root)
            opened = module._open_public
            def only_U(c, role):
                self.assertEqual(role, 'U'); return opened(c, role)
            with self.backend(module, source, applied=False), patch.object(module, '_open_public', side_effect=only_U), \
                 patch.object(module, '_apply_alignment', side_effect=AssertionError('No query transform on identity')):
                done = module.predict(config, 'cpu')
            self.assertFalse(done['adapter_applied']); self.assertEqual(done['Q_IQ_forward_samples'], 0)
            with np.load(root / 'new/predict/predictions.npz') as saved:
                np.testing.assert_array_equal(saved['knownness'][0], saved['knownness'][1])
                np.testing.assert_array_equal(saved['closed_prediction'][0], saved['closed_prediction'][1])
            targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
            result = module.score(root / 'new/predict', truth, root / 'score', targets)
            self.assertFalse(result['adaptation_success']); self.assertFalse(result['anonymous_registration_evaluated'])
            self.assertIn('seen_in_U', result['states']['D1']['methods']['mlp16_oe']['by_unknown_role'])

    def test_smoke_failure_keeps_U_and_query_reference_closed(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, source, _ = self.fixture(Path(tmp))
            with self.backend(module, source), patch.object(module, '_synthetic_smoke', side_effect=RuntimeError('smoke')), \
                 patch.object(module, '_open_public', side_effect=AssertionError('IQ closed')), \
                 patch.object(module, '_reference', side_effect=AssertionError('D0 closed')), self.assertRaises(RuntimeError):
                module.predict(config, 'cpu')

    def test_extra_truth_unknown_count_wrong_condition_or_role_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, _, truth = self.fixture(Path(tmp))
            for key in ['truth', 'unknown_count', 'unknown_ratio', 'labels', 'classes', 'clean_reference', 'paired_reference']:
                with self.subTest(key=key), self.assertRaises(ValueError): module.validate_config(dict(config, **{key: str(truth)}))
            for change in [dict(condition='P0'), dict(U_public_root=config['Q_public_root']),
                           dict(Q_public_root=config['U_public_root']), dict(d0_prediction_root='wrong/predict'),
                           dict(dura_checkpoint='G.pt')]:
                with self.subTest(change=change), self.assertRaises(ValueError): module.validate_config(dict(config, **change))

    def test_wrong_U_capsule_schema_or_formal_ids_never_opens_IQ(self):
        module = self.module()
        for problem in ['capsule', 'schema', 'formal', 'private_file']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, source, _ = self.fixture(root); public = Path(config['U_public_root'])
                manifest = json.loads((public / 'manifest.json').read_text())
                if problem == 'capsule': manifest['capsule_id'] = 'wrong'
                elif problem == 'schema': manifest['schema'] = 'phase3_min_v1'
                elif problem == 'formal': np.save(public / 'ids.npy', np.asarray([f'p3f-{i:024x}' for i in range(384)]))
                else: (public / 'truth.npz').write_bytes(b'private')
                self.write(public / 'manifest.json', manifest)
                with self.backend(module, source), patch.object(module, '_open_public', side_effect=AssertionError('IQ closed')), self.assertRaises(ValueError):
                    module.predict(config, 'cpu')

    def test_source_and_alignment_mutation_cannot_complete(self):
        module = self.module()
        for problem in ['source', 'model', 'R']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, source, _ = self.fixture(root)
                with self.backend(module, source) as (reader, _, _):
                    fit = module._fit_alignment; apply = module._apply_alignment
                    def mutate_fit(*args):
                        value = fit(*args)
                        if problem == 'source': source['thresholds'][0] += 1
                        if problem == 'model': reader.fixed.add_(1)
                        return value
                    def mutate_R(branches, state, geometry):
                        value = apply(branches, state, geometry); state['rotations']['fused160'][0, 0] += 1; return value
                    with patch.object(module, '_fit_alignment', side_effect=mutate_fit), \
                         patch.object(module, '_apply_alignment', side_effect=mutate_R if problem == 'R' else apply), self.assertRaises(ValueError):
                        module.predict(config, 'cpu')
                self.assertFalse((root / 'new/predict/complete.json').exists())

    def test_D0_identity_threshold_ID_or_method_mismatch_stops_before_Q_IQ(self):
        module = self.module()
        for problem in ['source', 'capsule', 'threshold', 'ids', 'methods', 'wrong_G_condition']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, source, _ = self.fixture(root); old = Path(config['d0_prediction_root'])
                if problem in ['source', 'capsule', 'wrong_G_condition']:
                    marker = json.loads((old / 'complete.json').read_text())
                    if problem == 'source': marker['encoder_state_identity']['updates'] = 333
                    elif problem == 'capsule': marker['public_capsule_id'] = 'other'
                    else: marker['dura_version'] = 'a-different-condition'
                    self.write(old / 'complete.json', marker)
                else:
                    with np.load(old / 'predictions.npz') as stored: values = {key: stored[key].copy() for key in stored.files}
                    if problem == 'threshold': values['thresholds'][0] += 1
                    elif problem == 'ids': values['ids'] = values['ids'][::-1]
                    else: values['method_names'] = values['method_names'][::-1]
                    np.savez(old / 'predictions.npz', **values)
                opened = module._open_public
                def only_U(c, role):
                    self.assertEqual(role, 'U'); return opened(c, role)
                with self.backend(module, source), patch.object(module, '_open_public', side_effect=only_U), self.assertRaises(ValueError):
                    module.predict(config, 'cpu')

    def test_score_requires_fit_and_all_predictions_before_truth(self):
        module = self.module()
        targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
        for problem in ['fit_missing', 'prediction_missing', 'decision', 'baseline']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, source, truth = self.fixture(root)
                with self.backend(module, source, applied=False): module.predict(config, 'cpu')
                output = root / 'new/predict'
                if problem == 'fit_missing': (output / 'adapter/complete.json').unlink()
                elif problem == 'prediction_missing': (output / 'complete.json').unlink()
                else:
                    with np.load(output / 'predictions.npz') as stored: values = {k: stored[k].copy() for k in stored.files}
                    if problem == 'decision': values['prediction'][1, 0, 0] = -1
                    else: values['baseline_closed_prediction'][1, 0] = 2
                    np.savez(output / 'predictions.npz', **values)
                with patch.object(module, '_truth', side_effect=AssertionError('Truth stays closed')), self.assertRaises(ValueError):
                    module.score(output, truth, root / 'score', targets)

    def test_cli_first_line_reports_actual_process(self):
        module = self.module()
        import os
        with io.StringIO() as output, contextlib.redirect_stdout(output), \
             patch.object(module, '_read', return_value={}), patch.object(module, 'predict', return_value=dict(status='stub')):
            module.main(['predict', '--config', 'synthetic.json', '--device', 'cpu'])
            first = json.loads(output.getvalue().splitlines()[0])
        self.assertEqual(first['pid'], os.getpid()); self.assertEqual(Path(first['cwd']), Path.cwd())

    def test_real_algorithm_aggregate_roundtrip_supports_different_future_count(self):
        module = self.module()
        labels = np.repeat(np.arange(3), 128)
        geometry, branches = dict(schema='source_six_branch_geometry_v1', class_count=6, branches={}), {}
        from experiments.pamf_phase3_open_world import source_open_readout as source
        geometry['schema'] = source.GEOMETRY_SCHEMA
        for name, dim in DIMS.items():
            prototypes = np.eye(6, dim)
            rotation = np.eye(dim); rotation[:2, :2] = [[np.cos(.2), -np.sin(.2)], [np.sin(.2), np.cos(.2)]]
            branches[name] = (prototypes[labels] @ rotation).astype(np.float32)
            geometry['branches'][name] = dict(mean=np.zeros(dim), scale=np.ones(dim), prototypes=prototypes, whitener=np.eye(dim))
        logits = np.zeros((384, 6), np.float32); logits[np.arange(384), labels] = 4
        scores = np.zeros((384, 7), np.float64); scores[:, 0] = .95; scores[:, 4] = .95
        thresholds = np.asarray([.8, 1e6, 1e6, 1e6, .5, 1e6, 1e6], np.float64)
        fitted = module._fit_alignment(branches, logits, scores, thresholds, geometry)
        self.assertTrue(fitted['applied']); self.assertNotIn('core_mask', fitted); self.assertNotIn('selected_indices', fitted)
        future = {name: value[:7].copy() for name, value in branches.items()}
        expected = module._apply_alignment(future, fitted, geometry)
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / 'alignment.npz').open('wb') as stream: np.savez(stream, **module._adapter_arrays(fitted))
            loaded = module._load_alignment(tmp)
            actual = module._apply_alignment(future, loaded, geometry)
        for name in DIMS:
            self.assertEqual(actual[name].dtype, np.float32); np.testing.assert_array_equal(actual[name], expected[name])

    def test_later_unknown_protected_even_when_overall_unknown_and_known_targets_pass(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, source, truth = self.fixture(root)
            old_path = Path(config['d0_prediction_root']) / 'predictions.npz'
            with np.load(old_path) as data: old = {key: data[key].copy() for key in data.files}
            # A valid D0 working point with one correct known rejected gives a
            # measurable D1 gain; there is no target-truth fitting in predict.
            old['knownness'][0, 0] = .5; old['accepted'][0, 0] = False; old['prediction'][0, 0] = -1
            np.savez(old_path, **old)
            with self.backend(module, source), patch.object(module, '_apply_alignment',
                    side_effect=lambda branches, *_: {key: value.copy() for key, value in branches.items()}):
                module.predict(config, 'cpu')
            new_path = root / 'new/predict/predictions.npz'
            with np.load(new_path) as data: pred = {key: data[key].copy() for key in data.files}
            later_rows = np.arange(8, 864, 9)[:5]
            pred['knownness'][1, later_rows] = 3.; pred['accepted'][1, later_rows] = True
            pred['prediction'][1, later_rows] = pred['closed_prediction'][1, later_rows, None]
            np.savez(new_path, **pred)
            targets = Path(__file__).resolve().parents[1] / 'configs/pamf_phase3_targets_20261029.json'
            result = module.score(root / 'new/predict', truth, root / 'score', targets)
            self.assertTrue(result['states']['D1']['primary_goals']['pass'])
            primary = result['states']['D1']['methods']['mlp16_oe']
            self.assertGreater(primary['overall']['unknown_macro_rejection'], .95)
            self.assertLess(primary['by_unknown_role']['later_unknown']['unknown_macro_rejection'], .95)
            self.assertTrue(result['success_checks']['old_open_gain_positive'])
            self.assertFalse(result['success_checks']['later_unknown_protection'])
            self.assertFalse(result['adaptation_success'])

    def test_cross_storage_roots_keep_U_Q_roles_and_handles_isolated(self):
        module = self.module()
        for condition in ['C0', 'P1']:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, source, _ = self.fixture(root, condition)
                # Q is an existing immutable release; U belongs to the new
                # run's pool. Relocate only the synthetic three-file public.
                old_Q = Path(config['Q_public_root'])
                public = root / 'old-immutable-release/data/development/evaluation/public'
                public.mkdir(parents=True)
                for item in old_Q.iterdir(): (public / item.name).write_bytes(item.read_bytes())
                config['Q_public_root'] = str(public)
                self.assertIs(module.validate_config(config), config)
                with self.backend(module, source, applied=False):
                    done = module.predict(config, 'cpu')
                self.assertEqual(done['count'], 864)
                self.assertFalse(done['adapter_applied'])
                for change in [dict(U_public_root=config['Q_public_root']),
                               dict(Q_public_root=config['U_public_root']),
                               dict(Q_capsule_id=config['U_capsule_id']),
                               dict(Q_split_id=config['U_split_id'])]:
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        module.validate_config(dict(config, **change))

    def test_applied_native_head_uses_safe_numpy_ABI_bridge_and_new_logits(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, source, _ = self.fixture(root)
            original_as_tensor = torch.as_tensor
            def no_numpy_ABI(value, *args, **kwargs):
                if isinstance(value, np.ndarray):
                    raise RuntimeError('Simulated Torch2.1 NumPy2 ABI: numpy is not available')
                return original_as_tensor(value, *args, **kwargs)
            with self.backend(module, source) as (reader, original, _), \
                 patch.object(torch, 'as_tensor', side_effect=no_numpy_ABI), \
                 patch.object(torch, 'from_numpy', side_effect=AssertionError('No NumPy ABI bridge')):
                done = module.predict(config, 'cpu')
            self.assertTrue(done['adapter_applied'])
            self.assertEqual(done['Q_IQ_forward_samples'], 864)
            self.assertEqual(len(original.seen), 2)
            with np.load(root / 'new/predict/predictions.npz') as saved:
                np.testing.assert_array_equal(saved['closed_prediction'][1], ((np.arange(864) + 1) % 9) % 6)
                self.assertFalse(np.array_equal(saved['closed_prediction'][0], saved['closed_prediction'][1]))
                self.assertTrue(np.isfinite(saved['knownness']).all())


if __name__ == '__main__':
    unittest.main()
