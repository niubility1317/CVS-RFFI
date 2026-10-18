"""Focused synthetic RNG, waveform scope, parent/final and actual sixL tests."""
import argparse
import copy
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
import torch

def local_dependencies(path):
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / 'code')]
    package = types.ModuleType('experiments')
    package.__path__ = [str(Path(path) / 'experiments'), str(root / 'experiments')]
    sys.modules['experiments'] = package

def rng():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()

def restore(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])

class Checks(unittest.TestCase):
    def assert_rng_equal(self, a, b):
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1][0], b[1][0])
        np.testing.assert_array_equal(a[1][1], b[1][1])
        self.assertEqual(a[1][2:], b[1][2:])
        self.assertTrue(torch.equal(a[2], b[2]))

    def augmented(self):
        from . import design as d
        from experiments.cvs_phase1_overlay.model import native_modules
        native = native_modules()
        # Obtain actual inherited E20 schedule, without checkpoint/data reads.
        args = d.old.make_args(d.old.config(next(r for r in d.old.rows() if r['arm'] == 'B01'), 20), 'cpu')
        cfg = native.build_aug_base_cfg(args)
        aug = native.make_augmentor(cfg)
        native.configure_augmentor_for_epoch(aug, cfg, 13, args)
        return aug

    def test_native_prefix_and_all_global_random_poststates(self):
        from .augmentation import Wrapper
        x = torch.randn(128, 2, 256, generator=torch.Generator().manual_seed(6001))
        y = torch.arange(128) % 6
        for seed in range(12):
            aug = self.augmented()
            random.seed(seed + 6002)
            np.random.seed(seed + 6002)
            torch.manual_seed(seed + 6002)
            start = rng()
            captured = []
            original = aug._apply_general_channel
            def prefix(v):
                out = original(v)
                captured.append(out.clone())
                return out
            aug._apply_general_channel = prefix
            expected_full = aug(x, labels=y, no_pa=False)
            expected_end = rng()
            aug._apply_general_channel = original
            restore(start)
            records = []
            actual = Wrapper(aug, records)(x, labels=y, no_pa=False)
            self.assert_rng_equal(expected_end, rng())
            self.assertTrue(torch.equal(actual, captured[0]))
            self.assertEqual(records[0]['actual_PA_injected_packets'], 0)
            self.assertTrue(records[0]['exact_native_nonPA_prefix'])
            self.assertEqual(records[0]['shadow_PA_changed_packets'], int((expected_full != actual).flatten(1).any(1).sum()))
            self.assertGreater(aug.p_pa, 0)

    def test_shadow_consumes_numpy_and_protects_prefix_from_inplace_PA(self):
        from .augmentation import Wrapper
        aug = self.augmented()
        # Test-only guaranteed selection. Production probabilities are unmodified.
        aug.p_pa = 1.
        x = torch.randn(128, 2, 256, generator=torch.Generator().manual_seed(6003))
        before, y = x.clone(), torch.arange(128) % 6
        original = aug.simulate_pa
        def future(v, *a, **kw):
            np.random.random(3)
            out, strength = original(v, *a, **kw)
            v.add_(.125)
            return out + .125, strength
        aug.simulate_pa = future
        random.seed(6004)
        np.random.seed(6004)
        torch.manual_seed(6004)
        start = rng()
        aug(x, labels=y, no_pa=False)
        expected_end = rng()
        restore(start)
        records = []
        Wrapper(aug, records)(x, labels=y, no_pa=False)
        self.assert_rng_equal(expected_end, rng())
        self.assertTrue(torch.equal(x, before))
        self.assertEqual(records[0]['shadow_PA_selected_packets'], 128)
        self.assertEqual(records[0]['shadow_PA_calls'], 1)
        self.assertEqual(records[0]['shadow_PA_changed_packets'], 128)

    def test_unselected_PA_mask_still_keeps_global_rng(self):
        from .augmentation import Wrapper
        aug = self.augmented()
        aug.p_pa = 1e-30
        x, y = torch.ones(128, 2, 256), torch.arange(128) % 6
        start = rng()
        expected = aug(x, labels=y, no_pa=False)
        expected_end = rng()
        restore(start)
        records = []
        result = Wrapper(aug, records)(x, labels=y, no_pa=False)
        self.assert_rng_equal(expected_end, rng())
        self.assertTrue(torch.equal(expected, result))
        self.assertEqual(records[0]['shadow_PA_calls'], 0)
        self.assertEqual(records[0]['shadow_PA_selected_packets'], 0)

    def test_scope_rejects_U_As_raw_and_PA_switch(self):
        from .augmentation import Wrapper
        aug = self.augmented()
        wrapper = Wrapper(aug, [])
        x, y = torch.ones(128, 2, 256), torch.arange(128) % 6
        for arguments in [dict(labels=y, no_pa=True), dict(labels=None, no_pa=False),
                          dict(labels=-torch.ones_like(y), no_pa=False),
                          dict(labels=y, no_pa=False, return_pa_strength=True)]:
            with self.assertRaises(ValueError):
                wrapper(x, **arguments)
        with self.assertRaises(ValueError):
            wrapper(x[:64], labels=y[:64], no_pa=False)

    def test_exact_corresponding_parent_and_native_tail_contract(self):
        from . import design as d
        for row in d.rows():
            c = d.config(row)
            d.validate(c)
            old = d.old_config(row['arm'])
            self.assertEqual((c['epochs'], c['updates'], c['epoch_offset'], c['step_offset'], c['L_NN_per_step']), (10, 2220, 20, 4440, 768))
            self.assertEqual(c['parent_source'], old['parent_source'])
            self.assertEqual(Path(c['parent_source']).parts[-2:], ('B01-s2026092701' if c['As'] else 'B00-s2026092701', 'stage20'))
            self.assertEqual(c['scene_prior']['probabilities'], [1 / 3] * 3)
            self.assertFalse(c['new_Ag'])
            self.assertEqual(c['total_L_loss_scale'], 1.25)
            a = d.make_args(c, 'cpu')
            self.assertEqual((a.epochs, a.label_epochs, a.pseudo_epochs, a.lr), (10, 0, 10, .0002))
            self.assertTrue(a.aug_enable_pa_normal)
            self.assertGreater(a.aug_p_pa, 0)
            self.assertFalse(a.from_scratch)
            self.assertFalse(a.teacher_ckpt)
            for key, value in [('model_seed', 2026092702), ('epochs', 20), ('As', not c['As']), ('parent_source', '/wrong'), ('L_NN_per_step', 512)]:
                wrong = copy.deepcopy(c)
                wrong[key] = value
                with self.assertRaises(ValueError):
                    d.validate(wrong)
            with self.assertRaises(ValueError):
                d.config(row, 20)

    def test_final_checkpoint_and_role_negative_cases(self):
        from . import design as d, protocol as p
        for row in d.rows():
            c = d.config(row)
            roles = {'L_s': ['toyL'], 'U_s': ['toyU'], 'V': ['toyV']}
            done = dict(config=c, epoch=10, optimizer_steps=2220, target_access=False)
            frozen = dict(config=c, selection='final_only', target_access=False)
            initial = dict(scratch_only=False, target_contact=False, checkpoint_sources=c['checkpoint_sources'], source_roles='EXACT_MATCH')
            p.final_metadata(c, done, frozen, initial, dict(role_ids=roles), roles)
            cp = dict(epoch=10, a1_ema_successful_updates=2220, run_id=c['run_id'], candidate_id=c['row_id'], checkpoint_selection='final_only', ema_model={}, args=dict(baseline_ckpt=c['parent_source'] + '/final_ssdg.pth', teacher_ckpt=''))
            p.checkpoint(c, cp)
            for key, value in [('run_id', d.REFERENCE_RUN), ('candidate_id', d.rows()[1 - d.rows().index(row)]['row_id']), ('epoch', 20), ('ema_model', None)]:
                wrong = copy.deepcopy(cp)
                wrong[key] = value
                with self.assertRaises(ValueError):
                    p.checkpoint(c, wrong)
            with self.assertRaises(ValueError):
                p.final_metadata(c, done, frozen, initial, dict(role_ids={}), roles)

    def test_source_FAIL_and_missing_peer_still_require_all7_JSON_bool(self):
        from unittest.mock import patch
        from . import design as d, report
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'new'
            ref = Path(tmp) / 'runs' / d.REFERENCE_RUN
            no, yes = d.rows()
            paths = [base / no['row_id'] / 'source/completion.json', base / yes['row_id'] / 'source/completion.json',
                     base / yes['row_id'] / 'source_qualification/complete.json', ref / 'B01-s2026092701/source_qualification/complete.json']
            inputs = [dict(config=d.config(no), target_access=False, final_source_metrics=dict(source_val_count=27000, source_val_accuracy=.98)),
                      dict(config=d.config(yes), target_access=False, final_source_metrics=dict(source_val_count=27000, source_val_accuracy=.978)),
                      dict(config=d.config(yes), target_access=False, formal_scores_read=False, training_updates=0, NN=12600, source_accuracies=[[.9, .76, .62]] * 2),
                      dict(config=d.old_config('AS_PAOFF'), target_access=False, formal_scores_read=False, source_accuracies=[[.9, .74, .60]] * 2)]
            for path, value in zip(paths, inputs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding='utf8')
            with patch.object(d, 'BASE', base), patch.object(d, 'PROJECT', tmp), patch.object(d, 'config', side_effect=lambda row: inputs[0 if row['arm'] == 'NOAS_PAOFF' else 1]['config']):
                result = report.source_decision()
                self.assertFalse(result['source_qualified'])
                self.assertTrue(result['formal_evaluation_always_required'])
                self.assertTrue(all(type(v) is bool for v in result['checks'].values()))
                json.dumps(result, allow_nan=False)
                paths[0].unlink()
                partial = report.source_decision()
                self.assertEqual(partial['status'], 'SOURCE_READOUT_PARTIAL')
                self.assertTrue(partial['formal_evaluation_always_required'])

    def test_actual_sixL_both_arms_default_head_and_Sinc(self):
        from .preflight import synthetic
        result = [synthetic(arm) for arm in ('NOAS_PAOFF', 'AS_PAOFF')]
        self.assertEqual(sum(v['NN'] for v in result), 1536)
        self.assertEqual([v['As_IQ'] for v in result], [0, 64])
        self.assertTrue(all(v['default_head_joint_gradient'] and v['Sinc_joint_gradient'] for v in result))
        self.assertTrue(all(v['same_generated_IQ_each_pair'] for v in result))

    def test_native_factory_scope_preserves_parent_E13_and_restores(self):
        from . import design as d
        from .augmentation import Wrapper
        from .observer import Observer
        from experiments.cvs_phase1_overlay.model import native_modules
        native = native_modules()
        args = d.old.make_args(d.old.config(next(r for r in d.old.rows() if r['arm'] == 'B01'), 20), 'cpu')
        cfg = native.build_aug_base_cfg(args)
        make, configure = native.make_augmentor, native.configure_augmentor_for_epoch
        reference = make(cfg)
        configure(reference, cfg, 13, args)
        o = Observer.__new__(Observer)
        o.parent, o.PA_records, o.augmentors_created = dict(args=vars(args)), [], 0
        o.optimizer = None
        with o.native_scope(native):
            aug = native.make_augmentor(cfg)
            self.assertIsInstance(aug, Wrapper)
            native.configure_augmentor_for_epoch(aug, cfg, 1, d.make_args(d.config(d.rows()[1]), 'cpu'))
            self.assertEqual(aug.p_pa, reference.p_pa)
            self.assertEqual(aug.max_time_shift, reference.max_time_shift)
            self.assertEqual(o.augmentors_created, 1)
        self.assertIs(native.make_augmentor, make)
        self.assertIs(native.configure_augmentor_for_epoch, configure)

    def test_dispatch_one_failed_row_finishes_other_and_decides_before_collect(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        from . import design as d, dispatch, report, preflight
        actions, launched = [], []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'new'
            class Child:
                def __init__(self, command, **kwargs):
                    self.row = command[-1]
                    self.pid = 60000 + len(launched)
                    self.returncode = None
                    launched.append(self.row)
                def poll(self):
                    if self.row.startswith('NOAS'):
                        self.returncode = 1
                    else:
                        self.returncode = 0
                        d.write(base / self.row / 'complete.json', dict(status='TRAINED_AND_SCORED'))
                    return self.returncode
            def decide():
                actions.append('source')
                return dict(source_qualified=False)
            def collect():
                actions.append('formal_collect')
                return dict(status='FORMAL_PARTIAL')
            with patch.object(d, 'BASE', base), patch.object(d, 'PROJECT', tmp), \
                 patch.object(dispatch, 'lock', side_effect=lambda *a, **kw: nullcontext()), \
                 patch.object(dispatch, 'capacity', return_value={0: dict(pids={1233539}, free_mb=20000)}), \
                 patch.object(dispatch.subprocess, 'Popen', side_effect=Child), \
                 patch.object(dispatch.time, 'sleep'), patch.object(preflight, 'smoke'), \
                 patch.object(report, 'source_decision', side_effect=decide), \
                 patch.object(report, 'collect', side_effect=collect):
                with self.assertRaisesRegex(RuntimeError, 'remaining fixed rows finished'):
                    dispatch.dispatch()
            self.assertEqual(launched, [r['row_id'] for r in d.rows()])
            self.assertEqual(actions, ['source', 'formal_collect'])
            failure = d.read(base / 'failure.json')
            self.assertTrue(failure['all_other_fixed_rows_finished'])
            self.assertEqual(d.read(base / 'complete.json')['rows'], [d.rows()[1]['row_id']])

    def test_source_readout_exception_still_executes_all7_prediction_stage(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        from . import design as d, dispatch, observer
        from experiments.cvs_equivariant_identity import precision
        qualifier = types.ModuleType('experiments.cvs_pa_waveform60.qualification')
        def fail(_):
            raise RuntimeError('synthetic source-readout failure')
        qualifier.run = fail
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(d, 'BASE', Path(tmp)), \
                 patch.object(dispatch, 'lock', side_effect=lambda *a, **kw: nullcontext()), \
                 patch.object(dispatch, 'capacity', return_value={0: dict(pids=set(), free_mb=20000)}), \
                 patch.dict(dispatch.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                 patch.object(observer, 'train'), \
                 patch.object(precision, 'numerical_context', side_effect=lambda *a: nullcontext()), \
                 patch.dict(sys.modules, {qualifier.__name__: qualifier}), \
                 patch.object(dispatch, 'execute') as executed:
                dispatch.worker(d.rows()[0]['row_id'])
            executed.assert_called_once_with('predict', d.rows()[0]['row_id'])

def run():
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Focused PA waveform checks failed')
    return dict(status='PASS', tests=result.testsRun, synthetic_NN=1536,
                real_IQ=0, checkpoint_reads=0, target_access=False, optimizer_updates=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dependencies', type=Path, required=True)
    args = parser.parse_args()
    local_dependencies(args.dependencies)
    name = 'experiments.cvs_pa_waveform60'
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('__init__.py'), submodule_search_locations=[str(Path(__file__).parent)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    __package__ = name
    print(json.dumps(run()))
