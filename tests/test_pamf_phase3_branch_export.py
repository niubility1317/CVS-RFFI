import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_phase3_open_world.source_development import build_model


class RawLift(torch.nn.Module):
    def forward(self, x):
        return x.repeat(1, 12, 1)


class FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_gate = torch.nn.Identity()
        self.pa_lift = RawLift()
        self.cls_head = torch.nn.Linear(160, 4, bias=False)

    def forward(self, x):
        self.freq_gate(x[:, 0, :128].reshape(len(x), 4, 32))
        self.pa_lift(x)
        z = x[:, 0, :160]
        return dict(t_emb=z, f_emb=z + .1, feat_pa=z + .2, feat_joint=z + .3,
                    logits=torch.full((len(x), 4), -999., device=x.device))


class FakeResponse(torch.nn.Module):
    def forward(self, x):
        z = x[:, 0, :29].clone()
        z[:, 27] = 0.
        return z


class FakeView(torch.nn.Module):
    def forward(self, x):
        return x[:, :112]


class FakePool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.views = torch.nn.ModuleList([FakeView() for _ in range(3)])
        self.output = torch.nn.Linear(336, 160, bias=False)

    def forward(self, x):
        return self.output(torch.cat([view(x) for view in self.views], dim=1))


class FakeReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.id_backbone = FakeCore()
        self.response = FakeResponse()
        self.sequence_pool = FakePool()
        self.response_projection = torch.nn.Linear(29, 160, bias=False)

    def features(self, x):
        aux = self.core.id_backbone(x)
        return aux['feat_joint'] + self.response_projection(self.response(x)) + self.sequence_pool(aux['t_emb'])

    def classify_features(self, z):
        return self.core.id_backbone.cls_head(z)


class BranchExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/branch_export.py'
        self.assertTrue(path.is_file(), 'Source branch exporter is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.branch_export')

    def local_runtime(self):
        root = Path(__file__).resolve().parents[1]
        frozen = root / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        return dict(base_release=str(root), frozen_sources={key: str(frozen / key) for key in ('F', 'AF', 'BASE', 'FOLD', 'G')})

    def fixture(self, root):
        parent = root / 'parent'
        parent.mkdir()
        source_contract = root / 'source_contract.json'
        source_contract.write_text('{}', encoding='utf-8')
        runtime = dict(base_release='base', frozen_sources={key: key for key in ('F', 'AF', 'BASE', 'FOLD', 'G')})
        fold = dict(unknown_source_labels=[0, 1], heldout_source_rx=8)
        previous = dict(run_id='parent-source', row_id='fold01', model_seed=2026092701, code_commit='parent-commit',
            code_runtime=runtime, source_cache='source-cache', source_contract=str(source_contract), fold=fold,
            output_root=str(parent), epochs=20, batch_size=128, shuffle_seed=2026102802,
            optimizer=dict(name='AdamW', lr=.0003, weight_decay=.0001, betas=[.9, .999], eps=1e-8),
            schedule='cosine_per_step', label_smoothing=0.)
        manifest = dict(unknown_source_labels=[0, 1], heldout_source_rx=8)
        done = dict(status='SOURCE_DEVELOPMENT_FROZEN', source_only=True, target_reads=0, U_reads=0,
                    epochs=20, updates=540, checkpoint_initialization='scratch', train_count=3360, validation_count=27000,
                    head_joint=True, checkpoint_selection='fixed_final_epoch20', config=previous, manifest=manifest,
                    checkpoint_ref='source_model.pt')
        (parent / 'complete.json').write_text(json.dumps(done), encoding='utf-8')
        reader = FakeReader()
        torch.save(dict(state_dict=reader.state_dict(), config=previous, manifest=manifest, epoch=20, updates=540,
                        source_only=True, checkpoint_initialization='scratch', target_reads=0, U_reads=0), parent / 'source_model.pt')
        config = dict(run_id='branch-source', row_id='branches', code_commit='new-commit', code_runtime=runtime,
            source_model_root=str(parent), source_cache='source-cache', source_contract=str(source_contract),
            model_seed=2026092701, fold=fold, batch_size=128, output_root=str(root / 'output'))
        train_x, validation_x = torch.zeros(3360, 2, 256), torch.zeros(27000, 2, 256)
        data = dict(train_x=train_x, train_y=np.arange(3360, dtype=np.int64) % 4,
                    validation_x=validation_x, validation_y=np.arange(27000, dtype=np.int64) % 6,
                    train_ids=np.asarray(['train-' + str(i) for i in range(3360)]),
                    validation_ids=np.asarray(['validation-' + str(i) for i in range(27000)]),
                    validation_receivers=np.asarray([1, 3, 4, 6, 8] * 5400, dtype=np.int64),
                    manifest=manifest, counts=dict(train=3360, validation=27000))
        return config, reader, data

    def hook_counts(self, model):
        return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]

    def test_real_scratch_extracts_branches_without_changing_state(self):
        module = self.module()
        with patch('torch.load', side_effect=AssertionError('No weights')), patch('numpy.load', side_effect=AssertionError('No data')):
            model = build_model(self.local_runtime(), 2026092701).eval().requires_grad_(False)
            x = torch.randn(3, 2, 256)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            hooks_before = self.hook_counts(model)
            values, logits = module.extract_branches(model, x)
            self.assertEqual(list(values), list(module.BRANCHES))
            for name, dimension in module.DIMENSIONS.items():
                self.assertEqual(tuple(values[name].shape), (3, dimension))
                self.assertTrue(bool(torch.isfinite(values[name]).all()))
            self.assertTrue(torch.equal(values['reference29'][:, 27], torch.zeros(3)))
            self.assertTrue(torch.allclose(logits, model.classify_features(values['fused160'])))
            self.assertEqual(self.hook_counts(model), hooks_before)
            self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
            one, _ = module.extract_branches(model, x[:1])
            changed, _ = module.extract_branches(model, torch.cat([x[:1], torch.randn_like(x[1:])]))
            for name in values:
                self.assertTrue(torch.allclose(one[name], values[name][:1], atol=2e-5, rtol=2e-5), name)
                self.assertTrue(torch.allclose(changed[name][:1], values[name][:1], atol=2e-5, rtol=2e-5), name)

    def test_full_logits_and_raw_gate_input_are_used(self):
        module = self.module()
        model = FakeReader().eval().requires_grad_(False)
        x = torch.randn(2, 2, 256)
        gate = model.core.id_backbone.freq_gate
        hook = gate.register_forward_hook(lambda m, args, value: value + 99.)
        try:
            values, logits = module.extract_branches(model, x)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(values['spectrum128'], x[:, 0, :128]))
        self.assertTrue(torch.equal(logits, model.classify_features(values['fused160'])))
        self.assertFalse(bool((logits == -999.).all()))
        lifted = x.repeat(1, 12, 1)
        expected = torch.cat([lifted.mean(-1), lifted.std(-1, unbiased=False)], dim=1)
        self.assertTrue(torch.allclose(values['pa_moments48'], expected))

    def test_exception_removes_all_temporary_hooks(self):
        module = self.module()
        model = FakeReader().eval().requires_grad_(False)
        before = self.hook_counts(model)
        with patch.object(model, 'features', side_effect=RuntimeError('Synthetic forward failure')), self.assertRaises(RuntimeError):
            module.extract_branches(model, torch.zeros(2, 2, 256))
        self.assertEqual(self.hook_counts(model), before)

    def test_nonzero_cfo_coordinate_is_rejected_and_hooks_are_removed(self):
        module = self.module()
        model = FakeReader().eval().requires_grad_(False)
        def unmask(m, args, value):
            result = value.clone()
            result[:, 27] = 1.
            return result
        handle = model.response.register_forward_hook(unmask)
        before = self.hook_counts(model)
        try:
            with self.assertRaises(ValueError):
                module.extract_branches(model, torch.zeros(2, 2, 256))
            self.assertEqual(self.hook_counts(model), before)
        finally:
            handle.remove()

    def test_parent_identity_mismatch_is_rejected_before_checkpoint_read(self):
        module = self.module()
        for key, value in [('model_seed', 2026092702), ('source_cache', 'wrong-cache'), ('source_contract', 'wrong-contract'),
                           ('fold', dict(unknown_source_labels=[2, 3], heldout_source_rx=8)),
                           ('code_runtime', dict(base_release='other', frozen_sources={key: key for key in ('F', 'AF', 'BASE', 'FOLD', 'G')}))]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                config, _, _ = self.fixture(Path(tmp))
                path = Path(config['source_model_root']) / 'complete.json'
                done = json.loads(path.read_text())
                done['config'][key] = value
                path.write_text(json.dumps(done), encoding='utf-8')
                with patch('torch.load', side_effect=AssertionError('Parent mismatch must keep weights closed')), self.assertRaises(ValueError):
                    module.export(config, 'cpu')

    def test_six_class_checkpoint_is_rejected_before_source_data(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, reader, _ = self.fixture(Path(tmp))
            path = Path(config['source_model_root']) / 'source_model.pt'
            payload = torch.load(path, weights_only=False)
            payload['state_dict']['core.id_backbone.cls_head.weight'] = torch.zeros(6, 160)
            torch.save(payload, path)
            with patch.object(module, 'build_model', return_value=reader), patch.object(module, '_load_smoke_x', side_effect=AssertionError('No data')), self.assertRaises(ValueError):
                module.export(config, 'cpu')

    def test_export_writes_exact_schema_and_marker_last(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, reader, data = self.fixture(root)
            original_save = np.savez
            def save(path, **values):
                self.assertFalse((root / 'output/complete.json').exists())
                self.assertTrue((root / 'output/smoke.json').is_file())
                return original_save(path, **values)
            def load_fold(c, contract):
                self.assertTrue((root / 'output/smoke.json').is_file(), 'V must stay closed until smoke passes')
                return data
            with patch.object(module, 'build_model', return_value=reader), patch.object(module, '_load_smoke_x', return_value=data['train_x'][:2]), \
                 patch.object(module, '_load_fold', side_effect=load_fold), patch.object(module.np, 'savez', side_effect=save):
                result = module.export(config, 'cpu')
            self.assertEqual(result['status'], 'SOURCE_BRANCHES_EXPORTED')
            self.assertEqual((result['parent_epochs'], result['parent_updates']), (20, 540))
            self.assertFalse(result['head_changed'])
            self.assertEqual(result['optimizer_updates'], 0)
            expected = {'train_y', 'train_ids', 'validation_logits', 'validation_y', 'validation_ids', 'validation_receivers'}
            expected |= {role + '_features__' + name for role in ('train', 'validation') for name in module.BRANCHES}
            with np.load(root / 'output/source_branches.npz', allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), expected)
                self.assertEqual(len(archive.files), 22)
                self.assertEqual(archive['validation_logits'].shape, (27000, 4))
                for name, dimension in module.DIMENSIONS.items():
                    self.assertEqual(archive['train_features__' + name].shape, (3360, dimension))
                    self.assertEqual(archive['validation_features__' + name].dtype, np.dtype('float32'))

    def test_output_and_input_boundaries(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _, _ = self.fixture(root)
            for key in ('target', 'truth', 'public_root', 'checkpoint', 'teacher', 'unknown_classes'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: 'forbidden'}))
            (root / 'output').mkdir()
            with patch('torch.load', side_effect=AssertionError('No reads on existing output')), self.assertRaises(FileExistsError):
                module.export(config, 'cpu')

    def test_export_failure_leaves_no_complete_marker(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, reader, data = self.fixture(root)
            with patch.object(module, 'build_model', return_value=reader), patch.object(module, '_load_smoke_x', return_value=data['train_x'][:2]), \
                 patch.object(module, '_load_fold', return_value=data), patch.object(module, '_export_arrays', side_effect=OSError('Incomplete export')):
                with self.assertRaises(OSError):
                    module.export(config, 'cpu')
            self.assertFalse((root / 'output/complete.json').exists())

    def test_cli_help(self):
        self.module()
        result = subprocess.run([sys.executable, '-B', '-m', 'experiments.pamf_phase3_open_world.branch_export', '--help'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--config', result.stdout)


if __name__ == '__main__':
    unittest.main()
