import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


class Core(torch.nn.Module):
    def __init__(self, classes):
        super().__init__(); self.cls_head = torch.nn.Linear(160, classes, bias=False)
        self.freq_gate = torch.nn.Identity(); self.pa_lift = torch.nn.Identity()

    def forward(self, x):
        self.freq_gate(x[:, 0, :128].reshape(len(x), 4, 32))
        self.pa_lift(x.repeat(1, 12, 1))
        z = x[:, 0, :160]
        return dict(t_emb=z, f_emb=z + 1, feat_pa=z + 2, feat_joint=z + 3,
                    logits=torch.full((len(x), self.cls_head.weight.shape[0]), -999.))


class Response(torch.nn.Module):
    def forward(self, x):
        z = x[:, 0, :29].clone(); z[:, 27] = 0.; return z


class View(torch.nn.Module):
    def forward(self, x):
        return x[:, :112]


class Pool(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.views = torch.nn.ModuleList([View() for _ in range(3)])
        self.output_projection = torch.nn.Linear(336, 160, bias=False)

    def forward(self, x):
        return self.output_projection(torch.cat([view(x) for view in self.views], dim=1))


class Reader(torch.nn.Module):
    def __init__(self, classes=6):
        super().__init__(); self.core = torch.nn.Module(); self.core.id_backbone = Core(classes)
        self.response = Response(); self.sequence_pool = Pool()
        self.response_projection = torch.nn.Linear(29, 160, bias=False)
        remaining = (232993 if classes == 6 else 232673) - sum(p.numel() for p in self.parameters())
        self.fixture_padding = torch.nn.Parameter(torch.zeros(remaining))
        self.seen = 0

    def features(self, x):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('Frozen no_grad export required')
        self.seen += len(x)
        aux = self.core.id_backbone(x)
        return aux['feat_joint'] + self.response_projection(self.response(x)) + self.sequence_pool(aux['t_emb'])

    def classify_features(self, z):
        return self.core.id_backbone.cls_head(z)


SMALL = dict(source_L=6, source_V=6, exposure_train=6, novel_validation=6)


class SourceOpenExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_open_export')

    def fixture(self, root):
        contract = dict(classes=['old%d' % i for i in range(6)], counts=dict(L_s=6, V=6),
            source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3],
            role_ids={role: [role + '-physical-%d' % i for i in range(6)] for role in ['L_s', 'V']})
        path = root / 'contract.json'; path.write_text(json.dumps(contract), encoding='utf-8')
        oe = root / 'oe'; oe.mkdir()
        handles = {}
        for role in ('exposure_train', 'novel_validation'):
            public = oe / role / 'public'; public.mkdir(parents=True)
            np.save(public / 'iq.npy', np.ones((6, 2, 256), dtype=np.float32))
            np.save(public / 'ids.npy', np.asarray(['opaque-' + role + '-%d' % i for i in range(6)]))
            rx, day = np.asarray([1, 3, 4, 6, 8, 1], dtype=np.int64), np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int64)
            np.savez(public / 'metadata.npz', rx_i=rx, day_i=day, domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1)
            handles[role] = dict(public_root=str(public), count=6, capsule_id='capsule-' + role,
                split_id='split-' + role, data_status='VALIDATED_ONCE', iq_shape=[6, 2, 256], iq_dtype='float32')
        marker = dict(status='SOURCE_OE_DATA_VALIDATED', data_status='VALIDATED_ONCE', source_only=True,
            target_reads=0, counts={role: 6 for role in handles}, original_L_s_reads=0, original_U_s_reads=0,
            original_V_reads=0, model_reads=0, model_predictions=0, optimizer_updates=0, channel_observations_generated=0,
            source_domains=dict(receiver_names=['1-19', '18-2', '19-2', '2-19', '3-19'],
                receiver_indices=[1, 3, 4, 6, 8], dates=['2021_03_08', '2021_03_15', '2021_03_23'], day_indices=[1, 2, 3], domain_count=15),
            config={'exposure_train_classes': 'PRIVATE_DO_NOT_FORWARD'}, **handles)
        (oe / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
        config = dict(run_id='source-open', row_id='s1', model_seed=2026092701, base_release='base',
            frozen_sources={name: name for name in ('F', 'AF', 'BASE', 'FOLD', 'G')}, source_output='own-full6',
            source_contract=str(path), registered_classes=contract['classes'], code_commit='commit', batch_size=128,
            output_root=str(root / 'output'), source_cache='source-cache', retention=.99, oe_data_root=str(oe))
        def source(config, role, contract):
            self.assertIn(role, ['L_s', 'V'])
            rx, day = np.array([1, 3, 4, 6, 8, 1]), np.array([1, 2, 3, 1, 2, 3])
            return SimpleNamespace(x=torch.randn(6, 2, 256), metadata=dict(ids=np.asarray(contract['role_ids'][role]),
                y=np.arange(6), rx_i=rx, day_i=day, domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1,
                eq_i=np.ones(6, dtype=np.int64)))
        return config, contract, source, Reader().eval().requires_grad_(False)

    def test_export_roles_schema_smoke_order_no_truth_and_marker_last(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, contract, source, reader = self.fixture(root)
            calls, writes = [], []
            def source_checked(c, role, contract):
                if role == 'V':
                    self.assertTrue((root / 'output/smoke.json').is_file())
                calls.append(role); return source(c, role, contract)
            original_oe, original_write, original_read, original_load = module._oe_handles, module._write, module._read, np.load
            def oe_checked(*args):
                self.assertTrue((root / 'output/smoke.json').is_file()); return original_oe(*args)
            def read_checked(path):
                self.assertNotIn('truth', str(path)); self.assertNotIn('builder_report', str(path)); return original_read(path)
            def load_checked(path, *args, **kwargs):
                self.assertNotIn('truth', str(path)); self.assertNotIn('builder_report', str(path))
                return original_load(path, *args, **kwargs)
            def writing(path, value):
                self.assertFalse((root / 'output/complete.json').exists()); writes.append(Path(path).name)
                if Path(path).name == 'complete.json':
                    self.assertTrue((root / 'output/novel_validation/public/features.npz').is_file())
                return original_write(path, value)
            with patch.object(module, 'COUNTS', SMALL), patch.object(module.runner, '_load_model', return_value=reader), \
                 patch.object(module.runner, '_source_dataset', side_effect=source_checked), \
                 patch.object(module, '_oe_handles', side_effect=oe_checked), patch.object(module, '_write', side_effect=writing), \
                 patch.object(module, '_read', side_effect=read_checked), patch.object(module.np, 'load', side_effect=load_checked), \
                 patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('tolist bridge only')):
                result = module.export(config, 'cpu')
            self.assertEqual(calls, ['L_s', 'V']); self.assertEqual(reader.seen, 26)
            self.assertEqual(result['counts'], SMALL); self.assertEqual(result['model_classes'], 6)
            self.assertEqual(result['parameters'], 232993); self.assertEqual(result['status'], 'SOURCE_OPEN_FEATURES_EXPORTED')
            self.assertEqual(writes[0], 'smoke.json'); self.assertEqual(writes[-1], 'complete.json')
            for role, ref in result['feature_refs'].items():
                path = root / 'output' / ref
                if role in ['exposure_train', 'novel_validation']:
                    self.assertEqual({p.name for p in path.parent.iterdir()}, {'features.npz'})
                with np.load(path, allow_pickle=False) as stored:
                    expected = {'ids', 'logits', 'rx_i', 'day_i'} | {'features__' + name for name in module.BRANCHES}
                    if role.startswith('source_'):
                        expected.add('y')
                        np.testing.assert_array_equal(stored['ids'], contract['role_ids']['L_s' if role == 'source_L' else 'V'])
                    self.assertEqual(set(stored.files), expected)
                    self.assertEqual(stored['logits'].shape, (6, 6))
                    for name, dimension in module.DIMENSIONS.items():
                        self.assertEqual(stored['features__' + name].shape, (6, dimension))
                        self.assertEqual(stored['features__' + name].dtype, np.dtype('float32'))
                    self.assertTrue((stored['features__reference29'][:, 27] == 0).all())
            self.assertNotIn('config', result['source_OE_handles']['exposure_train'])

    def test_real_scratch_six_head_and_same_forward_state_preservation(self):
        module = self.module()
        from experiments.pamf_phase3_open_world.source_development import build_model
        root = Path(__file__).resolve().parents[1]
        frozen = root / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        runtime = dict(base_release=str(root), frozen_sources={name: str(frozen / name) for name in ('F', 'AF', 'BASE', 'FOLD', 'G')})
        with patch('torch.load', side_effect=AssertionError('No real checkpoint')), patch('numpy.load', side_effect=AssertionError('No data')):
            reader = build_model(runtime, 2026092701)
            reader.core.id_backbone.cls_head.weight = torch.nn.Parameter(torch.randn(6, 160))
            reader.eval().requires_grad_(False)
            before = {name: value.clone() for name, value in reader.state_dict().items()}
            x = torch.randn(2, 2, 256)
            branches, logits = module.extract_branches(reader, x)
            self.assertEqual(tuple(logits.shape), (2, 6))
            self.assertEqual(sum(p.numel() for p in reader.parameters()), 232993)
            self.assertTrue(torch.allclose(logits, reader.classify_features(branches['fused160'])))
            self.assertTrue(all(torch.equal(before[name], value) for name, value in reader.state_dict().items()))

    def test_four_head_rejected_and_local_hooks_removed_on_failure(self):
        module = self.module()
        reader = Reader(4).eval().requires_grad_(False)
        with self.assertRaises(ValueError):
            module.extract_branches(reader, torch.ones(2, 2, 256))
        six = Reader().eval().requires_grad_(False)
        baseline = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in six.modules()]
        with patch.object(six.response, 'forward', side_effect=RuntimeError('synthetic')), self.assertRaises(RuntimeError):
            module.extract_branches(six, torch.ones(2, 2, 256))
        self.assertEqual(baseline, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in six.modules()])
        with patch.object(six.response, 'forward', side_effect=lambda x: x[:, 0, :29].clone()), self.assertRaises(ValueError):
            module.extract_branches(six, torch.ones(2, 2, 256))
        self.assertEqual(baseline, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in six.modules()])

    def test_smoke_failure_blocks_v_oe_and_collision_blocks_all_reads(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, contract, source, reader = self.fixture(root)
            def only_l(c, role, contract):
                self.assertEqual(role, 'L_s'); return source(c, role, contract)
            with patch.object(module, 'COUNTS', SMALL), patch.object(module.runner, '_load_model', return_value=reader), \
                 patch.object(module.runner, '_source_dataset', side_effect=only_l), \
                 patch.object(reader, 'features', side_effect=RuntimeError('smoke failure')), \
                 patch.object(module, '_oe_handles', side_effect=AssertionError('OE forbidden before smoke')), self.assertRaises(RuntimeError):
                module.export(config, 'cpu')
            self.assertFalse((root / 'output/complete.json').exists())
            with patch.object(module.runner, '_prepare', side_effect=AssertionError('No reads after collision')), self.assertRaises(FileExistsError):
                module.export(config, 'cpu')

    def test_source_order_oe_extra_label_and_domain_mismatch_fail_without_complete(self):
        module = self.module()
        for failure in ('source_order', 'oe_label', 'oe_domain', 'oe_marker', 'oe_count', 'oe_missing'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); config, contract, source, reader = self.fixture(root)
                def dataset(c, role, contract):
                    data = source(c, role, contract)
                    if failure == 'source_order' and role == 'V':
                        data.metadata['ids'] = data.metadata['ids'][::-1].copy()
                    return data
                path = root / 'oe/exposure_train/public/metadata.npz'
                if failure in ('oe_label', 'oe_domain'):
                    with np.load(path, allow_pickle=False) as stored:
                        arrays = {name: stored[name].copy() for name in stored.files}
                    if failure == 'oe_label': arrays['y'] = np.zeros(6, dtype=np.int64)
                    else: arrays['domain'][0] = 99
                    np.savez(path, **arrays)
                if failure == 'oe_marker':
                    path = root / 'oe/complete.json'; marker = json.loads(path.read_text()); marker['target_reads'] = 1
                    path.write_text(json.dumps(marker), encoding='utf-8')
                if failure == 'oe_count':
                    path = root / 'oe/complete.json'; marker = json.loads(path.read_text()); marker['exposure_train']['count'] = 7
                    path.write_text(json.dumps(marker), encoding='utf-8')
                if failure == 'oe_missing':
                    (root / 'oe/exposure_train/public/iq.npy').unlink()
                with patch.object(module, 'COUNTS', SMALL), patch.object(module.runner, '_load_model', return_value=reader), \
                     patch.object(module.runner, '_source_dataset', side_effect=dataset), self.assertRaises(ValueError):
                    module.export(config, 'cpu')
                self.assertFalse((root / 'output/complete.json').exists())

    def test_relocated_validated_oe_uses_current_root_and_ignores_old_builder_paths(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, contract, source, reader = self.fixture(root)
            marker_path = root / 'oe/complete.json'; marker = json.loads(marker_path.read_text())
            for role in ('exposure_train', 'novel_validation'):
                marker[role]['public_root'] = 'Z:/UNREACHABLE_OLD_WINDOWS/' + role + '/public'
                marker[role]['truth_path'] = 'Z:/UNREACHABLE_OLD_WINDOWS/' + role + '/truth/truth.npz'
            marker_path.write_text(json.dumps(marker), encoding='utf-8')
            original = np.load
            def current_files_only(path, *args, **kwargs):
                self.assertNotIn('UNREACHABLE_OLD_WINDOWS', str(path)); return original(path, *args, **kwargs)
            with patch.object(module, 'COUNTS', SMALL), patch.object(module.runner, '_load_model', return_value=reader), \
                 patch.object(module.runner, '_source_dataset', side_effect=source), \
                 patch.object(module.np, 'load', side_effect=current_files_only):
                result = module.export(config, 'cpu')
            self.assertEqual(result['status'], 'SOURCE_OPEN_FEATURES_EXPORTED')
            self.assertEqual(result['source_OE_handles']['exposure_train']['capsule_id'], 'capsule-exposure_train')


if __name__ == '__main__':
    unittest.main()
