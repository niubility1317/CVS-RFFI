import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class FakeReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.id_backbone = torch.nn.Module()
        self.core.id_backbone.cls_head = torch.nn.Linear(160, 4, bias=False)
        self.forward_labels = []

    def features(self, x):
        if self.training or torch.is_grad_enabled():
            raise AssertionError('Only frozen eval no_grad forwards are legal')
        self.forward_labels.extend(x[:, 0, 0].tolist())
        return x[:, 0, :160].clone()

    def classify_features(self, features):
        return self.core.id_backbone.cls_head(features)


class StreamExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.stream_export')

    def fixture(self, root):
        from experiments.pamf_phase3_open_world.source_fold import select_fold
        cache, parent, branches = root / 'cache', root / 'parent', root / 'branches'
        for path in (cache, parent, branches):
            path.mkdir()
        metadata = {}
        for role, cell in [('L_s', 70), ('V', 300)]:
            rows = [(y, rx, day) for y in range(6) for rx in [1, 3, 4, 6, 8] for day in [1, 2, 3] for _ in range(cell)]
            y, rx, day = np.asarray(rows).T
            n = len(rows)
            metadata[role] = dict(y=y, rx_i=rx, day_i=day, eq_i=np.ones(n, dtype=np.int64),
                sig_i=np.arange(n), base_index=np.arange(n),
                domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1,
                ids=np.asarray([role + '-canonical-%05d' % i for i in range(n)]))
            np.savez(cache / (role + '_metadata.npz'), **metadata[role])
        contract = dict(source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3], num_classes=6,
            target_access_before_freeze=False, target_constructed=False,
            classes=['tx%d' % i for i in range(6)], counts=dict(L_s=6300, V=27000),
            role_ids={role: data['ids'].tolist() for role, data in metadata.items()})
        contract_path = root / 'contract.json'; contract_path.write_text(json.dumps(contract), encoding='utf-8')
        fold = dict(unknown_source_labels=[0, 1], heldout_source_rx=8)
        selected = select_fold(metadata['L_s'], metadata['V'], contract, fold)
        runtime = dict(base_release='base', frozen_sources={key: key for key in ['F', 'AF', 'BASE', 'FOLD', 'G']})
        previous = dict(run_id='parent-run', row_id='fold01', code_commit='parent-commit', code_runtime=runtime,
            source_cache=str(cache), source_contract=str(contract_path), fold=fold, model_seed=2026092701,
            output_root=str(parent), epochs=20, batch_size=128, shuffle_seed=2026102802,
            optimizer=dict(name='AdamW', lr=.0003, weight_decay=.0001, betas=[.9, .999], eps=1e-8),
            schedule='cosine_per_step', label_smoothing=0.)
        parent_marker = dict(status='SOURCE_DEVELOPMENT_FROZEN', source_only=True, target_reads=0, U_reads=0,
            epochs=20, updates=540, checkpoint_initialization='scratch', train_count=3360,
            validation_count=27000, head_joint=True, checkpoint_selection='fixed_final_epoch20',
            checkpoint_ref='source_model.pt', config=previous, manifest=selected['manifest'])
        (parent / 'complete.json').write_text(json.dumps(parent_marker), encoding='utf-8')
        model = FakeReader()
        checkpoint = dict(state_dict=model.state_dict(), config=previous, manifest=selected['manifest'],
            epoch=20, updates=540, source_only=True, checkpoint_initialization='scratch', target_reads=0, U_reads=0)
        torch.save(checkpoint, parent / 'source_model.pt')
        base = dict(run_id='joint-run', row_id='joint', code_commit='new-commit', code_runtime=runtime,
            source_model_root=str(parent), source_cache=str(cache), source_contract=str(contract_path),
            model_seed=2026092701, fold=fold, batch_size=128, output_root=str(root / 'output'))
        branch_config = dict(base, run_id='branch-run', output_root=str(branches))
        branch_marker = dict(status='SOURCE_BRANCHES_EXPORTED', source_only=True, target_reads=0,
            optimizer_updates=0, head_changed=False, parent_epochs=20, parent_updates=540,
            train_count=3360, validation_count=27000, heldout_source_rx=8,
            config=branch_config, source_manifest=selected['manifest'], features_ref='source_branches.npz')
        (branches / 'complete.json').write_text(json.dumps(branch_marker), encoding='utf-8')
        train = np.zeros((3360, 160), dtype=np.float32); train[np.arange(3360), selected['train_y']] = 1.
        validation = np.zeros((27000, 160), dtype=np.float32)
        validation[np.arange(27000), selected['validation_y']] = 1.
        np.savez(branches / 'source_branches.npz', train_features__fused160=train,
            train_y=selected['train_y'], train_ids=selected['train_ids'], validation_features__fused160=validation,
            validation_logits=np.zeros((27000, 4), dtype=np.float32), validation_y=selected['validation_y'],
            validation_ids=selected['validation_ids'], validation_receivers=selected['validation_receivers'],
            ignored_other_branch=np.asarray([object()]))
        labeled_x = torch.zeros(6300, 2, 256)
        labeled_x[:, 0, 0] = torch.tensor(metadata['L_s']['y'].tolist(), dtype=torch.float32)
        return dict(base, branch_export_root=str(branches)), contract, metadata, selected, model, labeled_x

    def test_complete_fields_roles_and_label1_never_forwarded(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, contract, meta, selected, model, x = self.fixture(root)
            with patch.object(module.branch_export, 'build_model', return_value=model), \
                 patch.object(module, '_load_labeled', return_value=(meta['L_s'], x)), \
                 patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('Use the tolist bridge')):
                result = module.export(config, 'cpu')
            self.assertEqual(result['status'], 'SOURCE_STREAM_INPUTS_EXPORTED')
            self.assertEqual(result['calibration_count'], 14400)
            self.assertEqual(len(model.forward_labels), 1052)
            self.assertNotIn(1., model.forward_labels)
            self.assertEqual(model.forward_labels.count(0.), 210)
            output = root / 'output'
            with np.load(output / 'source_inputs.npz', allow_pickle=False) as inputs:
                self.assertEqual(set(inputs.files), {'train_features', 'train_y', 'calibration_features'})
                self.assertEqual(inputs['train_features'].shape, (3360, 160))
                self.assertEqual(inputs['calibration_features'].shape, (14400, 160))
                self.assertTrue(np.all(inputs['calibration_features'][:, 4:] == 0))
            seen = set()
            for role, count, labels in [('discovery', 1050, set(range(5))), ('evaluation', 5400, set(range(6)))]:
                public = output / role / 'public'
                self.assertEqual({p.name for p in public.iterdir()}, {'features.npz'})
                with np.load(public / 'features.npz', allow_pickle=False) as features:
                    self.assertEqual(set(features.files), {'ids', 'features', 'logits'})
                    self.assertEqual(features['features'].shape, (count, 160))
                    self.assertEqual(features['logits'].shape, (count, 4))
                    self.assertEqual(features['features'].dtype, np.dtype('float32'))
                    self.assertEqual(features['ids'].dtype.kind, 'U')
                    ids = features['ids'].copy()
                self.assertFalse(seen & set(ids.tolist())); seen.update(ids.tolist())
                self.assertTrue(all(value.startswith('p3s-') and len(value) == 28 for value in ids))
                with np.load(output / role / 'truth/truth.npz', allow_pickle=False) as truth:
                    self.assertEqual(set(truth.files), {'ids', 'y'})
                    np.testing.assert_array_equal(truth['ids'], ids)
                    self.assertEqual(set(truth['y'].tolist()), labels)
            report = json.loads((output / 'builder_report.json').read_text())
            self.assertTrue(report['full_L_loaded_for_private_selection_only'])
            self.assertFalse(report['new_blind_target_evidence'])
            self.assertTrue((output / 'complete.json').is_file())

    def test_smoke_precedes_frozen_v_read_and_completion_is_last(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, contract, meta, selected, model, x = self.fixture(root)
            original, writes = module._write, []
            def writing(path, value):
                self.assertFalse((root / 'output/complete.json').exists())
                writes.append(Path(path).name)
                if Path(path).name == 'complete.json':
                    self.assertTrue((root / 'output/evaluation/public/features.npz').is_file())
                    self.assertTrue((root / 'output/evaluation/truth/truth.npz').is_file())
                    self.assertTrue((root / 'output/fold_manifest.json').is_file())
                return original(path, value)
            frozen = module._load_frozen
            def after_smoke(*args):
                self.assertTrue((root / 'output/smoke.json').is_file())
                self.assertEqual(len(model.forward_labels), 2)
                return frozen(*args)
            with patch.object(module.branch_export, 'build_model', return_value=model), \
                 patch.object(module, '_load_labeled', return_value=(meta['L_s'], x)), \
                 patch.object(module, '_load_frozen', side_effect=after_smoke), patch.object(module, '_write', side_effect=writing):
                module.export(config, 'cpu')
            self.assertEqual(writes[0], 'smoke.json')
            self.assertEqual(writes[-1], 'complete.json')

    def test_parent_and_checkpoint_errors_precede_source_io(self):
        module = self.module()
        for failure in ('parent', 'checkpoint'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, contract, meta, selected, model, x = self.fixture(root)
                if failure == 'parent':
                    marker = json.loads((root / 'parent/complete.json').read_text()); marker['target_reads'] = 1
                    (root / 'parent/complete.json').write_text(json.dumps(marker), encoding='utf-8')
                else:
                    checkpoint = torch.load(root / 'parent/source_model.pt', weights_only=False)
                    checkpoint['state_dict']['core.id_backbone.cls_head.weight'] = torch.zeros(6, 160)
                    torch.save(checkpoint, root / 'parent/source_model.pt')
                with patch.object(module.branch_export, 'build_model', return_value=model), \
                     patch.object(module, '_load_labeled', side_effect=AssertionError('Source must not open')), self.assertRaises(ValueError):
                    module.export(config, 'cpu')
                self.assertFalse((root / 'output/complete.json').exists())

    def test_branch_identity_mismatch_and_physical_ids_are_rejected(self):
        module = self.module()
        for field in ('identity', 'train_ids', 'validation_ids'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, contract, meta, selected, model, x = self.fixture(root)
                if field == 'identity':
                    marker = json.loads((root / 'branches/complete.json').read_text())
                    marker['config']['model_seed'] += 1
                    (root / 'branches/complete.json').write_text(json.dumps(marker), encoding='utf-8')
                else:
                    path = root / 'branches/source_branches.npz'
                    with np.load(path, allow_pickle=False) as stored:
                        arrays = {k: stored[k].copy() for k in stored.files if k != 'ignored_other_branch'}
                    arrays[field] = arrays[field][::-1].copy()
                    np.savez(path, **arrays)
                with patch.object(module.branch_export, 'build_model', return_value=model), \
                     patch.object(module, '_load_labeled', return_value=(meta['L_s'], x)), self.assertRaises(ValueError):
                    module.export(config, 'cpu')
                self.assertFalse((root / 'output/complete.json').exists())

    def test_smoke_failure_blocks_v_and_output_collision_precedes_reads(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, contract, meta, selected, model, x = self.fixture(root)
            with patch.object(module.branch_export, 'build_model', return_value=model), \
                 patch.object(module, '_load_labeled', return_value=(meta['L_s'], x)), \
                 patch.object(model, 'features', side_effect=RuntimeError('synthetic failure')), \
                 patch.object(module, '_load_frozen', side_effect=AssertionError('V may not open')), self.assertRaises(RuntimeError):
                module.export(config, 'cpu')
            self.assertFalse((root / 'output/complete.json').exists())
            with patch.object(module.branch_export, '_parent', side_effect=AssertionError('Collision before all reads')), self.assertRaises(FileExistsError):
                module.export(config, 'cpu')

    def test_metadata_selector_and_config_do_not_accept_extra_roles(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, contract, meta, selected, model, x = self.fixture(root)
            for key in ('target', 'U_s', 'teacher_checkpoint'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: 'forbidden'}))
            invalid = dict(meta['L_s']); invalid['rx_i'] = invalid['rx_i'].copy(); invalid['rx_i'][0] = 99
            with patch.object(module.branch_export, 'build_model', return_value=model), \
                 patch.object(module, '_load_labeled', return_value=(invalid, x)), self.assertRaises(ValueError):
                module.export(config, 'cpu')
            self.assertEqual(len(model.forward_labels), 0)


if __name__ == '__main__':
    unittest.main()
