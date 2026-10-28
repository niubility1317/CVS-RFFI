import copy
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


class ToySinc(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_coordinate = nn.Parameter(torch.tensor([[.3], [.7]]))
        self.band_coordinate = nn.Parameter(torch.tensor([[.2], [.5]]))

    def forward(self, x):
        return x * (self.low_coordinate.sum() + self.band_coordinate.sum())


class ToyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.id_backbone = nn.Module()
        self.core.id_backbone.sinc = ToySinc()
        self.bn = nn.BatchNorm1d(2)
        self.core.id_backbone.cls_head = nn.Linear(2, 6, bias=False)
        self.seen = []

    def features(self, x):
        self.seen.append((self.training, len(x), torch.is_grad_enabled()))
        # Exercise RNG restoration even when OE is in eval mode.
        torch.rand(1)
        z = self.core.id_backbone.sinc(x.mean(-1))
        return self.bn(z)

    def classify_features(self, z):
        return self.core.id_backbone.cls_head(z)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.encoder = ToyReader()


class EmbeddingTrainTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_open_embedding_train.py'
        self.assertTrue(path.is_file(), 'Source joint embedding training is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_open_embedding_train')

    def config(self, output='new-output', arm='supervised_only'):
        value = dict(run_id='embedding-r01', row_id=arm, model_seed=2026092701,
            base_release='base', frozen_sources={k: k for k in ('F', 'AF', 'BASE', 'FOLD', 'G')},
            source_output='original-E50', source_contract='source_contract.json', source_cache='source-cache',
            registered_classes=['14-10', '14-7', '20-15', '20-19', '6-15', '8-20'],
            code_commit='a' * 40, batch_size=128, output_root=output, arm=arm, steps=222,
            lr=.0001, L_seed=2026092701, OE_seed=2026102907)
        if arm == 'uniform_oe':
            value['diverse_data_root'] = 'source21'
        return value

    def test_exact_inputs_fixed_budget_and_ce_never_has_oe_path(self):
        module = self.module()
        module.validate_config(self.config())
        module.validate_config(self.config(arm='uniform_oe'))
        for extra in ('target', 'source_V', 'U_s', 'checkpoint', 'teacher', 'novel_validation', 'diverse_data_root'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                module.validate_config(dict(self.config(), **{extra: 'forbidden'}))
        for key, bad in [('steps', 221), ('lr', .001), ('L_seed', 392005), ('OE_seed', 1), ('batch_size', 64)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_config(dict(self.config(), **{key: bad}))

    def test_uniform_loss_has_exact_formula_and_moves_logits_towards_uniform(self):
        module = self.module()
        logits = torch.tensor([[4., 0., -1., 2., 1., -2.]], requires_grad=True)
        loss = module.uniform_oe_loss(logits)
        self.assertAlmostEqual(float(loss.detach()), float((-logits.log_softmax(1).mean() - math.log(6)).detach()), places=6)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad, (logits.softmax(1) - 1 / 6) / len(logits)))
        self.assertAlmostEqual(float(module.uniform_oe_loss(torch.zeros(2, 6))), 0., places=6)
        for bad in (torch.zeros(2, 4), torch.full((2, 6), float('nan'))):
            with self.assertRaises(ValueError):
                module.uniform_oe_loss(bad)

    def test_oe_eval_keeps_gradients_bn_flags_and_l_rng(self):
        module = self.module()
        model = ToyModel().train()
        reader = model.id_backbone.encoder
        before = {k: v.clone() for k, v in model.named_buffers()}
        rng = torch.random.get_rng_state().clone()
        loss = module.oe_forward(model, torch.randn(2, 2, 256, generator=torch.Generator().manual_seed(1)))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        loss.backward()
        self.assertTrue(model.training and reader.training)
        self.assertEqual(reader.seen[-1], (False, 2, True))
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertGreater(float(reader.core.id_backbone.sinc.low_coordinate.grad.norm()), 0.)
        self.assertGreater(float(reader.core.id_backbone.cls_head.weight.grad.norm()), 0.)

    def test_gradient_smoke_restores_state_rng_and_clears_grads_even_on_failure(self):
        module = self.module()
        model = ToyModel().train()
        x = torch.randn(2, 2, 256)
        y = torch.tensor([0, 1])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        result = module.gradient_smoke(model, x, y)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertGreater(result['sinc_gradient_norm'], 0.)
        self.assertGreater(result['head_gradient_norm'], 0.)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        with patch.object(model.id_backbone.encoder, 'features', side_effect=RuntimeError('failed forward')):
            with self.assertRaises(RuntimeError):
                module.gradient_smoke(model, x, y)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_domain_sampler_uses_private_rng_uniform_domains_and_restorable_smoke(self):
        module = self.module()
        domains = np.repeat(np.arange(15), np.arange(1, 16))
        sampler = module.DomainSampler(domains, 2026102907)
        rng = torch.random.get_rng_state().clone()
        state = sampler.generator.get_state().clone()
        first = sampler.sample(64)
        sampler.generator.set_state(state)
        self.assertTrue(np.array_equal(first, sampler.sample(64)))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        values = sampler.sample(30000)
        counts = np.bincount(domains[values], minlength=15)
        self.assertLess(int(counts.max() - counts.min()), 450)
        with self.assertRaises(ValueError):
            module.DomainSampler(np.arange(14), 1)

    def test_parent_schedule_is_e51_without_u_and_keeps_parent_immutable(self):
        module = self.module()
        original = dict(model_seed=2026092701, source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3],
            ablation='F21', regime='native_cric', U_use=True, own_EMA=False, native_U_gate=False,
            repl_row=dict(structure='SHUFFLE_TIME', L_A=True, L_B=True, U=True),
            core_mechanisms=None)
        saved = copy.deepcopy(original)
        value = module._training_config(self.config(), original)
        self.assertEqual(original, saved)
        self.assertEqual(value['step_offset'], 11100)
        self.assertEqual(value['epoch_offset'], 50)
        self.assertEqual(value['ablation'], 'F20')
        self.assertEqual(value['regime'], 'matched_sat_supervised')
        self.assertFalse(value['U_use'] or value['native_unlabeled_loop'] or value['own_EMA'])
        for step in (1, 222):
            self.assertEqual((value['step_offset'] + step - 1) // 222 + 1, 51)

    def test_native_supervised_function_is_reused_and_counts_real_forward_roles(self):
        module = self.module()
        from contextlib import contextmanager
        model = ToyModel().train()
        calls = []
        @contextmanager
        def context():
            calls.append('L_context')
            yield
        model._phase1_labeled_feature_context = context
        def forward(x, **kw):
            calls.append(('ordinary', len(x), kw['return_aux']))
            return {'tx_logits': model.id_backbone.encoder.classify_features(model.id_backbone.encoder.features(x))}
        model.forward = forward
        def hook(**kw):
            calls.append(('native_hook', kw['optimizer_step'], kw['labeled_count']))
            self.assertIs(kw['labeled_inputs'], raw)
            return kw['identity_loss'] * .5, {'CRIC_supcon_weight': .1, 'T1_L_packets': 768, 'T1_PL_seen': 0}
        model._source_labeled_feature_hook = hook
        raw = torch.randn(128, 2, 256)
        loss, metrics = module.supervised_loss(model, raw, torch.arange(128) % 6,
            torch.arange(128) % 15, {'rx_i': torch.ones(128)}, lambda x, **kw: x, False, 1)
        self.assertGreater(float(loss.detach()), 0.)
        self.assertEqual(calls, ['L_context', ('ordinary', 128, True), ('native_hook', 1, 128)])
        self.assertEqual(metrics['T1_L_packets'], 768)

    def test_toy_loop_fixed_updates_ce_avoids_oe_and_oe_has_no_bn_effect(self):
        module = self.module()
        batches = [(torch.randn(128, 2, 256), torch.arange(128) % 6, torch.arange(128) % 15, {})] * 3
        for use_oe in (False, True):
            with self.subTest(use_oe=use_oe):
                model = ToyModel().train()
                opt = torch.optim.AdamW(model.parameters(), lr=.0001)
                def labeled(m, raw, y, d, meta, aug, no_pa, step):
                    self.assertTrue(m.training)
                    logits = m.id_backbone.encoder.classify_features(m.id_backbone.encoder.features(raw))
                    return torch.nn.functional.cross_entropy(logits, y), {'CRIC_supcon_weight': .1, 'T1_L_packets': 768, 'T1_PL_seen': 0}
                oe_calls = []
                def get_oe(count):
                    oe_calls.append(count)
                    self.assertEqual(count, 64)
                    return torch.randn(count, 2, 256, generator=torch.Generator().manual_seed(3))
                trace = module._train_steps(model, batches, opt, lambda x, **kw: x, False, 'cpu',
                    3, get_oe if use_oe else None, lambda o, p: torch.nn.utils.clip_grad_norm_(p, 5.),
                    labeled_loss=labeled)
                self.assertEqual(len(trace), 3)
                self.assertEqual(oe_calls, [64] * 3 if use_oe else [])
                self.assertEqual(int(model.id_backbone.encoder.bn.num_batches_tracked), 3)
                self.assertEqual(trace[-1]['step'], 3)
                self.assertEqual(trace[-1]['cumulative_updates'], 11103)
                self.assertEqual(trace[-1]['oe_samples'], 64 if use_oe else 0)

    def test_output_protection_and_no_complete_on_failure(self):
        module = self.module()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'run'
            root.mkdir()
            with patch.object(module, 'make_trainable_model', side_effect=AssertionError('No model IO')) as loader:
                with self.assertRaises(FileExistsError):
                    module.train(self.config(str(root)), 'cpu')
                self.assertEqual(loader.call_count, 0)
            failed = Path(td) / 'failed'
            with patch.object(module, 'make_trainable_model', side_effect=RuntimeError('bad parent')):
                with self.assertRaises(RuntimeError):
                    module.train(self.config(str(failed)), 'cpu')
            self.assertFalse((failed / 'complete.json').exists())

    def test_synthetic_native_full_six_model_has_sinc_head_gradients_and_center_units_step(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        config = self.config()
        config['base_release'] = str(root)
        frozen = root / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        config['frozen_sources'] = {key: str(frozen / key) for key in ('F', 'AF', 'BASE', 'FOLD', 'G')}
        # Immutable code only. No checkpoint, cache, IQ or truth file is read.
        module.runner._activate(config)
        parent = dict(run_id='parent', row_id='parent', model_seed=2026092701, method_id='pamf_t1_e50',
            source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3], num_classes=6,
            view_comparison={'view_count': 3}, paper_method='synthetic interface test',
            repl_row={'group': 'main', 'structure': 'SHUFFLE_TIME', 'L_A': True, 'L_B': True, 'U': True})
        with patch('torch.load', side_effect=AssertionError('Real weight reads forbidden')):
            model = module._new_native(config, module._training_config(config, parent), 'cpu')
        module._assert_trainable(model)
        self.assertEqual(sum(p.numel() for p in model.id_backbone.encoder.parameters()), 232993)
        x = torch.randn(2, 2, 256, generator=torch.Generator().manual_seed(42))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        result = module.gradient_smoke(model, x, torch.tensor([0, 1]))
        self.assertGreater(result['sinc_gradient_norm'], 0.)
        self.assertGreater(result['head_gradient_norm'], 0.)
        oe = module.gradient_smoke(model, x)
        self.assertGreater(oe['sinc_gradient_norm'], 0.)
        self.assertGreater(oe['head_gradient_norm'], 0.)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(state[key], value), key)
        from experiments.cric_dura_comparison.f_optimizer import build, clip_physical
        opt = build(model, list(model.parameters()), 'CENTER_UNITS', lr=.0001, weight_decay=.0001, foreach=False)
        self.assertFalse(opt.state)
        self.assertEqual(opt.param_groups[1]['lr'], .1)
        model._cd_practical = lambda values, meta, epoch, step: values
        batch = (torch.randn(128, 2, 256), torch.arange(128) % 6, torch.arange(128) % 15,
            {'rx_i': torch.tensor([1, 3, 4, 6, 8] * 26)[:128]})
        trace = module._train_steps(model, [batch], opt, lambda values, **kw: values, False, 'cpu',
            1, None, lambda optimizer, params: clip_physical(optimizer, params, 5.))
        self.assertEqual(opt._sinc_unit_steps, 1)
        self.assertEqual(opt._sinc_physical_clip_calls, 1)
        self.assertEqual(model._cd_calls, 1)
        self.assertEqual(trace[0]['native_metrics']['CRIC_supcon_weight'], .1)
        self.assertEqual(trace[0]['native_metrics']['T1_PL_seen'], 0)
        self.assertEqual(trace[0]['L_packet_views'], 768)
        self.assertTrue(any(not torch.equal(state[key], value) for key, value in model.state_dict().items()))

    def test_increment_loader_rejects_wrong_parent_arm_or_budget_before_state_load(self):
        module = self.module()
        config = self.config()
        model = ToyModel()
        model._phase3_parent_identity = {'run': 'verified-E50'}
        model._phase3_source_handle = {'source': 'original-LUV'}
        payload = dict(schema=module.SCHEMA, identity=module._checkpoint_identity(config),
            original_parent_identity=model._phase3_parent_identity, source_data_handle=model._phase3_source_handle,
            updates=222, cumulative_updates=11322, optimizer_reset=True, U_reads=0, target_reads=0,
            validation_reads=0, model=copy.deepcopy(model.state_dict()))
        for key, wrong in [('identity', {}), ('original_parent_identity', {}), ('updates', 221),
                           ('U_reads', 1), ('validation_reads', 1), ('cumulative_updates', 11100)]:
            with self.subTest(key=key), patch.object(module, 'make_trainable_model', return_value=model), \
                    patch('torch.load', return_value=dict(payload, **{key: wrong})), self.assertRaises(ValueError):
                module.load_trained_model(config, 'new-final.pt', 'cpu')
        with patch.object(module, 'make_trainable_model', return_value=model), \
                patch('torch.load', return_value=payload), patch.object(module, '_assert_trainable'):
            loaded = module.load_trained_model(config, 'new-final.pt', 'cpu')
        self.assertFalse(loaded.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_training_completes_222_steps_and_writes_marker_last_without_ce_oe_io(self):
        module = self.module()
        from contextlib import contextmanager
        import tempfile
        model = ToyModel()
        model._phase3_parent_identity = {'parent': 'verified-E50'}
        model._phase3_parent_args = {'weight_decay': .0001}
        model._phase3_source_handle = {'source': 'original-LUV'}
        model._cd_calls = model._feature_l_steps = model._cd_U_seen = model._cd_U_accepted = 0
        @contextmanager
        def context():
            model._feature_l_steps += 1
            yield
        model._phase1_labeled_feature_context = context
        def forward(x, **kw):
            reader = model.id_backbone.encoder
            return {'tx_logits': reader.classify_features(reader.features(x))}
        model.forward = forward
        def hook(**kw):
            model._cd_calls += 1
            return kw['identity_loss'] * 0, {'CRIC_supcon_weight': .1, 'T1_L_packets': 768, 'T1_PL_seen': 0}
        model._source_labeled_feature_hook = hook
        class Data(torch.utils.data.Dataset):
            role = 'L_s'
            def __init__(self):
                self.x = torch.randn(6300, 2, 256)
                self.metadata = {'y': np.arange(6300) % 6, 'ids': np.asarray(['L' + str(i) for i in range(6300)])}
            def __len__(self):
                return len(self.x)
            def __getitem__(self, i):
                return self.x[i], int(self.metadata['y'][i]), i % 15, {}
        data = Data()
        def build(m, params, mode, **kw):
            self.assertEqual(mode, 'CENTER_UNITS')
            opt = torch.optim.AdamW(params, **kw)
            opt._sinc_unit_steps = opt._sinc_physical_clip_calls = 0
            old = opt.step
            def step(*a, **k):
                result = old(*a, **k)
                opt._sinc_unit_steps += 1
                return result
            opt.step = step
            return opt
        def clip(opt, params, maxnorm):
            self.assertEqual(maxnorm, 5.)
            opt._sinc_physical_clip_calls += 1
            return torch.nn.utils.clip_grad_norm_(params, maxnorm)
        writes = []
        original_write = module.runner._write
        def write(path, value):
            writes.append(Path(path).name)
            if Path(path).name == 'complete.json':
                self.assertTrue((Path(path).parent / 'final.pt').is_file())
                self.assertTrue((Path(path).parent / 'training.json').is_file())
                self.assertEqual(len((Path(path).parent / 'steps.jsonl').read_text().splitlines()), 222)
            original_write(path, value)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'run'
            with patch.object(module, 'make_trainable_model', return_value=model), \
                    patch.object(module, '_contract', return_value={}), \
                    patch.object(module, '_load_labeled', return_value=data), \
                    patch.object(module, '_augmentor', return_value=(lambda x, **kw: x, False, {})), \
                    patch.object(module, '_load_oe', side_effect=AssertionError('Supervised control must never read OE')) as oe, \
                    patch('experiments.cric_dura_comparison.f_optimizer.build', side_effect=build), \
                    patch('experiments.cric_dura_comparison.f_optimizer.clip_physical', side_effect=clip), \
                    patch.object(module.runner, '_write', side_effect=write), patch('builtins.print'):
                result = module.train(self.config(str(root)), 'cpu')
            self.assertEqual(result['status'], 'SOURCE_OPEN_EMBEDDING_FROZEN')
            self.assertEqual(result['updates'], 222)
            self.assertEqual(result['cumulative_updates'], 11322)
            self.assertEqual(result['OE_sampled_exposures'], 0)
            self.assertEqual(result['L_sampled_exposures'], 28416)
            self.assertEqual(oe.call_count, 0)
            self.assertEqual(writes[-1], 'complete.json')
            value = torch.load(root / 'final.pt', map_location='cpu', weights_only=False)
            self.assertEqual(value['identity']['arm'], 'supervised_only')
            self.assertEqual(value['original_parent_identity'], model._phase3_parent_identity)
            self.assertEqual(value['updates'], 222)

    def test_bad_parent_is_rejected_before_l_oe_or_training(self):
        module = self.module()
        config = self.config()
        wrong = dict(run_id='wrong-parent', row_id=module.PARENT_ROW, model_seed=2026092701,
            source_cache=config['source_cache'], split_seed=392005, repl_row={'L_A': True, 'L_B': True})
        with patch.object(module.runner, '_load_model', return_value=object()), \
                patch.object(module.runner, '_read', return_value={'config': wrong}), \
                patch('torch.load', side_effect=AssertionError('No further CP read')) as load:
            with self.assertRaises(ValueError):
                module._verified_parent(config, {})
            self.assertEqual(load.call_count, 0)

    def test_actual_parent_layout_allows_moved_validated_cache_and_binds_contract_roles(self):
        module = self.module()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Shape of r01 actual_parent_metadata.json: selected_original has a
            # summary null for split_seed, while the actual config has no key.
            metadata = dict(complete_status='SOURCE_FROZEN',
                config_keys=['run_id', 'row_id', 'source_cache', 'source_contract', 'repl_row', 'code_commit'],
                selected_original=dict(run_id=module.PARENT_RUN, row_id=module.PARENT_ROW, split_seed=None,
                    source_cache=str(root / 'old-CRIC-data/F0'), source_contract=str(root / 'parent-contract.json'),
                    repl_row=dict(L_A=True, L_B=True, U=True, structure='SHUFFLE_TIME'), code_commit='b' * 40),
                source_cache_expected=str(root / 'validated-full2x2-data/source/F0'), source_cache_same_resolved=False)
            original = {key: value for key, value in metadata['selected_original'].items() if key in metadata['config_keys']}
            config = self.config()
            config.update(source_cache=metadata['source_cache_expected'], source_contract=str(root / 'current-contract.json'),
                source_output=str(root / 'original-E50'))
            contract = dict(classes=module.CLASSES, source_rxs=module.RX, source_days=[1, 2, 3], split_seed=392005,
                counts=dict(L_s=6300, U_s=56700, V=27000),
                role_ids={role: [role + ':' + str(i) for i in range(n)] for role, n in (('L_s', 6300), ('U_s', 56700), ('V', 27000))})
            Path(config['source_contract']).write_text(json.dumps(contract), encoding='utf-8')
            Path(original['source_contract']).write_text(json.dumps(contract), encoding='utf-8')
            self.assertNotIn('split_seed', original)
            self.assertNotEqual(original['source_cache'], config['source_cache'])
            with patch.object(module.runner, '_load_model', return_value=object()) as native, \
                    patch.object(module.runner, '_read', wraps=module.runner._read) as read, \
                    patch('torch.load', return_value={'args': {'weight_decay': .0001}, 'model': {}}):
                # Only the already-validated metadata is substituted; no IQ is opened.
                complete = root / 'original-E50/complete.json'
                complete.parent.mkdir()
                complete.write_text(json.dumps({'status': 'SOURCE_FROZEN', 'config': original}), encoding='utf-8')
                parsed = module._contract(config)
                actual, _, identity = module._verified_parent(config, parsed)
                self.assertEqual(actual, original)
                self.assertEqual(identity['source_cache'], config['source_cache'])
                native.assert_called_once_with(config, parsed, 'cpu')
                self.assertTrue(any(Path(call.args[0]) == Path(original['source_contract']) for call in read.call_args_list))
            for bad in (None, 2026092701, True):
                with self.subTest(contract_split_seed=bad):
                    Path(config['source_contract']).write_text(json.dumps(dict(contract, split_seed=bad)), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        module._contract(config)
            # Counts/classes stay unchanged: a foreign physical ID must still fail.
            changed = copy.deepcopy(contract)
            changed['role_ids']['L_s'][0] = 'foreign-physical-ID'
            Path(config['source_contract']).write_text(json.dumps(changed), encoding='utf-8')
            parsed = module._contract(config)
            with patch.object(module.runner, '_load_model', return_value=object()), \
                    patch('torch.load', side_effect=AssertionError('No payload load after role mismatch')) as load:
                with self.assertRaisesRegex(ValueError, 'role_ids'):
                    module._verified_parent(config, parsed)
                self.assertEqual(load.call_count, 0)


if __name__ == '__main__':
    unittest.main()
