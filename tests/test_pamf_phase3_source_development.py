import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


class SourceDevelopmentModelTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_development.py'
        self.assertTrue(path.is_file(), 'Four-class scratch source development constructor is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_development')

    def runtime(self):
        root = Path(__file__).resolve().parents[1]
        frozen = root / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        self.assertTrue(frozen.is_dir(), 'Verified local immutable code fixture is missing')
        return dict(base_release=str(root), frozen_sources={key: str(frozen / key) for key in ('F', 'AF', 'BASE', 'FOLD', 'G')})

    def test_real_scratch_graph_has_four_class_joint_ce_gradients(self):
        module = self.module()
        with patch('torch.load', side_effect=AssertionError('Checkpoint access forbidden')) as guard:
            model = module.build_model(self.runtime(), 2026092701)
            model.train()
            z = model.features(torch.randn(4, 2, 256))
            logits = model.classify_features(z)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(4), label_smoothing=0.)
            loss.backward()
        self.assertEqual(guard.call_count, 0)
        self.assertEqual(tuple(z.shape), (4, 160))
        self.assertEqual(tuple(logits.shape), (4, 4))
        self.assertTrue(bool(torch.isfinite(loss)))
        head = model.core.id_backbone.cls_head
        self.assertEqual(tuple(head.weight.shape), (4, 160))
        self.assertEqual(head.scale, 30.)
        self.assertFalse(hasattr(head, 'bias'))
        self.assertGreater(float(head.weight.grad.norm()), 0.)
        self.assertGreater(float(model.sequence_pool.output_projection.weight.grad.norm()), 0.)
        self.assertTrue(all(p.device.type == 'cpu' and p.requires_grad for p in model.parameters()))

    def test_architecture_order_and_parameter_count(self):
        module = self.module()
        with patch('torch.load', side_effect=AssertionError('Checkpoint access forbidden')):
            model = module.build_model(self.runtime(), 2026092701)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 232673)
        self.assertEqual(len(model.sequence_pool.views), 3)
        self.assertEqual(tuple(model.sequence_pool.output_projection.weight.shape), (160, 336))
        self.assertTrue(all(v.window_order.tolist() == [0, 5, 3, 6, 2, 1, 4, 7] for v in model.sequence_pool.views))
        self.assertEqual(model.core.id_backbone.sinc.mode, 'CENTER_25K')
        self.assertEqual(type(model.response).__name__, 'NoExplicitCFO')
        self.assertFalse(hasattr(model, 'dom_backbone'))
        self.assertFalse(hasattr(model, 'teacher'))
        self.assertFalse(hasattr(model, '_source_labeled_feature_hook'))
        self.assertEqual(model.source_development_contract['initialization'], 'scratch_only')
        self.assertEqual(model.source_development_contract['checkpoint_reads'], 0)

    def test_same_seed_repeats_and_different_seed_changes_initialization(self):
        module = self.module()
        first = module.build_model(self.runtime(), 2026092701)
        second = module.build_model(self.runtime(), 2026092701)
        different = module.build_model(self.runtime(), 2026092702)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        self.assertFalse(torch.equal(first.core.id_backbone.cls_head.weight, different.core.id_backbone.cls_head.weight))
        self.assertFalse(torch.equal(first.sequence_pool.output_projection.weight, different.sequence_pool.output_projection.weight))

    def test_constructor_preserves_caller_cpu_rng(self):
        module = self.module()
        before = torch.random.get_rng_state().clone()
        module.build_model(self.runtime(), 2026092701)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_checkpoint_target_and_data_inputs_fail_before_activation(self):
        module = self.module()
        for key in ('checkpoint', 'teacher_checkpoint', 'source_output', 'source_cache', 'public_root', 'truth', 'data', 'target', 'unknown_classes'):
            runtime = dict(self.runtime(), **{key: 'forbidden'})
            with self.subTest(key=key), patch.object(module, '_activate_code', side_effect=AssertionError('Must not activate')), self.assertRaises(ValueError):
                module.build_model(runtime, 2026092701)

    def test_invalid_seed_and_class_count_fail(self):
        module = self.module()
        for seed, count in ((True, 4), (0, 4), (2026092701, 6), (2026092701, True), (2026092701, 0)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                module.build_model(self.runtime(), seed, count)


if __name__ == '__main__':
    unittest.main()
