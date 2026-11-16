import importlib.util
import unittest
from unittest.mock import patch


class SourceCEAnchorTests(unittest.TestCase):
    def test_identity_loss_is_captured_for_only_the_original_hook_call(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.pamf_repl_weight.source_ce_anchor'))
        from experiments.pamf_repl_weight.source_ce_anchor import capture_identity_loss
        class Model:
            def hook(self, **kwargs):
                self.seen = self._diagnostic_identity_loss
                return 'original result'
        model = Model()
        model._source_labeled_feature_hook = model.hook
        marker = object()
        capture_identity_loss(model)
        self.assertEqual(model._source_labeled_feature_hook(identity_loss=marker), 'original result')
        self.assertIs(model.seen, marker)
        self.assertFalse(hasattr(model, '_diagnostic_identity_loss'))
        with self.assertRaises(ValueError):
            model._source_labeled_feature_hook()

    def test_guarded_full_L_alignment_does_not_imply_CE_alignment(self):
        if importlib.util.find_spec('experiments.pamf_repl_weight.source_ce_anchor') is None:
            self.skipTest('Diagnostic not implemented yet')
        import torch
        from experiments.pamf_repl_weight.source_ce_anchor import anchor_facts
        param = torch.nn.Parameter(torch.zeros(2))
        named = [('x.cls_head.weight', param)]
        ce = (torch.tensor([0., 1.]),)
        full = (torch.tensor([1., 0.]),)
        raw = (torch.tensor([0., -1.]),)
        combined = (torch.tensor([1., -1.]),)
        baseline_step = (torch.tensor([1., 0.]),)
        joint_step = (torch.tensor([1., -1.]),)
        facts = anchor_facts(named, ce, full, raw, combined, baseline_step, joint_step)['cls_head']
        self.assertEqual(facts['CE_vs_full_L']['cosine'], 0.)
        self.assertEqual(facts['CE_vs_guarded_U']['cosine'], -1.)
        self.assertEqual(facts['CE_vs_extra_adamw_descent']['cosine'], -1.)

    def test_capture_encloses_the_final_hard_pl_hook_and_guide(self):
        from experiments.cric_dura_comparison import t1_model
        from experiments.pamf_ssl_research import hard_pl_train
        from experiments.pamf_repl_weight.source_ce_anchor import installed

        class Model:
            def _source_labeled_feature_hook(self, **kwargs):
                return 0., {}

        class ProbeObjective:
            def __init__(self, config):
                self.metrics = {}
                self.captured = None
            def guide(self, labeled, student):
                self.captured = student._diagnostic_identity_loss
                return labeled

        holder = {}
        with patch.object(t1_model, 'install', side_effect=lambda model, config: model):
            with installed():
                with hard_pl_train.installed({}, holder, objective_cls=ProbeObjective):
                    model = t1_model.install(Model(), {})
                    self.assertTrue(model._source_labeled_feature_hook._accepts_labeled_inputs)
                    model._source_labeled_feature_hook(identity_loss=3.)
                    self.assertEqual(holder['objective'].captured, 3.)
                    self.assertFalse(hasattr(model, '_diagnostic_identity_loss'))


if __name__ == '__main__':
    unittest.main()
