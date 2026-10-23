from copy import deepcopy
import unittest
import torch

from experiments.cric_dura_comparison.f_model import contrast
from experiments.pamf_ssl_research.plan import configuration as base_config, SEEDS
from tests.test_pamf_ssl_research import template
from experiments.pamf_ssl_research.u_cric import configuration, validate
from experiments.pamf_ssl_research.u_cric_objective import soft_cross_receiver_contrast, UnlabeledCRICObjective


class UnlabeledCRICTest(unittest.TestCase):
    def test_one_hot_targets_reproduce_existing_labeled_cric(self):
        torch.manual_seed(7)
        h = torch.randn(8, 4, 12, requires_grad=True)
        y = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        rx = torch.tensor([1, 3, 1, 3, 4, 6, 4, 6])
        p = torch.nn.functional.one_hot(y, 6).float().requires_grad_()
        w = torch.ones(8, requires_grad=True)
        value, detail = soft_cross_receiver_contrast(h, p, w, rx, [1, 3, 4, 6, 8], .2)
        old, _ = contrast({'time_down': h}, y, rx)
        torch.testing.assert_close(value, old)
        self.assertEqual(detail['valid_anchors'], 8)
        value.backward()
        self.assertTrue(bool((h.grad.norm(dim=(1, 2)) > 0).all()))
        self.assertIsNone(p.grad)
        self.assertIsNone(w.grad)

    def test_soft_targets_and_confidence_change_loss_without_teacher_gradient(self):
        torch.manual_seed(9)
        h = torch.randn(6, 10, requires_grad=True)
        p = torch.tensor([[.9, .1], [.8, .2], [.1, .9], [.2, .8], [.85, .15], [.15, .85]], requires_grad=True)
        w = torch.tensor([1., .7, .5, .9, 0., .8], requires_grad=True)
        rx = torch.tensor([1, 3, 1, 3, 4, 4])
        loss, detail = soft_cross_receiver_contrast(h, p, w, rx, [1, 3, 4, 6, 8], .2)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss.detach()), 0.)
        self.assertEqual(detail['valid_anchors'], 5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(h.grad).all()))
        self.assertIsNone(p.grad)
        self.assertIsNone(w.grad)
        self.assertEqual(float(h.grad[4].norm()), 0.)

    def test_missing_cross_rx_or_negative_evidence_returns_finite_zero(self):
        for rx, p, w in (
            (torch.ones(4, dtype=torch.long), torch.eye(2).repeat(2, 1), torch.ones(4)),
            (torch.tensor([1, 3, 1, 3]), torch.tensor([[1., 0.]]).repeat(4, 1), torch.ones(4)),
            (torch.tensor([1, 3, 1, 3]), torch.eye(2).repeat(2, 1), torch.zeros(4)),
        ):
            h = torch.randn(4, 3, 8, requires_grad=True)
            value, detail = soft_cross_receiver_contrast(h, p, w, rx, [1, 3, 4, 6, 8], .2)
            self.assertEqual(float(value.detach()), 0.)
            self.assertEqual(detail['valid_anchors'], 0)
            value.backward()
            self.assertTrue(bool(torch.isfinite(h.grad).all()))

    def test_non_source_rx_and_malformed_probabilities_are_rejected(self):
        h = torch.randn(4, 8)
        p = torch.eye(2).repeat(2, 1)
        with self.assertRaisesRegex(ValueError, 'source receiver'):
            soft_cross_receiver_contrast(h, p, torch.ones(4), torch.tensor([1, 3, 4, 9]), [1, 3, 4, 6, 8], .2)
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            soft_cross_receiver_contrast(h, p * 2, torch.ones(4), torch.tensor([1, 3, 4, 6]), [1, 3, 4, 6, 8], .2)

    def test_fixed_source_fork_recipe_and_hidden_target_boundary(self):
        parent = base_config(template(), 'RXPSM', SEEDS[0], '/p', 'old')
        c = configuration(parent, '/p', 'new')
        self.assertEqual(c['u_cric']['levels'], ['time_down', 't2.pool'])
        self.assertEqual(c['u_cric']['weight'], .10)
        self.assertEqual(c['u_cric']['temperature'], .2)
        self.assertNotIn('paired_u', c)
        bad = deepcopy(c)
        bad['u_cric']['weight'] = .2
        with self.assertRaisesRegex(ValueError, 'U-CRIC recipe'): validate(bad)
        bad = deepcopy(c)
        bad['truth'] = '/target/truth'
        with self.assertRaisesRegex(ValueError, 'source handles'): validate(bad)
        parent['row_id'] = 'PSM-s' + str(SEEDS[0])
        with self.assertRaisesRegex(ValueError, 'same-seed RXPSM E30'): configuration(parent, '/p', 'new')

    def test_exposed_u_truth_is_rejected_and_temporary_hooks_are_removed(self):
        from tests.test_pamf_ssl_paired_u import Student
        student = Student()
        modules = [torch.nn.Identity(), torch.nn.Identity()]
        student.get_submodule = lambda name: modules[0 if name.endswith('time_down') else 1]
        parent = base_config(template(), 'RXPSM', SEEDS[0], '/p', 'old')
        obj = UnlabeledCRICObjective(student, configuration(parent, '/p', 'new'))
        obj.current_epoch = 31
        student._cd_U_iter = iter([(torch.randn(4, 2, 256), torch.zeros(4, dtype=torch.long), None, None)])
        with self.assertRaisesRegex(ValueError, 'U TX truth exposed'): obj.loss(student, 'cpu')
        self.assertEqual(obj.u_cric_calls, 0)
        self.assertIsNone(obj._u_maps)
        self.assertTrue(all(not module._forward_hooks for module in modules))

    def test_original_parent_state_and_new_layer_counts_roundtrip(self):
        from tests.test_pamf_ssl_paired_u import Student
        from experiments.pamf_ssl_research.objective import Objective
        student = Student()
        parent = base_config(template(), 'RXPSM', SEEDS[0], '/p', 'old')
        c = configuration(parent, '/p', 'new')
        state = Objective(student, parent).state_dict()
        state.update(ema_updates=6660, seen=963900, physical_seen=481950, calls=3774)
        obj = UnlabeledCRICObjective(student, c)
        obj.load_state_dict(state)
        self.assertEqual(obj.ema_updates, 6660)
        self.assertEqual(obj.u_cric_calls, 0)
        obj.u_cric_calls, obj.u_cric_seen = 13, 3328
        obj.u_cric_anchors = dict(time_down=3000, **{'t2.pool': 2999})
        restored = UnlabeledCRICObjective(student, c)
        restored.load_state_dict(obj.state_dict())
        self.assertEqual(restored.u_cric_calls, 13)
        self.assertEqual(restored.u_cric_seen, 3328)
        self.assertEqual(restored.u_cric_anchors['t2.pool'], 2999)


if __name__ == '__main__': unittest.main()
