"""Numerical and optimizer regressions for the explicitly repaired ISSL variant."""
import copy
import unittest
import torch
from torch.nn import functional as F
from issl_fixed import (distillation_loss, contrastive_loss, freeze_teacher,
                        momentum_update, append_queue, ssl_step, validate_config, train_supervised)


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(4)
        self.fc = torch.nn.Linear(4, 6, bias=False)

    def forward(self, x):
        features = self.bn(x)
        return self.fc(features), features


class FixedLossTests(unittest.TestCase):
    def test_kd_preserves_author_value_and_reaches_student_only(self):
        student = torch.tensor([[1., -1., 2.], [2., 0., -2.]], requires_grad=True)
        teacher = torch.tensor([[2., 1., -1.], [0., 3., 1.]], requires_grad=True)
        expected = -(F.softmax(teacher.detach()/20, 1) * F.log_softmax(student/20, 1)).sum(1).mean()
        actual = distillation_loss(student, teacher, 20)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(student.grad.norm()), 0)
        self.assertIsNone(teacher.grad)

    def test_kd_rejects_bad_temperature_or_shape(self):
        for temperature in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                distillation_loss(torch.ones(2, 3), torch.ones(2, 3), temperature)
        with self.assertRaises(ValueError):
            distillation_loss(torch.ones(2, 3), torch.ones(2, 4), 2)

    def test_contrastive_logits_use_transposed_normalized_queue(self):
        q = torch.tensor([[1., 2., 3.], [3., 1., 2.]], requires_grad=True)
        k = torch.tensor([[2., 1., 3.], [1., 2., 3.]], requires_grad=True)
        queue = torch.arange(1., 13.).view(4, 3).requires_grad_()
        qn, kn, negatives = [F.normalize(value, dim=1) for value in (q, k, queue)]
        expected = F.cross_entropy(torch.cat(((qn*kn).sum(1, keepdim=True), qn @ negatives.T), 1)/5, torch.zeros(2, dtype=torch.long))
        actual = contrastive_loss(q, k, queue, 5)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(q.grad.norm()), 0)
        self.assertIsNone(k.grad)
        self.assertIsNone(queue.grad)

    def test_contrastive_loss_is_invariant_to_positive_rescaling(self):
        q, k, queue = torch.randn(2, 3), torch.randn(2, 3), torch.randn(5, 3)
        torch.testing.assert_close(contrastive_loss(q, k, queue, 5), contrastive_loss(q*7, k*11, queue*100, 5))

    def test_zero_vectors_are_finite(self):
        q = torch.zeros(2, 3, requires_grad=True)
        loss = contrastive_loss(q, torch.zeros_like(q), torch.zeros(4, 3), 5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(q.grad).all())

    def test_queue_is_detached_normalized_and_exactly_bounded(self):
        queue = torch.arange(1., 16.).view(5, 3)
        new = torch.tensor([[5., 2., 1.], [4., 2., 3.]], requires_grad=True)
        updated = append_queue(queue, new, 6)
        expected = F.normalize(torch.cat((queue, new.detach()))[-6:], dim=1)
        torch.testing.assert_close(updated, expected)
        self.assertFalse(updated.requires_grad)
        self.assertEqual(updated.shape, (6, 3))

    def test_queue_rejects_invalid_capacity(self):
        with self.assertRaises(ValueError):
            append_queue(torch.ones(2, 3), torch.ones(2, 3), 0)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_teacher_freeze_preserves_bn_buffers_and_gradients(self):
        teacher = freeze_teacher(TinyNet())
        before = copy.deepcopy(teacher.state_dict())
        with torch.no_grad(): teacher(torch.randn(8, 4))
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        for name, value in teacher.state_dict().items(): torch.testing.assert_close(value, before[name])

    def test_momentum_updates_parameters_and_bn_buffers(self):
        student, key = TinyNet(), freeze_teacher(TinyNet())
        before = copy.deepcopy(key.state_dict())
        student.bn.running_mean.fill_(4)
        student.bn.num_batches_tracked.fill_(3)
        momentum_update(key, student, .9)
        for name, value in key.state_dict().items():
            expected = .9*before[name] + .1*student.state_dict()[name] if value.is_floating_point() else student.state_dict()[name]
            torch.testing.assert_close(value, expected)
        self.assertFalse(key.training)

    def test_ssl_clears_stale_grads_and_updates_new_head(self):
        first = TinyNet()
        second = copy.deepcopy(first)
        key1, key2 = freeze_teacher(copy.deepcopy(first)), freeze_teacher(copy.deepcopy(first))
        for parameter in first.parameters(): parameter.grad = torch.full_like(parameter, 100)
        qx, kx, queue = torch.randn(8, 4), torch.randn(8, 4), torch.randn(10, 6)
        original = first.fc.weight.detach().clone()
        left, stats1 = ssl_step(first, key1, torch.optim.SGD(first.parameters(), .1), qx, kx, queue, old_classes=3, capacity=10)
        right, stats2 = ssl_step(second, key2, torch.optim.SGD(second.parameters(), .1), qx, kx, queue, old_classes=3, capacity=10)
        for p, other in zip(first.parameters(), second.parameters()): torch.testing.assert_close(p, other)
        torch.testing.assert_close(left, right)
        self.assertGreater(stats1['kd_logit_grad_norm'], 0)
        self.assertGreater(stats1['new_head_grad_norm'], 0)
        self.assertFalse(torch.equal(first.fc.weight[3:], original[3:]))
        self.assertTrue(all(p.grad is None for p in key1.parameters()))
        self.assertTrue(stats1['zero_grad'] and stats1['teacher_frozen'])

    def test_repeated_ssl_steps_do_not_reuse_graph(self):
        model = TinyNet()
        key = freeze_teacher(copy.deepcopy(model))
        optimizer = torch.optim.Adam(model.parameters(), 1e-5)
        queue = torch.randn(7, 6)
        for _ in range(3):
            queue, stats = ssl_step(model, key, optimizer, torch.randn(8, 4), torch.randn(8, 4), queue, old_classes=3, capacity=10)
            self.assertTrue(torch.isfinite(torch.tensor(stats['loss'])))
        self.assertEqual(len(queue), 10)

    def test_config_rejects_missing_or_invalid_budgets_before_launch(self):
        config = dict(implementation='issl_fixed_v1', base_epochs=2, ssl_epochs=2,
                      downstream_epochs=2, incremental_epochs=2, base_lr=.03, ssl_lr=1e-5,
                      downstream_lr=.03, incremental_lr=1e-4, data='data.npz', output='new',
                      issl_source='source', run_id='new-run', seed=9)
        validate_config(config)
        for field, value in [('base_epochs', 0), ('ssl_epochs', True), ('seed', None), ('base_lr', float('nan'))]:
            with self.assertRaises(ValueError): validate_config(config | {field: value})

    def test_supervised_new_classes_receive_gradients_and_teacher_is_immutable(self):
        import numpy as np
        class SignalNet(TinyNet):
            def forward(self, x):
                return super().forward(torch.cat((x.mean(-1), x.std(-1)), dim=1))
        class Capture:
            def __init__(self): self.rows=[]
            def log(self, **row): self.rows.append(row)
        model = SignalNet()
        teacher = freeze_teacher(SignalNet())
        teacher.fc = torch.nn.Linear(4, 3, bias=False)
        before = copy.deepcopy(teacher.state_dict())
        head_before = model.fc.weight.detach().clone()
        x = np.random.default_rng(2).normal(size=(128,256,2)).astype(np.float32)
        labels = np.arange(128) % 6
        log = Capture()
        train_supervised(model, x, labels, 1, 1e-4, 'incremental_baseline', torch.device('cpu'), log, teacher)
        step = log.rows[0]
        self.assertGreater(step['new_head_grad_norm'], 0)
        self.assertGreater(step['kd_logit_grad_norm'], 0)
        self.assertFalse(torch.equal(model.fc.weight[3:], head_before[3:]))
        for name, value in teacher.state_dict().items(): torch.testing.assert_close(value, before[name])
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
