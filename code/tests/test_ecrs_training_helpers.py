import json
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cvsrffi.ecrs_training import (
    EffectiveStepLR, ResponseEMA, assemble_ecrs_losses, capture_rng_state,
    cross_rx_triplet_loss, masked_response_ce, one_way_u_consistency,
    restore_rng_state,
)


class TrainingHelpersTests(unittest.TestCase):
    def test_masked_ce_and_all_u(self):
        logits = torch.randn(5, 3, requires_grad=True)
        labels = torch.tensor([0, -1, 999, 2, 1])
        mask = torch.tensor([True, False, False, True, False])
        loss, n = masked_response_ce(logits, labels, mask)
        self.assertEqual(n, 2)
        torch.testing.assert_close(loss, F.cross_entropy(logits[[0, 3]], labels[[0, 3]]))
        loss.backward()
        self.assertEqual(float(logits.grad[[1, 2, 4]].abs().sum()), 0.)
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        zero, n = masked_response_ce(logits, labels, torch.zeros(5, dtype=torch.bool))
        self.assertEqual(n, 0)
        self.assertEqual(float(zero.detach()), 0.)
        self.assertTrue(zero.requires_grad)

    def test_triplets_equal_bruteforce_anchor_mean_and_gradients(self):
        torch.manual_seed(3)
        z = torch.randn(9, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 999])
        rx = torch.tensor([0, 1, 0, 1, 0, 1, 1, 1, 0])
        day = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 0])
        view = torch.zeros(9, dtype=torch.long)
        mask = torch.tensor([True] * 8 + [False])
        actual, count = cross_rx_triplet_loss(z, labels, rx, day, view, mask)
        dist = 1 - F.normalize(z, dim=1) @ F.normalize(z, dim=1).T
        anchors = []
        for i in range(8):
            pairs = [F.relu(.2 + dist[i, j] - dist[i, k])
                     for j in range(8) for k in range(8)
                     if labels[i] == labels[j] and rx[i] != rx[j]
                     and labels[i] != labels[k] and rx[i] == rx[k]
                     and day[i] == day[k] and view[i] == view[k]]
            if pairs:
                anchors.append(torch.stack(pairs).mean())
        expected = torch.stack(anchors).mean()
        self.assertEqual(count, len(anchors))
        torch.testing.assert_close(actual, expected)
        ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, z)[0]
        torch.testing.assert_close(ga, ge)
        self.assertGreater(float(ga.abs().sum()), 0.)
        self.assertEqual(float(ga[-1].abs().sum()), 0.)
        zero, count = cross_rx_triplet_loss(z, labels, rx * 0, day, view, mask)
        self.assertEqual(count, 0)
        self.assertEqual(float(zero.detach()), 0.)

    def test_u_student_only_and_named_independent_losses(self):
        encoder = nn.Linear(3, 4)
        physical = torch.randn(4, 3, requires_grad=True)
        student = encoder(physical.detach())
        teacher = torch.randn(4, 4, requires_grad=True)
        loss, n = one_way_u_consistency(student, teacher)
        loss.backward()
        self.assertEqual(n, 4)
        self.assertGreater(float(encoder.weight.grad.abs().sum()), 0.)
        self.assertIsNone(physical.grad)
        self.assertIsNone(teacher.grad)
        output = assemble_ecrs_losses(logits=torch.randn(4, 3), z_resp=encoder(physical.detach()),
                    labels=torch.full((4,), -1), label_mask=torch.zeros(4, dtype=torch.bool),
                    student_leo=encoder(physical.detach()), teacher_clean=teacher,
                    enabled={'resp_ce': False, 'cross_rx': False, 'u_pair': True})
        self.assertFalse(output['telemetry']['resp_ce']['executed'])
        self.assertTrue(output['telemetry']['u_pair']['executed'])
        json.dumps(output['telemetry'])
        torch.testing.assert_close(output['total'], .03 * output['losses']['u_pair'])

    def test_ema_eval_immutable_and_successful_updates(self):
        student = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.4))
        ema = ResponseEMA(student, decay=.9)
        before = {k: v.clone() for k, v in ema.module.state_dict().items()}
        ema.train()
        a, b = ema(torch.randn(4, 3)), ema(torch.randn(4, 3))
        self.assertFalse(a.requires_grad)
        for key, value in before.items():
            torch.testing.assert_close(ema.module.state_dict()[key], value)
        with torch.no_grad():
            student[0].weight.add_(1)
        ema.update(student, successful=False)
        torch.testing.assert_close(ema.module[0].weight, before['0.weight'])
        ema.update(student, successful=True)
        torch.testing.assert_close(ema.module[0].weight, before['0.weight'] + .1)
        self.assertEqual(int(ema.updates), 1)
        restored = ResponseEMA(student, decay=.5)
        restored.load_state_dict(ema.state_dict())
        self.assertEqual(restored.decay, .9)

    def test_effective_lr_skip_inactive_and_resume(self):
        a, b = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([{'params': [a]}, {'params': [b]}], lr=.1, momentum=.9)
        configs = {1: {'peak_lr': .2, 'min_lr': .01, 'warmup_steps': 2, 'total_steps': 10}}
        scheduler = EffectiveStepLR(opt, configs)
        original_group = opt.param_groups[1]
        scheduler.step(successful=True)
        self.assertEqual(scheduler.steps[1], 0)
        b.grad = torch.ones_like(b)
        scheduler.step(successful=False)
        self.assertEqual(scheduler.steps[1], 0)
        opt.step()
        scheduler.step(successful=True)
        self.assertEqual(scheduler.steps[1], 1)
        self.assertIs(opt.param_groups[1], original_group)
        self.assertEqual(opt.param_groups[0]['lr'], .1)
        state = scheduler.state_dict()
        b.grad = torch.zeros_like(b)
        opt.step()
        scheduler.step()
        expected = opt.param_groups[1]['lr']
        scheduler.load_state_dict(state)
        scheduler.step()
        self.assertEqual(opt.param_groups[1]['lr'], expected)
        self.assertIn('momentum_buffer', opt.state[b])

    def test_rng_restores_exact_next_values(self):
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])


if __name__ == '__main__':
    unittest.main()
