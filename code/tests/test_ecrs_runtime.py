import copy
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from cvsrffi.ecrs_runtime import ECRSRuntime


class TinyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))


class TinyModel(nn.Module):
    ecrs_version = 'v2'

    def __init__(self):
        super().__init__()
        self.ecrs = TinyBranch()
        self.ecrs_response_head = nn.Linear(4, 2)
        self.fused_head = nn.Linear(4, 2)
        self.raw_head = nn.Linear(3, 2)
        self.response_calls = 0

    def response_encoder(self):
        return self.ecrs.encoder

    def forward(self, *args, **kwargs):
        raise AssertionError('Runtime must not execute the full backbone')

    def forward_response(self, x):
        self.response_calls += 1
        r = x.detach()
        z = F.normalize(self.ecrs.encoder(r), dim=-1)
        return {'encoder_input': r, 'z_resp': z,
                'resp_tx_logits': self.ecrs_response_head(z),
                'quality_valid': torch.ones(x.shape[0], dtype=torch.bool)}


class RuntimeTests(unittest.TestCase):
    def test_compute_only_has_no_rng_forward_lr_or_teacher_side_effect(self):
        model = TinyModel()
        model.ecrs_compute_only = True
        def forbidden_encoder():
            raise AssertionError('Compute-only must not access the student encoder')
        model.response_encoder = forbidden_encoder
        opt = torch.optim.SGD(model.parameters(), lr=.123)
        rng = torch.get_rng_state().clone()
        runtime = ECRSRuntime(model, optimizer=opt,
            group_configs={0: {'peak_lr': .9, 'total_steps': 10}},
            u_rx=[0, 1], u_day=[0, 0])
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertIsNone(runtime.ema)
        self.assertIsNone(runtime.u_queue)
        self.assertIsNone(runtime.clock)
        self.assertEqual(opt.param_groups[0]['lr'], .123)
        out = runtime.revision_losses(model, {}, torch.ones(2, 3), None, {})
        self.assertFalse(out['total'].requires_grad)
        self.assertEqual(float(out['total']), 0.)
        self.assertEqual(model.response_calls, 0)
        self.assertFalse(runtime.after_optimizer_step(model, True)['ema_updated'])
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_regular_runtime_ema_initialization_preserves_rng(self):
        model = TinyModel()
        state = torch.get_rng_state().clone()
        ECRSRuntime(model, u_rx=[0, 1], u_day=[0, 0])
        torch.testing.assert_close(torch.get_rng_state(), state)

    def test_pure_u_updates_only_student_and_ema_after_success(self):
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        runtime = ECRSRuntime(model)
        runtime.set_epoch(1, {'enabled': True, 'u_pair': True, 'resp_cls': True, 'fusion': True})
        clean, leo = torch.randn(6, 3), torch.randn(6, 3)
        before_bn = model.ecrs.encoder[1].num_batches_tracked.clone()
        before = model.ecrs.encoder[0].weight.detach().clone()
        result = runtime.revision_losses(model, None, None, None, {}, u_clean=clean, u_leo=leo)
        self.assertEqual(model.response_calls, 2)
        self.assertEqual(int(model.ecrs.encoder[1].num_batches_tracked - before_bn), 1)
        self.assertEqual(result['telemetry']['u_pair']['valid_count'], 6)
        self.assertEqual(result['telemetry']['resp_ce']['valid_count'], 0)
        self.assertEqual(result['telemetry']['fused_ce']['valid_count'], 0)
        result['total'].backward()
        self.assertIsNone(model.raw_head.weight.grad)
        self.assertIsNone(model.ecrs_response_head.weight.grad)
        self.assertFalse(runtime.after_optimizer_step(model, False)['ema_updated'])
        opt.step()
        self.assertTrue(runtime.after_optimizer_step(model, True)['ema_updated'])
        self.assertFalse(torch.equal(before, model.ecrs.encoder[0].weight))
        self.assertEqual(int(runtime.ema.updates), 1)
        model.eval()
        self.assertFalse(runtime.after_optimizer_step(model, True)['ema_updated'])
        self.assertEqual(int(runtime.ema.updates), 1)
        with self.assertRaises(RuntimeError):
            runtime.revision_losses(model, None, None, None, {}, u_clean=clean, u_leo=leo)

    def test_supervised_reuse_named_additive_losses(self):
        model = TinyModel()
        runtime = ECRSRuntime(model)
        runtime.set_epoch(91, {'enabled': True, 'resp_cls': True, 'cross_rx': True, 'fusion': True})
        x = torch.randn(4, 3)
        clean, leo = model.forward_response(x), model.forward_response(x + .1)
        clean['tx_logits_fused'] = model.fused_head(clean['z_resp'].detach())
        leo['tx_logits_fused'] = model.fused_head(leo['z_resp'].detach())
        y = torch.tensor([0, 0, 1, 1])
        meta = {'receiver_id': torch.tensor([0, 1, 0, 1]), 'day_id': torch.zeros(4, dtype=torch.long),
                'label_mask': torch.ones(4, dtype=torch.bool)}
        result = runtime.revision_losses(model, clean, x, y, meta, out_leo=leo)
        self.assertEqual(model.response_calls, 2)
        self.assertEqual(result['telemetry']['resp_ce']['valid_count'], 8)
        self.assertEqual(result['telemetry']['cross_rx']['valid_count'], 8)
        self.assertEqual(result['telemetry']['fused_ce']['valid_count'], 8)
        expected = sum(runtime.weights[k] * v for k, v in result['losses'].items())
        torch.testing.assert_close(result['total'], expected)
        json.dumps(result['telemetry'])
        result['total'].backward()
        self.assertGreater(float(model.ecrs.encoder[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.fused_head.weight.grad.abs().sum()), 0.)
        self.assertIsNone(model.raw_head.weight.grad)

    def test_epoch_boundary_restore_queue_rng_ema_clock(self):
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        kwargs = {'weights': {'u_pair': .04}, 'u_rx': [0, 0, 1, 1], 'u_day': [0, 0, 0, 0],
                  'group_configs': {0: {'total_steps': 20, 'warmup_steps': 2, 'peak_lr': .1}}}
        runtime = ECRSRuntime(model, optimizer=opt, **kwargs)
        runtime.set_epoch(3, {'enabled': True, 'u_pair': True})
        runtime.next_u_indices(3)
        loss = runtime.revision_losses(model, None, None, None, {},
                                      u_clean=torch.randn(4, 3), u_leo=torch.randn(4, 3))['total']
        loss.backward()
        opt.step()
        runtime.after_optimizer_step(model, True)
        checkpoint = copy.deepcopy({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'runtime': runtime.state_dict()})
        expected_indices, expected_rng = runtime.next_u_indices(7), torch.rand(5)
        restored_model = TinyModel()
        restored_model.load_state_dict(checkpoint['model'])
        restored_opt = torch.optim.SGD(restored_model.parameters(), lr=.1)
        restored = ECRSRuntime(restored_model, optimizer=restored_opt, **kwargs)
        restored_opt.load_state_dict(checkpoint['optimizer'])
        restored.load_state_dict(checkpoint['runtime'])
        self.assertEqual(restored.next_u_indices(7), expected_indices)
        torch.testing.assert_close(torch.rand(5), expected_rng)
        self.assertEqual(restored.clock.steps, runtime.clock.steps)
        self.assertEqual(int(restored.ema.updates), 1)
        self.assertEqual(restored.epoch, 3)
        restored.set_epoch(4, {'enabled': True, 'u_pair': True})
        self.assertEqual(restored.batch_index, 0)
        self.assertEqual(restored.u_queue.coverage['draw_count'], 10)


if __name__ == '__main__':
    unittest.main()
