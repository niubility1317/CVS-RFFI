"""Focused synthetic checks; these do not establish identification benefits."""
import random
import unittest

import torch

from .augmentation import (MAX_RADIUS, apply_response,
                           session_response)
from .design import plan
from .objective import mixed_objective, rdrop_per_sample


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(8, 2, 256)

    def test_identity_and_zero(self):
        out, _ = apply_response(self.x, [0j] * 8)
        self.assertTrue(torch.equal(out, self.x))
        out, _ = apply_response(torch.zeros_like(self.x), [.06j] * 8)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_rms_preserved_but_waveform_changed(self):
        out, _ = apply_response(self.x, [.04 + .04j] * 8)
        self.assertTrue(torch.allclose(out.square().sum(1).mean(1),
                                       self.x.square().sum(1).mean(1), atol=5e-7))
        self.assertGreater(float((out - self.x).square().mean()), 1e-4)

    def test_edge_effect_excludes_reference_window_before_scalar_restoration(self):
        reflected, r = apply_response(self.x, [.06j] * 8)
        edged, e = apply_response(self.x, [.06j] * 8, boundary='edge')
        reflected = reflected / r['scalar_power_restoration'][:, None, None]
        edged = edged / e['scalar_power_restoration'][:, None, None]
        self.assertTrue(torch.allclose(reflected[:, :, 1:255], edged[:, :, 1:255], atol=5e-7))
        self.assertGreater(float((reflected[:, :, 0] - edged[:, :, 0]).abs().max()), 0)

    def test_finite_response_frequency_bounds(self):
        for a in (.06, -.06, .06j, -.06j):
            h = 1 + a * torch.linspace(-1, 1, 101)
            self.assertGreaterEqual(float(h.abs().min()) + 1e-6, 1 - MAX_RADIUS)
            self.assertLessEqual(float(h.abs().max()), 1 + MAX_RADIUS + 1e-6)

    def test_no_label_or_global_rng_dependency(self):
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
        call = lambda **kw: session_response(21, stream='train', epoch=2, rx=1, day=2, eq=0, **kw)
        self.assertEqual(call(), call())
        with self.assertRaises(TypeError):
            call(tx=3)
        apply_response(self.x, [call()] * 8)
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertNotEqual(call(), session_response(21, stream='source_probe', epoch=2, rx=1, day=2, eq=0))
        self.assertNotEqual(call(), session_response(21, stream='train', epoch=3, rx=1, day=2, eq=0))

    def test_reject_invalid_input_and_label_dependent_metadata(self):
        for x, coeff in ((self.x, [.1j] * 8), (self.x, [float('nan')] * 8),
                         (self.x[:, :, :128], [0j] * 8), (self.x * float('nan'), [0j] * 8)):
            with self.assertRaises(ValueError):
                apply_response(x, coeff)
        with self.assertRaises(ValueError):
            apply_response(self.x, [0j] * 8, boundary='circular')
        with self.assertRaises(ValueError):
            session_response(1, stream='target', epoch=1, rx=1, day=1, eq=0)

    def test_loss_normalization_and_default_logits_gradients(self):
        y = torch.arange(8) % 6
        logits = [torch.randn(8, 6, requires_grad=True) for _ in range(6)]
        loss, m = mixed_objective(logits[:2], logits[2:4], logits[4:], y)
        expected = sum(w * rdrop_per_sample(*logits[j:j+2], y)[0].mean()
                       for w, j in ((.25, 0), (.25, 2), (.5, 4)))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in logits))
        self.assertEqual(m['labeled_forward_packets'], 48)
        with self.assertRaises(ValueError):
            mixed_objective(logits[:2], logits[2:4], logits[4:], torch.full_like(y, -1))
        with self.assertRaises(ValueError):
            rdrop_per_sample(torch.empty(0, 6), torch.empty(0, 6), torch.empty(0, dtype=torch.long))

    def test_plan_is_e20_scratch_and_unlaunched(self):
        p = plan()
        self.assertEqual(p['experiments_launched'], 0)
        for row in p['rows']:
            self.assertEqual(row['total_epochs'], 20)
            self.assertEqual(row['optimizer_steps'], 4440)
            self.assertEqual(row['parent_training_epochs'], 0)
            self.assertEqual(row['source_roles'], dict(L_s=6300, U_s=56700, V=27000))
            self.assertEqual(row['test_views'][0], 'clean')
            self.assertEqual(len(row['test_views']), 7)
            self.assertFalse(row['new_U_augmentation'])
        factors = [r for r in p['rows'] if r['arm'] != 'R_LEGACY']
        self.assertEqual({r['labeled_forward_packets'] for r in factors}, {3409920})
        self.assertAlmostEqual(sum(p['training_scene_probabilities'].values()), 1.)
        self.assertTrue(all(v > 0 for v in p['training_scene_probabilities'].values()))
        self.assertEqual({(r['A_g'], r['A_s']) for r in factors},
                         {(False, False), (True, False), (False, True), (True, True)})


if __name__ == '__main__':
    unittest.main()
