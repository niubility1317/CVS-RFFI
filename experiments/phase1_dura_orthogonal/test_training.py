import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.phase1_dura_orthogonal import ground
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.train_source import extract_context, source_loss


class IdentityAction:
    def __call__(self, value):
        return value

    def adjoint(self, value):
        return value


class FakeDura:
    def estimate_context(self, x):
        return None, IdentityAction(), torch.full((len(x),), .1)

    def stage_outputs(self, x, context):
        anchor = x * .5
        return [anchor], None, context[1], None


class SourceTrainingTest(unittest.TestCase):
    def test_gpu_requires_one_selected_device_and_scoring_hides_it(self):
        with self.assertRaises(ValueError):
            training_environments('cuda:0', {})
        with self.assertRaises(ValueError):
            training_environments('cuda:0', {'CUDA_VISIBLE_DEVICES': '1,2'})
        train, score = training_environments('cuda:0', {'CUDA_VISIBLE_DEVICES': '7'})
        self.assertEqual(train['CUDA_VISIBLE_DEVICES'], '7')
        self.assertEqual(score['CUDA_VISIBLE_DEVICES'], '')
        train, score = training_environments('cpu', {'CUDA_VISIBLE_DEVICES': '7'})
        self.assertEqual(train['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(score['CUDA_VISIBLE_DEVICES'], '')

    def test_identity_action_produces_the_observation_residual(self):
        x = torch.randn(3, 2, 256)
        anchor, direction, noise = extract_context(FakeDura(), x)
        torch.testing.assert_close(anchor, x*.5)
        torch.testing.assert_close(direction, x*.5)
        torch.testing.assert_close(noise, torch.full((3,), .1))

    def test_iq_loss_prefers_perfect_clean_and_propagates_gradients(self):
        clean = torch.ones(2, 2, 256)
        anchor = clean*.8
        perfect = source_loss(clean, anchor, clean)
        imperfect = anchor.clone().requires_grad_(True)
        loss = source_loss(imperfect, anchor, clean)
        self.assertLess(float(perfect), float(loss.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(imperfect.grad).all())
        self.assertGreater(float(imperfect.grad.abs().sum()), 0)

    def test_dura_guard_penalizes_degradation_and_preserves_improvement(self):
        clean = torch.ones(2, 2, 256)
        anchor = clean*.95
        worse = clean*.9
        better = clean*.99
        self.assertGreater(float(source_loss(worse, anchor, clean, objective='dura_guard')),
                           float(source_loss(worse, anchor, clean, objective='iq')))
        torch.testing.assert_close(source_loss(better, anchor, clean, objective='dura_guard'),
                                   source_loss(better, anchor, clean, objective='iq'))
        with self.assertRaises(ValueError):
            source_loss(better, anchor, clean, objective='unregistered')

    def test_band_guard_emphasizes_middle_frequency_error(self):
        clean = torch.ones(1, 2, 256)
        t = torch.arange(256, dtype=torch.float32)
        phase = 2*torch.pi*40*t/256
        delta = torch.stack((phase.cos(), phase.sin()))[None] * .1
        output = (clean+delta).requires_grad_()
        base = source_loss(output, clean, clean, objective='dura_guard')
        weighted = source_loss(output, clean, clean, objective='band_guard')
        torch.testing.assert_close(weighted-base, torch.tensor(.0025), atol=1e-7, rtol=1e-5)
        weighted.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        dc = clean+.1
        torch.testing.assert_close(source_loss(dc, clean, clean, objective='band_guard'),
                                   source_loss(dc, clean, clean, objective='dura_guard'))

    def test_nonlinear_guard_is_zero_at_match_and_has_finite_gradient(self):
        clean = torch.ones(2, 2, 256)
        torch.testing.assert_close(source_loss(clean, clean, clean, objective='nonlinear_guard'),
                                   torch.tensor(0.))
        output = (clean*.9).requires_grad_()
        loss = source_loss(output, clean, clean, objective='nonlinear_guard')
        self.assertGreater(float(loss.detach()), float(source_loss(output, clean, clean, objective='dura_guard').detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertGreater(float(output.grad.abs().sum()), 0.)

    def test_incomplete_prediction_is_rejected_before_truth_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'candidate'
            folder = out/'flow/test_prediction'
            folder.mkdir(parents=True)
            (folder/'complete.json').write_text('{"status":"INCOMPLETE"}', encoding='utf8')
            original = ground.core.read
            def read(path):
                if str(path).endswith('truth.json'):
                    raise AssertionError('Truth must not be read for incomplete predictions')
                return original(path)
            with patch.object(ground.core, 'read', side_effect=read):
                with self.assertRaises(ValueError):
                    ground.paired_score(root, out, 'fixed')


if __name__ == '__main__':
    unittest.main()
