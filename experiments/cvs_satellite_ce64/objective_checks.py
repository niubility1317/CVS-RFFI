"""Focused tensor-only checks for satellite CE replacement; no IQ or model."""
import json
import time
import unittest
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from .objective import ARMS, captured_native_views, replacement, validate_head, validate_objective


class ObjectiveChecks(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(6401)
        self.first = torch.randn(64, 6, generator=generator, dtype=torch.float64, requires_grad=True)
        self.second = torch.randn(64, 6, generator=generator, dtype=torch.float64, requires_grad=True)
        self.labels = torch.arange(64, dtype=torch.long).remainder(6)

    def ce(self, logits, temperature=1.):
        return F.cross_entropy(logits / temperature, self.labels, label_smoothing=.01)

    def test_half_exact_formula(self):
        delta, record = replacement(self.first, self.second, self.labels, 'CE_HALF', ARMS['CE_HALF'])
        original = .5 * (self.ce(self.first) + self.ce(self.second))
        torch.testing.assert_close(delta, -.125 * original)
        self.assertAlmostEqual(record['new_sat_CE'], .5 * record['old_sat_CE'])
        self.assertTrue(delta.requires_grad)

    def test_temperature_exact_formula_without_T_squared(self):
        delta, record = replacement(self.first, self.second, self.labels, 'CE_T2', ARMS['CE_T2'])
        old = .5 * (self.ce(self.first) + self.ce(self.second))
        new = .5 * (self.ce(self.first, 2.) + self.ce(self.second, 2.))
        torch.testing.assert_close(delta, .25 * (new - old))
        self.assertAlmostEqual(record['new_sat_CE'], float(new))
        self.assertFalse(record['T_squared'])
        self.assertNotAlmostEqual(record['new_sat_CE'], 4. * float(new))

    def test_two_separate_pass_CE_not_CE_of_mean_logits(self):
        _, record = replacement(self.first, self.second, self.labels, 'CE_T2', ARMS['CE_T2'])
        expected = .5 * (self.ce(self.first, 2.) + self.ce(self.second, 2.))
        self.assertAlmostEqual(record['new_sat_CE'], float(expected))
        self.assertGreater(abs(record['new_sat_CE'] - float(self.ce(.5 * (self.first + self.second), 2.))), .01)
        self.assertEqual(record['passes'], 2)
        self.assertTrue(record['two_separate_CE_mean'])

    def test_label_smoothing_matches_uniform_six_class_target(self):
        target = F.one_hot(self.labels, 6).to(self.first.dtype) * .99 + .01 / 6
        expected = -(target * (self.first / 2.).log_softmax(1)).sum(1).mean()
        torch.testing.assert_close(self.ce(self.first, 2.), expected)
        _, record = replacement(self.first, self.second, self.labels, 'CE_T2', ARMS['CE_T2'])
        self.assertEqual(record['label_smoothing'], .01)

    def test_only_satellite_gradients_change_and_full_formula_matches(self):
        generator = torch.Generator().manual_seed(6402)
        for arm, config in ARMS.items():
            with self.subTest(arm=arm):
                z0 = torch.randn(128, 6, generator=generator, dtype=torch.float64, requires_grad=True)
                z1 = torch.randn(128, 6, generator=generator, dtype=torch.float64, requires_grad=True)
                labels = torch.arange(128).remainder(6)
                mask = (torch.arange(128) + 17).remainder(2) != 0
                p, q = z0.log_softmax(1), z1.log_softmax(1)
                kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1).mean()
                common_raw_supcon = z0.square().mean() * .13 + z1.square().mean() * .07
                original = common_raw_supcon + .5 * (.5 * (F.cross_entropy(z0, labels, label_smoothing=.01)
                                    + F.cross_entropy(z1, labels, label_smoothing=.01)) + kl)
                correction, _ = replacement(z0[mask], z1[mask], labels[mask], arm, config)
                after = original + correction
                clean_pair = .5 * (F.cross_entropy(z0[~mask], labels[~mask], label_smoothing=.01)
                                    + F.cross_entropy(z1[~mask], labels[~mask], label_smoothing=.01))
                sat_pair = .5 * config['weight'] * (
                    F.cross_entropy(z0[mask] / config['temperature'], labels[mask], label_smoothing=.01)
                    + F.cross_entropy(z1[mask] / config['temperature'], labels[mask], label_smoothing=.01))
                explicit = common_raw_supcon + .25 * clean_pair + .25 * sat_pair + .5 * kl
                torch.testing.assert_close(after, explicit)
                old_grad = torch.autograd.grad(original, (z0, z1), retain_graph=True)
                new_grad = torch.autograd.grad(after, (z0, z1), retain_graph=True)
                expected_grad = torch.autograd.grad(explicit, (z0, z1), retain_graph=True)
                delta_grad = torch.autograd.grad(correction, (z0, z1))
                for old, new, expected, change in zip(old_grad, new_grad, expected_grad, delta_grad):
                    torch.testing.assert_close(new, expected)
                    torch.testing.assert_close(new[~mask], old[~mask])
                    self.assertEqual(int(torch.count_nonzero(change[~mask])), 0)
                    self.assertGreater(float(change[mask].abs().sum()), 0.)

    def test_T2_logit_chain_rule_and_half_prefactor_only(self):
        target = F.one_hot(self.labels, 6).to(self.first.dtype) * .99 + .01 / 6
        expected_T2 = ((self.first / 2.).softmax(1) - target) / (2. * 64.)
        expected_half = .5 * (self.first.softmax(1) - target) / 64.
        actual_T2, = torch.autograd.grad(self.ce(self.first, 2.), self.first)
        actual_half, = torch.autograd.grad(.5 * self.ce(self.first), self.first)
        torch.testing.assert_close(actual_T2, expected_T2)
        torch.testing.assert_close(actual_half, expected_half)
        self.assertGreater(float((actual_T2 - actual_half).abs().sum()), .01)

    def test_reject_unknown_or_changed_objective(self):
        for arm, config in [('OTHER', ARMS['CE_T2']), ('CE_T2', dict(ARMS['CE_T2'], temperature=3.)),
                            ('CE_HALF', dict(ARMS['CE_HALF'], smoothing=0.)),
                            ('CE_T2', dict(ARMS['CE_T2'], T_squared=True))]:
            with self.subTest(arm=arm, config=config), self.assertRaises(ValueError):
                validate_objective(arm, config)

    def test_reject_wrong_slots_labels_and_nonfinite(self):
        invalid = [(self.first[:63], self.second[:63], self.labels[:63]),
                   (self.first[:, :5], self.second[:, :5], self.labels),
                   (self.first, self.second, self.labels.float()),
                   (self.first, self.second, self.labels + 6)]
        bad = self.first.detach().clone()
        bad[0, 0] = float('nan')
        invalid.append((bad, self.second, self.labels))
        for first, second, labels in invalid:
            with self.subTest(shape=tuple(first.shape), dtype=str(labels.dtype)), self.assertRaises(ValueError):
                replacement(first, second, labels, 'CE_T2', ARMS['CE_T2'])

    def test_original_head_unchanged_or_rejected(self):
        weight = torch.zeros(6, 160)
        head = SimpleNamespace(scale=30., weight=weight)
        validate_head(head)
        self.assertIs(head.weight, weight)
        for bad in [SimpleNamespace(scale=15., weight=weight),
                    SimpleNamespace(scale=30., weight=torch.zeros(6, 161)),
                    SimpleNamespace(scale=30.)]:
            with self.assertRaises(ValueError):
                validate_head(bad)

    def test_capture_reuses_calls_rng_parameters_and_live_logits(self):
        class Channel:
            def __init__(self):
                self.calls = 0
            def apply(self):
                self.calls += 1
                return None, [{} for _ in range(64)]
        class Native:
            def __init__(self):
                self.calls = 0
                self._full_channel = Channel()
                self.logits = torch.zeros(128, 6, requires_grad=True)
            def forward_identity_only(self):
                self.calls += 1
                return {'tx_logits': self.logits}
        native = Native()
        rng = torch.random.get_rng_state().clone()
        original_attributes = set(vars(native))
        with captured_native_views(native) as (logits, records):
            for _ in range(5):
                native.forward_identity_only()
            native._full_channel.apply()
        self.assertEqual(native.calls, 5)
        self.assertEqual(native._full_channel.calls, 1)
        self.assertEqual(len(logits), 5)
        self.assertEqual(len(records), 1)
        self.assertTrue(all(z is native.logits and z.requires_grad for z in logits))
        self.assertEqual(set(vars(native)), original_attributes)
        self.assertNotIn('apply', vars(native._full_channel))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_capture_restores_instance_shadows_after_failure(self):
        channel = SimpleNamespace(apply=lambda: (None, [{} for _ in range(63)]))
        forward = lambda: {'tx_logits': torch.zeros(128, 6)}
        native = SimpleNamespace(_full_channel=channel, forward_identity_only=forward)
        previous_apply = channel.apply
        with self.assertRaises(ValueError):
            with captured_native_views(native):
                native._full_channel.apply()
        self.assertIs(native.forward_identity_only, forward)
        self.assertIs(channel.apply, previous_apply)

    def test_record_coefficient_and_cost_contract(self):
        _, record = replacement(self.first, self.second, self.labels, 'CE_T2', ARMS['CE_T2'])
        self.assertEqual(record['mixed_group_weight'] * record['satellite_fraction'], .25)
        self.assertEqual(record['per_pass_coefficient'] * record['passes'], .25)
        self.assertEqual(record['satellite_slots'], 64)
        self.assertEqual(record['mixed_slots'], 128)
        self.assertEqual(record['extra_NN'], 0)
        self.assertEqual(record['extra_parameters'], 0)
        self.assertTrue(record['symmetric_KL_unchanged'])
        self.assertTrue(record['clean_raw_ordinary_CE_unchanged'])
        self.assertTrue(record['all_satellite_CE_retained'])


def run():
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObjectiveChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  wall_seconds=time.perf_counter() - start,
                  IQ=0, checkpoint_reads=0, model_forwards=0, optimizer_updates=0)
    if not result.wasSuccessful():
        raise AssertionError(report)
    return report


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
