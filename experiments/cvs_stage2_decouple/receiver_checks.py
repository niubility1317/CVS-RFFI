"""Focused checks for cross-fit leakage, balanced reliability and real margins."""
import copy
import io
import json
import unittest

import torch

from .receiver import (ReceiverBank, FOLDS, _source, _build_groups,
                       _balanced_group_loss, _reliability, fit_receiver,
                       one_sided_huber)


def fixture(per_group=8, rx_count=3):
    generator = torch.Generator().manual_seed(19)
    rows = [(tx, rx, day, j) for tx in range(6) for day in (0, 1)
            for rx in range(rx_count) for j in range(per_group)]
    state = torch.randn(len(rows), 40, generator=generator) * .02
    state[:, 32] = .6
    y = torch.tensor([r[0] for r in rows])
    rx = torch.tensor([r[1] for r in rows])
    day = torch.tensor([r[2] for r in rows])
    state[:, 0] = rx.float()
    state[:, 1] = y.float() * .1
    margin = 3. * rx + .1 * y + .3 * day
    return dict(state=state, y=y, rx=rx, day=day,
                ids=['p' + str(i) for i in range(len(rows))],
                parent_margin=margin.float(), parent_correct=torch.ones(len(rows), dtype=torch.bool))


class ReceiverChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cache = fixture()
        cls.payload = fit_receiver(cls.cache, 712, fit_steps=40)

    def test_input_contract_excludes_duplicates(self):
        cache = fixture()
        cache['ids'][1] = cache['ids'][0]
        with self.assertRaisesRegex(ValueError, 'unique physical'):
            fit_receiver(cache, 0, fit_steps=1)
        cache = fixture()
        cache['state'][0, 32] = 1.5
        with self.assertRaisesRegex(ValueError, 'coherence'):
            fit_receiver(cache, 0, fit_steps=1)

    def test_only_fit_tx_standardization_and_training(self):
        altered = copy.deepcopy(self.cache)
        held_mask = (altered['y'] == 0) | (altered['y'] == 1)
        altered['state'][held_mask, 3:32] += 100
        altered['parent_margin'][held_mask] += 33
        changed = fit_receiver(altered, 712, fit_steps=40)
        original, other = self.payload['folds'][0], changed['folds'][0]
        self.assertEqual(original['fit_tx'], [2, 3, 4, 5])
        self.assertEqual(original['heldout_tx'], [0, 1])
        self.assertTrue(torch.equal(original['standard_mean'], other['standard_mean']))
        self.assertTrue(torch.equal(original['standard_scale'], other['standard_scale']))
        for key in original['model']:
            self.assertTrue(torch.equal(original['model'][key], other['model'][key]), key)
        allowed = {self.cache['ids'][i] for i in range(len(self.cache['ids']))
                   if int(self.cache['y'][i]) in (2, 3, 4, 5)}
        self.assertEqual(set(original['standard_fit_ids']), allowed)
        self.assertEqual(tuple(tuple(f['heldout_tx']) for f in self.payload['folds']), FOLDS)

    def test_group_mean_not_packet_fit_and_balanced_reductions(self):
        groups = [dict(indices=[0, 1], target=2.), dict(indices=[2], target=-2.),
                  dict(indices=[3, 4, 5], target=0.)]
        blocks = [dict(group_ids=[0, 1]), dict(group_ids=[2])]
        values = torch.tensor([0., 4., 0., 2., 2., 2.], requires_grad=True)
        loss = _balanced_group_loss(values, groups, blocks)
        # First group is exact in mean even though individual samples differ.
        expected = ((0. + 1.5) / 2. + 1.5) / 2.
        self.assertAlmostEqual(float(loss), expected)
        loss.backward()
        self.assertEqual(float(values.grad[:2].abs().sum()), 0.)

    def test_reliability_common_denominator_and_zero_baseline(self):
        groups = [dict(target=0.), dict(target=2.),
                  dict(target=0.), dict(target=4.), dict(target=8.)]
        blocks = [dict(group_ids=[0, 1]), dict(group_ids=[2, 3, 4])]
        predictions = {i: .5 * g['target'] for i, g in enumerate(groups)}
        evidence = _reliability(groups, blocks, predictions)
        # Equal block average: (4 + mean(16,64,16))/2=18, not pair mean25.
        self.assertAlmostEqual(evidence['mse_zero'], 18.)
        self.assertAlmostEqual(evidence['mse_pred'], 4.5)
        self.assertAlmostEqual(evidence['c'], .75)
        self.assertEqual(evidence['relation_pairs'], 4)
        self.assertEqual(evidence['direction_accuracy'], 1.)
        zero = _reliability([dict(target=0.), dict(target=0.)],
                            [dict(group_ids=[0, 1])], {0: 0., 1: 0.})
        self.assertEqual(zero['c'], 0.)
        self.assertIsNone(zero['direction_accuracy'])
        empty = _reliability([], [], {})
        self.assertIsNone(empty['mse_pred'])
        self.assertEqual(empty['c'], 0.)

    def test_fixed_quality_merge_no_cross_day_or_tx(self):
        cache = fixture(per_group=8, rx_count=2)
        for i in range(len(cache['state'])):
            # Four IDs in each adjacent bin; only merging restores >=8.
            cache['state'][i, 32] = .1 if i % 8 < 4 else .3
        groups, blocks, decisions, tau = _build_groups(_source(cache))
        self.assertTrue(all(d['quality_bins'] == 2 for d in decisions))
        for group in groups:
            rows = group['indices']
            self.assertEqual(len(rows), 8)
            self.assertEqual(set(cache['y'][rows].tolist()), {group['tx']})
            self.assertEqual(set(cache['day'][rows].tolist()), {group['day']})
        for tx in range(6):
            expected = torch.quantile(torch.tensor([g['parent_mean'] for g in groups if g['tx'] == tx]), .75)
            self.assertEqual(float(tau[tx]), float(expected))

    def test_sampling_is_unique_and_independent_of_r(self):
        original = ReceiverBank(self.payload)
        changed = copy.deepcopy(self.payload)
        for group in changed['groups']:
            group['prediction'] *= -100
        for fold in changed['folds']:
            fold['c'] = 0.
        altered = ReceiverBank(changed)
        gen_a, gen_b = torch.Generator().manual_seed(44), torch.Generator().manual_seed(44)
        for _ in range(12):
            a, b = original.sample_relations(gen_a), altered.sample_relations(gen_b)
            self.assertTrue(torch.equal(a['indices'], b['indices']))
            self.assertEqual(len(set(a['indices'].tolist())), 32)
            self.assertEqual(len({r['tx'] for r in a['relations']}), 2)
            for relation in a['relations']:
                strong = a['indices'][relation['strong_positions']]
                weak = a['indices'][relation['weak_positions']]
                both = torch.cat((strong, weak))
                self.assertEqual(set(self.cache['y'][both].tolist()), {relation['tx']})
                self.assertEqual(set(self.cache['day'][both].tolist()), {relation['day']})
                self.assertEqual(len(set(self.cache['rx'][strong].tolist())), 1)
                self.assertNotEqual(int(self.cache['rx'][strong[0]]), int(self.cache['rx'][weak[0]]))

    def test_r_loss_fixed_denominator_c_once_and_gradient(self):
        bank = ReceiverBank(self.payload)
        sample = bank.sample_relations(torch.Generator().manual_seed(14))
        active, inactive = sample['relations']
        active.update(valid=True, active=True, c=.4, delta=.5, tau=6., reason='active')
        inactive.update(active=False, c=0., reason='zero_reliability')
        margins = torch.zeros(128, requires_grad=True)
        with torch.no_grad():
            margins[torch.tensor(active['strong_positions']) + 96] = 5.
            margins[torch.tensor(active['weak_positions']) + 96] = 1.
        loss, evidence = bank.identity_loss(margins, sample)
        # z=5-1-.5=3.5; H=3; one c=.4; fixed denominator2 =>.6.
        self.assertAlmostEqual(float(loss), .6, places=6)
        loss.backward()
        self.assertEqual(float(margins.grad[torch.tensor(active['strong_positions']) + 96].abs().sum()), 0.)
        weak_grad = margins.grad[torch.tensor(active['weak_positions']) + 96]
        self.assertTrue(torch.allclose(weak_grad, torch.full((8,), -.4 / 2 / 8)))
        self.assertEqual(float(margins.grad[:96].abs().sum()), 0.)
        self.assertEqual(evidence['fixed_denominator'], 2)
        self.assertEqual(evidence['reliability_applications'], 1)
        # The same differentiable real margin path reaches an identity parameter.
        parameter = torch.tensor(1., requires_grad=True)
        scaled = margins.detach() * parameter
        routed, _ = bank.identity_loss(scaled, sample)
        routed.backward()
        self.assertNotEqual(float(parameter.grad), 0.)

    def test_satisfied_constraints_and_empty_groups_return_zero(self):
        cache = fixture(per_group=7, rx_count=2)
        payload = fit_receiver(cache, 17, fit_steps=10)
        self.assertEqual(payload['groups'], [])
        self.assertTrue(torch.isnan(payload['tau_y']).all())
        self.assertTrue(all(f['actual_fit_steps'] == 0 and f['c'] == 0. for f in payload['folds']))
        bank = ReceiverBank(payload)
        sample = bank.sample_relations(torch.Generator().manual_seed(9))
        self.assertEqual(len(sample['indices']), 32)
        self.assertEqual(sample['diagnostics']['fallback_rows'], 32)
        margins = torch.ones(32, requires_grad=True)
        loss, _ = bank.identity_loss(margins, sample, offset=0)
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(margins.grad.abs().sum()), 0.)
        z = torch.tensor([-2., 0., .5, 2.], requires_grad=True)
        one_sided_huber(z).sum().backward()
        self.assertTrue(torch.equal(z.grad, torch.tensor([0., 0., .5, 1.])))

    def test_serialization_and_rng_preservation(self):
        before = torch.random.get_rng_state().clone()
        payload = fit_receiver(self.cache, 31, fit_steps=2)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        self.assertTrue(torch.equal(restored['tau_y'], payload['tau_y']))
        self.assertEqual(restored['parent_forward_calls'], 0)
        self.assertTrue(all(f['actual_fit_steps'] == 2 for f in restored['folds']))


if __name__ == '__main__':
    result = unittest.main(exit=False)
    print(json.dumps(dict(status='PASS' if result.result.wasSuccessful() else 'FAIL',
                          tests=result.result.testsRun, failures=len(result.result.failures),
                          errors=len(result.result.errors))))
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
