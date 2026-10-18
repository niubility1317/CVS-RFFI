"""Focused tensor checks; no actual model, IQ, checkpoint or optimizer update."""
import copy
import io
import math
import time
import unittest

import torch
from torch.nn import functional as F

from .teaching import ARMS, teaching_loss, weight


POLICY = dict(tau=.95, max_weight=.16, ramp_epochs=5)
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')


class TeachingChecks(unittest.TestCase):
    def inputs(self):
        student = torch.tensor([[0., 1., 0., 0., 0., 0.],
                                [0., 0., 3., 0., 0., 0.],
                                [0., 0., 0., 5., 0., 0.],
                                [0., 0., 0., 0., 2., 0.]], requires_grad=True)
        teacher = torch.zeros(4, 6)
        teacher[:, 0] = torch.tensor([6., 7., 8., 9.])
        teacher.requires_grad_()
        labels = torch.full((4,), -1, dtype=torch.long)
        records = [dict(scene=SCENES[i % 3], state_end=('LOS', 'PARTIAL', 'BLOCKED')[i % 3],
                        receiver_lock_status=('locked', 'degraded', 'unlocked')[i % 3],
                        signal_to_added_noise_db=snr)
                   for i, snr in enumerate((-10., 0., 10., 20.))]
        return student, teacher, labels, records

    def call(self, arm='OBSERVATION_WEIGHTED', epoch=5, **changes):
        names = ('student', 'teacher', 'hidden_y', 'metadata')
        values = dict(zip(names, self.inputs()))
        values.update(changes)
        return teaching_loss(**values, arm=arm, local_epoch=epoch, policy=POLICY)

    def test_registered_arms_and_local_epoch_ramp(self):
        self.assertEqual(ARMS, ('MASS_UNIFORM', 'OBSERVATION_WEIGHTED'))
        for arm in ARMS:
            self.assertEqual([weight(arm, e, POLICY) for e in (1, 2, 5, 10)], [.032, .064, .16, .16])
        for arm, epoch, policy in ((ARMS[0], 0, POLICY), (ARMS[1], 21, POLICY),
                                   ('unregistered', 5, POLICY), (ARMS[0], True, POLICY),
                                   (ARMS[0], 1.5, POLICY), (ARMS[0], 1, dict(POLICY, tau=.9)),
                                   (ARMS[0], 1, dict(POLICY, max_weight=.2)),
                                   (ARMS[0], 1, dict(POLICY, ramp_epochs=3))):
            with self.subTest(arm=arm, epoch=epoch, policy=policy):
                with self.assertRaises(ValueError):
                    weight(arm, epoch, policy)

    def test_formula_full_subset_mean_and_equal_quality_mass(self):
        s, t, y, records = self.inputs()
        p = t.detach().softmax(1)
        q_t = ((p.max(1).values - .95) / .05).clamp(0., 1.)
        q_n = torch.sigmoid(torch.tensor([-10., 0., 10., 20.]).sub(5.).div(5.))
        a = (q_t * q_n).sum() / q_t.sum()
        per = F.kl_div(s.log_softmax(1), p, reduction='none').sum(1)
        recs = []
        for arm, q in zip(ARMS, (q_t * a, q_t * q_n)):
            loss, rec = teaching_loss(s, t, y, records, arm, 5, POLICY)
            self.assertTrue(torch.allclose(loss, .16 * (q * per).mean(), atol=1e-6))
            self.assertAlmostEqual(rec['KL'], float((q * per).mean()), places=6)
            self.assertAlmostEqual(rec['q_mass'], float(q.sum()), places=6)
            self.assertAlmostEqual(rec['mass_match']['uniform_multiplier'], float(a), places=6)
            self.assertLess(rec['mass_match']['absolute_difference'], 1e-6)
            self.assertEqual(rec['q_teacher']['count'], 4)
            self.assertEqual(rec['q_observation']['count'], 4)
            self.assertEqual(rec['effective_q']['count'], 4)
            self.assertFalse(rec['selected_count_normalization'])
            self.assertEqual(rec['temperature'], 1.)
            recs.append(rec)
        self.assertAlmostEqual(recs[0]['q_mass'], recs[1]['q_mass'], places=6)

    def test_redistribution_is_not_merely_a_scalar_loss_reduction(self):
        s, t, y, records = self.inputs()
        losses, gradients = [], []
        for arm in ARMS:
            loss, _ = teaching_loss(s, t, y, records, arm, 5, POLICY)
            losses.append(float(loss.detach()))
            gradients.append(torch.autograd.grad(loss, s, retain_graph=True)[0])
        self.assertNotAlmostEqual(*losses, places=4)
        ratios = gradients[1][:, 0] / gradients[0][:, 0]
        self.assertGreater(float(ratios.max() - ratios.min()), .5)

    def test_only_student_receives_gradient_and_no_inputs_are_changed(self):
        s, t, y, records = self.inputs()
        before_s, before_t, before_records = s.detach().clone(), t.detach().clone(), copy.deepcopy(records)
        loss, rec = teaching_loss(s, t, y, records, ARMS[1], 3, POLICY)
        loss.backward()
        self.assertGreater(float(s.grad.abs().sum()), 0.)
        self.assertIsNone(t.grad)
        self.assertTrue(torch.equal(s.detach(), before_s))
        self.assertTrue(torch.equal(t.detach(), before_t))
        self.assertEqual(records, before_records)
        self.assertFalse(rec['U_TX_truth_read'])
        self.assertTrue(rec['old_U_masks_unchanged'])

    def test_no_student_confidence_gate(self):
        s, t, y, records = self.inputs()
        s = torch.zeros_like(s, requires_grad=True)
        loss, rec = teaching_loss(s, t, y, records, ARMS[1], 5, POLICY)
        loss.backward()
        self.assertEqual(rec['positive_q'], 4)
        self.assertFalse(rec['urban_confidence_filter'])
        self.assertTrue(bool((s.grad.abs().sum(1) > 0).all()))

    def test_zero_teacher_confidence_is_finite_and_has_zero_gradient(self):
        s, t, y, records = self.inputs()
        t = torch.zeros_like(t, requires_grad=True)
        for arm in ARMS:
            student = s.detach().clone().requires_grad_()
            loss, rec = teaching_loss(student, t, y, records, arm, 5, POLICY)
            self.assertEqual(float(loss), 0.)
            self.assertTrue(loss.requires_grad)
            self.assertEqual(rec['mass_match']['uniform_multiplier'], 0.)
            self.assertEqual(rec['q_mass'], 0.)
            self.assertEqual(rec['positive_q'], 0)
            loss.backward()
            self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))
            self.assertIsNone(t.grad)

    def test_none_observation_is_zero_without_deleting_sample(self):
        s, t, y, records = self.inputs()
        records[0]['signal_to_added_noise_db'] = None
        loss, rec = teaching_loss(s, t, y, records, ARMS[1], 5, POLICY)
        loss.backward()
        self.assertEqual(rec['none_snr_count'], 1)
        self.assertEqual(rec['q_observation']['count'], 4)
        self.assertEqual(rec['q_observation']['positive'], 3)
        self.assertEqual(sum(row['count'] for row in rec['strata']), 4)
        self.assertTrue(torch.equal(s.grad[0], torch.zeros_like(s.grad[0])))
        self.assertEqual(sum(row['none_snr_count'] for row in rec['strata']), 1)
        uniform_loss, uniform = teaching_loss(s, t, y, records, ARMS[0], 5, POLICY)
        uniform_grad = torch.autograd.grad(uniform_loss, s)[0]
        self.assertGreater(float(uniform_grad[0].abs().sum()), 0.)
        self.assertAlmostEqual(uniform['q_mass'], rec['q_mass'], places=6)

    def test_all_none_observations_zero_both_arms(self):
        s, t, y, records = self.inputs()
        for row in records: row['signal_to_added_noise_db'] = None
        for arm in ARMS:
            loss, rec = teaching_loss(s, t, y, records, arm, 10, POLICY)
            self.assertEqual(float(loss), 0.)
            self.assertEqual(rec['none_snr_count'], 4)
            self.assertEqual(rec['q_mass'], 0.)
            self.assertTrue(all(row['signal_to_added_noise_db_mean'] is None for row in rec['strata']))

    def test_all_three_scenes_and_joint_strata_are_retained(self):
        _, rec = self.call()
        self.assertEqual({r['scene'] for r in rec['strata']}, set(SCENES))
        self.assertEqual(sum(r['count'] for r in rec['strata']), 4)
        self.assertAlmostEqual(sum(r['q_mass'] for r in rec['strata']), rec['q_mass'], places=6)
        self.assertAlmostEqual(sum(r['q_teacher_mass'] for r in rec['strata']), rec['q_teacher']['mass'], places=6)
        self.assertAlmostEqual(sum(r['q_observation_mass'] for r in rec['strata']), rec['q_observation']['mass'], places=6)
        self.assertEqual(sum(rec['effective_q']['histogram']), 4)
        self.assertFalse(rec['observation_proxy_is_recoverability_truth'])

    def test_metadata_stays_in_input_order(self):
        s, t, y, records = self.inputs()
        first, _ = teaching_loss(s, t, y, records, ARMS[1], 5, POLICY)
        second, _ = teaching_loss(s, t, y, list(reversed(records)), ARMS[1], 5, POLICY)
        self.assertNotAlmostEqual(float(first.detach()), float(second.detach()), places=4)

    def test_finite_extreme_snr_has_bounded_weights(self):
        s, t, y, records = self.inputs()
        for row, value in zip(records, (-1e6, 1e6, -100., 100.)):
            row['signal_to_added_noise_db'] = value
        loss, rec = teaching_loss(s, t, y, records, ARMS[1], 5, POLICY)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertGreaterEqual(rec['q_observation']['quantiles'][0], 0.)
        self.assertLessEqual(rec['q_observation']['quantiles'][-1], 1.)

    def test_hidden_truth_dtype_and_shape_negative_protocol_cases(self):
        for labels in (torch.zeros(4, dtype=torch.long), torch.full((4,), -1.),
                       torch.full((4, 1), -1, dtype=torch.long), torch.full((3,), -1, dtype=torch.long)):
            with self.subTest(shape=tuple(labels.shape), dtype=str(labels.dtype)):
                with self.assertRaises(ValueError): self.call(hidden_y=labels)

    def test_invalid_or_nonfinite_logits_rejected(self):
        for name in ('student', 'teacher'):
            for value in (torch.zeros(4, 5), torch.full((4, 6), float('nan')),
                          torch.full((4, 6), float('inf')), torch.zeros(4, 6, dtype=torch.long)):
                with self.subTest(name=name, dtype=str(value.dtype), shape=tuple(value.shape)):
                    with self.assertRaises(ValueError): self.call(**{name: value})
        with self.assertRaises(ValueError):
            self.call(student=torch.zeros(0, 6), teacher=torch.zeros(0, 6),
                      hidden_y=torch.empty(0, dtype=torch.long), metadata=[])

    def test_metadata_structure_fields_and_roles_negative_cases(self):
        _, _, _, records = self.inputs()
        for invalid in (records[:-1], {}, [None] * 4, 'not-records'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError): self.call(metadata=invalid)
        for key in ('scene', 'state_end', 'receiver_lock_status', 'signal_to_added_noise_db'):
            broken = copy.deepcopy(records)
            del broken[0][key]
            with self.subTest(missing=key):
                with self.assertRaises(ValueError): self.call(metadata=broken)
        for key, value in (('scene', 'practical_high_urban'), ('state_end', 'unknown'),
                           ('receiver_lock_status', 'unknown'), ('scene', None)):
            broken = copy.deepcopy(records)
            broken[0][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError): self.call(metadata=broken)

    def test_nonfinite_or_nonnumeric_snr_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf'), '5', True, complex(1, 1)):
            _, _, _, records = self.inputs()
            records[0]['signal_to_added_noise_db'] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.call(metadata=records)


def run():
    started = time.perf_counter()
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TeachingChecks))
    if not result.wasSuccessful():
        raise AssertionError(output.getvalue())
    return dict(status='PASS', tests=result.testsRun, skipped=len(result.skipped),
                CVS_NN=0, real_IQ=0, CP=0, optimizer_updates=0,
                mass_matched=True, whole_subset_mean=True, teacher_stop_gradient=True,
                hidden_TX=True, observation_proxy_not_recoverability_truth=True,
                wall_seconds=time.perf_counter() - started)


if __name__ == '__main__':
    import json
    print(json.dumps(run()))
