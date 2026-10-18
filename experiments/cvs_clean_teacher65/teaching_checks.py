"""Tensor/toy checks only: hidden labels, attribution, masks, frozen state and RNG."""
import copy
import io
import tempfile
import time
from pathlib import Path
from types import MethodType
import unittest
import torch
from torch.nn import functional as F
from . import design as d
from .teaching import ARMS, TEACHERS, teaching_loss, weight
from .rng import private_rng

POLICY = dict(tau=.95, max_weight=.16, ramp_epochs=5)


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 6)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = 8.
        self.drop = torch.nn.Dropout(.5)
        self.register_buffer('persistent_marker', torch.tensor([.7]))
        self.packets = 0

    def forward_identity_only(self, x, domain_labels=None):
        self.packets += len(x)
        features = x.mean(-1) if x.ndim == 3 else x
        return dict(tx_logits=self.head(self.drop(features)))


class TeachingChecks(unittest.TestCase):
    def inputs(self, n=4):
        student = torch.zeros(n, 6, requires_grad=True)
        one = torch.zeros(n, 6, requires_grad=True)
        with torch.no_grad(): one[:, 0] = 8.
        two = one.detach().clone().roll(1, 1).requires_grad_()
        three = one.detach().clone().roll(2, 1).requires_grad_()
        return student, dict(zip(TEACHERS, (one, two, three))), torch.full((n,), -1, dtype=torch.long)

    def test_batchmean_not_selected_count_and_detached_teacher(self):
        student, teachers, y = self.inputs()
        with torch.no_grad():
            for t in teachers.values(): t[1:] = 0.
        loss, rec = teaching_loss(student, teachers, y, 'FROZEN_AS20_EMA', 5, POLICY)
        p = teachers['FROZEN_AS20_EMA'].detach().softmax(1)
        q = ((p.max(1).values - .95) / .05).clamp(0, 1)
        expected = .16 * (q * F.kl_div(student.log_softmax(1), p, reduction='none').sum(1)).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(rec['positive_q'], 1)
        self.assertLess(rec['q_mean'], .25)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.)
        self.assertTrue(all(t.grad is None for t in teachers.values()))
        self.assertTrue(torch.equal(student.grad[1:], torch.zeros_like(student.grad[1:])))

    def test_each_arm_uses_its_teacher_and_zero_executes_all_readouts(self):
        student, teachers, y = self.inputs()
        for arm in ARMS:
            loss, rec = teaching_loss(student, teachers, y, arm, 1, POLICY)
            self.assertEqual(set(rec['teachers']), set(TEACHERS))
            self.assertTrue(all(sum(r['pseudo_counts']) == 4 for r in rec['teachers'].values()))
            self.assertEqual(rec['selected_teacher'], TEACHERS[0] if arm == 'ZERO_KL' else arm)
            self.assertEqual(rec['weight'], 0. if arm == 'ZERO_KL' else .032)
            self.assertEqual(float(loss), 0. if arm == 'ZERO_KL' else rec['weighted_KL'])
            self.assertTrue(loss.requires_grad)
        self.assertEqual(len(rec['teacher_disagreement']), 3)
        self.assertTrue(all(v == 1. for v in rec['teacher_disagreement'].values()))

    def test_ramp_uses_new_local_epochs_only(self):
        self.assertEqual([weight('EVOLVING_AS_EMA', e, POLICY) for e in (1, 2, 5, 10)], [.032, .064, .16, .16])
        self.assertEqual(weight('ZERO_KL', 10, POLICY), 0.)
        for e in (0, 14, 21):
            with self.assertRaises(ValueError): weight('EVOLVING_AS_EMA', e, POLICY)

    def test_hidden_labels_unknown_arm_missing_nonfinite_teacher_rejected(self):
        s, t, y = self.inputs()
        with self.assertRaises(ValueError): teaching_loss(s, t, torch.zeros_like(y), ARMS[1], 1, POLICY)
        with self.assertRaises(ValueError): teaching_loss(s, t, y, 'renamed', 1, POLICY)
        with self.assertRaises(ValueError): teaching_loss(s, {TEACHERS[0]: t[TEACHERS[0]]}, y, ARMS[1], 1, POLICY)
        t[TEACHERS[0]] = torch.full_like(s, float('nan'))
        with self.assertRaises(ValueError): teaching_loss(s, t, y, ARMS[1], 1, POLICY)
        with self.assertRaises(ValueError): weight(ARMS[1], 1, dict(POLICY, tau=.9))

    def test_student_confidence_does_not_filter_teacher_supervision(self):
        s, t, y = self.inputs()
        # The urban student is exactly uniform, while all clean teachers trust a class.
        loss, rec = teaching_loss(s, t, y, ARMS[1], 10, POLICY)
        self.assertEqual(rec['positive_q'], len(y))
        self.assertFalse(rec['urban_confidence_filter'])
        loss.backward()
        self.assertTrue(bool((s.grad.abs().sum(1) > 0).all()))

    def test_private_extra_stream_preserves_native_rng(self):
        torch.manual_seed(810)
        x = torch.ones(12, 8)
        start = torch.get_rng_state().clone()
        with private_rng(x, 4441): first = F.dropout(x, .5, True)
        self.assertTrue(torch.equal(start, torch.get_rng_state()))
        with private_rng(x, 4441): second = F.dropout(x, .5, True)
        self.assertTrue(torch.equal(first, second))
        with private_rng(x, 4442): third = F.dropout(x, .5, True)
        self.assertFalse(torch.equal(first, third))

    def test_external_holder_does_not_enter_student_ema_or_optimizer(self):
        from . import model as api
        student = ToyHead()
        bank = api.FrozenTeachers(dict(B01=ToyHead(), B00=ToyHead()))
        before = {k: v.clone() for k, v in student.state_dict().items()}
        api._FROZEN_TEACHERS[student] = bank
        own_ema = copy.deepcopy(student)
        self.assertEqual(set(before), set(student.state_dict()))
        with self.assertRaises(ValueError): api.teacher_bank(own_ema)
        ids = {id(p) for p in student.parameters()}
        self.assertFalse(ids & {id(p) for t in bank.teachers.values() for p in t.parameters()})
        optimizer = torch.optim.SGD(student.parameters(), lr=.1)
        self.assertEqual(ids, {id(p) for g in optimizer.param_groups for p in g['params']})
        x, domains = torch.ones(7, 2, 256), torch.zeros(7, dtype=torch.long)
        rng = torch.get_rng_state().clone()
        values = bank.forward(x, domains)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(not z.requires_grad for z in values.values()))
        self.assertTrue(bank.assert_frozen())
        self.assertEqual(bank.summary()['forward_NN'], dict(B01=7, B00=7))
        self.assertEqual(bank.summary()['total_forward_NN'], 14)
        self.assertTrue(all(torch.equal(v, student.state_dict()[k]) for k, v in before.items()))
        with torch.no_grad(): next(bank.teachers['B00'].parameters()).add_(.01)
        with self.assertRaises(ValueError): bank.assert_frozen()

    def test_native_mask_live_gradient_and_prefix128_tail124(self):
        from . import model as api
        from experiments.cvs_fullsource_concat20.supcon import RX_BY_DOMAIN
        for arm in ARMS:
            for n, local_step in ((256, 1), (124, 222)):
                with tempfile.TemporaryDirectory() as folder:
                    student = ToyHead().train()
                    student._feature_l_steps = local_step
                    student._clean_teacher_config = dict(arm=arm, output_root=folder, U_teaching=POLICY)
                    student._clean_teacher_pending = None
                    student._clean_teacher_packets = 0
                    student._clean_teacher_records = []
                    student._clean_teacher_physics = False
                    class Channel:
                        sessions = {}
                        def apply(self, x, meta, epoch, step):
                            if epoch != 21 or step != 4440 + local_step: raise AssertionError('Cumulative stream differs')
                            self.packets = len(x)
                            return x + .1, [dict(scene='toy') for _ in x]
                    student._clean_teacher_channel = Channel()
                    student._full_channel = Channel()
                    bank = api.FrozenTeachers(dict(B01=ToyHead(), B00=ToyHead()))
                    api._FROZEN_TEACHERS[student] = bank
                    own_ema = ToyHead().eval().requires_grad_(False)
                    x = torch.arange(n * 512, dtype=torch.float32).reshape(n, 2, 256) / (n * 512)
                    domains = torch.arange(n) % 15
                    meta = dict(eq_i=torch.ones(n, dtype=torch.long), rx_i=torch.tensor(RX_BY_DOMAIN)[domains],
                                day_i=domains % 3 + 1, sig_i=torch.arange(n) + 4000, base_index=torch.arange(n))
                    t = torch.zeros(n, 6); t[:, 0] = 8.
                    live_native = student.forward_identity_only(x)['tx_logits']
                    base, strong = torch.arange(n) % 3 == 0, torch.arange(n) % 5 != 0
                    rng = torch.get_rng_state().clone()
                    mask, _ = api.select(student, model=student, teacher=own_ema, x_u=x,
                        teacher_logits=t, student_logits=live_native, pseudo=t.argmax(1), base_mask=base,
                        strong_mask=strong, extra=(domains, meta), domains=domains, optimizer_step=local_step)
                    self.assertTrue(torch.equal(mask, base & strong))
                    loss, _ = api.extra_hook(student, model=student, teacher=own_ema,
                        student_output=dict(tx_logits=live_native), teacher_output=dict(tx_logits=t),
                        extra=(domains, meta), domains=domains, optimizer_step=local_step,
                        identity_loss=live_native.sum() * 0., pseudo_loss=live_native.sum() * 0.,
                        entropy_loss=live_native.sum() * 0.)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertTrue(loss.requires_grad)
                    loss.backward()
                    if arm == 'ZERO_KL': self.assertEqual(float(student.head.weight.grad.abs().sum()), 0.)
                    else: self.assertGreater(float(student.head.weight.grad.abs().sum()), 0.)
                    k, record = min(n, 128), student._clean_teacher_records[-1]
                    self.assertEqual(student._clean_teacher_packets, k)
                    self.assertEqual(record['native_n'], n)
                    self.assertEqual(record['additional_n'], k)
                    self.assertEqual(record['selected_physical_ids'][-1][-1], k - 1)
                    self.assertEqual(record['original_selected'], int((base & strong).sum()))
                    self.assertEqual(record['extra_frozen_teacher_NN'], 2 * k)
                    self.assertEqual(bank.summary()['forward_NN'], dict(B01=k, B00=k))
                    self.assertTrue(bank.assert_frozen())
                    self.assertIsNone(student._clean_teacher_pending)
                    self.assertTrue((Path(folder) / 'U_physics.jsonl.gz').exists())


def run():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TeachingChecks)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(output.getvalue())
    return dict(status='PASS', tests=result.testsRun, skipped=len(result.skipped), CVS_NN=0,
                real_IQ=0, CP=0, optimizer_updates=0, wall_seconds=time.perf_counter() - started,
                original_U_mask_unchanged=True, native_prefix128_tail124=True,
                frozen_teachers_external_to_student_EMA=True, hidden_TX=True)


if __name__ == '__main__':
    import json
    print(json.dumps(run()))
