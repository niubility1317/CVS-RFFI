from copy import deepcopy
import unittest
import torch

from experiments.pamf_ssl_research.plan import configuration as base_config, SEEDS
from experiments.pamf_ssl_research.view_anchor import configuration, physical_alignment
from experiments.pamf_ssl_research.view_anchor_train import validate_parent_metadata
from tests.test_pamf_ssl_research import template


class PhysicalViewAnchorTest(unittest.TestCase):
    def test_actual_u_objective_adds_only_declared_e31_loss_and_keeps_own_state(self):
        from experiments.pamf_ssl_research.objective import Objective
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature = torch.nn.Linear(2, 160)
                self.head = torch.nn.Linear(160, 6)
                with torch.no_grad():
                    self.head.bias[0] = 15.
            def features(self, x):
                return self.feature(x.mean(-1))
            def classify_features(self, z):
                return self.head(z)
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.id_backbone = torch.nn.Module()
                self.id_backbone.encoder = Encoder()
                self._cd_U_seen = self._cd_U_accepted = 0
                self._cd_practical = lambda x, *args: x + .3
            def forward_identity_only(self, x, **kwargs):
                z = self.id_backbone.encoder.features(x)
                return dict(z_id=z, tx_logits=self.id_backbone.encoder.classify_features(z))
        parent = base_config(template(), 'RXPSM', SEEDS[0], '/p', 'old')
        c = configuration(parent, '/p', 'new')
        first = Student()
        second = deepcopy(first)
        x = torch.randn(8, 2, 256)
        meta = dict(rx_i=torch.tensor([1, 3, 4, 6, 8, 1, 3, 4]))
        objects, losses = [], []
        for epoch, student in ((30, first), (31, second)):
            student._cd_U_loader = [(x, torch.full((8,), -1), torch.zeros(8, dtype=torch.long), meta)]
            student._cd_U_iter = iter(student._cd_U_loader)
            obj = Objective(student, c)
            obj.ema_updates, obj.seen, obj.physical_seen, obj.calls = 6660, 963900, 481950, 3774
            obj.current_epoch, obj.current_step = epoch, 6661
            obj.labeled = (x, torch.zeros(8, dtype=torch.long), meta['rx_i'])
            torch.manual_seed(81)  # Identical views isolate the newly scheduled term.
            losses.append(obj.loss(student, 'cpu'))
            objects.append(obj)
        self.assertEqual(objects[0].view_anchor_calls, 0)
        self.assertGreater(float(objects[1].metrics['SSL_view_anchor_loss']), 0.)
        torch.testing.assert_close(losses[1] - losses[0], objects[1].metrics['SSL_view_anchor_loss'])
        losses[1].backward()
        self.assertGreater(float(second.id_backbone.encoder.feature.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in objects[1].teacher.parameters()))
        objects[1].after_update(second)
        restored = Objective(second, c)
        restored.load_state_dict(objects[1].state_dict())
        self.assertEqual(restored.ema_updates, 6661)
        self.assertEqual(restored.seen, 963908)
        self.assertEqual(restored.view_anchor_calls, 1)
        self.assertEqual(restored.view_anchor_physical_seen, 4)

    def test_gradient_is_physical_student_only_and_uses_physical_denominator(self):
        student = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        teacher = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
        weight = torch.tensor([.8, .9], requires_grad=True)
        loss = physical_alignment(student, teacher, weight, torch.tensor([True, False]))
        self.assertAlmostEqual(float(loss.detach()), .8, places=6)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(weight.grad)
        self.assertGreater(float(student.grad[0].norm()), 0.)
        self.assertEqual(float(student.grad[1].norm()), 0.)

    def test_absent_physical_views_have_finite_zero_loss(self):
        student = torch.randn(4, 160, requires_grad=True)
        loss = physical_alignment(student, torch.randn_like(student), torch.ones(4), torch.zeros(4, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(float(student.grad.norm()), 0.)

    def test_fixed_parent_config_rejects_other_method_and_target_inputs(self):
        parent = base_config(template(), 'RXPSM', SEEDS[0], '/p', 'old')
        c = configuration(parent, '/p', 'new')
        self.assertEqual(c['optimizer_updates'], 11100)
        self.assertEqual(c['view_anchor']['start_epoch'], 31)
        self.assertEqual(c['view_anchor']['weight'], .1)
        for key, value in (('ssl_method', 'PSM'), ('tests', {'target': 'forbidden'})):
            bad = dict(parent, **{key: value})
            with self.assertRaises(ValueError):
                configuration(bad, '/p', 'new')

    def test_parent_metadata_rejects_seed_contract_and_truth_mismatch(self):
        from pathlib import Path
        parent = base_config(template(), 'RXPSM', SEEDS[0], '/p', 'old')
        c = configuration(parent, '/p', 'new')
        parents = []
        previous = None
        for end, epochs, offset in ((20, 20, 0), (30, 10, 20)):
            folder = Path(c['fork_parent_source']) / ('segment%d' % end)
            pc = dict(parent, parent_source=previous, step_offset=offset * 222, epoch_offset=offset)
            done = dict(config=pc, status='SOURCE_FROZEN', epoch=epochs, updates=epochs * 222,
                        target_access=False, U_truth_read=False, head_fit_updates=0)
            parents.append((folder, done, {'source_roles': 'fixed'}))
            previous = str(folder)
        validate_parent_metadata(c, parents, {'source_roles': 'fixed'})
        for field, value in (('model_seed', SEEDS[1]), ('run_id', 'another-run')):
            bad = deepcopy(parents)
            bad[1][1]['config'][field] = value
            with self.assertRaises(ValueError):
                validate_parent_metadata(c, bad, {'source_roles': 'fixed'})
        bad = deepcopy(parents)
        bad[1][1]['target_access'] = True
        with self.assertRaises(ValueError):
            validate_parent_metadata(c, bad, {'source_roles': 'fixed'})
        with self.assertRaises(ValueError):
            validate_parent_metadata(c, parents, {'source_roles': 'different'})
