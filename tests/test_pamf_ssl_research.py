import unittest
import torch
from torch.nn import functional as F
from experiments.pamf_ssl_research.objective import confidence_weight, ReceiverPrototypes, Objective
from experiments.pamf_ssl_research.plan import configuration, SEEDS, METHODS, GPU_BY_SEED
from experiments.pamf_ssl_research.control import validate_rows


def template():
    return dict(num_classes=6, source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3],
                optimizer_updates=11100, epochs=50, trajectory=[20, 10, 20])


class SSLResearchTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for mixed-device regression')
    def test_teacher_moves_dynamic_cpu_layers_to_the_student_device(self):
        student = torch.nn.Module()
        student.id_backbone = torch.nn.Module()
        encoder = torch.nn.Module()
        encoder.old_layer = torch.nn.Linear(2, 160).to('cuda:0')
        encoder.dynamic_layer = torch.nn.Linear(160, 6)  # Existing installer adds CPU modules.
        student.id_backbone.encoder = encoder
        c = configuration(template(), METHODS[0], SEEDS[0], '/p', 'c')
        objective = Objective(student, c)
        self.assertTrue(all(p.device.type == 'cuda' for p in objective.teacher.parameters()))

    def test_actual_unlabeled_loss_backward_and_own_state_round_trip(self):
        import io
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
                self._cd_practical = lambda x, *args: x + .1
            def forward_identity_only(self, x, **kwargs):
                z = self.id_backbone.encoder.features(x)
                return dict(z_id=z, tx_logits=self.id_backbone.encoder.classify_features(z))
        for method in METHODS:
            c = configuration(template(), method, SEEDS[0], '/p', 'c')
            student = Student()
            x = torch.randn(8, 2, 256)
            meta = dict(rx_i=torch.tensor([1, 3, 4, 6, 8, 1, 3, 4]))
            student._cd_U_loader = [(x, torch.full((8,), -1), torch.zeros(8, dtype=torch.long), meta)]
            student._cd_U_iter = iter(student._cd_U_loader)
            obj = Objective(student, c)
            obj.current_epoch, obj.current_step = 14, 2887
            obj.labeled = (x, torch.zeros(8, dtype=torch.long), meta['rx_i'])
            loss = obj.loss(student, 'cpu')
            loss.backward()
            self.assertGreater(float(loss.detach()), 0.)
            self.assertGreater(obj.effective, 0.)
            self.assertEqual(obj.physical_seen, 4)
            self.assertGreater(float(student.id_backbone.encoder.head.weight.grad.norm()), 0.)
            obj.after_update(student)
            stream = io.BytesIO()
            torch.save(obj.state_dict(), stream)
            stream.seek(0)
            restored = Objective(student, c)
            restored.load_state_dict(torch.load(stream, weights_only=False))
            self.assertEqual(restored.ema_updates, 1)
            self.assertEqual(restored.seen, 8)
            torch.testing.assert_close(restored.bank.counts, obj.bank.counts)
            self.assertTrue(all(p.grad is None for p in restored.teacher.parameters()))

    def test_weights_are_detached_bounded_and_low_confidence_excluded(self):
        p = torch.tensor([.3, .7, .85, .95, .99], requires_grad=True)
        w = confidence_weight(p, torch.tensor(.95), torch.tensor(.01))
        self.assertFalse(w.requires_grad)
        self.assertEqual(w[0], 0.)
        self.assertTrue(bool(((0 <= w) & (w <= 1)).all()))
        self.assertTrue(bool((w[1:] >= w[:-1]).all()))

    def test_own_receiver_cannot_influence_cross_receiver_references(self):
        bank = ReceiverPrototypes([1, 3, 4], 2, 2, 'cpu')
        bank.counts.fill_(1)
        bank.centers[:] = torch.eye(2)
        z = torch.tensor([[1., 0.]])
        fallback = torch.tensor([[.5, .5]])
        before, _, valid = bank.distribution(z, torch.tensor([1]), fallback)
        bank.centers[0] = torch.flip(torch.eye(2), [0])
        after, _, _ = bank.distribution(z, torch.tensor([1]), fallback)
        torch.testing.assert_close(before, after)
        self.assertTrue(valid.all())

    def test_incomplete_bank_falls_back_without_invented_class_probabilities(self):
        bank = ReceiverPrototypes([1, 3, 4], 2, 2, 'cpu')
        p = torch.tensor([[.8, .2]])
        q, _, valid = bank.distribution(torch.ones(1, 2), torch.tensor([1]), p)
        torch.testing.assert_close(p, q)
        self.assertFalse(valid.any())

    def test_prototype_updates_reject_hidden_labels_and_non_source_receivers(self):
        bank = ReceiverPrototypes([1, 3, 4], 2, 2, 'cpu')
        for y, rx in ((-1, 1), (0, 7)):
            with self.assertRaises(ValueError):
                bank.update(torch.ones(1, 2), torch.tensor([y]), torch.tensor([rx]))

    def test_class_permutation_equivariance_and_finite_gradient(self):
        bank = ReceiverPrototypes([1, 3, 4], 3, 4, 'cpu')
        bank.counts.fill_(1)
        bank.centers = F.normalize(torch.randn(3, 3, 4), dim=-1)
        z = torch.randn(7, 4)
        p = torch.randn(7, 3).softmax(1)
        q, _, _ = bank.distribution(z, torch.ones(7, dtype=torch.long), p)
        permutation = torch.tensor([2, 0, 1])
        bank.centers = bank.centers[:, permutation]
        bank.counts = bank.counts[:, permutation]
        r, _, _ = bank.distribution(z, torch.ones(7, dtype=torch.long), p[:, permutation])
        torch.testing.assert_close(r, q[:, permutation])
        logits = torch.randn(7, 3, requires_grad=True)
        loss = -(q.detach() * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_rejects_target_inputs_and_preserves_budget(self):
        c = configuration(template(), METHODS[0], SEEDS[0], '/p', 'c')
        self.assertEqual(c['optimizer_updates'], 11100)
        for key in ('truth', 'tests', 'target_truth', 'views_root', 'teacher_checkpoint'):
            t = dict(template(), **{key: 'forbidden'})
            with self.assertRaises(ValueError):
                configuration(t, METHODS[0], SEEDS[0], '/p', 'c')

    def test_seed_card_binding_and_no_extra_rows(self):
        rows = [dict(row_id=m + str(s), model_seed=s, method=m, gpu=GPU_BY_SEED[s]) for s in SEEDS for m in METHODS]
        validate_rows(rows)
        rows[0]['gpu'] = 2
        with self.assertRaises(ValueError):
            validate_rows(rows)


if __name__ == '__main__':
    unittest.main()
