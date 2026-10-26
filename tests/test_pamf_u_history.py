import unittest
from unittest.mock import patch
import torch
from experiments.pamf_u_history.history import History, METHODS, KEYS


class HistoryTests(unittest.TestCase):
    def setup_history(self, n=2):
        return History([(1, 3, 1, i, i) for i in range(n)]), torch.arange(n)

    def test_positive_is_exact_ordinary_pl_without_prior(self):
        for method in METHODS:
            h, ids = self.setup_history()
            logits = torch.tensor([[6., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]], requires_grad=True)
            p = logits.detach().softmax(1)
            expected = (torch.nn.functional.cross_entropy(logits, p.argmax(1), reduction='none') * (p.max(1).values >= .95)).mean()
            value, _ = h.loss(logits, p, ids, method, update=444)
            self.assertTrue(torch.equal(value, expected))
            self.assertEqual(h.negative_seen + h.temporal_seen, 0)

    def test_previous_two_visits_only_and_negative_on_rejected(self):
        h, ids = self.setup_history()
        p = torch.tensor([[.6, .3, .025, .025, .025, .025], [.98, .004, .004, .004, .004, .004]])
        logits = p.log().requires_grad_()
        h.loss(logits, p, ids, 'PL-History', update=1)
        h.loss(logits, p, ids, 'PL-History', update=2)
        self.assertEqual(h.negative_seen, 0)
        value, stats = h.loss(logits, p, ids, 'PL-History', update=444)
        self.assertEqual(stats['UH_negative_batch'], 1)
        self.assertEqual(stats['UH_temporal_batch'], 2)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()), 0.)

    def test_current_prediction_cannot_replace_historical_target(self):
        h, ids = self.setup_history(1)
        a = torch.tensor([[.8, .1, .025, .025, .025, .025]])
        b = torch.tensor([[.1, .8, .025, .025, .025, .025]])
        h.loss(a.log(), a, ids, 'PL-Temporal', update=1)
        h.loss(a.log(), a, ids, 'PL-Temporal', update=2)
        _, stats = h.loss(b.log(), b, ids, 'PL-Temporal', update=444)
        self.assertGreater(float(stats['UH_temporal_loss']), 1.)

    def test_extreme_logits_remain_finite(self):
        h, ids = self.setup_history(1)
        p = torch.tensor([[.6, .3, .025, .025, .025, .025]])
        for _ in range(2):
            h.loss(p.log(), p, ids, 'PL-Negative', update=1)
        logits = torch.tensor([[-1000., -1000., 1000., -1000., -1000., -1000.]], requires_grad=True)
        loss, _ = h.loss(logits, p, ids, 'PL-Negative', update=444)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_label_permutation_equivariance(self):
        p = torch.tensor([[.6, .3, .025, .025, .025, .025], [.7, .2, .025, .025, .025, .025]])
        permutation = torch.tensor([2, 5, 0, 1, 4, 3])
        values = []
        for q in (p, p[:, permutation]):
            h, ids = self.setup_history()
            for _ in range(3):
                loss, _ = h.loss(q.log(), q, ids, 'PL-History', update=444)
            values.append(loss)
        self.assertTrue(torch.allclose(*values, atol=1e-7))

    def test_truth_foreign_role_and_physical_records_rejected(self):
        h, _ = self.setup_history()
        m = {k: torch.tensor([row[j] for row in h.lookup]) for j, k in enumerate(KEYS)}
        m['tx_label_visible'] = torch.tensor([False, False])
        for labels, role, training in ((torch.tensor([0, -1]), 'U_s', True), (torch.tensor([-1, -1]), 'V', True), (torch.tensor([-1, -1]), 'U_s', False)):
            with self.assertRaises(ValueError):
                h.indices(m, labels, role=role, training=training)
        m['base_index'][0] = 999
        with self.assertRaises(ValueError):
            h.indices(m, torch.tensor([-1, -1]), role='U_s', training=True)
        self.assertEqual(h.seen, 0)

    def test_pilot_contains_no_comparator_training(self):
        from experiments.pamf_u_history import source
        self.assertEqual(source.METHODS, METHODS)
        for method in ('SIG1-PL', 'SIG1-Sup', 'ordinary_PL', 'supervised'):
            with self.assertRaises(ValueError):
                source.config({}, 'test', method)

    def test_completed_prefix_configuration_and_segment_parent_isolation(self):
        from experiments.pamf_u_history import source
        context = dict(project='PROJECT', python='PYTHON', frozen_sources={}, G=dict(kind='original_F0_LU', roles='LU'), cases=[
            dict(row_id='F0-ordinary_PL-s'+str(seed), protocol='F0', model_seed=seed,
                 parent_source='PROJECT/ordinary/segment20', expected_source_contract='PROJECT/contract.json',
                 source_config=dict(ablation='F21', num_classes=6)) for seed in (2026092701, 2026092702, 2026092703)])
        header = dict(config=dict(source_cache='PROJECT/source', source_contract='PROJECT/contract.json',
                                  source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3], num_classes=6))
        for method in METHODS:
            with patch.object(source, 'read', return_value=header):
                c = source.config(context, 'test', method)
            root_out = c['output_root']
            for epochs, offset, parent in ((10, 20, c['shared_SIG1_E20']), (20, 30, root_out+'/segment30')):
                segment = dict(c, output_root=root_out+'/segment'+str(offset+epochs), segment_epochs=epochs,
                               epoch_offset=offset, step_offset=offset*222, parent_source=parent)
                source.validate_config(segment)
                segment['parent_source'] = 'foreign-run/segment30'
                with self.assertRaises(ValueError):
                    source.validate_config(segment)
            self.assertEqual(c['sig_time_views'], 1)
            self.assertTrue(c['U_use'])


if __name__ == '__main__':
    unittest.main()
