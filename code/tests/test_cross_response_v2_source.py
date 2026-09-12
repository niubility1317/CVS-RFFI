from types import SimpleNamespace
import unittest

import torch

from cvsrffi.cross_response.predictor import SharedResponsePredictor
from cvsrffi.cross_response.readouts import ResponseReadouts
from cvsrffi.cross_response.schema import SampleRecord
from cvsrffi.cross_response.source_baselines import (
    build_donor_reuse_tasks, fit_source_baselines, prediction_errors,
)
from cvsrffi.cross_response.source_eval import evaluate_source_response_v2
from cvsrffi.cross_response.statistics import WaveformStatistics, FixedSourceNormalizer
from cvsrffi.cross_response.replay_audit import (
    first_divergence, snapshot_training_state, summarize_gradient_audits,
)


def records(ntx=6, nrx=5, n=4):
    return [SampleRecord(i, t, r, 1, 'clean', (t, r, s))
            for i, (t, r, s) in enumerate((t, r, s) for t in range(ntx)
                for r in range(nrx) for s in range(n))]


class SourceTests(unittest.TestCase):
    def test_strict_whole_source_reuse_physical_counts_and_rx_topology(self):
        plans = build_donor_reuse_tasks(records(), k=2, max_tasks=1)
        for kind, item in plans.items():
            self.assertEqual(item['status'], 'AVAILABLE')
            task = item['tasks'][0]
            task.validate()
            self.assertEqual(len(task.donor_records[0]), len(task.donor_records[1]))
            self.assertFalse({r.physical_sample_id for r in task.donor_records[0]} &
                             {r.physical_sample_id for r in task.donor_records[1]})
            self.assertEqual(len(task.donor_tx[0]), 2)
            self.assertEqual(len(task.donor_rx[0]), 2)
        tx, rx = plans['tx_reuse']['tasks'][0], plans['rx_reuse']['tasks'][0]
        self.assertEqual(len(set(tx.query_tx+tx.donor_tx[0]+tx.donor_tx[1])), 6)
        self.assertEqual(len(set(rx.query_rx+rx.donor_rx[0]+rx.donor_rx[1])), 5)
        self.assertEqual(plans['rx_reuse']['query_shape'], [2, 1])
        self.assertFalse(plans['rx_reuse']['independent_query_interaction'])

    def test_insufficient_records_or_source_axes_are_na_not_padded(self):
        few = build_donor_reuse_tasks(records(n=2), k=2, max_tasks=1)
        self.assertEqual(few['tx_reuse']['status'], 'N/A')
        self.assertEqual(few['rx_reuse']['status'], 'N/A')
        few_tx = build_donor_reuse_tasks(records(ntx=5), k=2, max_tasks=1)
        self.assertEqual(few_tx['tx_reuse']['status'], 'N/A')
        self.assertEqual(few_tx['rx_reuse']['status'], 'AVAILABLE')

    def test_matched_fit_is_actual_learning_with_same_budget_and_no_caller_change(self):
        torch.manual_seed(33)
        predictor = SharedResponsePredictor(2, 1)
        with torch.no_grad():
            for p in predictor.parameters():
                p.zero_()
        tx, rx = torch.eye(2), torch.tensor([[1., 0.], [0., 1.]])
        target = torch.tensor([[[2.], [-2.]], [[2.], [-2.]]])
        tasks = [{'tx': tx, 'rx': rx, 'target': target}]
        before = {k: v.clone() for k, v in predictor.state_dict().items()}
        report = fit_source_baselines(predictor, tasks, tasks, steps=30, lr=.05, seed=3,
            fit_physical_ids=[1, 2], eval_physical_ids=[3, 4],
            fit_role='source_train', eval_role='source_validation')
        self.assertLess(report['rx_only']['errors'][0]['mse'], 1.)
        self.assertGreater(report['tx_only']['errors'][0]['mse'], 3.9)
        self.assertEqual(report['rx_only']['fit_task_indices'], report['full_matched_fit']['fit_task_indices'])
        self.assertLess(report['rx_only']['parameter_count'], report['full_matched_fit']['parameter_count'])
        self.assertFalse(report['head_only']['independent_evidence'])
        self.assertIsNone(first_divergence(before, predictor.state_dict()))
        with self.assertRaises(ValueError):
            fit_source_baselines(predictor, tasks, tasks, steps=1, lr=.1, seed=1,
                fit_physical_ids=[1], eval_physical_ids=[1], fit_role='source_train', eval_role='source_validation')

    def test_error_decomposition_and_distinct_ratio_aggregates(self):
        prediction = torch.randn(3, 2, 4)
        e = prediction_errors(prediction, torch.zeros_like(prediction))
        self.assertAlmostEqual(e['mse'], e['grand_mse']+e['tx_mse']+e['rx_mse']+e['interaction_mse'])
        reports = [{'weighted_norms': {'a': 1., 'b': 1.}, 'pairs': {'a/b': {'norm_ratio': 1.}}},
                   {'weighted_norms': {'a': 9., 'b': 3.}, 'pairs': {'a/b': {'norm_ratio': 3.}}}]
        out = summarize_gradient_audits(reports)['pairs']['a/b']
        self.assertEqual(out['mean_step_norm_ratio'], 2.)
        self.assertEqual(out['ratio_of_mean_norms'], 2.5)

    def test_dataset_entrypoint_source_fit_v2_and_costs(self):
        class Dataset:
            transform = None
            def __init__(self, split, offset):
                self.split_source = split
                self.index = [SimpleNamespace(tx_i=t, rx_i=r, day_i=1, eq_i=0, sig_i=s+offset)
                              for t in range(6) for r in range(5) for s in range(4)]
            def raw_iq(self, index):
                r = self.index[index]
                t = torch.arange(8).float()
                return torch.stack((torch.sin(t+r.tx_i)*(.5+r.sig_i/100), torch.cos(t+r.rx_i)))
            def __getitem__(self, index):
                r = self.index[index]
                return self.raw_iq(index), r.tx_i, r.rx_i
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 3)
            def forward(self, iq, **kwargs):
                z = self.linear(iq.flatten(1))
                return {'z_id': z, 'z_dom': z}
        model, readouts = Model(), ResponseReadouts(3, 3, 2)
        predictor = SharedResponsePredictor(2, 5)
        statistics = WaveformStatistics('iq', input_length=8)
        normalizer = FixedSourceNormalizer(5).fit(torch.randn(10, 5))
        train, validation = Dataset('ssdg_labeled_tx_visible', 0), Dataset('ssdg_source_v_cal', 20)
        before = snapshot_training_state(model, None, extra_state={'readouts': readouts, 'predictor': predictor})
        report = evaluate_source_response_v2(model, readouts, predictor, statistics, normalizer,
            validation, {'K': 2, 'source_eval_max_blocks': 1, 'source_baseline_fit_steps': 2,
                         'source_baseline_fit_lr': .01}, 'cpu', {i: i for i in range(5)},
            source_role='source_validation', source_train_dataset=train)
        self.assertEqual(report['status'], 'VERIFIED')
        self.assertEqual(report['cost']['backbone_forwards'], 8)
        self.assertEqual(report['cost']['auxiliary_fit_forwards'], 6)
        self.assertEqual(report['reuse']['rx_reuse']['tasks'][0]['branches'][0]['shuffled_rx'], None)
        self.assertIsNone(report['necessity_gate_passed'])
        self.assertIsNone(first_divergence(before, snapshot_training_state(model, None, extra_state={'readouts': readouts, 'predictor': predictor})))


if __name__ == '__main__':
    unittest.main()
