"""Independent synthetic evidence checks for paired support summaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools'), str(ROOT/'tests')]
import evaluate_d92_sequential_residual_probe as probe
import summarize_d92_sequential_residual_probe as summary
from test_evaluate_d92_sequential_residual_probe import synthetic, as_record


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic()
        cls.steps = []
        result = probe.probe_sequential_residual(**cls.inputs, step_callback=cls.steps.append)
        cls.record, cls.split = as_record(result, cls.inputs, 2)

    def test_independent_recomputation_and_resource_counts(self):
        stages, metrics = summary.verify_record(self.record, self.split, self.inputs['old_classes'])
        self.assertEqual(len(stages), 20)
        self.assertEqual(metrics['proxy'], probe.parent_mean(self.record['oneshot_proxy']['trials']))
        resources = {}
        summary.accumulate_resources(resources, self.record, stages)
        self.assertEqual(resources['base_fit_stage_count'], 8)
        self.assertEqual(resources['residual_fit_stage_count'], 12)
        self.assertGreater(resources['residual_actual_prepare_seconds_sum'], 0.)
        self.assertNotIn('residual_prepare_seconds_sum', resources)

    def test_wrong_counts_scores_and_physical_ids_rejected(self):
        for mutation in ('optimizer', 'held', 'scores', 'shared_b', 'scale'):
            record = deepcopy(self.record)
            path = record['folds'][0]
            if mutation == 'optimizer': record['optimizer_steps'] += 1
            if mutation == 'held': path['c_ids'][0] = path['c_training_ids'][0]
            if mutation == 'scores': path['paths']['R_seq']['c_scores'][0][0] += 20
            if mutation == 'shared_b': path['paths']['R_reset']['b_scores'][0][0] += 20
            if mutation == 'scale': path['residual_stages'][2]['q'] += .1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summary.verify_record(record, self.split, self.inputs['old_classes'])

    def test_k1_missing_metrics_are_explicit(self):
        inputs = synthetic(k=1)
        record, split = as_record(probe.probe_sequential_residual(**inputs), inputs, 1)
        self.assertEqual(summary.verify_record(record, split, inputs['old_classes']), ([], dict(oof=None, proxy=None)))
        record['optimizer_steps'] = 64
        with self.assertRaisesRegex(ValueError, 'K1'):
            summary.verify_record(record, split, inputs['old_classes'])

    def test_step_stream_has_exactly_64_updates_each_and_matching_compact(self):
        stages, _ = summary.verify_record(self.record, self.split, self.inputs['old_classes'])
        expected = [(self.split['split_id'], stage) for stage in stages if stage['event'] == 'RESIDUAL_FIT']
        full = [dict(row, split_id=self.split['split_id']) for row in self.steps]
        with tempfile.TemporaryDirectory() as temporary:
            lane = Path(temporary)
            def save(rows):
                (lane/'training_steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
                (lane/'training_steps_compact.jsonl').write_text(''.join(json.dumps(probe.scalars(row))+'\n' for row in rows), encoding='utf-8')
            save(full)
            self.assertEqual(summary.verify_step_stream(lane, expected), 12*64)
            broken = deepcopy(full); broken[0]['step'] = 2; save(broken)
            with self.assertRaisesRegex(ValueError, 'sequence'):
                summary.verify_step_stream(lane, expected)

    def test_incomplete_pilot_never_opens_score_stream(self):
        spec = dict(execution=dict(remote_run_root='unused'), rows=[])
        values = [dict(spec=spec, commit='same'), dict(status='FAILED', model_rows=4, completed_rows=3, episodes=120, commit='same'), {}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(summary, 'validate_spec'), \
             patch.object(summary, 'read', side_effect=values), patch.object(summary, 'jsonlines') as scores:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                summary.summarize(spec=spec, output=str(Path(temporary)/'out'))
            scores.assert_not_called()

    def test_six_correctness_transitions_partition_old_samples(self):
        for scope in ('oof',):
            for value in self.record[scope]['paths'].values():
                fractions = [v for key, v in value['metrics'].items() if key.startswith('correctness_B_Cold_C_')]
                self.assertEqual(len(fractions), 6)
                self.assertAlmostEqual(sum(fractions), 1.)


if __name__ == '__main__': unittest.main()
