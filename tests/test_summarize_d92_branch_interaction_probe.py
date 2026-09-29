import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'code')]
import evaluate_d92_branch_interaction_probe as entry
import summarize_d92_branch_interaction_probe as summary
from test_evaluate_d92_branch_interaction_probe import fixture


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')


def complete_fixture(root):
    run = root/'run'; run.mkdir(); rows = []; cohorts = {}
    for cohort in ('rx1','rx3'):
        for seed in range(2026092701,2026092705):
            rid = f'{cohort}-{seed}'; lane = run/rid; lane.mkdir()
            args = fixture(lane,ks=(1,2)); args['output'] = lane/'probe'
            args['expected_model_seed'] = seed
            for name in ('features_complete.json','startup.json','checkpoint_provenance.json'):
                path = args['support_features']/name; value = json.loads(path.read_text(encoding='utf-8'))
                value['model_seed'] = seed
                if 'provenance' in value: value['provenance']['model_seed'] = seed
                write(path,value)
            with contextlib.redirect_stdout(io.StringIO()): entry.evaluate(**args)
            rows.append(dict(row_id=rid,cohort=cohort,seeds=dict(model=seed),expected_checkpoint_sha256=args['expected_checkpoint_sha256']))
            cohorts[cohort] = dict(matrix=args['config']['matrix'],capsule_id=args['expected_capsule_id'],expected_split_count=2)
    spec = dict(run_id='synthetic',execution=dict(remote_run_root=str(run)),rows=rows,
                probe=dict(cohorts=cohorts,total_episodes=16))
    write(run/'startup.json',dict(spec=spec,query_access=False,source_sample_access=False,commit='abc',started=1.))
    write(run/'complete.json',dict(status='SUPPORT_PROBE_COMPLETE',completed_rows=8,model_rows=8,episodes=16,
        query_access=False,source_sample_access=False,query_performance_claim=False,optimizer_steps=0,commit='abc',finished=2.))
    write(run/'state.json',{row['row_id']:dict(status='SUPPORT_PROBE_COMPLETE',episodes=2) for row in rows})
    return spec,run


class InteractionSummaryTests(unittest.TestCase):
    def test_candidate_screen_uses_both_controls_strict_gain_and_each_k_guard(self):
        overall = {}; by_k = {}
        for control in ('linear','energy_control'):
            pair = 'interaction_minus_'+control
            for metric in ('h','new_accuracy'): summary.add(overall,('new_present','paired',pair+'.'+metric),.02)
            for k in (5,10,20):
                summary.add(by_k,('new_present',k,'paired',pair+'.old_accuracy'),-.009)
                for metric in ('h','new_accuracy'): summary.add(by_k,('new_present',k,'paired',pair+'.'+metric),.02)
        self.assertTrue(summary.candidate_assessment(overall,by_k,full_matrix=True)['passes'])
        key = ('new_present',10,'paired','interaction_minus_energy_control.old_accuracy')
        by_k.pop(key); summary.add(by_k,key,-.011)
        self.assertFalse(summary.candidate_assessment(overall,by_k,full_matrix=True)['passes'])
        self.assertIsNone(summary.candidate_assessment(overall,by_k,full_matrix=False)['passes'])

    def test_positive_overall_cannot_hide_a_negative_or_zero_k_improvement(self):
        for control in ('linear','energy_control'):
            for metric in ('h','new_accuracy'):
                for failed_k in (5,10,20):
                    for failing_value in (-.01,0.):
                        with self.subTest(control=control,metric=metric,k=failed_k,value=failing_value):
                            overall = {}; by_k = {}
                            for reference in ('linear','energy_control'):
                                pair = 'interaction_minus_'+reference
                                for measured in ('h','new_accuracy'):
                                    values = []
                                    for k in (5,10,20):
                                        value = failing_value if (reference,measured,k) == (control,metric,failed_k) else .03
                                        values.append(value)
                                        summary.add(by_k,('new_present',k,'paired',pair+'.'+measured),value)
                                    summary.add(overall,('new_present','paired',pair+'.'+measured),sum(values)/3)
                                for k in (5,10,20): summary.add(by_k,('new_present',k,'paired',pair+'.old_accuracy'),0.)
                            result = summary.candidate_assessment(overall,by_k,full_matrix=True)
                            self.assertGreater(result['comparisons'][control]['mean_h_delta'],0)
                            self.assertGreater(result['comparisons'][control]['mean_new_accuracy_delta'],0)
                            self.assertTrue(result['available'])
                            self.assertFalse(result['comparisons'][control]['passes'])
                            self.assertFalse(result['passes'])

    def test_complete_stream_is_paired_and_reads_no_arrays_or_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec,run = complete_fixture(root); original = Path.open; opened = []
            def guarded(path,*a,**kw):
                mode = a[0] if a else kw.get('mode','r')
                if 'r' in mode:
                    opened.append(path)
                    self.assertIn(path.name,{'startup.json','complete.json','state.json','probe_complete.json',
                                            'features_complete.json','fit_trace.jsonl','compact.jsonl'})
                return original(path,*a,**kw)
            with patch.object(Path,'open',guarded): result = summary.summarize(spec=spec,output=root/'summary')
            self.assertTrue(opened)
            self.assertEqual(result['coverage']['episodes'],16)
            self.assertEqual(result['coverage']['oof_episodes'],8)
            self.assertEqual(result['coverage']['factorizations'],48)
            pair = next(v for v in result['statistics'] if v['metric'] == 'interaction_minus_linear.h')
            self.assertEqual(pair['count'],8)
            self.assertFalse(result['automatic_promotion']); self.assertIsNone(result['selected_arm'])
            self.assertTrue((root/'summary/by_newcount.csv').exists())
            with self.assertRaises(FileExistsError): summary.summarize(spec=spec,output=root/'summary')

    def test_missing_episode_or_corrupt_paired_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); spec,run = complete_fixture(root)
            path = run/spec['rows'][0]['row_id']/'probe/fit_trace.jsonl'
            lines = path.read_text(encoding='utf-8').splitlines()
            value = json.loads(lines[1]); value['paired']['interaction_minus_linear']['rows'][0]['correct_delta'] = 99
            lines[1] = json.dumps(value); path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Paired physical'): summary.summarize(spec=spec,output=root/'summary')
            self.assertFalse((root/'summary').exists())
            path.write_text(lines[0]+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Trace/compact'): summary.summarize(spec=spec,output=root/'summary')

    def test_allold_new_and_h_remain_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp),ks=(2,))
            arrays,tasks,old,*_ = entry.load_support(support_features=args['support_features'],capsule=args['capsule'],
                expected_capsule_id=args['expected_capsule_id'],expected_checkpoint_sha256=args['expected_checkpoint_sha256'],
                expected_model_seed=args['expected_model_seed'],config=dict(algorithm=entry.CACHE_VALIDATION_CONFIG,matrix=args['config']['matrix']))
            split,positions,labels = tasks[0]
            record = entry.probe_branch_interaction(**{k:v[positions] for k,v in arrays.items()},support_labels=labels,
                support_ids=split['support_ids'],classes=split['registered_classes'],old_classes=split['registered_classes'])
            record.update(new_count=0,query_rows_used=0,source_rows_used=0)
            summary.verify_oof(record)
            for kind,key,value in summary.episode_metrics(record):
                if key.endswith('.h') or key.endswith('.new_accuracy'): self.assertIsNone(value)


if __name__ == '__main__': unittest.main()
