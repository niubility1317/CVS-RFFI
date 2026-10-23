"""Prevent source/target mixing, raw reruns and DURA substitution for the raw goal."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.pamf_ssl_research.dura_reference import (
    DURA_RELATIVE, METHODS, READOUTS, RUN, prepare, ready)
from experiments.pamf_ssl_research.dura_reference_report import report
from experiments.pamf_ssl_research.plan import SEEDS, VIEWS
from tests.test_pamf_ssl_goal_counts import save


def context():
    g = dict(kind='original_F0_LU', roles='LU', checkpoint='/project/runs/' + DURA_RELATIVE,
             source_contract='/contract.json')
    value = {str(seed): {arm: dict(predict=dict(readouts={'raw': {}, 'DURA-LU': deepcopy(g)},
                    source_contract='/contract.json'), score=dict(output_root='/baseline/' + arm + str(seed)))
                        for arm in ('F20', 'F21')} for seed in SEEDS}
    value['_reference_rows'] = []
    for method in METHODS:
        for seed in SEEDS:
            row = method + '-s' + str(seed)
            source_run = 'old-' + method
            cp = dict(run_id=source_run, row_id=row, source_run_id=source_run, source_row_id=row,
                      model_seed=seed, readouts={'raw': {}}, source_contract='/contract.json',
                      output_root='/old/' + row + '/predictions', source_output='/old/' + row + '/source',
                      tests=dict(main=dict(views_root='/received')))
            cs = dict(run_id=source_run, row_id=row, model_seed=seed, output_root='/old/' + row + '/scores',
                      readouts=['raw'], tests=dict(main=dict(views_root='/received', truth='/truth.json')))
            value['_reference_rows'].append(dict(method=method, model_seed=seed, prediction=cp, scoring=cs))
    return value


def ready_config(root):
    c = dict(source_output=str(root/'source'), raw_predictions_from=str(root/'raw'),
             source_run_id='old', source_row_id='PSM-s2026092701', model_seed=2026092701)
    save(root/'source/complete.json', dict(status='SOURCE_FROZEN', epoch=50, updates=11100,
         target_access=False, config=dict(run_id='old', row_id=c['source_row_id'], model_seed=c['model_seed'])))
    save(root/'raw/complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', run_id='old',
         row_id=c['source_row_id'], truth_read=False, query_fit=False, views=list(VIEWS),
         readouts=['raw'], tests=dict(main=dict(count=168000))))
    return c


def synthetic_scores(method, seed):
    baseline = method in ('F20', 'F21')
    rows, pairs = [], []
    for view in VIEWS:
        # Only DURA passes; raw must still fail the research goal.
        raw_correct = 140000 if view == 'clean' else 120000
        restored_correct = (140000 if view == 'clean' else 125000) if baseline else (
            142000 if view == 'clean' else 128000)
        for readout, correct in zip(READOUTS, (raw_correct, restored_correct)):
            cm = [[0]*6 for _ in range(6)]
            cm[0][0], cm[0][1] = correct, 168000-correct
            rows.append(dict(test='main', readout=readout, view=view, dimension='overall', stratum='ALL',
                count=168000, accuracy=correct/168000, macro_f1=.75, confusion=cm))
        net = restored_correct-raw_correct
        pairs.append(dict(test='main', readout='DURA-LU', view=view, dimension='overall', stratum='ALL',
                          count=168000, repairs=max(net, 0), damage=max(-net, 0), net=net))
    return dict(status='SCORED', truth_last=True, run_id='baseline' if baseline else RUN,
                row_id=method+'-s'+str(seed), model_seed=seed, results=rows, paired_raw=pairs)


def report_fixture(root):
    rt, _ = prepare(context(), '/project', '/release', 'commit')
    rt['run_root'] = str(root/'run')
    save(root/'run/state.json', dict(status='ALL_SCORED'))
    for seed in SEEDS:
        for arm in ('F20', 'F21'):
            folder = root/arm/str(seed)
            save(folder/'scores.json', synthetic_scores(arm, seed))
            rt['baseline_scores'][str(seed)][arm] = str(folder)
    for row in rt['rows']:
        folder = root/row['row_id']
        save(folder/'scores.json', synthetic_scores(row['method'], row['model_seed']))
        config = dict(run_id=RUN, row_id=row['row_id'], model_seed=row['model_seed'],
                      readouts=list(READOUTS), output_root=str(folder))
        row['score_config'] = str(folder/'score-config.json')
        save(Path(row['score_config']), config)
    return rt


class DuraReferenceTest(unittest.TestCase):
    def test_child_reservation_counts_before_nvidia_smi_registration(self):
        from experiments.pamf_ssl_research.dura_reference_control import has_capacity
        active = {'first': dict(child=SimpleNamespace(pid=222), gpu=6)}
        with patch('experiments.pamf_ssl_research.dura_reference_control.subprocess.check_output',
                   side_effect=['111\n', '20000\n']):
            self.assertFalse(has_capacity(6, active))
        with patch('experiments.pamf_ssl_research.dura_reference_control.subprocess.check_output',
                   side_effect=['222\n', '20000\n']):
            self.assertTrue(has_capacity(6, active))

    def test_exact_nine_rows_reuse_raw_and_preserve_training_config(self):
        c = context()
        previous = deepcopy(c)
        rt, blobs = prepare(c, '/project', '/release', 'commit')
        self.assertEqual(c, previous)
        self.assertEqual(len(rt['rows']), 9)
        self.assertFalse(any('train_config' in row for row in rt['rows']))
        for row in rt['rows']:
            cp = blobs['configs/'+row['row_id']+'-predict.json']
            old = next(x['prediction'] for x in c['_reference_rows'] if x['prediction']['row_id']==row['row_id'])
            self.assertEqual(cp['raw_predictions_from'], old['output_root'])
            self.assertEqual(cp['source_output'], old['source_output'])
            self.assertEqual(cp['readouts']['DURA-LU']['checkpoint'], '/project/runs/'+DURA_RELATIVE)
            self.assertNotIn('truth', cp['tests']['main'])

    def test_changed_restorer_or_incomplete_method_matrix_is_rejected(self):
        c = context()
        c['2026092701']['F21']['predict']['readouts']['DURA-LU']['checkpoint'] = '/new.pt'
        with self.assertRaisesRegex(ValueError, 'fixed original DURA'):
            prepare(c, '/project', '/release', 'commit')
        c = context()
        c['_reference_rows'].pop()
        with self.assertRaisesRegex(ValueError, 'three-method/three-seed'):
            prepare(c, '/project', '/release', 'commit')

    def test_readiness_waits_without_creating_or_predicting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = dict(source_output=str(root/'not-yet'), raw_predictions_from=str(root/'raw'))
            self.assertFalse(ready(c))
            self.assertEqual(list(root.iterdir()), [])
            c = ready_config(root)
            self.assertTrue(ready(c))

    def test_wrong_source_seed_or_partial_raw_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = ready_config(root)
            c['model_seed'] += 1
            with self.assertRaisesRegex(ValueError, 'source identity'):
                ready(c)
            c = ready_config(root)
            p = root/'raw/complete.json'
            v = json.loads(p.read_text())
            v['views'] = v['views'][:-1]
            save(p, v)
            with self.assertRaisesRegex(ValueError, 'full raw predictions'):
                ready(c)

    def test_truth_is_rejected_before_source_io(self):
        with self.assertRaisesRegex(ValueError, 'rejects truth'):
            ready(dict(truth='/truth.json'))

    def test_dura_positive_result_never_substitutes_for_raw_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = report_fixture(root)
            done = report(rt, root/'report')
            a = json.loads((root/'report/analysis.json').read_text())
            self.assertFalse(done['goal_met'])
            self.assertTrue(all(v['thresholds_pass'] for v in a['verdicts'] if v['readout']=='DURA-LU'))
            self.assertTrue(all(not v['thresholds_pass'] for v in a['verdicts'] if v['readout']=='raw'))
            self.assertEqual({r['readout'] for r in a['summary']}, set(READOUTS))

    def test_missing_dura_scene_or_false_repair_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = report_fixture(root)
            row = rt['rows'][0]
            c = json.loads(Path(row['score_config']).read_text())
            p = Path(c['output_root'])/'scores.json'
            v = json.loads(p.read_text())
            original = deepcopy(v)
            v['results'] = [r for r in v['results'] if not (r['readout']=='DURA-LU' and r['view']=='clean')]
            save(p, v)
            with self.assertRaisesRegex(ValueError, 'seven-view coverage'):
                report(rt, root/'missing-report')
            original['paired_raw'][0]['net'] += 1
            save(p, original)
            with self.assertRaisesRegex(ValueError, 'repairs/damage'):
                report(rt, root/'invalid-report')


if __name__ == '__main__':
    unittest.main()
