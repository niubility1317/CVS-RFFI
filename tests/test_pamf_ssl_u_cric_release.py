from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_ssl_research.plan import SEEDS
from experiments.pamf_ssl_research.u_cric import RUN, PAIR_RUN, paired_dispatch_settled
from experiments.pamf_ssl_research.u_cric_publish import prepare
from tests.test_pamf_ssl_paired_u_release import fixture
from tests.test_pamf_ssl_goal_counts import save


class UnlabeledCRICReleaseTest(unittest.TestCase):
    def test_source_only_training_and_two_readouts_for_three_fixed_seeds(self):
        c = fixture()
        before = deepcopy(c)
        rt, blobs = prepare(c, '/p', '/p/releases/'+RUN, 'new')
        self.assertEqual(c, before)
        self.assertEqual(rt['methods'], ['UCRIC'])
        self.assertEqual(rt['max_compute_processes_per_gpu'], 3)
        self.assertEqual(rt['gpu_predecessor_state'], '/p/runs/'+PAIR_RUN+'/state.json')
        self.assertEqual(rt['prior_score_owner_state'], rt['gpu_predecessor_state'])
        self.assertEqual(len(rt['rows']), 3)
        for row in rt['rows']:
            stem = 'configs/'+row['row_id']
            t, p, s = (blobs[stem+'-'+kind+'.json'] for kind in ('train', 'predict', 'score'))
            self.assertEqual(t['algorithm_id'], 'pamf_u_cric_v1')
            self.assertNotIn('paired_u', t)
            self.assertNotIn('tests', t)
            self.assertNotIn('truth', p['tests']['main'])
            self.assertEqual(list(p['readouts']), ['raw', 'DURA-LU'])
            self.assertEqual(s['readouts'], ['raw', 'DURA-LU'])
            self.assertEqual(t['output_root'], p['source_output'])
            self.assertIn('/'+RUN+'/', p['output_root'])

    def test_predecessor_must_have_dispatched_and_not_have_pending_gpu_work(self):
        state = dict(jobs={'PAIREDU-s'+str(SEEDS[0]): dict(status='QUEUED')})
        with patch('experiments.cric_dura_comparison.io.read', return_value=state):
            self.assertFalse(paired_dispatch_settled(dict(gpu_predecessor_state='state'), SEEDS[0]))
            state['jobs']['PAIREDU-s'+str(SEEDS[0])]['status'] = 'RUNNING'
            self.assertTrue(paired_dispatch_settled(dict(gpu_predecessor_state='state'), SEEDS[0]))
            self.assertFalse(paired_dispatch_settled(dict(gpu_predecessor_state='state'), SEEDS[1]))

    def test_second_slot_smoke_waits_for_predecessor_then_allows_two_existing_apps(self):
        from experiments.pamf_ssl_research.control import run
        class ReachedSmoke(Exception): pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt, _ = prepare(fixture(), '/p', str(root), 'new')
            rt.update(run_root=str(root/'new'), runtime_file=str(root/'runtime.json'))
            def smoke(command, **kwargs):
                self.assertIn('experiments.pamf_ssl_research.u_cric_worker', command)
                raise ReachedSmoke
            with patch('experiments.pamf_ssl_research.u_cric.paired_dispatch_settled', side_effect=[False, True]), \
                 patch('experiments.pamf_ssl_research.control.capacity', return_value=(2, 19000)), \
                 patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
                 patch('experiments.pamf_ssl_research.control.subprocess.run', side_effect=smoke), \
                 patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
                with self.assertRaises(ReachedSmoke): run(rt)
                sleep.assert_called_once_with(20)
                launch.assert_not_called()

    def test_completed_dual_report_keeps_actual_ucric_identity(self):
        from experiments.pamf_ssl_research.dura_reference_report import report
        from tests.test_pamf_ssl_dura_reference import report_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = report_fixture(root)
            rt['run_id'], rt['methods'] = RUN, ['UCRIC']
            rt['rows'] = [r for r in rt['rows'] if r['method'] == 'PSM']
            for row in rt['rows']:
                row.update(method='UCRIC', row_id='UCRIC-s'+str(row['model_seed']))
                c = json.loads(Path(row['score_config']).read_text())
                c.update(run_id=RUN, row_id=row['row_id'])
                save(Path(row['score_config']), c)
                p = Path(c['output_root'])/'scores.json'
                value = json.loads(p.read_text())
                value.update(run_id=RUN, row_id=row['row_id'])
                save(p, value)
            report(rt, root/'report')
            a = json.loads((root/'report/analysis.json').read_text())
            self.assertEqual({v['method'] for v in a['verdicts']}, {'UCRIC'})
            self.assertFalse(a['goal_met'])
            self.assertIn('| UCRIC |', (root/'report/report.md').read_text(encoding='utf-8'))


if __name__ == '__main__': unittest.main()
