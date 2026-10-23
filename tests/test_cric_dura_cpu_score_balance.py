import tempfile
import unittest
from pathlib import Path

from experiments.cric_dura_comparison import multiprocess_control as mc


class CPUScoreBalance(unittest.TestCase):
    def candidates(self, eligible, active, jobs, limit=4, host_ok=True):
        method = getattr(mc, 'cpu_score_candidates', None)
        self.assertIsNotNone(method, 'Bounded parallel CPU scoring is missing')
        return method(eligible, active, jobs, limit, host_ok)

    def test_fills_only_free_CPU_slots_without_relaunching_active_job(self):
        jobs = {str(i): dict(id=str(i), kind='score', dependencies=[]) for i in range(7)}
        active = {'0': dict(pid=100, gpu=None)}
        selected = self.candidates(list(jobs.values()), active, jobs)
        self.assertEqual([j['id'] for j in selected], ['1', '2', '3'])
        self.assertEqual(active, {'0': dict(pid=100, gpu=None)})

    def test_dependencies_and_non_score_jobs_cannot_enter_CPU_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'prediction_complete.json'
            jobs = {'score': dict(id='score', kind='score', dependencies=[str(marker)]),
                    'GPU': dict(id='GPU', kind='predict', dependencies=[])}
            self.assertEqual(self.candidates(list(jobs.values()), {}, jobs), [])
            marker.write_text('{}', encoding='utf-8')
            self.assertEqual([j['id'] for j in self.candidates(list(jobs.values()), {}, jobs)], ['score'])

    def test_pressure_or_full_pool_only_blocks_new_admission(self):
        jobs = {str(i): dict(id=str(i), kind='score', dependencies=[]) for i in range(5)}
        active = {str(i): dict(pid=100+i, gpu=None) for i in range(4)}
        self.assertEqual(self.candidates(list(jobs.values()), active, jobs), [])
        self.assertEqual(self.candidates(list(jobs.values()), {}, jobs, host_ok=False), [])
        self.assertEqual(len(active), 4)

    def test_default_policy_stays_serial_and_bounded_opt_in_rejects_bad_values(self):
        policy = getattr(mc, 'cpu_score_settings', None)
        self.assertIsNotNone(policy, 'Explicit CPU resource policy is missing')
        self.assertEqual(policy({}), (1, 30.0))
        self.assertEqual(policy(dict(max_cpu_score_workers=4, poll_seconds=1)), (4, 1.0))
        for bad in (0, 9, True, 2.5):
            with self.assertRaises(ValueError):
                policy(dict(max_cpu_score_workers=bad))
        for bad in (0, 31, float('nan'), True):
            with self.assertRaises(ValueError):
                policy(dict(poll_seconds=bad))

    def test_handover_keeps_resolved_science_jobs_and_disables_one_time_extensions(self):
        try:
            from experiments.cric_dura_comparison.publish_cpu_balance import runtime_for
        except ImportError:
            self.fail('CPU-only resource handover is missing')
        previous = dict(project='/project', run_id='old', release_root='/project/releases/old',
            control_root='/project/runs/old', runtime_file='/project/releases/old/runtime.json',
            jobs='/project/runs/old/resolved_jobs.json', source_jobs='/old-original-jobs.json',
            worker_release='/science', worker_commit='science-commit', allowed_gpus=list(range(8)),
            physics_device_recovery=True, core_extension=True)
        rt = runtime_for(previous, 'new-controller-commit')
        self.assertEqual(rt['source_jobs'], '/project/runs/old/resolved_jobs.json')
        self.assertEqual(rt['jobs'], previous['jobs'])
        self.assertEqual(rt['worker_release'], '/science')
        self.assertEqual(rt['worker_commit'], 'science-commit')
        self.assertEqual(rt['max_cpu_score_workers'], 4)
        self.assertEqual(rt['poll_seconds'], 1)
        self.assertFalse(rt['physics_device_recovery'])
        self.assertFalse(rt['core_extension'])
        self.assertEqual(previous['source_jobs'], '/old-original-jobs.json')


if __name__ == '__main__':
    unittest.main()
