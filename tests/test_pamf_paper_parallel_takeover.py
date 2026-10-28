import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.pamf_paper_completion.parallel_takeover import process_alive, reconcile, scan_live, slots


class TakeoverTests(unittest.TestCase):
    def test_adopts_live_worker_even_if_old_queue_did_not_record_it(self):
        jobs = [{'id': 'train__a', 'kind': 'train', 'config': '/configs/a', 'completion': '/done/a'}]
        state = {'jobs': {'train__a': {'status': 'QUEUED'}}}
        live = {'/configs/a': {'pid': 10, 'gpu': 0}}
        result = reconcile(jobs, state, live, lambda j: False, lambda j: True)
        self.assertEqual(result['jobs']['train__a']['pid'], 10)
        self.assertEqual(result['jobs']['train__a']['status'], 'RUNNING')

    def test_unfinished_old_worker_is_never_launched_again(self):
        jobs = [{'id': 'train__a', 'kind': 'train', 'config': '/configs/a', 'completion': '/done/a'}]
        state = {'jobs': {'train__a': {'status': 'RUNNING', 'pid': 10, 'gpu': 0}}}
        with self.assertRaises(ValueError):
            reconcile(jobs, state, {}, lambda j: False, lambda j: True)

    def test_queued_existing_log_without_process_refuses_relaunch(self):
        jobs = [{'id': 'train__a', 'kind': 'train', 'config': '/configs/a', 'completion': '/done/a'}]
        state = {'jobs': {'train__a': {'status': 'QUEUED'}}}
        with self.assertRaises(ValueError):
            reconcile(jobs, state, {}, lambda j: False, lambda j: True)

    def test_completed_artifact_closes_a_finished_worker(self):
        jobs = [{'id': 'train__a', 'kind': 'train', 'config': '/configs/a', 'completion': '/done/a'}]
        state = {'jobs': {'train__a': {'status': 'RUNNING', 'pid': 10, 'gpu': 0}}}
        result = reconcile(jobs, state, {}, lambda j: True, lambda j: True)
        self.assertEqual(result['jobs']['train__a']['status'], 'COMPLETE')

    def test_slots_skip_foreign_and_cap_at_three(self):
        self.assertEqual(slots({1, 2}, {1}, foreign=False, free_mib=16000), 1)
        self.assertEqual(slots({1, 2, 3}, {1}, foreign=False, free_mib=16000), 0)
        self.assertEqual(slots({1}, {1}, foreign=True, free_mib=16000), 0)
        self.assertEqual(slots({1}, {1}, foreign=False, free_mib=11000), 0)

    def test_unrelated_process_environment_is_never_read(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp) / '123'
            proc.mkdir()
            (proc / 'cmdline').write_bytes(b'/usr/lib/systemd/systemd\x00--user\x00')
            with patch('experiments.pamf_paper_completion.parallel_takeover.os.getuid', return_value=proc.stat().st_uid, create=True), \
                 patch('experiments.pamf_paper_completion.parallel_takeover.process_info', side_effect=PermissionError):
                self.assertEqual(scan_live({'run_root': '/run'}, [], proc_root=Path(temp)), {})

    def test_other_run_worker_environment_is_never_read(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp) / '123'
            proc.mkdir()
            (proc / 'cmdline').write_bytes(b'python\x00-m\x00experiments.pamf_paper_completion.runner\x00worker\x00--config\x00/other/configs/a\x00')
            with patch('experiments.pamf_paper_completion.parallel_takeover.os.getuid', return_value=proc.stat().st_uid, create=True), \
                 patch('experiments.pamf_paper_completion.parallel_takeover.process_info', side_effect=PermissionError):
                self.assertEqual(scan_live({'run_root': '/run'}, [], proc_root=Path(temp)), {})

    def test_dead_owner_detection_never_reads_environ(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc = root / '123'
            proc.mkdir()
            (proc / 'stat').write_text('123 (owner) Z 1')
            self.assertFalse(process_alive(123, proc_root=root))
            (proc / 'stat').write_text('123 (owner) S 1')
            self.assertTrue(process_alive(123, proc_root=root))
            self.assertFalse(process_alive(456, proc_root=root))

