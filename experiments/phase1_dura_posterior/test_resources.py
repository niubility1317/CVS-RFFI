import unittest
import tempfile
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.phase1_dura_posterior.resources import select_gpu, gpu_lease


class ResourceTest(unittest.TestCase):
    def test_respects_existing_processes_and_free_memory(self):
        inventory = [(0, 'gpu-a', 20000), (1, 'gpu-b', 9000), (2, 'gpu-c', 15000)]
        processes = [('gpu-a', 10), ('gpu-a', 11), ('gpu-b', 12), ('gpu-c', 13)]
        self.assertEqual(select_gpu(inventory, processes), 2)
        self.assertIsNone(select_gpu(inventory[:2], processes[:3]))

    def test_invalid_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            select_gpu([(0, 'gpu-a', 20000), (0, 'gpu-b', 20000)], [])
        with self.assertRaises(ValueError):
            select_gpu([(0, 'gpu-a', 20000)], [('unknown', 1)])

    def test_capacity_is_rechecked_after_lock_and_lock_is_released_on_error(self):
        events = []
        handles = []
        def flock(handle, operation):
            handles.append(handle)
            events.append(('flock', operation))
        fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
                                flock=flock)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lease.lock'
            with (patch.dict('sys.modules', {'fcntl': fcntl}),
                  patch('experiments.phase1_dura_posterior.resources.wait_for_gpu',
                        side_effect=lambda status: events.append(('capacity', None)) or 3)):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    with gpu_lease(path, lambda *args, **kwargs: None) as (gpu, lock_fd):
                        self.assertEqual(gpu, 3)
                        self.assertIsInstance(lock_fd, int)
                        events.append(('worker', None))
                        raise RuntimeError('worker failed')
        self.assertEqual(events, [('flock', 3), ('capacity', None), ('worker', None)])
        self.assertTrue(handles[0].closed)

    @unittest.skipUnless(os.name == 'posix', 'fcntl lock inheritance requires POSIX')
    def test_worker_inherits_lock_after_controller_closes_its_descriptor(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lease.lock'
            child = None
            try:
                with patch('experiments.phase1_dura_posterior.resources.wait_for_gpu', return_value=0):
                    with gpu_lease(path, lambda *args, **kwargs: None) as (_, lock_fd):
                        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)'],
                                                 stdin=subprocess.PIPE, pass_fds=(lock_fd,))
                with path.open('a', encoding='ascii') as competitor:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    child.stdin.write(b'x')
                    child.stdin.close()
                    self.assertEqual(child.wait(timeout=10), 0)
                    fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if child is not None and child.poll() is None:
                    child.stdin.close()
                    child.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
