import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.phase1_dura_volterra import source_controller


class CpuControllerTest(unittest.TestCase):
    def test_cpu_environment_caps_threads_and_disables_gpu(self):
        env = source_controller.environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            self.assertEqual(env[key], '2')

    def test_smoke_failure_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); release = root/'release'; controller = root/'controller'
            release.mkdir(); controller.mkdir(); (release/'release_commit.txt').write_text('fixed\n')
            states = []
            with (patch.object(source_controller, 'ROOT', root), patch.object(source_controller, 'RELEASE', release),
                  patch.object(source_controller, 'CONTROL', controller),
                  patch.object(source_controller, 'status', side_effect=lambda state, **kw: states.append(state)),
                  patch.object(source_controller, 'run', side_effect=RuntimeError('smoke failed')) as run,
                  patch('sys.argv', ['source_controller', '--commit', 'fixed'])):
                with self.assertRaisesRegex(RuntimeError, 'smoke failed'): source_controller.main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(states[-1], 'FAILED')
            self.assertFalse((root/'runs'/source_controller.RUN).exists())


if __name__ == '__main__': unittest.main()
