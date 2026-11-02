from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.phase1_dura_posterior import source_controller as dura_controller
from experiments.phase1_dura_correlation import source_controller


class ControllerTest(unittest.TestCase):
    def test_correlation_and_dura_share_gpu_lease(self):
        self.assertEqual(source_controller.GPU_LOCK, dura_controller.GPU_LOCK)

    def test_failed_real_source_smoke_does_not_start_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, controller = root/'release', root/'controller'
            release.mkdir()
            controller.mkdir()
            (release/'release_commit.txt').write_text('fixed\n', encoding='ascii')
            states = []
            with (patch.object(source_controller, 'ROOT', root),
                  patch.object(source_controller, 'RELEASE', release),
                  patch.object(source_controller, 'CONTROLLER', controller),
                  patch.object(source_controller, 'wait_for_cpu'),
                  patch.object(source_controller, 'gpu_lease', return_value=nullcontext((0, 7))),
                  patch.object(source_controller, 'status', side_effect=lambda state, **kw: states.append(state)),
                  patch.object(source_controller, 'run_module', side_effect=[None, RuntimeError('smoke failed')]) as run,
                  patch('sys.argv', ['source_controller', '--commit', 'fixed'])):
                with self.assertRaisesRegex(RuntimeError, 'smoke failed'):
                    source_controller.main()
            self.assertEqual(run.call_count, 2)
            self.assertIn('--smoke', run.call_args.args[1])
            self.assertEqual(states[-1], 'FAILED')
            self.assertFalse((root/'runs'/source_controller.RUN).exists())


if __name__ == '__main__':
    unittest.main()
