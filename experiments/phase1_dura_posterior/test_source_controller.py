import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.phase1_dura_posterior import source_controller as controller


class SourceControllerTest(unittest.TestCase):
    def test_one_technical_failure_does_not_cancel_the_other_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root/'release'
            output = root/'controller'
            release.mkdir()
            output.mkdir()
            (release/'release_commit.txt').write_text('fixed\n', encoding='ascii')
            calls = []
            states = []

            def arm(kind, name, commit):
                calls.append(kind)
                if kind == 'mean':
                    raise RuntimeError('arm-specific problem')
                return {'source': name, 'L_s_scored': 1}

            with (patch.object(controller, 'RELEASE', release),
                  patch.object(controller, 'OUTPUT', output),
                  patch.object(controller, 'wait_for_real_smoke'),
                  patch.object(controller, 'run_arm', side_effect=arm),
                  patch.object(controller, 'status', side_effect=lambda state, **kw: states.append((state, kw))),
                  patch('sys.argv', ['source_controller', '--commit', 'fixed'])):
                controller.main()
            self.assertEqual(calls, ['mean', 'nonlocal'])
            self.assertEqual(states[-1][0], 'PARTIAL_TECHNICAL_FAILURE')
            self.assertIn('mean', states[-1][1]['failures'])
            self.assertIn('nonlocal', states[-1][1]['completed'])

    def test_failed_training_smoke_never_starts_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'controller'
            output.mkdir()
            with (patch.object(controller, 'ROOT', root),
                  patch.object(controller, 'OUTPUT', output),
                  patch.object(controller, 'wait_for_gpu', return_value=0),
                  patch.object(controller, 'status'),
                  patch.object(controller, 'run_module', side_effect=RuntimeError('smoke failed')) as run):
                with self.assertRaisesRegex(RuntimeError, 'smoke failed'):
                    controller.run_arm('mean', 'new-source', 'fixed')
            self.assertEqual(run.call_count, 1)
            self.assertIn('--smoke', run.call_args.args[1])
            self.assertFalse((root/'runs/new-source').exists())


if __name__ == '__main__':
    unittest.main()
