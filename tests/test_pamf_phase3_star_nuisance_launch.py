import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pamf_phase3_open_world import star_nuisance_launch as launch


class LaunchOwnership(unittest.TestCase):
    def test_existing_output_cannot_spawn_or_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'owned'; root.mkdir(); token=root/'preserved'; token.write_text('old')
            runtime=Path(directory)/'runtime.json'; runtime.write_text(json.dumps({'run_root':str(root)}))
            with patch.object(launch,'validate',side_effect=lambda x:x), patch.object(launch.subprocess,'Popen') as spawn:
                with self.assertRaises(FileExistsError):launch.start(runtime)
                spawn.assert_not_called()
            self.assertEqual(token.read_text(),'old')

    def test_driver_failure_is_terminal_and_does_not_start_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'logs').mkdir(); (root/'configs').mkdir()
            runtime=root/'runtime.json'; runtime.write_text(json.dumps(dict(run_root=str(root),release_root=str(root),
                python='python',code_commit='a'*40,allowed_gpus=[0],minimum_free_mb=4096,novel_truth='private.npz')))
            with patch.object(launch,'validate',side_effect=lambda x:x), patch.object(launch.subprocess,'check_output',return_value='0, 1024\n'), patch.object(launch.subprocess,'Popen') as spawn:
                launch.driver(runtime); spawn.assert_not_called()
            state=json.loads((root/'pipeline_state.json').read_text())
            self.assertEqual(state['status'],'FAILED_PRESERVED')
            self.assertEqual(state['jobs'],{})


if __name__=='__main__':unittest.main()
