import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from experiments.pamf_phase3_open_world import launch


class Phase3LaunchTests(unittest.TestCase):
    def test_foreign_workers_do_not_exclude_a_card(self):
        self.assertTrue(launch.admissible(12000, owned_workers=0))
        self.assertTrue(launch.admissible(6000, owned_workers=1))
        self.assertFalse(launch.admissible(12000, owned_workers=2))
        self.assertFalse(launch.admissible(3000, owned_workers=0))

    def test_source_calibration_precedes_target_and_truth_is_last(self):
        self.assertEqual(launch.STAGES, ('smoke', 'calibrate', 'build', 'predict', 'score'))

    def test_runtime_requires_nonoverwriting_owned_root(self):
        with self.assertRaises(ValueError):
            launch.validate_runtime(dict(run_id='bad', max_added_workers_per_gpu=3))

    def test_dispatch_automatically_closes_all_stages_in_truth_last_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'pilot'
            runtime = dict(run_id='pilot', run_root=str(root), release_root=folder, python='python',
                code_commit='a' * 40, launch_owner='test-owner', allowed_gpus=[2], max_added_workers_per_gpu=2,
                model=dict(source_cache='source', retention=.99),
                data=dict(source_root='legacy', known_classes=['a','b','c','d','e','f'],
                    unknown_classes=['new'], receivers=[0], evaluation_seed=1, expected_query_count=4))
            observed = []
            class Child:
                pid = 123
                def __init__(self, command, **kwargs):
                    if command[4].endswith('.runner'):
                        kind = command[5]
                        cfg = json.loads(Path(command[7]).read_text(encoding='utf-8'))
                        output = Path(cfg['output_root'])
                        status = dict(smoke='PASS', calibrate='SOURCE_CALIBRATION_FROZEN',
                                      predict='ALL_PREDICTIONS_COMPLETE')[kind]
                        output.mkdir()
                        (output/'complete.json').write_text(json.dumps(dict(status=status)), encoding='utf-8')
                    elif command[4].endswith('.data'):
                        kind = 'build'
                        output = root/'data/evaluation/public'
                        output.mkdir(parents=True)
                        (output/'manifest.json').write_text(json.dumps(dict(data_status='VALIDATED_ONCE',count=4)),encoding='utf-8')
                    else:
                        kind = 'score'
                        self.assert_prediction_complete = (root/'predict/complete.json').is_file()
                        if not self.assert_prediction_complete:
                            raise AssertionError('Truth scorer was launched before predictions completed')
                        output = root/'score'
                        output.mkdir()
                        (output/'complete.json').write_text(json.dumps(dict(status='SCORED_COMPLETE')),encoding='utf-8')
                    observed.append(kind)
                def wait(self):
                    return 0
            with patch.object(launch, 'choose_gpu', return_value=2), patch.object(launch.subprocess, 'Popen', Child):
                launch.dispatch(runtime)
            self.assertEqual(observed, list(launch.STAGES))
            self.assertEqual(launch.read(root/'pipeline_state.json')['status'], 'ANALYZED')
            with self.assertRaises(FileExistsError):
                launch.dispatch(runtime)


if __name__ == '__main__':
    unittest.main()
