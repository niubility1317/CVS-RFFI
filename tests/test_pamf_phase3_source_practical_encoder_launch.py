import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class EncoderLaunchTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_encoder_launch')

    def runtime(self,root):
        return dict(run_id='test',run_root=str(root/'test'),release_root=str(root/'release'),python='python',code_commit='a'*40,
            launch_owner='owner',allowed_gpus=[0,1],max_added_workers_per_gpu=2,minimum_free_mb=4096,max_concurrent_rows=2,
            targets_path=str(root/'targets.json'),cache_config={},rows={a:dict(train_config={},source_config={},
                source_novel_truth='source_private',development={c:dict(predict_config={},truth_path=c+'_private') for c in ('C0','P1')})
                for a in ('clean_increment','practical_increment')})

    def test_fixed_stage_order_and_private_truth_only_in_scorer(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));jobs=module.row_commands(rt,'clean_increment')
            self.assertEqual([j['name'] for j in jobs],['train','source','source_score','C0_predict','C0_score','P1_predict','P1_score'])
            for job in jobs:
                is_score=job['name'].endswith('score')
                self.assertEqual(job['gpu'],not is_score)
                self.assertEqual('--truth' in job['argv'] or '--novel-truth' in job['argv'],is_score)
                if not is_score:
                    self.assertNotIn('source_private',job['argv']);self.assertNotIn('C0_private',job['argv'])

    def test_gpu_reservation_never_adds_third_and_uses_remaining_memory(self):
        module=self.module()
        rt=dict(allowed_gpus=[0,1],minimum_free_mb=4096)
        snapshot='0, 10000\n1, 8000\n'
        with patch.object(module.subprocess,'check_output',return_value=snapshot):
            self.assertEqual(module.choose_gpu(rt,[0,0]),1)
            self.assertEqual(module.choose_gpu(rt,[0]),1)

    def test_existing_root_never_spawns_or_overwrites(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));p=Path(rt['run_root']);p.mkdir();(p/'keep').write_text('keep')
            with patch.object(module,'validate_runtime',return_value=rt),patch.object(module.subprocess,'Popen') as child:
                with self.assertRaises(FileExistsError):module.launch(rt,'runtime.json')
            child.assert_not_called();self.assertEqual((p/'keep').read_text(),'keep')


if __name__=='__main__':unittest.main()
