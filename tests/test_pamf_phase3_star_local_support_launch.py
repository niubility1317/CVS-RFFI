import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests import test_pamf_phase3_star_consensus_run as fixtures


class LocalSupportLaunchTests(unittest.TestCase):
    def fixture(self,root):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_local_support_launch');run=root/'run'
        c=fixtures.StarConsensusRunTests().config(root);c.update(run_id=m.RUN,code_commit='a'*40,teacher_root=str(root/'teacher'),output_root=str(run/'rows/original_F0/predict'))
        c['source_config'].update(run_id=m.RUN,row_id='star-local-support-s2026092701',output_root=c['output_root'])
        r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(run),release_root=str(root/'release'),python='python',launch_owner='test',
            allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,max_concurrent_rows=1,
            rows={'original_F0':{'source_config':c}},novel_truth='private.npz')
        path=root/'runtime.json';path.write_text(json.dumps(r));return m,r,path

    def test_runtime_fixed_single_row_root_and_capacity(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));self.assertIs(m.validate(r),r)
            for defect in ('field','root','capacity','rows','concurrency'):
                with self.subTest(defect=defect):
                    bad=copy.deepcopy(r)
                    if defect=='field':bad['rows']['original_F0']['source_config']['target_root']='forbidden'
                    elif defect=='root':bad['rows']['original_F0']['source_config']['output_root']='other-owner'
                    elif defect=='capacity':bad['max_added_workers_per_gpu']=3
                    elif defect=='concurrency':bad['max_concurrent_rows']=2
                    else:bad['rows']['joint']=bad['rows']['original_F0']
                    with self.assertRaises(ValueError):m.validate(bad)

    def test_successful_source_to_independent_cpu_score_closes_terminal_fields(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/original_F0').mkdir(parents=True);commands=[]
            def job(rt,state,sp,name,cmd,gpu):
                commands.append((name,cmd,gpu))
                if name.endswith('_score'):
                    out=root/'rows/original_F0/score';out.mkdir();(out/'complete.json').write_text(json.dumps(dict(
                        status='STAR_LOCAL_SUPPORT_SCORES_COMPLETE',run_id=m.RUN,source_gate_passed=True,eligible_primary=True,
                        next_step='SOURCE_LOCAL_SUPPORT_READY_FOR_TARGET_FALSIFICATION')))
                return True
            with patch.object(m,'_job',side_effect=job),patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'original_F0',2)
            self.assertEqual([c[0] for c in commands],['original_F0_source','original_F0_score']);self.assertIsNone(commands[1][2])
            self.assertIn('experiments.pamf_phase3_open_world.star_local_support_run',commands[0][1]);self.assertNotIn('run',commands[0][1])
            row=json.loads((root/'rows/original_F0/state.json').read_text());self.assertEqual(row['status'],'ANALYZED_SOURCE_ONLY')
            self.assertTrue(row['source_gate_passed']);self.assertTrue(row['eligible_primary'])
            with patch.object(m,'_job',return_value=True),patch.object(m,'_gpu_free',return_value=[2]):m.driver(path)
            pipeline=json.loads((root/'pipeline_state.json').read_text());self.assertEqual(pipeline['status'],'ANALYZED_SOURCE_ONLY')
            self.assertTrue(pipeline['eligible_primary']);self.assertEqual(pipeline['primary'],'original_F0::rx_consensus_relative')

    def test_source_failure_preserves_partial_no_score_no_restart(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/original_F0').mkdir(parents=True)
            with patch.object(m,'_job',return_value=False) as job,patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'original_F0',2)
            self.assertEqual(job.call_count,1);self.assertEqual(json.loads((root/'rows/original_F0/state.json').read_text())['status'],'FAILED_PRESERVED')
            with self.assertRaises(FileExistsError):m.start(path)


if __name__=='__main__':unittest.main()
