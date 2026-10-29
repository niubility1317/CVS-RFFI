import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_phase3_open_world import star_class_risk_launch as launch


class CPUThresholdChain(unittest.TestCase):
    def runtime(self,root):
        return dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root='/release',python='/python',
                    launch_owner='owned',source_config={},novel_truth='/separate/source_truth.npz')

    def invoke(self,root,job):
        root.mkdir();(root/'logs').mkdir();cfg=root/'runtime.json'
        cfg.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
        with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch,'_job',side_effect=job):
            launch.driver(cfg)
        return json.loads((root/'pipeline_state.json').read_text())

    def test_failed_prediction_never_connects_truth(self):
        calls=[]
        def job(r,state,path,name,command,gpu):calls.append((name,gpu));return False
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,[('source_predict',None)]);self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_existing_root_never_relaunches(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'runtime.json';cfg.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
            with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch.subprocess,'Popen') as child:
                with self.assertRaises(FileExistsError):launch.start(cfg)
            child.assert_not_called()

    def terminal(self,wrong=False):
        from experiments.pamf_phase3_open_world import star_class_risk_run as predictor,star_class_risk_score as scorer
        calls=[]
        def job(r,state,path,name,command,gpu):
            calls.append((name,gpu));self.assertEqual(launch._env(r,gpu)['CUDA_VISIBLE_DEVICES'],'')
            output=Path(r['run_root'])/('prediction' if name=='source_predict' else 'score');output.mkdir()
            result=dict(status=predictor.STATUS if name=='source_predict' else scorer.STATUS,run_id=launch.RUN,
                code_commit=('b' if wrong else 'a')*40,source_gate_passed=False,next_step='NO_PROMOTION')
            (output/'complete.json').write_text(json.dumps(result),encoding='utf-8');return True
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,[('source_predict',None),('source_score',None)])
        return state

    def test_cpu_terminal_science_failure_is_analyzed_not_restarted(self):
        state=self.terminal();self.assertEqual(state['status'],'ANALYZED_SOURCE_ONLY');self.assertFalse(state['source_gate_passed'])

    def test_wrong_terminal_code_cannot_become_analyzed(self):
        self.assertEqual(self.terminal(wrong=True)['status'],'FAILED_PRESERVED')


if __name__=='__main__':unittest.main()
