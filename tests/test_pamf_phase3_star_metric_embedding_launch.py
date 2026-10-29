import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tests import test_pamf_phase3_star_consensus_run as fixtures


class MetricEmbeddingLaunchTests(unittest.TestCase):
    def fixture(self,root):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_metric_embedding_launch');run=root/'run';rows={}
        for arm in m.ARMS:
            c=fixtures.StarConsensusRunTests().config(root);c.update(run_id=m.RUN,arm=arm,code_commit='a'*40,parent_prediction_root=str(root/'parent'),output_root=str(run/'rows'/arm/'predict'))
            c['source_config'].update(run_id=m.RUN,row_id='star-metric-embedding-'+arm+'-s2026092701',output_root=c['output_root'])
            rows[arm]={'source_config':c}
        r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(run),release_root=str(root/'release'),python='python',launch_owner='test',allowed_gpus=list(range(8)),
            minimum_free_mb=4096,max_added_workers_per_gpu=2,max_concurrent_rows=2,rows=rows,novel_truth='private.npz')
        path=root/'runtime.json';path.write_text(json.dumps(r));return m,r,path

    def test_three_arm_runtime_root_and_capacity_are_fixed(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));self.assertIs(m.validate(r),r)
            for defect in ('field','root','arm','capacity','concurrency','rows'):
                with self.subTest(defect=defect):
                    bad=copy.deepcopy(r)
                    if defect=='field':bad['rows'][m.PRIMARY_ARM]['source_config']['query_truth']='forbidden'
                    elif defect=='root':bad['rows'][m.PRIMARY_ARM]['source_config']['output_root']='other-owner'
                    elif defect=='arm':bad['rows'][m.PRIMARY_ARM]['source_config']['arm']='identity'
                    elif defect=='capacity':bad['max_added_workers_per_gpu']=3
                    elif defect=='concurrency':bad['max_concurrent_rows']=3
                    else:del bad['rows']['identity']
                    with self.assertRaises(ValueError):m.validate(bad)

    def test_source_immediately_enters_cpu_score_with_correct_terminal_fields(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);arm=m.PRIMARY_ARM;(root/'rows'/arm).mkdir(parents=True);calls=[]
            def job(rt,state,sp,name,cmd,gpu):
                calls.append((name,cmd,gpu))
                if name.endswith('_score'):
                    out=root/'rows'/arm/'score';out.mkdir();(out/'complete.json').write_text(json.dumps(dict(status='STAR_METRIC_EMBEDDING_SCORES_COMPLETE',
                        run_id=m.RUN,arm=arm,source_gate_passed=True,arm_source_gate_passed=True,eligible_primary=True,next_step='SOURCE_METRIC_EMBEDDING_READY_FOR_TARGET_FALSIFICATION')))
                return True
            with patch.object(m,'_job',side_effect=job),patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,arm,2)
            self.assertEqual([v[0] for v in calls],[arm+'_source',arm+'_score']);self.assertIsNone(calls[1][2]);self.assertNotIn('run',calls[0][1])
            self.assertIn('experiments.pamf_phase3_open_world.star_metric_embedding_run',calls[0][1])
            row=json.loads((root/'rows'/arm/'state.json').read_text());self.assertEqual(row['status'],'ANALYZED_SOURCE_ONLY');self.assertTrue(row['eligible_primary'])

    def test_failure_preserves_partial_and_never_scores_restarts_or_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/identity').mkdir(parents=True)
            with patch.object(m,'_job',return_value=False) as job,patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'identity',2)
            self.assertEqual(job.call_count,1);self.assertEqual(json.loads((root/'rows/identity/state.json').read_text())['status'],'FAILED_PRESERVED')
            with self.assertRaises(FileExistsError):m.start(path)

    def test_two_distinct_cards_all_rows_continue_and_controls_cannot_replace_main(self):
        for failed_control in (False,True):
            with self.subTest(failed_control=failed_control),tempfile.TemporaryDirectory() as td:
                m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'logs').mkdir(parents=True);active=0;max_active=0;launched=[]
                for arm in m.ARMS:(root/'rows'/arm).mkdir(parents=True)
                class Process:
                    def __init__(self,command,**kwargs):
                        nonlocal active,max_active
                        self.pid=100+len(launched);self.done=False;arm=command[command.index('--arm')+1];gpu=int(command[command.index('--gpu')+1]);launched.append((arm,gpu))
                        active+=1;max_active=max(max_active,active);out=root/'rows'/arm
                        (out/'state.json').write_text(json.dumps(dict(status='FAILED_PRESERVED' if failed_control and arm=='identity' else 'ANALYZED_SOURCE_ONLY')))
                        (out/'score').mkdir();passed=arm!=m.PRIMARY_ARM
                        (out/'score/complete.json').write_text(json.dumps(dict(source_gate_passed=passed,eligible_primary=False,next_step='CONTROL_ONLY_COMPLETE' if passed else 'NO_PROMOTION')))
                    def poll(self):
                        nonlocal active
                        if not self.done:active-=1;self.done=True
                        return 0
                with patch.object(m.subprocess,'Popen',Process),patch.object(m,'_gpu_free',return_value=[2,3]),patch.object(m.time,'sleep'):m.driver(path)
                self.assertEqual(len(launched),3);self.assertEqual(max_active,2);self.assertNotEqual(launched[0][1],launched[1][1])
                state=json.loads((root/'pipeline_state.json').read_text())
                self.assertEqual(state['status'],'FAILED_PRESERVED' if failed_control else 'ANALYZED_SOURCE_ONLY')
                if not failed_control:self.assertFalse(state['source_gate_passed']);self.assertFalse(state['eligible_primary']);self.assertEqual(state['next_step'],'NO_PROMOTION')
                if not failed_control:self.assertEqual(state['primary'],'compact_oe_structure::predicted_identity_mahalanobis')


if __name__=='__main__':unittest.main()
