"""Owned two-row relation experiment: source/target predictions before truth."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class RelationOELaunchTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_relation_oe_launch')

    def runtime(self,m,root):
        cache=root.parent/m.CACHE_RUN/'cache';teacher=root.parent/m.TEACHER_RUN/'teacher';rows={}
        for arm in m.ARMS:
            t=dict(run_id=m.RUN,code_commit='a'*40,arm=arm,native_config={},L_root=str(cache/'L'),OE_root=str(cache/'OE'),
                teacher_root=str(teacher),output_root=str(root/'rows'/arm/'train'),steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
            e=dict(run_id=m.RUN,code_commit='a'*40,arm=arm,train_config=t,checkpoint=str(root/'rows'/arm/'train/final.pt'),cache_root=str(cache),
                output_root=str(root/'rows'/arm/'predict'),batch_size=128,Q_public_root='/Q-public',Q_capsule_id='capsule',Q_split_id='split',
                dura_checkpoint='/dura/final.pt',dura_release='/dura-release')
            rows[arm]=dict(train_config=t,eval_config=e)
        return dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root='/release',python='/python',launch_owner='root-owner',
            allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,max_concurrent_rows=2,rows=rows,
            novel_truth='/private/novel_truth.npz',target_truth='/private/target_truth.npz')

    def validators(self):
        tf={'run_id','code_commit','arm','native_config','L_root','OE_root','teacher_root','output_root','steps','lr','L_seed','OE_seed'}
        ef={'run_id','code_commit','arm','train_config','checkpoint','cache_root','output_root','batch_size','Q_public_root','Q_capsule_id','Q_split_id','dura_checkpoint','dura_release'}
        def validate(fields):
            def check(config):
                if set(config)!=fields:raise ValueError('Exact public schema required')
                return config
            return check
        return patch.dict(sys.modules,{
            'experiments.pamf_phase3_open_world.star_relation_oe_train':types.SimpleNamespace(validate_config=validate(tf)),
            'experiments.pamf_phase3_open_world.star_relation_oe_eval':types.SimpleNamespace(validate_config=validate(ef))})

    def root(self,root):
        root.mkdir();(root/'logs').mkdir();(root/'rows').mkdir();(root/'configs').mkdir()
        for arm in ('relation_control','relation_oe'):(root/'rows'/arm).mkdir()

    def invoke(self,m,root,job,arm='relation_oe'):
        self.root(root);r=self.runtime(m,root);runtime=root.parent/'runtime.json';runtime.write_text(json.dumps(r),encoding='utf-8')
        with self.validators(),patch.object(m,'_gpu_free',return_value=[2,4]),patch.object(m,'_job',side_effect=job):
            m.row_driver(runtime,arm,2)
        return json.loads((root/'rows'/arm/'state.json').read_text()),r

    def test_schema_truth_and_immutable_parent_role_paths(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            r=self.runtime(m,Path(td)/m.RUN)
            with self.validators():self.assertIs(m.validate(r),r)
            for kind in ('teacher_export','runtime_truth','eval_truth','train_truth','teacher_path','cache_path','checkpoint','nested_train','code','output'):
                bad=deepcopy(r)
                if kind=='teacher_export':bad['teacher_config']={}
                elif kind=='runtime_truth':bad['target_truth']=''
                elif kind=='eval_truth':bad['rows']['relation_oe']['eval_config']['target_truth']='forbidden'
                elif kind=='train_truth':bad['rows']['relation_oe']['train_config']['novel_truth']='forbidden'
                elif kind=='teacher_path':bad['rows']['relation_oe']['train_config']['teacher_root']='/newteacher'
                elif kind=='cache_path':bad['rows']['relation_oe']['eval_config']['cache_root']='/newcache'
                elif kind=='checkpoint':bad['rows']['relation_oe']['eval_config']['checkpoint']='/other/final.pt'
                elif kind=='nested_train':bad['rows']['relation_oe']['eval_config']['train_config']={}
                elif kind=='code':bad['rows']['relation_oe']['eval_config']['code_commit']='b'*40
                else:bad['rows']['relation_oe']['eval_config']['output_root']='/other/owned'
                with self.subTest(kind=kind),self.validators(),self.assertRaises(ValueError):m.validate(bad)

    def test_row_always_evaluates_target_then_scores_even_when_source_gate_false(self):
        m=self.module();calls=[]
        def job(r,state,path,name,command,gpu):
            calls.append((name,command,gpu))
            if name.endswith('_eval'):
                p=Path(r['run_root'])/'rows/relation_oe/predict';p.mkdir()
                (p/'complete.json').write_text(json.dumps(dict(source_gate_passed=False,target_predictions_complete=True)))
            if name.endswith('_score'):
                p=Path(r['run_root'])/'rows/relation_oe/score';p.mkdir()
                (p/'complete.json').write_text(json.dumps(dict(status=m.SCORE_STATUS,run_id=m.RUN,code_commit=r['code_commit'],arm='relation_oe',eligible_primary=False,next_step='NO_PROMOTION')))
            return True
        with tempfile.TemporaryDirectory() as td:state,r=self.invoke(m,Path(td)/'run',job)
        self.assertEqual([x[0] for x in calls],['relation_oe_train','relation_oe_eval','relation_oe_score'])
        self.assertEqual(state['status'],'ANALYZED_DEVELOPMENT');self.assertFalse(state['eligible_primary'])
        self.assertEqual(calls[0][1][-4:],['--config',str(Path(r['run_root'])/'configs/relation_oe_train_config.json'),'--device','cuda:0'])
        self.assertIn('train',calls[0][1]);self.assertIn('run',calls[1][1])
        for name,command,gpu in calls[:2]:
            self.assertNotIn(r['novel_truth'],command);self.assertNotIn(r['target_truth'],command);self.assertEqual(gpu,2)
        self.assertIn(r['novel_truth'],calls[2][1]);self.assertIn(r['target_truth'],calls[2][1]);self.assertIsNone(calls[2][2])

    def test_each_technical_stage_failure_preserves_partial_and_stops_only_that_row(self):
        m=self.module()
        for failed in ('train','eval','score'):
            calls=[]
            def job(r,state,path,name,command,gpu):calls.append(name);return not name.endswith('_'+failed)
            with self.subTest(stage=failed),tempfile.TemporaryDirectory() as td:state,_=self.invoke(m,Path(td)/'run',job)
            expected=['relation_oe_train','relation_oe_eval','relation_oe_score'][:('train','eval','score').index(failed)+1]
            self.assertEqual(calls,expected);self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_wrong_score_terminal_identity_is_not_analyzed(self):
        m=self.module()
        def job(r,state,path,name,command,gpu):
            if name.endswith('_score'):
                p=Path(r['run_root'])/'rows/relation_oe/score';p.mkdir()
                (p/'complete.json').write_text(json.dumps(dict(status=m.SCORE_STATUS,run_id=m.RUN,code_commit='b'*40,arm='relation_oe',eligible_primary=False,next_step='NO_PROMOTION')))
            return True
        with tempfile.TemporaryDirectory() as td:state,_=self.invoke(m,Path(td)/'run',job)
        self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_driver_uses_two_distinct_cards_no_teacher_export_and_keeps_healthy_row(self):
        m=self.module();launched=[]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';self.root(root);r=self.runtime(m,root);runtime=root.parent/'runtime.json';runtime.write_text(json.dumps(r))
            def spawn(command,**kwargs):
                arm=command[command.index('--arm')+1];gpu=int(command[command.index('--gpu')+1]);launched.append((arm,gpu))
                status='FAILED_PRESERVED' if arm=='relation_control' else 'ANALYZED_DEVELOPMENT'
                (root/'rows'/arm/'state.json').write_text(json.dumps(dict(status=status)))
                return types.SimpleNamespace(pid=100+len(launched),poll=lambda:0)
            with self.validators(),patch.object(m,'_gpu_free',return_value=[2,4]),patch.object(m.subprocess,'Popen',side_effect=spawn), \
                 patch.object(m,'_job') as job,patch.object(m.time,'sleep'):
                m.driver(runtime)
            state=json.loads((root/'pipeline_state.json').read_text())
            self.assertEqual(launched,[('relation_control',2),('relation_oe',4)]);job.assert_not_called()
            self.assertEqual(state['status'],'FAILED_PRESERVED');self.assertEqual(state['jobs']['relation_oe']['status'],'COMPLETE')
            self.assertEqual(state['jobs']['relation_control']['status'],'FAILED')

    def test_driver_fixed_main_reads_only_eligible_next_step_without_source_gate(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';self.root(root);r=self.runtime(m,root);runtime=root.parent/'runtime.json';runtime.write_text(json.dumps(r))
            def spawn(command,**kwargs):
                arm=command[command.index('--arm')+1];(root/'rows'/arm/'state.json').write_text(json.dumps(dict(status='ANALYZED_DEVELOPMENT')))
                p=root/'rows'/arm/'score';p.mkdir();(p/'complete.json').write_text(json.dumps(dict(status=m.SCORE_STATUS,run_id=m.RUN,
                    code_commit=r['code_commit'],arm=arm,eligible_primary=arm=='relation_control',next_step='NO_PROMOTION')))
                return types.SimpleNamespace(pid=111,poll=lambda:0)
            with self.validators(),patch.object(m,'_gpu_free',return_value=[2,4]),patch.object(m.subprocess,'Popen',side_effect=spawn):m.driver(runtime)
            state=json.loads((root/'pipeline_state.json').read_text());self.assertEqual(state['status'],'ANALYZED_DEVELOPMENT')
            self.assertEqual(state['primary'],'relation_oe::entropy_global');self.assertFalse(state['eligible_primary']);self.assertFalse(state['final_phase3_complete'])

    def test_existing_owned_row_state_is_not_overwritten_or_restarted(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';self.root(root);r=self.runtime(m,root);runtime=root.parent/'runtime.json';runtime.write_text(json.dumps(r))
            state=root/'rows/relation_oe/state.json';state.write_text('{"existing_healthy_pid":123}')
            with self.validators(),patch.object(m,'_job') as job,self.assertRaises(FileExistsError):m.row_driver(runtime,'relation_oe',2)
            job.assert_not_called();self.assertEqual(state.read_text(),'{"existing_healthy_pid":123}')

    def test_start_never_replaces_existing_root_or_exports_a_new_teacher(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run';r=self.runtime(m,root);runtime=Path(td)/'runtime.json';runtime.write_text(json.dumps(r))
            with self.validators(),patch.object(m.subprocess,'Popen',return_value=types.SimpleNamespace(pid=321)) as spawn:m.start(runtime)
            self.assertFalse((root/'configs/teacher.json').exists());self.assertEqual(len(list((root/'configs').glob('*.json'))),4)
            for p in (root/'configs').glob('*.json'):
                content=p.read_text();self.assertNotIn(r['novel_truth'],content);self.assertNotIn(r['target_truth'],content)
            self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            with self.validators(),patch.object(m.subprocess,'Popen') as spawn,self.assertRaises(FileExistsError):m.start(runtime)
            spawn.assert_not_called()


if __name__=='__main__':unittest.main()
