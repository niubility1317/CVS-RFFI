"""Three owned rows; GPU prediction precedes CPU truth-side scoring."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class OpenWorldLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_open_world_launch')

    def runtime(self, m, root, concurrency=3):
        cache = root.parent / m.CACHE_RUN / 'cache'
        rows = {}
        for arm in m.ARMS:
            t = dict(run_id=m.RUN, code_commit='a'*40, arm=arm, native_config={}, L_root=str(cache/'L'),
                OE_root=str(cache/'OE'), output_root=str(root/'rows'/arm/'train'), steps=2220, lr=.0001,
                sampling_seed=2026103017, slot_seed=2026103018)
            e = dict(run_id=m.RUN, code_commit='a'*40, arm=arm, train_config=t,
                checkpoint=str(root/'rows'/arm/'train/final.pt'), cache_root=str(cache),
                output_root=str(root/'rows'/arm/'predict'), batch_size=128, Q_public_root='/Q-public',
                Q_capsule_id='capsule', Q_split_id='split', dura_checkpoint='/dura/final.pt', dura_release='/dura-release')
            rows[arm] = dict(train_config=t, eval_config=e)
        return dict(run_id=m.RUN, code_commit='a'*40, run_root=str(root), release_root='/release', python='/python',
            launch_owner='root-owner', allowed_gpus=list(range(8)), minimum_free_mb=8192,
            max_added_workers_per_gpu=2, max_concurrent_rows=concurrency, rows=rows,
            novel_truth='/private/novel_truth.npz', target_truth='/private/target_truth.npz')

    def validators(self):
        tf = {'run_id','code_commit','arm','native_config','L_root','OE_root','output_root','steps','lr','sampling_seed','slot_seed'}
        ef = {'run_id','code_commit','arm','train_config','checkpoint','cache_root','output_root','batch_size',
            'Q_public_root','Q_capsule_id','Q_split_id','dura_checkpoint','dura_release'}
        def validator(fields):
            def check(c):
                if not isinstance(c, dict) or set(c) != fields:
                    raise ValueError('Exact role schema required')
                return c
            return check
        return patch.dict(sys.modules, {
            'experiments.pamf_phase3_open_world.star_open_world_train':types.SimpleNamespace(validate_config=validator(tf)),
            'experiments.pamf_phase3_open_world.star_open_world_eval':types.SimpleNamespace(validate_config=validator(ef))})

    def prepare(self, m, root, concurrency=3):
        root.mkdir(); (root/'logs').mkdir(); (root/'rows').mkdir(); (root/'configs').mkdir()
        for arm in m.ARMS:
            (root/'rows'/arm).mkdir()
        r = self.runtime(m, root, concurrency)
        path = root.parent/'runtime.json'; path.write_text(json.dumps(r), encoding='utf-8')
        return r, path

    def score_marker(self, m, r, arm, eligible=False):
        p = Path(r['run_root'])/'rows'/arm/'score'; p.mkdir()
        value = dict(status=m.SCORE_STATUS, run_id=m.RUN, code_commit=r['code_commit'], arm=arm,
            eligible_primary=eligible, next_step='ELIGIBLE_NEXT_UDA_FALSIFICATION' if eligible else 'NO_PROMOTION',
            truth_last=True, formal_target_accessed=False, final_phase3_complete=False)
        (p/'complete.json').write_text(json.dumps(value), encoding='utf-8')

    def stage_marker(self, m, r, arm, stage):
        p = Path(r['run_root'])/'rows'/arm/('train' if stage=='train' else 'predict'); p.mkdir()
        common = dict(run_id=m.RUN, code_commit=r['code_commit'], arm=arm)
        if stage == 'train':
            value = dict(common, status=m.TRAIN_STATUS, updates=2220, cumulative_updates=13320,
                target_reads=0, validation_reads=0, novel_validation_reads=0, original_U_s_reads=0,
                identity_head_replaced=False)
        else:
            value = dict(common, status=m.EVAL_STATUS, known_count=27000, novel_count=3150, target_count=864,
                target_Q_fit=False, source_truth_reads=0, target_truth_reads=0, formal_target_accessed=False,
                thresholds_frozen_before_novel_and_target=True, source_unknown_success_target_entry_gate=False)
        (p/'complete.json').write_text(json.dumps(value), encoding='utf-8')

    def test_strict_three_row_schema_and_role_permissions(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as td:
            r = self.runtime(m, Path(td)/'run')
            with self.validators():
                self.assertIs(m.validate(r), r)
                self.assertIs(m.validate(self.runtime(m, Path(td)/'run', 2))['max_concurrent_rows'], 2)
            for kind in ('teacher_export','train_truth','eval_truth','cache','checkpoint','nested','code','steps','seed','slots','memory','concurrency','missing_arm'):
                bad = deepcopy(r); row = bad['rows']['M']
                if kind == 'teacher_export': bad['teacher_config'] = {}
                elif kind == 'train_truth': row['train_config']['target_truth'] = '/forbidden'
                elif kind == 'eval_truth': row['eval_config']['novel_truth'] = '/forbidden'
                elif kind == 'cache': row['eval_config']['cache_root'] = '/othercache'
                elif kind == 'checkpoint': row['eval_config']['checkpoint'] = '/other/final.pt'
                elif kind == 'nested': row['eval_config']['train_config'] = {}
                elif kind == 'code': row['eval_config']['code_commit'] = 'b'*40
                elif kind == 'steps': row['train_config']['steps'] = 222
                elif kind == 'seed': row['train_config']['sampling_seed'] = 1
                elif kind == 'slots': bad['max_added_workers_per_gpu'] = 3
                elif kind == 'memory': bad['minimum_free_mb'] = 4096
                elif kind == 'concurrency': bad['max_concurrent_rows'] = 4
                else: del bad['rows']['S0']
                with self.subTest(kind=kind), self.validators(), self.assertRaises(ValueError):
                    m.validate(bad)

    def test_row_full_target_evaluation_and_cpu_truth_after_source_failure(self):
        m = self.module(); calls = []
        with tempfile.TemporaryDirectory() as td:
            r, runtime = self.prepare(m, Path(td)/'run')
            def job(rr, state, path, name, command, gpu):
                calls.append((name, command, gpu))
                if name.endswith('_train'): self.stage_marker(m, r, 'M', 'train')
                elif name.endswith('_eval'): self.stage_marker(m, r, 'M', 'eval')
                else: self.score_marker(m, r, 'M', False)
                return True
            with self.validators(), patch.object(m, '_gpu_free', return_value=[2]), patch.object(m, '_job', side_effect=job):
                m.row_driver(runtime, 'M', 2)
            state = m._read(Path(r['run_root'])/'rows/M/state.json')
        self.assertEqual([v[0] for v in calls], ['M_train','M_eval','M_score'])
        self.assertEqual(state['status'], 'ANALYZED_DEVELOPMENT'); self.assertFalse(state['eligible_primary'])
        self.assertEqual(calls[0][1][4:6], ['experiments.pamf_phase3_open_world.star_open_world_train','--config'])
        self.assertEqual(calls[1][1][4:6], ['experiments.pamf_phase3_open_world.star_open_world_eval','--config'])
        for _, command, gpu in calls[:2]:
            self.assertEqual(gpu, 2)
            self.assertNotIn(r['novel_truth'], command); self.assertNotIn(r['target_truth'], command)
        self.assertIn(r['novel_truth'], calls[-1][1]); self.assertIn(r['target_truth'], calls[-1][1])
        self.assertIsNone(calls[-1][2]); self.assertEqual(m._env(r)['CUDA_VISIBLE_DEVICES'], '')

    def test_technical_or_wrong_complete_stops_only_that_row(self):
        m = self.module()
        for failure in ('train','eval','score','wrong_train','wrong_eval','wrong_score'):
            calls = []
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                r, runtime = self.prepare(m, Path(td)/'run')
                def job(rr, state, path, name, command, gpu):
                    stage = name.split('_')[-1]; calls.append(stage)
                    if failure == stage: return False
                    if stage in ('train','eval'): self.stage_marker(m, r, 'M', stage)
                    else: self.score_marker(m, r, 'M')
                    if failure == 'wrong_'+stage:
                        p = Path(r['run_root'])/'rows/M'/dict(train='train',eval='predict',score='score')[stage]/'complete.json'
                        v = m._read(p); v['arm']='S0'; p.write_text(json.dumps(v))
                    return True
                with self.validators(), patch.object(m, '_gpu_free', return_value=[2]), patch.object(m, '_job', side_effect=job):
                    m.row_driver(runtime, 'M', 2)
                state = m._read(Path(r['run_root'])/'rows/M/state.json')
                self.assertEqual(state['status'], 'FAILED_PRESERVED')
                self.assertEqual(calls, ['train','eval','score'][:('train','eval','score').index(failure.removeprefix('wrong_'))+1])

    def driver_case(self, m, root, free, failing=None, eligible_arm=None, concurrency=3):
        r, runtime = self.prepare(m, root, concurrency); launched=[]; polls={}
        def spawn(command, **kwargs):
            arm = command[command.index('--arm')+1]; gpu = int(command[command.index('--gpu')+1])
            launched.append((arm,gpu)); polls[arm]=0
            (root/'rows'/arm/'state.json').write_text(json.dumps(dict(status='FAILED_PRESERVED' if arm==failing else 'ANALYZED_DEVELOPMENT')))
            self.score_marker(m,r,arm,eligible=arm==eligible_arm)
            def poll():
                polls[arm] += 1
                return None if polls[arm]==1 else 0
            return types.SimpleNamespace(pid=100+len(launched), poll=poll)
        with self.validators(), patch.object(m, '_gpu_free', side_effect=free), patch.object(m.subprocess, 'Popen', side_effect=spawn), \
             patch.object(m.time,'sleep'), patch.object(m.os,'kill') as kill:
            m.driver(runtime)
        kill.assert_not_called()
        return m._read(root/'pipeline_state.json'), launched

    def test_actual_free_slots_prefer_distinct_cards_and_never_stop_foreign(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as td:
            state, launched = self.driver_case(m, Path(td)/'run', lambda r:[2,4], eligible_arm='S0')
        self.assertEqual(launched, [('S0',2),('S1',4),('M',2)])
        self.assertEqual(state['status'],'ANALYZED_DEVELOPMENT')
        self.assertEqual(state['primary'],m.PRIMARY); self.assertFalse(state['eligible_primary'])
        self.assertFalse(state['final_phase3_complete']); self.assertFalse(state['formal_target_accessed'])
        with tempfile.TemporaryDirectory() as td:
            state, launched = self.driver_case(m, Path(td)/'run', lambda r:[2], failing='S0')
        self.assertTrue(all(g==2 for _,g in launched)); self.assertEqual(len(launched),3)
        self.assertEqual(state['jobs']['S1']['status'],'COMPLETE'); self.assertEqual(state['jobs']['M']['status'],'COMPLETE')
        self.assertEqual(state['status'],'PARTIAL_FAILED_PRESERVED')

    def test_memory_is_rechecked_waits_and_two_row_concurrency_is_supported(self):
        m = self.module(); probes=[]
        def free(r):
            probes.append(1)
            return [] if len(probes)==1 else [2]
        with tempfile.TemporaryDirectory() as td:
            state, launched = self.driver_case(m, Path(td)/'run', free, eligible_arm='M', concurrency=2)
        self.assertEqual(state['status'],'ANALYZED_DEVELOPMENT'); self.assertTrue(state['eligible_primary'])
        self.assertEqual(launched,[('S0',2),('S1',2),('M',2)])
        self.assertGreater(len(probes),len(launched))
        r=dict(allowed_gpus=list(range(8)),minimum_free_mb=8192)
        with patch.object(m.subprocess,'check_output',return_value='2, 9000\n4, 20000\n7, 8191\n') as query:
            self.assertEqual(m._gpu_free(r),[4,2])
        self.assertNotIn('query-compute-apps',' '.join(query.call_args.args[0]))

    def test_output_protection_and_six_configs_without_teacher_or_truth(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'run'; r=self.runtime(m,root); runtime=Path(td)/'runtime.json';runtime.write_text(json.dumps(r))
            with self.validators(), patch.object(m.subprocess,'Popen',return_value=types.SimpleNamespace(pid=321)) as spawn:
                m.start(runtime)
            self.assertEqual(len(list((root/'configs').glob('*.json'))),6)
            self.assertFalse((root/'configs/teacher.json').exists())
            for p in (root/'configs').glob('*.json'):
                self.assertNotIn(r['novel_truth'],p.read_text());self.assertNotIn(r['target_truth'],p.read_text())
            self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            with self.validators(),patch.object(m.subprocess,'Popen') as spawn,self.assertRaises(FileExistsError):m.start(runtime)
            spawn.assert_not_called()
            p=root/'rows/M/state.json';p.write_text('{"healthy_pid":123}')
            with self.validators(),patch.object(m,'_job') as job,self.assertRaises(FileExistsError):m.row_driver(runtime,'M',2)
            job.assert_not_called();self.assertEqual(p.read_text(),'{"healthy_pid":123}')


if __name__=='__main__':unittest.main()
