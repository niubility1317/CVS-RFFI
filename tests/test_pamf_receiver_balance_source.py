import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.pamf_channel_pl import receiver_balance_source as s
from experiments.pamf_channel_pl.receiver_balance_objective import labeled_prior,recipe
from tests.test_pamf_clean_h_source import config as original_config
from tests.test_pamf_no_pair_source import context as previous_context


def context(project='/project'):
    return dict(project=project,python='python',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},
        template=project+'/runs/'+s.reference.RUN+'/configs/'+s.BASE_METHOD+'.json',
        reference=project+'/runs/'+s.reference.RUN+'/evaluations/'+s.BASE_METHOD+'/complete.json',expected_source_contract='roles')


def runtime(project='/project'):
    release=project+'/releases/'+s.RELEASE
    return dict(run_id=s.RUN,project=project,python='python',release_root=release,run_root=project+'/runs/'+s.RUN,
        context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='commit',
        frozen_sources=context(project)['frozen_sources'],allowed_gpus=list(range(8)),launch_owner=s.OWNER)


class ReceiverBalanceSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setup_configs(self,project='/project'):
        self.old=json.loads(json.dumps(original_config(s.reference.BASE_METHOD)).replace('/project',project))
        with patch.object(s.reference,'read',return_value=self.old):self.previous=s.reference.template(previous_context(project),'previous')

    def test_exact_two_modes_preserve_original_parent_budget_gate_head_and_H(self):
        self.setup_configs()
        with patch.object(s.reference,'read',return_value=self.old),patch.object(s,'read',return_value=self.previous):
            for method in s.METHODS:
                c=s.template(context(),'commit',method);s.validate_config(c)
                self.assertEqual(c['receiver_balance'],recipe(method));self.assertEqual(c['clean_H'],self.previous['clean_H'])
                for key in ('shared_SIG1_E20','hard_pl_joint_contrast','hard_pl_gradient_guard','plain_PL_threshold','epochs','model_seed'):
                    self.assertEqual(c[key],self.previous[key])
                for change in (dict(target_inputs={}),dict(source_G={}),dict(model_seed=2026092702),dict(epochs=40),dict(shared_SIG1_E20='failed_E50'),dict(plain_PL_threshold=.8)):
                    with self.assertRaises(ValueError):s.validate_config(dict(c,**change))
                for offset,n in ((20,10),(30,20)):
                    part=dict(c,segment_epochs=n,epoch_offset=offset,step_offset=offset*222,
                        parent_source=c['shared_SIG1_E20'] if offset==20 else c['output_root']+'/segment30',output_root=c['output_root']+'/segment'+str(offset+n))
                    s.validate_config(part);part['parent_source']='foreign'
                    with self.assertRaises(ValueError):s.validate_config(part)

    def test_forbidden_G_target_or_foreign_context_and_runtime(self):
        c=context();s.validate_context(c)
        for key in ('G','target_inputs','truth'):
            with self.assertRaises(ValueError):s.validate_context(dict(c,**{key:'forbidden'}))
        c['frozen_sources']['G']='forbidden'
        with self.assertRaises(ValueError):s.validate_context(c)
        rt=runtime();s.validate_runtime(rt)
        for change in (dict(run_root='/foreign'),dict(allowed_gpus=[0]),dict(target_inputs={})):
            with self.assertRaises(ValueError):s.validate_runtime(dict(rt,**change))

    def test_gate_uses_completed_NoPair_counts_and_both_fixed_thresholds(self):
        reference=dict(raw_correct=26552,H3_correct=62148)
        for method in s.METHODS:
            for raw,h,passed in ((26568,62109,True),(26567,62148,False),(26574,62108,False)):
                self.assertEqual(s.source_decision(method,dict(raw_correct=raw,H3_correct=h),reference)['source_pass'],passed)
        with self.assertRaises(ValueError):s.source_decision(s.METHODS[0],reference,dict(raw_correct=26554,H3_correct=62190))

    def test_L_prior_uses_only_true_L6300_metadata_not_V_or_U(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids=np.arange(6300);y=np.tile(np.arange(6),1050);rx=np.repeat(np.arange(1,6),1260)
            np.savez(Path(tmp)/'L_s_metadata.npz',ids=ids,y=y,rx_i=rx)
            # A deliberately contradictory V decoy must not initialize frequency.
            np.savez(Path(tmp)/'V_metadata.npz',ids=ids,y=np.zeros(6300),rx_i=np.ones(6300))
            c=dict(source_cache=tmp,source_rxs=list(range(1,6)));contract=dict(role_ids=dict(L_s=ids.tolist()))
            self.assertEqual(labeled_prior(c,contract,'RX').tolist(),[1260]*5)
            self.assertEqual(labeled_prior(c,contract,'RX_TX').tolist(),[210]*30)
            contract['role_ids']['L_s'][0]=-1
            with self.assertRaises(ValueError):labeled_prior(c,contract,'RX')

    def test_three_jobs_no_control_training_or_target_and_source_evaluation_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=tmp.replace('\\','/');self.setup_configs(project);rt=runtime(project);Path(rt['run_root']).mkdir(parents=True);ctx=context(project)
            done=dict(method=s.BASE_METHOD,signature_views=1,source_output='old_source')
            complete=dict(target_access=False,U_truth_read=False,head_fit_updates=0,source_contract='roles')
            def reader(path):
                path=str(path).replace('\\','/')
                if path==rt['context']:return ctx
                if path==ctx['reference']:return done
                if path=='old_source/complete.json':return complete
                if path=='roles':return dict(roles='same')
                return self.previous
            with patch.object(s.reference,'read',return_value=self.old),patch.object(s,'read',reader),patch.object(s.base,'source_score',return_value=dict(raw_correct=26552,H3_correct=62148)):
                jobs=s.prepare(rt)
            self.assertEqual([j['kind'] for j in jobs],['smoke','train_eval','train_eval'])
            self.assertEqual([j['id'] for j in jobs[1:]],list(s.METHODS))
            self.assertTrue(all(j['entrypoint']=='experiments.pamf_channel_pl.receiver_balance_source' for j in jobs))
            for job in jobs[1:]:
                self.assertEqual(job['dependencies'],[str(Path(rt['run_root'])/'smoke.json')])
                config=json.loads(Path(job['config']).read_text());self.assertNotIn('tests',config);self.assertNotIn('source_G',config)
        c=dict(paper_method=s.METHODS[0],frozen_sources={},source_context='source_context',output_root='source',project='/project')
        with patch.object(s.base,'activate'),patch.object(s.base,'train') as train,patch.object(s.base,'evaluate') as evaluate:s.worker(c,'train_eval')
        train.assert_called_once_with(c,s.validate_config,s.ReceiverBalancedObjective)
        self.assertEqual(evaluate.call_args.args[0]['context'],'source_context');self.assertEqual(evaluate.call_args.args[0]['output_root'],'/project/runs/'+s.RUN+'/evaluations/'+s.METHODS[0])
        with patch.object(s.base,'activate'),patch.object(s.base,'train',side_effect=ValueError('training failed')),patch.object(s.base,'evaluate') as evaluate:
            with self.assertRaises(ValueError):s.worker(c,'train_eval')
        evaluate.assert_not_called()

    def test_global_capacity_keeps_controller_queued_without_GPU_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=tmp.replace('\\','/');Path(tmp,'runs').mkdir();rt=runtime(project)
            jobs=[dict(id='smoke',kind='smoke',dependencies=[],completion='absent',successful_statuses=['PASS'],gpu_pool=list(range(8)))]
            class Snapshot(Exception):pass
            with patch.object(s,'prepare',return_value=jobs),patch('experiments.pamf_repl_completion.runner.capacity',return_value=({101,102},20000)),patch('experiments.pamf_repl_completion.runner.processes',return_value={}),patch('experiments.pamf_peer_supervised.runner.launch') as launch,patch.object(s.time,'sleep',side_effect=Snapshot):
                with self.assertRaises(Snapshot):s.dispatch(rt)
            launch.assert_not_called();state=json.loads((Path(rt['run_root'])/'pipeline_state.json').read_text())
            self.assertEqual(state['counts'],{'QUEUED':1});self.assertFalse(state['systemic_stop_new_launches'])


if __name__=='__main__':unittest.main()
