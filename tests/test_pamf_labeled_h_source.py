import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_channel_pl import labeled_h_source as s
from tests.test_pamf_clean_h_source import config as original_config


def context(project='/project'):
    return dict(project=project,python='python',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},template=project+'/runs/'+s.base.RUN+'/configs/'+s.BASE_METHOD+'.json',reference=project+'/runs/'+s.base.RUN+'/evaluations/'+s.BASE_METHOD+'/complete.json',expected_source_contract='roles')


class LabeledHSourceTests(unittest.TestCase):
    def setUp(self):self.old=original_config(s.BASE_METHOD)

    def test_only_four_F_roots_and_known_completed_source_reference(self):
        c=context();s.validate_context(c)
        for k in ('G','target_inputs','truth'):
            with self.assertRaises(ValueError):s.validate_context(dict(c,**{k:'forbidden'}))
        c['frozen_sources']['G']='forbidden'
        with self.assertRaises(ValueError):s.validate_context(c)

    def test_exact_single_seed_recipe_and_own_E20_10_20_no_control_training(self):
        with patch.object(s,'read',return_value=self.old):
            c=s.template(context(),'commit');s.validate_config(c)
            self.assertEqual(c['shared_SIG1_E20'],self.old['shared_SIG1_E20']);self.assertEqual(c['clean_H'],self.old['clean_H']);self.assertEqual(c['model_seed'],2026092701)
            for k,v in [('epochs',40),('model_seed',2026092702),('shared_SIG1_E20','finished_E50'),('plain_PL_threshold',.8),('source_G',{}),('target_inputs',{})]:
                bad=deepcopy(c);bad[k]=v
                with self.assertRaises(ValueError):s.validate_config(bad)
            bad=deepcopy(c);bad['hard_pl_gradient_guard']['U_to_L_norm_cap']=2
            with self.assertRaises(ValueError):s.validate_config(bad)
            for offset,n in ((20,10),(30,20)):
                part=dict(c,segment_epochs=n,epoch_offset=offset,step_offset=offset*222,parent_source=c['shared_SIG1_E20'] if offset==20 else c['output_root']+'/segment30',output_root=c['output_root']+'/segment'+str(offset+n));s.validate_config(part)
                part['parent_source']='foreign'
                with self.assertRaises(ValueError):s.validate_config(part)

    def test_frozen_scientific_thresholds_require_both_and_actual_reference(self):
        ref=dict(raw_correct=26554,H3_correct=62190)
        for raw,h,expected in [(26568,62109,True),(26567,62190,False),(26574,62108,False)]:self.assertEqual(s.source_decision(dict(raw_correct=raw,H3_correct=h),ref)['source_pass'],expected)
        with self.assertRaises(ValueError):s.source_decision(ref,dict(raw_correct=26555,H3_correct=62190))

    def test_only_smoke_and_one_source_train_eval_in_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=tmp.replace('\\','/');root=Path(tmp)/'runs'/s.RUN;root.mkdir(parents=True);release=project+'/releases/'+s.RELEASE
            old=json.loads(json.dumps(self.old).replace('/project',project));ctx=context(project)
            rt=dict(run_id=s.RUN,project=project,python='python',release_root=release,run_root=project+'/runs/'+s.RUN,context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='commit',frozen_sources=ctx['frozen_sources'],allowed_gpus=list(range(8)),launch_owner='codex-pamf-L-anchor-H-source-20261027')
            def read(p):
                if str(p)==rt['context']:return ctx
                if str(p)==ctx['reference']:return 'reference'
                return old
            with patch.object(s,'read',read),patch.object(s.base,'source_score',return_value=dict(raw_correct=26554,H3_correct=62190)):jobs=s.prepare(rt)
            self.assertEqual([j['kind'] for j in jobs],['smoke','train_eval']);self.assertEqual(jobs[1]['dependencies'],[str(root/'smoke.json')]);self.assertTrue(all(j['entrypoint']=='experiments.pamf_channel_pl.labeled_h_source' for j in jobs))
            cfg=json.loads(Path(jobs[1]['config']).read_text());self.assertNotIn('tests',cfg);self.assertNotIn('source_G',cfg);self.assertEqual(cfg['labeled_H_anchors'],s.RECIPE)

    def test_worker_uses_new_factory_and_then_only_frozen_source_evaluation(self):
        c=dict(frozen_sources={},source_context='source_context',output_root='source',project='/project')
        with patch.object(s.base,'activate'),patch.object(s.base,'train') as train,patch.object(s.base,'evaluate') as evaluate:s.worker(c,'train_eval')
        train.assert_called_once_with(c,s.validate_config,s.LabeledHAnchor);args=evaluate.call_args.args[0];self.assertEqual(args['context'],'source_context');self.assertNotIn('tests',args);self.assertNotIn('truth',args)
        with patch.object(s.base,'activate'),patch.object(s.base,'train',side_effect=ValueError('source failed')),patch.object(s.base,'evaluate') as evaluate:
            with self.assertRaises(ValueError):s.worker(c,'train_eval')
        evaluate.assert_not_called()

    def test_controller_queues_without_launching_GPU_workers_when_global_slots_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=tmp.replace('\\','/');Path(tmp,'runs').mkdir();release=project+'/releases/'+s.RELEASE
            rt=dict(run_id=s.RUN,project=project,python='python',release_root=release,run_root=project+'/runs/'+s.RUN,context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='commit',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},allowed_gpus=list(range(8)),launch_owner='codex-pamf-L-anchor-H-source-20261027')
            jobs=[dict(id='smoke',kind='smoke',dependencies=[],completion='not_complete',successful_statuses=['PASS'],gpu_pool=list(range(8)))]
            class FirstQueueSnapshot(Exception):pass
            with patch.object(s,'prepare',return_value=jobs),patch('experiments.pamf_repl_completion.runner.capacity',return_value=({101,102},20000)),patch('experiments.pamf_repl_completion.runner.processes',return_value={}),patch('experiments.pamf_peer_supervised.runner.launch') as launch,patch.object(s.time,'sleep',side_effect=FirstQueueSnapshot):
                with self.assertRaises(FirstQueueSnapshot):s.dispatch(rt)
            launch.assert_not_called();state=json.loads((Path(rt['run_root'])/'pipeline_state.json').read_text())
            self.assertEqual(state['jobs']['smoke']['status'],'QUEUED');self.assertEqual(state['counts'],{'QUEUED':1});self.assertFalse(state['systemic_stop_new_launches'])


if __name__=='__main__':unittest.main()
