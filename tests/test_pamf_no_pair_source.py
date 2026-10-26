import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
from experiments.pamf_channel_pl import no_pair_source as s
from experiments.pamf_channel_pl.clean_h_objective import CleanHObjective
from experiments.pamf_channel_pl.sig1_repl_offset_objective import NoStyleNoRXE
from tests.test_pamf_clean_h_source import config as original_config


def context(project='/project'):
    return dict(project=project,python='python',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},
        template=project+'/runs/'+s.base.RUN+'/configs/'+s.BASE_METHOD+'.json',
        reference=project+'/runs/'+s.base.RUN+'/evaluations/'+s.BASE_METHOD+'/complete.json',expected_source_contract='roles')


def runtime(project='/project'):
    release=project+'/releases/'+s.RELEASE
    return dict(run_id=s.RUN,project=project,python='python',release_root=release,run_root=project+'/runs/'+s.RUN,
        context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit='commit',
        frozen_sources=context(project)['frozen_sources'],allowed_gpus=list(range(8)),launch_owner=s.OWNER)


class NoPairSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_loss_omits_peer_computation_and_preserves_original_U_gradient(self):
        obj=s.NoPairObjective.__new__(s.NoPairObjective);obj.pair_enabled=False;obj.pair_capture=None
        obj.pair_calls=obj.pair_packets=obj.pair_rejected=0;obj.L_calls=1;obj.H_L_calls=obj.H_U_calls=2;obj.H_L_packets=128;obj.H_U_packets=256
        obj.settings=dict(weight=6.);obj.metrics={};obj.c={};p=torch.tensor(2.,requires_grad=True)
        a=torch.randn(4,6,requires_grad=True);b=torch.randn(4,6,requires_grad=True);mask=torch.ones(4,dtype=torch.bool)
        def parent(this,*args):this.pending_gradient_loss=p.square();this.pair_capture.extend([(a,mask),(b,mask)]);return p*0.
        with patch.object(NoStyleNoRXE,'loss',parent),patch('experiments.pamf_channel_pl.clean_h_objective.physical_peer_loss',side_effect=AssertionError('disabled peer loss executed')):
            obj.loss(object(),'cpu')
        gp,ga,gb=torch.autograd.grad(obj.pending_gradient_loss,(p,a,b))
        self.assertEqual(float(gp),4.);self.assertEqual(float(ga.abs().sum()+gb.abs().sum()),0.)
        self.assertEqual(obj.metrics['HV_pair_enabled'],0);self.assertEqual(obj.metrics['HV_pair_weight'],0.);self.assertEqual(float(obj.metrics['HV_pair_loss']),0.)
        self.assertEqual(obj.pair_rejected,4);self.assertIsNone(obj.pair_capture)

    def test_objective_dual_L_is_preserved_and_only_pair_is_disabled(self):
        def init(this,c):this.c=c
        c=dict(paper_method=s.METHOD,hard_pl_no_uas=False,clean_H=s.RECIPE,source_contract='roles')
        with patch.object(NoStyleNoRXE,'__init__',init),patch('experiments.pamf_channel_pl.clean_h_objective.read',return_value={}):
            obj=s.NoPairObjective(c)
        self.assertTrue(obj.dual_L);self.assertFalse(obj.pair_enabled);self.assertIs(obj.c,c)
        with self.assertRaises(ValueError):s.NoPairObjective(dict(c,clean_H=dict(s.RECIPE,pair=True)))

    def test_exact_source_intervention_and_own_budget(self):
        old=original_config(s.BASE_METHOD)
        with patch.object(s,'read',return_value=old):
            c=s.template(context(),'commit');s.validate_config(c)
            self.assertEqual(c['clean_H'],s.RECIPE);self.assertEqual(c['shared_SIG1_E20'],old['shared_SIG1_E20'])
            self.assertEqual(c['hard_pl_joint_contrast'],old['hard_pl_joint_contrast']);self.assertEqual(c['hard_pl_gradient_guard'],old['hard_pl_gradient_guard'])
            for change in (dict(target_inputs={}),dict(source_G={}),dict(model_seed=2026092702),dict(epochs=40),dict(labeled_H_anchors={}),dict(clean_H=dict(s.RECIPE,pair=True))):
                with self.assertRaises(ValueError):s.validate_config(dict(c,**change))
            for offset,n in ((20,10),(30,20)):
                part=dict(c,segment_epochs=n,epoch_offset=offset,step_offset=offset*222,
                    parent_source=c['shared_SIG1_E20'] if offset==20 else c['output_root']+'/segment30',output_root=c['output_root']+'/segment'+str(offset+n))
                s.validate_config(part);part['parent_source']='foreign'
                with self.assertRaises(ValueError):s.validate_config(part)

    def test_G_target_and_foreign_reference_or_runtime_rejected(self):
        c=context();s.validate_context(c)
        for key in ('G','target_inputs','truth'):
            with self.assertRaises(ValueError):s.validate_context(dict(c,**{key:'forbidden'}))
        bad=deepcopy(c);bad['frozen_sources']['G']='forbidden'
        with self.assertRaises(ValueError):s.validate_context(bad)
        with self.assertRaises(ValueError):s.validate_context(dict(c,reference='/foreign/complete.json'))
        rt=runtime();s.validate_runtime(rt)
        for change in (dict(run_root='/foreign'),dict(allowed_gpus=[0]),dict(target_inputs={})):
            with self.assertRaises(ValueError):s.validate_runtime(dict(rt,**change))

    def test_source_gate_both_thresholds_and_fixed_reference(self):
        reference=dict(raw_correct=26554,H3_correct=62190)
        for raw,h,expected in ((26568,62109,True),(26567,62190,False),(26574,62108,False)):
            self.assertEqual(s.source_decision(dict(raw_correct=raw,H3_correct=h),reference)['source_pass'],expected)
        with self.assertRaises(ValueError):s.source_decision(reference,dict(raw_correct=26555,H3_correct=62190))

    def test_no_control_training_or_target_job_and_source_evaluation_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=tmp.replace('\\','/');rt=runtime(project);Path(rt['run_root']).mkdir(parents=True)
            old=json.loads(json.dumps(original_config(s.BASE_METHOD)).replace('/project',project));ctx=context(project)
            def reader(p):
                if str(p)==rt['context']:return ctx
                if str(p)==ctx['reference']:return 'reference'
                return old
            with patch.object(s,'read',reader),patch.object(s.base,'source_score',return_value=dict(raw_correct=26554,H3_correct=62190)):jobs=s.prepare(rt)
            self.assertEqual([j['kind'] for j in jobs],['smoke','train_eval']);self.assertEqual(jobs[1]['dependencies'],[str(Path(rt['run_root'])/'smoke.json')])
            self.assertTrue(all(j['entrypoint']=='experiments.pamf_channel_pl.no_pair_source' for j in jobs))
            c=json.loads(Path(jobs[1]['config']).read_text());self.assertNotIn('tests',c);self.assertNotIn('labeled_H_anchors',c)
        c=dict(frozen_sources={},source_context='source_context',output_root='source',project='/project')
        with patch.object(s.base,'activate'),patch.object(s.base,'train') as train,patch.object(s.base,'evaluate') as evaluate:s.worker(c,'train_eval')
        train.assert_called_once_with(c,s.validate_config,s.NoPairObjective);self.assertEqual(evaluate.call_args.args[0]['context'],'source_context')
        with patch.object(s.base,'activate'),patch.object(s.base,'train',side_effect=ValueError('training failed')),patch.object(s.base,'evaluate') as evaluate:
            with self.assertRaises(ValueError):s.worker(c,'train_eval')
        evaluate.assert_not_called()

    def test_no_GPU_capacity_leaves_CPU_controller_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=tmp.replace('\\','/');Path(tmp,'runs').mkdir();rt=runtime(project)
            jobs=[dict(id='smoke',kind='smoke',dependencies=[],completion='absent',successful_statuses=['PASS'],gpu_pool=list(range(8)))]
            class Snapshot(Exception):pass
            with patch.object(s,'prepare',return_value=jobs),patch('experiments.pamf_repl_completion.runner.capacity',return_value=({101,102},20000)),patch('experiments.pamf_repl_completion.runner.processes',return_value={}),patch('experiments.pamf_peer_supervised.runner.launch') as launch,patch.object(s.time,'sleep',side_effect=Snapshot):
                with self.assertRaises(Snapshot):s.dispatch(rt)
            launch.assert_not_called();state=json.loads((Path(rt['run_root'])/'pipeline_state.json').read_text())
            self.assertEqual(state['counts'],{'QUEUED':1});self.assertFalse(state['systemic_stop_new_launches'])


if __name__=='__main__':unittest.main()
