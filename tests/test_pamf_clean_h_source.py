import unittest,tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.pamf_channel_pl import clean_h_source as run
from experiments.pamf_channel_pl.clean_h_objective import CleanHObjective,physical_mask,view_plan,physical_peer_loss
from experiments.pamf_channel_pl.sig1_repl_offset_objective import NoStyleNoRXE
from tests.test_pamf_sig1_role_training import config as old_config


def context():
    project='/project'
    return dict(project=project,python='python',frozen_sources={k:k for k in ('F','AF','BASE','FOLD')},
        cases=[dict(model_seed=s,expected_source_contract='roles') for s in (2026092701,2026092702,2026092703)],
        recipe_source=project+'/runs/'+run.OLD_RUN+'/F/F0-REPL_NoStyleNoRXE-s2026092701/source/complete.json',
        references={m:project+'/runs/'+run.OLD_RUN+'/evaluations/'+m+'/complete.json' for m in run.REFERENCES})


def config(method):
    recipe=old_config('REPL_G_U');recipe.pop('restored_roles');recipe['source_G']={};recipe['G_inference']='legacy'
    with patch.object(run,'read',return_value=dict(status='SOURCE_FROZEN',epoch=50,target_access=False,U_truth_read=False,config=recipe)):
        return run.config(context(),'commit',method)


class CleanHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_detached_same_packet_loss_exact_gradients(self):
        a=torch.randn(8,6,dtype=torch.double,requires_grad=True);b=torch.randn(8,6,dtype=torch.double,requires_grad=True)
        mask=torch.tensor([1,0,0,0,1,0,0,0],dtype=torch.bool);ga,gb=torch.autograd.grad(physical_peer_loss(a,b,mask),(a,b))
        expected=.5*(a.softmax(1)-b.softmax(1))*mask[:,None]/8
        self.assertTrue(torch.allclose(ga,expected,atol=1e-14));self.assertTrue(torch.allclose(gb,-expected,atol=1e-14))
        self.assertTrue(bool((ga[~mask]==0).all()))

    def test_no_rejected_and_shared_wrong_predictions_do_not_create_hard_labels(self):
        for same,accepted in ((False,True),(True,False)):
            a=torch.tensor([[0.,9.,0,0,0,0]],requires_grad=True);b=(a.detach().clone() if same else torch.randn_like(a)).requires_grad_()
            ga,gb=torch.autograd.grad(physical_peer_loss(a,b,torch.tensor([not accepted])),(a,b))
            self.assertLess(float(ga.abs().max()+gb.abs().max()),2e-7)
        with self.assertRaises(ValueError):physical_peer_loss(torch.ones(2,6),torch.ones(2,5),torch.ones(2,dtype=torch.bool))

    def test_views_change_scene_and_epoch_with_exact_partial_batches(self):
        for step in (4441,4442,4443,6661):self.assertNotEqual(view_plan(step,'U','a')[0],view_plan(step,'U','b')[0])
        self.assertNotEqual(view_plan(4441,'L','a')[1],view_plan(4663,'L','a')[1])
        for n in (128,256,124):self.assertEqual(int(physical_mask(n,4441,'cpu').sum()),n//2)
        with self.assertRaises(ValueError):view_plan(4440,'U','a')
        with self.assertRaises(ValueError):view_plan(4441,'V','a')

    def test_U_TX_visibility_rejected(self):
        obj=CleanHObjective.__new__(CleanHObjective);x=torch.randn(4,2,256)
        obj.validate_unlabeled_batch((x,torch.full((4,),-1),None,dict(tx_label_visible=torch.zeros(4,dtype=torch.bool))))
        self.assertIs(obj.current_U,x)
        with self.assertRaises(ValueError):obj.validate_unlabeled_batch((x,torch.zeros(4,dtype=torch.long),None,dict(tx_label_visible=torch.zeros(4,dtype=torch.bool))))
        with self.assertRaises(ValueError):obj.H(x,dict(tx_label_visible=torch.ones(4,dtype=torch.bool)),physical_mask(4,1,'cpu'),4441,'U','a')

    def test_L_replaces_As_half_only_without_unused_native_H(self):
        obj=CleanHObjective.__new__(CleanHObjective);obj.dual_L=True;obj.active_L=False;obj.c=dict(step_offset=4440);obj.L_calls=0;calls=[]
        obj.H=lambda raw,meta,p,step,role,view:torch.full((64,2,256),2. if view=='a' else 3.)
        class Model:
            def _cd_practical(self,*args):raise AssertionError('unused native H generated')
            def forward_identity_only(self,x,*args,**kwargs):calls.append(x.clone());return x
        model=Model();original=model._cd_practical;raw=torch.randn(128,2,256);value=raw+1;physical=physical_mask(128,4441,'cpu')
        with obj.labeled(model,dict(labeled_count=128,labeled_inputs=raw,extra=(None,{}),optimizer_step=1)):
            self.assertTrue(torch.equal(model._cd_practical(raw[:64],{},21,4441),raw[:64]))
            for _ in range(5):model.forward_identity_only(value)
            model.forward_identity_only(torch.randn(256,2,256))
        for i in (0,3,4):self.assertTrue(torch.equal(calls[i],value))
        for i in (1,2):
            self.assertTrue(torch.equal(calls[i][~physical],value[~physical]));self.assertTrue(bool((calls[i][physical]==i+1).all()))
        self.assertEqual(obj.L_calls,1);self.assertFalse(obj.active_L);self.assertEqual(model._cd_practical.__func__,original.__func__)

    def test_U_changes_only_existing_mixed_physical_half_and_captures_H_logits(self):
        obj=CleanHObjective.__new__(CleanHObjective);obj.c=dict(step_offset=4440);obj._joint_forward_index=0;obj.current_U=torch.randn(4,2,256);obj.pair_capture=[]
        obj.H=lambda raw,meta,p,step,role,view:torch.full((2,2,256),2. if view=='a' else 3.)
        seen=[];logits=torch.randn(4,6);x=torch.randn(4,2,256);accepted=torch.tensor([True,False,False,True]);student=SimpleNamespace(_feature_l_steps=1)
        def parent(self,student,value,*args):self._joint_forward_index+=1;seen.append(value.clone());return logits
        with patch.object(NoStyleNoRXE,'unlabeled_student_forward',parent):
            for _ in range(6):obj.unlabeled_student_forward(student,x,None,None,accepted,{})
        p=physical_mask(4,4441,'cpu')
        for i in (0,1,4,5):self.assertTrue(torch.equal(seen[i],x))
        for i in (2,3):self.assertTrue(torch.equal(seen[i][~p],x[~p]));self.assertTrue(bool((seen[i][p]==i).all()))
        self.assertEqual(len(obj.pair_capture),2);self.assertTrue(torch.equal(obj.pair_capture[0][1],~accepted[p]))

    def test_configs_remove_G_metadata_and_reject_restoration_target_or_changed_recipe(self):
        for method in run.METHODS:
            c=config(method);run.validate_config(c);self.assertNotIn('source_G',c);self.assertNotIn('G_inference',c)
            for change in (dict(source_G={}),dict(G_inference=False),dict(tests={}),dict(sig_time_views=3),dict(hard_pl_no_uas=True),dict(own_EMA=True)):
                with self.assertRaises(ValueError):run.validate_config(dict(c,**change))
            bad=deepcopy(c);bad['frozen_sources']['G']='forbidden'
            with self.assertRaises(ValueError):run.validate_config(bad)

    def test_context_and_runtime_forbid_G_or_target(self):
        c=context();run.validate_context(c)
        with self.assertRaises(ValueError):run.validate_context(dict(c,G={}))
        rt=dict(run_id=run.RUN,project='/project',python='python',release_root='/project/releases/'+run.RELEASE,run_root='/project/runs/'+run.RUN,
            context='/project/releases/'+run.RELEASE+'/context.json',runtime_file='/project/releases/'+run.RELEASE+'/runtime.json',code_commit='c',frozen_sources=c['frozen_sources'],allowed_gpus=list(range(8)),launch_owner='codex-pamf-clean-h-source-20261026')
        run.validate_runtime(rt)
        for change in (dict(truth='x'),dict(frozen_sources=dict(c['frozen_sources'],G='G')),dict(run_root='/other')):
            with self.assertRaises(ValueError):run.validate_runtime(dict(rt,**change))

    def test_source_gate_uses_clean_priority_and_exact_boundaries(self):
        refs={m:dict(raw_correct=27000,H3_correct=75000) for m in run.REFERENCES}
        scores=dict(refs,**{m:dict(raw_correct=26973,H3_correct=75243) for m in run.METHODS})
        self.assertEqual(run.choose_source(scores)[1],run.METHODS[0])
        scores[run.METHODS[0]]['H3_correct']+=100;scores[run.METHODS[1]]['raw_correct']+=1
        self.assertEqual(run.choose_source(scores)[1],run.METHODS[1])
        for k in ('raw_correct','H3_correct'):
            bad=dict(refs,**{m:dict(raw_correct=26973,H3_correct=75243) for m in run.METHODS})
            for m in run.METHODS:bad[m][k]-=1
            self.assertIsNone(run.choose_source(bad)[1])

    def test_four_view_source_score_and_actual_prepared_entrypoints(self):
        end=dict(status='CLEAN_H_SOURCE_EVALUATED',source_V_seen=27000,parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_reads=0,target_reads=0,
            results=[dict(view=v,dimension='overall',n=27000,correct=26000) for v in run.VIEWS])
        self.assertEqual(run.source_score(end),dict(raw_correct=26000,H3_correct=78000))
        with self.assertRaises(ValueError):run.source_score(dict(end,target_reads=1))
        with self.assertRaises(ValueError):run.source_score(dict(end,results=end['results']+[dict(view='G_raw',dimension='overall',n=27000,correct=27000)]))
        ctx=context();source=dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,head_fit_updates=0,config=dict(source_contract='roles'))
        def reader(path):
            v=str(path).replace('\\','/')
            if v=='context':return ctx
            if v=='roles':return dict(roles='existing')
            if v=='source/complete.json':return source
            if v.endswith('complete.json'):return dict(source_output='source')
            raise AssertionError(v)
        with tempfile.TemporaryDirectory() as folder:
            rt=dict(context='context',run_root=folder,release_root='/release',python='python',code_commit='c')
            with patch.object(run,'read',reader),patch.object(run,'source_score',return_value=dict(raw_correct=1,H3_correct=2)),patch.object(run,'config',side_effect=lambda ctx,commit,m:dict(paper_method=m)):
                jobs=run.prepare(rt)
            self.assertEqual({j['entrypoint'] for j in jobs},{'experiments.pamf_channel_pl.clean_h_source'})
            self.assertEqual({j['id'] for j in jobs},{'smoke',*run.METHODS})


if __name__=='__main__':unittest.main()
