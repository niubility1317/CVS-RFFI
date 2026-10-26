import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.pamf_channel_pl import sig1_role_training as run
from experiments.pamf_channel_pl import sig1_repl_offset_source as old
from experiments.pamf_channel_pl.sig1_role_objective import RestoredRoles,ROLES,mask,replace_ordinary,view_plan
from experiments.pamf_channel_pl.sig1_repl_offset_objective import NoStyleNoRXE


def config(method):
    base=dict(model_seed=run.SEED,protocol_id='F0',method_id='pamf_t1_e50',num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],
        sig_time_views=1,regime='matched_sat_plain_pl',ablation='F21',U_use=True,U_start_epoch=14,shared_SIG1_E20=run.prefix('/project'),
        epochs=50,optimizer_updates=11100,trajectory=[20,10,20],plain_PL_weight=6.,plain_PL_threshold=.95,labeled_batch=128,unlabeled_batch=256,
        default_head_joint=True,hard_pl=dict(method='HPLT1U',weight=6.,threshold=.95),hard_pl_no_uas=True,
        hard_pl_mixstyle=dict(probability_per_level=0.),hard_pl_gradient_guard=dict(U_to_L_norm_cap=1.),hard_pl_joint_contrast=dict(weight=.05),
        project='/project',source_contract='roles',expected_source_contract='roles',source_cache='cache',
        repl_row=dict(structure='SIG_TIME',L_A=True,L_B=True,U=True,blocks=dict(style=False,joint=True,guard=True,extrapolation=False)))
    with patch.object(old,'config',return_value=base):return run.config(dict(project='/project'),'commit',method)


class RoleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_swap_preserves_physical_inputs_and_detaches_G(self):
        source=torch.randn(4,2,256,requires_grad=True);restored=torch.randn(2,2,256,requires_grad=True);physical=mask(4,1,'cpu')
        before=source.detach().clone();out=replace_ordinary(source,restored,physical)
        self.assertTrue(torch.equal(out[physical],source[physical]));self.assertTrue(torch.equal(out[~physical],restored))
        self.assertTrue(torch.equal(source,before));out.sum().backward();self.assertIsNone(restored.grad)
        self.assertTrue(bool((source.grad[~physical]==0).all()));self.assertTrue(bool((source.grad[physical]==1).all()))
        with self.assertRaises(ValueError):replace_ordinary(source,restored[:1],physical)

    def test_mask_keeps_full_and_partial_U_batches(self):
        for n in (128,256,124):self.assertEqual(int(mask(n,4441,'cpu').sum()),n//2)
        for n,step in ((127,1),(0,1),(128,0)):
            with self.assertRaises(ValueError):mask(n,step,'cpu')

    def test_views_have_distinct_scenes_and_epoch_freshness(self):
        for step in (4441,4442,4443,6661):
            a,na=view_plan(step,'L','a');b,nb=view_plan(step,'L','b');self.assertNotEqual(a,b);self.assertNotEqual(na,nb)
        self.assertNotEqual(view_plan(4441,'U','a')[1],view_plan(4663,'U','a')[1])
        with self.assertRaises(ValueError):view_plan(4440,'U','a')
        with self.assertRaises(ValueError):view_plan(4441,'V','a')

    def test_only_two_mixed_L_ordinary_halves_change(self):
        obj=RestoredRoles.__new__(RestoredRoles);obj.c=dict(step_offset=4440);obj.enable_L=True;obj.active_L=False;obj.L_calls=0
        obj.restored=lambda raw,meta,p,step,role,view:torch.full((64,2,256),2. if view=='a' else 3.)
        recorded=[]
        class Model:
            def forward_identity_only(self,x,*args,**kw):recorded.append(x.clone());return x
        model=Model();original=model.forward_identity_only;value=torch.randn(128,2,256);physical=mask(128,1,'cpu')
        kwargs=dict(labeled_count=128,labeled_inputs=torch.randn_like(value),extra=(None,{}),optimizer_step=1)
        with obj.labeled(model,kwargs):
            for _ in range(5):model.forward_identity_only(value)
            model.forward_identity_only(torch.randn(256,2,256))
        for i in (0,3,4):self.assertTrue(torch.equal(recorded[i],value))
        for i in (1,2):
            self.assertTrue(torch.equal(recorded[i][physical],value[physical]));self.assertTrue(bool((recorded[i][~physical]==i+1).all()))
        self.assertEqual(obj.L_calls,1);self.assertFalse(obj.active_L);self.assertEqual(model.forward_identity_only.__func__,original.__func__)

    def test_wrong_role_visibility_rejected_before_channel(self):
        obj=RestoredRoles.__new__(RestoredRoles);meta=dict(tx_label_visible=torch.ones(4,dtype=torch.bool))
        with self.assertRaises(ValueError):obj.restored(torch.randn(4,2,256),meta,mask(4,1,'cpu'),4441,'U','a')

    def test_only_U_mixed_positions_change_and_cross_RX_capture_pair_stays_raw(self):
        obj=RestoredRoles.__new__(RestoredRoles);obj.enable_U=True;obj._joint_forward_index=0;obj.c=dict(step_offset=4440);obj.current_raw_U=torch.randn(4,2,256)
        obj.restored=lambda raw,meta,p,step,role,view:torch.full((2,2,256),2. if view=='a' else 3.)
        value=torch.randn(4,2,256);physical=mask(4,4441,'cpu');model=SimpleNamespace(_feature_l_steps=1)
        def parent(self,student,x,*args):self._joint_forward_index+=1;return x
        with patch.object(NoStyleNoRXE,'unlabeled_student_forward',parent):
            out=[obj.unlabeled_student_forward(model,value,None,None,None,{}) for _ in range(6)]
        for i in (0,1,4,5):self.assertTrue(torch.equal(out[i],value))
        for i in (2,3):
            self.assertTrue(torch.equal(out[i][physical],value[physical]));self.assertTrue(bool((out[i][~physical]==i).all()))

    def test_three_matched_configs_reject_extra_mechanisms(self):
        for method in run.METHODS:
            c=config(method);run.validate_config(c);self.assertEqual(c['restored_roles']['enable_L'],ROLES[method][0])
            for change in (dict(tests={}),dict(sig_time_views=3),dict(plain_PL_weight=7),dict(own_EMA=True),dict(posthoc_head_fit=True)):
                with self.subTest(change=change),self.assertRaises(ValueError):run.validate_config(dict(c,**change))
            bad=deepcopy(c);bad['restored_roles']['noise']=True
            with self.assertRaises(ValueError):run.validate_config(bad)

    def test_only_own_complete_E20_trajectory_allowed(self):
        c=config(run.METHODS[0]);s=dict(c,segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=run.prefix('/project'),output_root=c['output_root']+'/segment30')
        run.validate_config(s)
        for change in (dict(parent_source='/foreign'),dict(step_offset=0),dict(segment_epochs=11)):
            with self.assertRaises(ValueError):run.validate_config(dict(s,**change))

    def test_exact_source_count_boundaries_and_lowest_G_cost_tie(self):
        refs={m:dict(raw_correct=27000,H3_correct=81000,G3_correct=75000) for m in run.REFERENCES}
        scores=dict(refs,**{m:dict(raw_correct=26973,H3_correct=80919,G3_correct=75243) for m in run.METHODS})
        decisions,selected=run.choose_source(scores);self.assertTrue(all(r['source_pass'] for r in decisions));self.assertEqual(selected,'REPL_G_L')
        for key in ('raw_correct','H3_correct','G3_correct'):
            bad=deepcopy(scores)
            for method in run.METHODS:bad[method][key]-=1
            self.assertIsNone(run.choose_source(bad)[1])

    def test_source_score_cannot_accept_partial_or_target_evaluation(self):
        c=dict(status='SIG1_REPL_SOURCE_EVALUATED',source_V_seen=27000,parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_reads=0,target_reads=0,
            results=[dict(view=view,dimension='overall',n=27000,correct=26000) for view in run.VIEWS])
        self.assertEqual(run.source_score(c)['H3_correct'],78000)
        with self.assertRaises(ValueError):run.source_score(dict(c,target_reads=1))
        with self.assertRaises(ValueError):run.source_score(dict(c,results=c['results'][:-1]))


if __name__=='__main__':unittest.main()
