import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import torch
from experiments.pamf_u_history.rx_mean_views import MeanBank,fit,transfers,Shift,RXS
from experiments.pamf_u_history.rx_mean_objective import soft_loss,combine,METHODS
from experiments.pamf_u_history.rx_mean_source import config,validate_train,validate_evaluation,prepare,SEEDS,RUN


class RXMeanTests(unittest.TestCase):
    def bank(self):return MeanBank(np.arange(20,dtype=np.float64).reshape(5,4)/7)

    def test_fit_uses_hiddenU_only_and_keeps_group_means(self):
        rx=np.repeat(RXS,3);x=np.arange(60,dtype=np.float64).reshape(15,4)
        bank,counts=fit(x,rx,role='U_s');np.testing.assert_allclose(bank.means,np.stack([x[rx==r].mean(0) for r in RXS]))
        self.assertEqual(set(counts.values()),{3});self.assertFalse(bank.means.flags.writeable)
        with self.assertRaises(ValueError):fit(x,rx,role='V')
        with self.assertRaises(ValueError):fit(x[rx!=8],rx[rx!=8],role='U_s')

    def test_action_is_bounded_group_constant_and_packet_independent(self):
        bank=self.bank();a,bounds=transfers(bank)
        np.testing.assert_array_equal(a[np.arange(5),np.arange(5)],np.ones((5,256)))
        self.assertLessEqual(bounds['max_abs_gain_db'],.5+1e-8);self.assertLessEqual(bounds['max_abs_phase_rad'],.025+1e-8)
        shift=Shift(bank,'cpu');x=torch.randn(3,2,256);rx=torch.tensor([1,3,1]);indices=torch.tensor([0,1,0])
        y,styles=shift(x,rx,indices,1);z,_=shift(x[:1],rx[:1],indices[:1],1)
        torch.testing.assert_close(y[:1],z);self.assertTrue(bool((styles!=rx).all()))
        scaled,_=shift(2*x,rx,indices,1);torch.testing.assert_close(scaled,2*y)
        with self.assertRaises(ValueError):shift(x,torch.tensor([0,3,1]),indices,1)

    def test_soft_consistency_scope_changes_gradient_only(self):
        u=torch.randn(7,6,requires_grad=True);l=torch.randn(3,6,requires_grad=True)
        up=torch.softmax(torch.randn(7,6),1);lp=torch.softmax(torch.randn(3,6),1)
        ul=soft_loss(u,up);ll=soft_loss(l,lp)
        a=combine(ul,ll,7,3,'U');b=combine(ul,ll,7,3,'LU');self.assertEqual(float(a.detach()),float(b.detach()))
        ga=torch.autograd.grad(a,(u,l),retain_graph=True,allow_unused=True);gb=torch.autograd.grad(b,(u,l),allow_unused=True)
        self.assertIsNone(ga[1]);self.assertGreater(float(ga[0].norm()),0);self.assertGreater(float(gb[1].norm()),0)
        with self.assertRaises(ValueError):soft_loss(u,up.requires_grad_())

    def context(self):
        from experiments.pamf_channel_pl.sig1_matched_baselines import prefix,RUN as BASELINE
        c=dict(project='/project',python='/python',frozen_sources={k:'/'+k for k in ('F','AF','BASE','FOLD')},
            channel_code='/physics',source_contract='/contract',source_cache='/cache',cases=[])
        for seed in SEEDS:
            rid='F0-SIG1-PL-s'+str(seed)
            a=dict(run_id=BASELINE,row_id=rid,model_seed=seed,paper_method='SIG1-PL',algorithm_id='pamf_sig1_matched_baselines_v1',
                method_id='pamf_t1_e50',protocol_id='F0',num_classes=6,source_rxs=list(RXS),source_days=[1,2,3],regime='matched_sat_plain_pl',ablation='F21',
                U_use=True,sig_time_views=1,U_start_epoch=14,shared_SIG1_E20=prefix('/project',seed),epochs=50,optimizer_updates=11100,trajectory=[20,10,20],
                default_head_joint=True,plain_PL_threshold=.95,plain_PL_weight=1.,unlabeled_batch=256,labeled_batch=128,
                repl_row=dict(row_id=rid,model_seed=seed,structure='SIG_TIME',U=True,L_A=True,L_B=True,blocks={}),project='/project',
                output_root='/project/runs/'+BASELINE+'/F/'+rid+'/source',source_contract='/contract',source_cache='/cache',frozen_sources=c['frozen_sources'])
            c['cases'].append(dict(model_seed=seed,baseline_config=a))
        return c

    def test_shared_recipe_and_prohibited_inputs(self):
        ctx=self.context();c=config(ctx,'commit',METHODS[0],SEEDS[1],.3);validate_train(c)
        for key,value in [('plain_PL_threshold',.8),('sig_time_views',3),('source_G','G'),('truth','target'),('shared_SIG1_E20','foreign')]:
            bad=deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate_train(bad)
        bad=deepcopy(c);bad['_calibration_base_override']='H'
        with self.assertRaises(ValueError):validate_train(bad)
        c.update(segment_epochs=10,epoch_offset=20,step_offset=4440,parent_source=c['shared_SIG1_E20'],output_root=c['output_root']+'/segment30');validate_train(c)

    def test_complete_matrix_and_prediction_truth_separation(self):
        ctx=self.context()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);context=root/'context.json';context.write_text(json.dumps(ctx),encoding='utf-8')
            ev=root/'evaluation.json';ev.write_text(json.dumps(dict(classes=list(range(6)),tests={t:dict(views_root='/'+t,truth='/truth') for t in ('cross_rx','cross_both')})),encoding='utf-8')
            rt=dict(context=str(context),evaluation_context=str(ev),run_root=str(root/'run'),release_root='/release',python='/python',code_commit='commit',run_id=RUN)
            jobs=prepare(rt);self.assertEqual(len(jobs),22)
            self.assertEqual(sum(j['kind']=='train' for j in jobs),6);self.assertEqual(sum(j['kind']=='calibrate' for j in jobs),3)
            for job in jobs:
                if job['kind']=='train':continue # Written only after actual calibration.
                c=json.loads(Path(job['config']).read_text())
                if job['kind']=='predict':
                    validate_evaluation(c);self.assertNotIn('truth',json.dumps(c));bad=deepcopy(c);bad['mean_bank']='/bank'
                    with self.assertRaises(ValueError):validate_evaluation(bad)
                if job['kind']=='score':
                    validate_evaluation(c,True);self.assertNotIn('source_cache',c);self.assertNotIn('source_output',c)
                if job['kind']=='calibrate':self.assertNotIn('evaluation_context',c)

    def test_native_runtime_import_happens_after_frozen_activation(self):
        import builtins
        from experiments.pamf_u_history.rx_mean_source import calibrate
        active=[];original=builtins.__import__
        class BoundaryReached(Exception):pass
        def importing(name,*args,**kwargs):
            if name=='experiments.cvs_phase1_stack.runtime':
                self.assertEqual(active,['frozen_F'])
                raise BoundaryReached()
            return original(name,*args,**kwargs)
        with patch('experiments.pamf_u_history.rx_mean_source.read',return_value=self.context()),patch('experiments.pamf_u_history.rx_mean_source.activate',side_effect=lambda c:active.append('frozen_F')):
            with patch('builtins.__import__',side_effect=importing),self.assertRaises(BoundaryReached):calibrate(dict(context='/source-only'))


if __name__=='__main__':unittest.main()
