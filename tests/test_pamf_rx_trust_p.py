import unittest
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from experiments.pamf_u_history.rx_trust_p_objective import candidates,routed_loss,Objective,Exposure,RECIPE
from experiments.pamf_u_history.rx_mean_max_objective import Objective as MaxObjective
from experiments.pamf_u_history.rx_trust_p_diagnostics import partial_statistics


class TrustPTest(unittest.TestCase):
    def probabilities(self):
        raw=torch.tensor([[.99,.002,.002,.002,.002,.002],[.99,.002,.002,.002,.002,.002],[.6,.08,.08,.08,.08,.08]])
        H=torch.tensor([[.5,.1,.1,.1,.1,.1],[.08,.9,.005,.005,.005,.005],[.3,.3,.1,.1,.1,.1]])
        return raw,H

    def test_singleton_union_and_original_confidence_route(self):
        raw,H=self.probabilities();accepted,mask=candidates(raw,H)
        self.assertEqual(accepted.tolist(),[True,True,False]);self.assertEqual(mask.sum(1).tolist(),[1,2,1])
        self.assertTrue(mask[torch.arange(3),raw.argmax(1)].all())
        order=torch.tensor([2,0,1]);a,m=candidates(raw[order],H[order]);torch.testing.assert_close(a,accepted[order]);torch.testing.assert_close(m,mask[order])

    def test_routed_value_gradient_and_full_U_denominator(self):
        raw,H=self.probabilities();logits=H.log().requires_grad_();loss,accepted,mask=routed_loss(logits,raw,H)
        hard=-logits.log_softmax(1)[0,0];partial=-torch.logsumexp(logits.log_softmax(1)[1,:2],0)
        local=F.kl_div(logits.log_softmax(1)[2],H[2],reduction='sum')
        torch.testing.assert_close(loss,(hard+partial+local)/3)
        grad=torch.autograd.grad(loss,logits)[0];p=logits.softmax(1).detach();conditional=p*mask/(p*mask).sum(1,keepdim=True)
        expected=torch.where(accepted[:,None],p-conditional,p-H)/3;torch.testing.assert_close(grad,expected,atol=1e-7,rtol=1e-5)
        self.assertLess(float(torch.linalg.norm(grad[1])),float(torch.linalg.norm(p[1]-F.one_hot(torch.tensor(0),6))))

    def test_no_accepted_samples_remain_same_H_soft_KL(self):
        q=torch.full((5,6),1/6);h=torch.softmax(torch.randn(5,6),1);z=torch.randn(5,6,requires_grad=True)
        loss,a,_=routed_loss(z,q,h);self.assertFalse(a.any());torch.testing.assert_close(loss,F.kl_div(z.log_softmax(1),h,reduction='batchmean'))

    def test_detached_normalized_six_class_targets_required(self):
        raw,H=self.probabilities()
        for a,b in [(raw.requires_grad_(),H),(raw.detach(),H*2),(raw.detach()[:,:5],H[:,:5])]:
            with self.assertRaises(ValueError):candidates(a,b)

    def test_max_default_loss_remains_exact_original_soft_KL(self):
        z=torch.randn(7,6,requires_grad=True);p=torch.softmax(torch.randn(7,6),1);obj=MaxObjective({})
        for mode in ('raw','H'):torch.testing.assert_close(obj.source_U_loss(z,p,mode),F.kl_div(z.log_softmax(1),p,reduction='batchmean'),rtol=0,atol=0)
        self.assertIsNot(Objective.source_U_loss,MaxObjective.source_U_loss)

    def test_source_set_inclusion_and_logit_gradient_match_torch_route(self):
        raw,H=self.probabilities();y=torch.tensor([0,1,0]);z=H.log().requires_grad_()
        r=partial_statistics(raw.log().numpy(),H.log().numpy(),y.numpy(),role='V')
        loss,a,mask=routed_loss(z,raw,H);g=torch.autograd.grad(loss,z)[0]
        self.assertEqual(r['accepted'],2);self.assertEqual(r['candidate_contains_truth'],2);self.assertEqual(r['raw_correct'],1)
        self.assertEqual(r['singleton'],1);self.assertEqual(r['pair'],1)
        self.assertAlmostEqual(r['partial_logit_gradient_norm_mean'],float(torch.linalg.norm(g[a]*3,dim=1).mean()),places=6)
        self.assertGreaterEqual(r['candidate_inclusion_pct'],r['raw_precision_pct'])

    def test_source_proxy_rejects_hidden_truth_role(self):
        raw,H=self.probabilities()
        with self.assertRaises(ValueError):partial_statistics(raw.log().numpy(),H.log().numpy(),torch.tensor([0,1,0]).numpy(),role='U_s')

    def test_H_routes_preserve_all_packets_and_actual_source_RX_counts(self):
        raw,H=self.probabilities();obj=Objective({});obj.raw_prob=raw;obj.accepted=raw.max(1).values>=.95
        obj.history=SimpleNamespace(P_singleton=0,P_pair=0,P_fallback=0,P_by_rx={str(r):dict(singleton=0,pair=0,fallback=0) for r in (1,3,4,6,8)},P_by_pseudo_class={str(k):0 for k in range(6)})
        obj.batch=(None,None,None,dict(rx_i=torch.tensor([1,3,8])))
        value=obj.source_U_loss(H.log().requires_grad_(),H,'H');self.assertTrue(torch.isfinite(value))
        self.assertEqual((obj.history.P_singleton,obj.history.P_pair,obj.history.P_fallback),(1,1,1))
        self.assertEqual(obj.history.P_by_rx['3']['pair'],1);self.assertEqual(obj.history.P_by_rx['8']['fallback'],1)
        self.assertEqual(sum(obj.history.P_by_pseudo_class.values()),2)
        obj.source_U_loss(raw.log().requires_grad_(),raw,'raw');self.assertEqual(obj.last['RXP_current_pair'],0)
        self.assertEqual((obj.history.P_singleton,obj.history.P_pair,obj.history.P_fallback),(1,1,1))
        with self.assertRaises(ValueError):Exposure(dict(y=[0]))


class TrustPPipelineTest(unittest.TestCase):
    def context(self):
        from test_pamf_rx_mean import RXMeanTests
        from experiments.pamf_u_history.rx_trust_p_source import PREDECESSOR
        c=RXMeanTests().context();c['predecessor_complete']=c['project']+'/runs/'+PREDECESSOR+'/complete.json';return c

    def test_shared_controls_and_own_continuation_paths(self):
        from experiments.pamf_u_history.rx_trust_p_source import config,SEEDS,validate_train,trajectory
        from experiments.pamf_u_history import rx_mean_source as mean
        c=config(self.context(),'commit',SEEDS[0],.4);validate_train(c)
        for key,value in [('plain_PL_threshold',.9),('sig_time_views',3),('source_G','G'),('target_truth','target')]:
            with self.assertRaises(ValueError):validate_train(dict(c,**{key:value}))
        bad=deepcopy(c);bad['rx_mean_recipe']['H_route_threshold']=.9
        with self.assertRaises(ValueError):validate_train(bad)
        own=c['output_root'];c.update(segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=own+'/segment30',output_root=own+'/segment50');validate_train(c)
        with self.assertRaises(ValueError):validate_train(dict(c,parent_source='/foreign'))
        old=mean.Objective
        with trajectory():self.assertIs(mean.Objective,Objective)
        self.assertIs(mean.Objective,old)

    def test_three_seeds_queue_after_Max_with_truth_isolated(self):
        from experiments.pamf_u_history.rx_trust_p_source import RUN,prepare
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);c=root/'context.json';c.write_text(json.dumps(self.context()));e=root/'evaluation.json'
            e.write_text(json.dumps(dict(classes=list(range(6)),tests={t:dict(views_root='/'+t,truth='/truth') for t in ('cross_rx','cross_both')})))
            jobs=prepare(dict(run_id=RUN,context=str(c),evaluation_context=str(e),run_root=str(root/'run'),release_root='/release',python='/python',code_commit='commit'))
            self.assertEqual(len(jobs),12)
            for job in jobs:
                self.assertEqual(job['entrypoint'],'experiments.pamf_u_history.rx_trust_p_source')
                if job['kind']=='train':continue
                value=json.loads(Path(job['config']).read_text())
                if job['kind']=='calibrate':
                    self.assertEqual(job['dependencies'],[self.context()['predecessor_complete']]);self.assertNotIn('evaluation_context',value)
                if job['kind']=='predict':self.assertNotIn('truth',json.dumps(value));self.assertNotIn('mean_bank',value)
                if job['kind']=='score':self.assertNotIn('source_output',value);self.assertNotIn('frozen_sources',value)

    def test_H_route_exposure_must_cover_all_and_not_change_compute(self):
        from experiments.pamf_u_history.rx_trust_p_source import check_exposure
        rx=(1,3,4,6,8);h=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,raw_base_U=850500,H_base_U=850500,
            mean_shift_L_pairs=426240,nonzero_shift_U=1701000,candidate_U_packets=6804000,teacher_U_packets=9355500,candidate_variants=4,
            selection_policy=RECIPE['style_selection'],selection_uses_U_TX=False,student_U_packets=3402000,extra_L_teacher_packets=426240,extra_L_student_packets=426240,
            by_RX_style={str(r)+'->'+str(rx[(i+1)%5]):340200 for i,r in enumerate(rx)},P_target_rule=RECIPE['H_target'],P_U_TX_truth_read=False,P_extra_F_forwards=0,
            P_H_singleton=500000,P_H_pair=300000,P_H_soft_fallback=50500,P_H_by_source_RX={str(r):dict(singleton=100000,pair=60000,fallback=10100) for r in rx},
            P_H_by_raw_pseudo_class={str(k):800000 if k==0 else 0 for k in range(6)})
        check_exposure(h)
        with self.assertRaises(ValueError):check_exposure(dict(h,P_H_pair=300001))
        bad=deepcopy(h);bad['P_H_by_source_RX']['1']['pair']-=1
        with self.assertRaises(ValueError):check_exposure(bad)


if __name__=='__main__':unittest.main()
