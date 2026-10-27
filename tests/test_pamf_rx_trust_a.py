from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import pytest
from experiments.pamf_u_history.rx_trust_a_objective import candidates,routed_loss,Objective,RECIPE


def probs():
    return (torch.tensor([[.99,.002,.002,.002,.002,.002],[.6,.08,.08,.08,.08,.08]]),
            torch.tensor([[.2,.5,.075,.075,.075,.075],[.3,.3,.1,.1,.1,.1]]))


def context():
    from test_pamf_rx_mean import RXMeanTests
    from experiments.pamf_u_history.rx_trust_a_source import PREDECESSOR
    c=RXMeanTests().context();c['predecessor_complete']=c['project']+'/runs/'+PREDECESSOR+'/complete.json'
    return c


def test_disagreeing_H_has_single_raw_anchor_and_corrective_logit_margin():
    raw,H=probs();z=H.log().requires_grad_();loss,a,mask=routed_loss(z,raw,H)
    assert a.tolist()==[True,False] and mask.sum(1).tolist()==[1,1]
    assert mask[0,0] and not mask[0,1]
    g=torch.autograd.grad(loss,z)[0]
    assert float(g[0,0]-g[0,1])<0 # Direct logit step shrinks wrong-minus-anchor margin.
    expected=(F.cross_entropy(z[:1],torch.tensor([0]))+F.kl_div(z[1:].log_softmax(1),H[1:],reduction='sum'))/2
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(g[0],(z[0].softmax(0).detach()-torch.tensor([1.,0.,0.,0.,0.,0.]))/2)


def test_rejected_U_still_exact_H_soft_KL_and_full_U_denominator():
    raw=torch.full((5,6),1/6);H=torch.softmax(torch.randn(5,6),1);z=torch.randn(5,6,requires_grad=True)
    loss,a,_=routed_loss(z,raw,H);assert not a.any()
    torch.testing.assert_close(loss,F.kl_div(z.log_softmax(1),H,reduction='batchmean'))


def test_detached_normalized_target_validation_and_permutation():
    raw,H=probs();a,m=candidates(raw,H);order=torch.tensor([1,0]);b,n=candidates(raw[order],H[order])
    torch.testing.assert_close(b,a[order]);torch.testing.assert_close(n,m[order])
    for x,y in ((raw.requires_grad_(),H),(raw.detach(),H*2),(raw.detach()[:,:5],H[:,:5])):
        with pytest.raises(ValueError):candidates(x,y)


def test_H_anchor_exposure_counts_raw_acceptance_and_original_RX_without_U_truth():
    raw,H=probs();obj=Objective({});obj.raw_prob=raw;obj.accepted=raw.max(1).values>=.95
    obj.history=SimpleNamespace(P_singleton=0,P_pair=0,P_fallback=0,P_by_rx={str(r):dict(singleton=0,pair=0,fallback=0) for r in (1,3,4,6,8)},P_by_pseudo_class={str(k):0 for k in range(6)})
    obj.batch=(None,None,None,dict(rx_i=torch.tensor([1,8])))
    assert torch.isfinite(obj.source_U_loss(H.log().requires_grad_(),H,'H'))
    assert (obj.history.P_singleton,obj.history.P_pair,obj.history.P_fallback)==(1,0,1)
    assert obj.history.P_by_rx['1']['singleton']==1 and obj.history.P_by_rx['8']['fallback']==1
    assert sum(obj.history.P_by_pseudo_class.values())==1
    obj.source_U_loss(raw.log().requires_grad_(),raw,'raw')
    assert (obj.history.P_singleton,obj.history.P_pair,obj.history.P_fallback)==(1,0,1)


def test_original_PL_full_model_scope_and_own_seed_parents_remain_fixed():
    from experiments.pamf_u_history.rx_trust_a_source import config,validate_train,SEEDS,trajectory
    from experiments.pamf_u_history import rx_mean_source as mean
    c=config(context(),'commit',SEEDS[0],.1);validate_train(c)
    for key,value in [('plain_PL_threshold',.9),('sig_time_views',3),('source_G','G'),('target_truth','target')]:
        with pytest.raises(ValueError):validate_train(dict(c,**{key:value}))
    own=c['output_root'];c.update(segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=own+'/segment30',output_root=own+'/segment50');validate_train(c)
    with pytest.raises(ValueError):validate_train(dict(c,parent_source='/foreign'))
    old=mean.Objective
    with trajectory():assert mean.Objective is Objective
    assert mean.Objective is old


def test_three_seeds_wait_for_PM_and_prediction_truth_isolation():
    from experiments.pamf_u_history.rx_trust_a_source import RUN,prepare
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);c=root/'context.json';c.write_text(json.dumps(context()));e=root/'evaluation.json'
        e.write_text(json.dumps(dict(classes=list(range(6)),tests={t:dict(views_root='/'+t,truth='/truth') for t in ('cross_rx','cross_both')})))
        jobs=prepare(dict(run_id=RUN,context=str(c),evaluation_context=str(e),run_root=str(root/'run'),release_root='/release',python='/python',code_commit='commit'))
        assert len(jobs)==12
        for job in jobs:
            assert job['entrypoint']=='experiments.pamf_u_history.rx_trust_a_source'
            if job['kind']=='train':continue
            value=json.loads(Path(job['config']).read_text())
            if job['kind']=='calibrate':
                assert job['dependencies']==[context()['predecessor_complete']] and 'evaluation_context' not in value
            if job['kind']=='predict':assert 'truth' not in json.dumps(value) and 'mean_bank' not in value
            if job['kind']=='score':assert 'source_output' not in value and 'frozen_sources' not in value


def test_source_mode_calibration_rule_matches_PM_algorithm_without_reusing_changed_loss_gradients():
    from experiments.pamf_u_history.rx_trust_a_source import mode_coefficient
    steps=[dict(step=i+1,base='raw' if i%2==0 else 'H',gradient=dict(norm=1 if i%2==0 else 4,base_norm=10)) for i in range(12)]
    alpha,medians=mode_coefficient(steps)
    assert alpha==pytest.approx(.25) and medians=={'raw':.1,'H':.4}
    assert RECIPE['coefficient_rule']=='min(1,.1/max_base_median(aux_norm/primary_norm));12train_batches6raw6H,noV'
    assert RECIPE['H_target']=='rawPLaccepted:hard_raw_argmax;otherwise_sameH_softKL'
