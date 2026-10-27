from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import pytest
from experiments.pamf_u_history.rx_trust_al_objective import Objective,RECIPE
from experiments.pamf_u_history.rx_trust_a_objective import Objective as AObjective,RECIPE as A_RECIPE
from experiments.pamf_u_history.rx_mean_max_objective import Objective as MaxObjective
from experiments.pamf_u_history.rx_mean_objective import soft_loss
from experiments.pamf_u_history.covariance_objective import combine


def objective(calibration=False):
    obj=Objective({'rx_mean_calibration':calibration})
    obj.history=SimpleNamespace(L_CE_seen=0,L_CE_by_base={'raw':0,'native_As':0},L_CE_by_true_class={str(k):0 for k in range(6)})
    return obj


def context():
    from test_pamf_rx_mean import RXMeanTests
    from experiments.pamf_u_history.rx_trust_al_source import PREDECESSOR
    c=RXMeanTests().context();c['predecessor_complete']=c['project']+'/runs/'+PREDECESSOR+'/complete.json'
    return c


def test_true_L_labels_overrule_wrong_same_base_prediction_with_original_smoothing():
    obj=objective();z=torch.tensor([[0.,3.,0.,0.,0.,0.]]).repeat(64,1).requires_grad_()
    wrong=z.detach().softmax(1);labels=torch.zeros(64,dtype=torch.long)
    value=obj.source_L_loss(z,wrong,labels,'H');gradient=torch.autograd.grad(value,z)[0]
    target=.99*F.one_hot(labels,6)+.01/6
    torch.testing.assert_close(gradient,(z.softmax(1).detach()-target)/64)
    assert float(gradient[0,0]-gradient[0,1])<0
    assert obj.history.L_CE_by_base=={'raw':0,'native_As':64}
    assert obj.history.L_CE_by_true_class['0']==64
    assert obj.last['AL_L_LS']==.01


@pytest.mark.parametrize('nu',[124,256])
def test_joint_L_gradient_added_without_changing_U_packet_normalization(nu):
    obj=objective(True);u=torch.tensor(.7,requires_grad=True);l=torch.tensor(.4,requires_grad=True)
    value=obj.combine_source_losses(u,l,nu,64)
    gu,gl=torch.autograd.grad(value,(u,l),retain_graph=True)
    assert gu.item()==pytest.approx(nu/(nu+64)) and gl.item()==pytest.approx(64/(nu+64))
    torch.testing.assert_close(value,combine(u,l,nu,64,'LU'),rtol=0,atol=0)
    assert set(obj.calibration_components)=={'U','L'}
    torch.testing.assert_close(sum(obj.calibration_components.values()),value)


def test_default_Max_hooks_preserve_exact_A_detached_L_semantics():
    obj=MaxObjective({});z=torch.randn(64,6,requires_grad=True);p=torch.softmax(torch.randn(64,6),1)
    l=obj.source_L_loss(z,p,torch.arange(64)%6,'H');u=torch.tensor(.7,requires_grad=True)
    torch.testing.assert_close(l,soft_loss(z,p),rtol=0,atol=0)
    value=obj.combine_source_losses(u,l,256,64)
    torch.testing.assert_close(value,combine(u,l,256,64,'U'),rtol=0,atol=0)
    assert torch.autograd.grad(value,z,allow_unused=True)[0] is None
    assert Objective.source_U_loss is AObjective.source_U_loss
    for key in ('H_target','H_route_threshold','raw_target','H_set_loss','coefficient_rule','view'):
        assert RECIPE[key]==A_RECIPE[key]


def test_visible_L_label_shape_and_mapping_required():
    obj=objective();z=torch.randn(64,6,requires_grad=True);p=z.detach().softmax(1);y=torch.zeros(64,dtype=torch.long)
    for labels in (y.float(),y[:63],y-1,y+6):
        with pytest.raises(ValueError):obj.source_L_loss(z,p,labels,'raw')
    with pytest.raises(ValueError):obj.source_L_loss(z,p.requires_grad_(),y,'raw')


def test_original_PL_SIG1_default_head_and_own_parent_scope_remain_fixed():
    from experiments.pamf_u_history.rx_trust_al_source import config,validate_train,SEEDS,trajectory
    from experiments.pamf_u_history import rx_mean_source as mean
    c=config(context(),'commit',SEEDS[0],.1);validate_train(c)
    for key,value in [('plain_PL_threshold',.9),('sig_time_views',3),('source_G','G'),('target_truth','target')]:
        with pytest.raises(ValueError):validate_train(dict(c,**{key:value}))
    bad=deepcopy(c);bad['rx_mean_recipe']['L_target']='U_truth'
    with pytest.raises(ValueError):validate_train(bad)
    own=c['output_root'];c.update(segment_epochs=20,epoch_offset=30,step_offset=6660,parent_source=own+'/segment30',output_root=own+'/segment50');validate_train(c)
    with pytest.raises(ValueError):validate_train(dict(c,parent_source='/foreign'))
    old=mean.Objective
    with trajectory():assert mean.Objective is Objective
    assert mean.Objective is old


def test_all_three_seeds_wait_for_A_and_predictor_never_gets_truth():
    from experiments.pamf_u_history.rx_trust_al_source import RUN,prepare
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);c=root/'context.json';c.write_text(json.dumps(context()));e=root/'evaluation.json'
        e.write_text(json.dumps(dict(classes=list(range(6)),tests={t:dict(views_root='/'+t,truth='/truth') for t in ('cross_rx','cross_both')})))
        jobs=prepare(dict(run_id=RUN,context=str(c),evaluation_context=str(e),run_root=str(root/'run'),release_root='/release',python='/python',code_commit='commit'))
        assert len(jobs)==12
        for job in jobs:
            assert job['entrypoint']=='experiments.pamf_u_history.rx_trust_al_source'
            if job['kind']=='train':continue
            value=json.loads(Path(job['config']).read_text())
            if job['kind']=='calibrate':assert job['dependencies']==[context()['predecessor_complete']] and 'evaluation_context' not in value
            if job['kind']=='predict':assert 'truth' not in json.dumps(value) and 'mean_bank' not in value
            if job['kind']=='score':assert 'source_output' not in value and 'frozen_sources' not in value
