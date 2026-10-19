import copy
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

torch.set_num_threads(2)


def module(name):
    assert importlib.util.find_spec('experiments.pamf_joint_coarse_restore') is not None, 'R23 implementation missing'
    return importlib.import_module('experiments.pamf_joint_coarse_restore.'+name)


class Estimator(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.tensor(.02))

    def forward(self, received, coarse):
        from experiments.pamf_action_restore.operators import Action
        value=torch.complex(coarse[:,0],coarse[:,1])
        main=.9+.1j+self.weight*value
        kernel=torch.ones((len(received),1),dtype=main.dtype,device=main.device)
        return Action(kernel,main,.02*main), .01+.001*coarse.square().mean((1,2))


class Context(nn.Module):
    def __init__(self):
        super().__init__();self.coarse=nn.Conv1d(2,2,1);self.estimator=Estimator()
        with torch.no_grad():
            self.coarse.weight.copy_(.8*torch.eye(2)[:,:,None]);self.coarse.bias.fill_(.01)

    def estimate(self, received):
        with torch.no_grad():base=self.coarse(received)
        action,variance=self.estimator(received,base)
        return base,action,variance


def packets(n=2):
    return .2*torch.randn((n,2,256),generator=torch.Generator().manual_seed(23001))


def build(candidate):
    torch.manual_seed(13)
    return module('models').JointCoarseRestorer(candidate,Context())


def test_initial_weights_and_outputs_equal_original_three_step_restorer():
    from experiments.pamf_unrolled_restore.models import UnrolledRestorer
    torch.manual_seed(13);baseline=UnrolledRestorer('unroll_unshared',Context()).eval()
    x=packets()
    for candidate in ('prior_only','joint_coarse'):
        model=build(candidate).eval()
        assert all(torch.equal(v,model.state_dict()[key]) for key,v in baseline.state_dict().items())
        assert torch.equal(model(x),baseline(x))


@pytest.mark.parametrize('candidate',['prior_only','joint_coarse'])
def test_gradients_and_one_step_change_only_authorized_parameters(candidate):
    model=build(candidate).train();before=copy.deepcopy(model.state_dict());x=packets()
    assert not model.frozen.estimator.training
    assert model.frozen.coarse.training==(candidate=='joint_coarse')
    stages,*_=model.stage_outputs(x)
    for stage in stages:stage.retain_grad()
    loss,_=module('training').iq_loss(stages[-1],.95*x+.03)
    loss.mean().backward()
    assert len(stages)==3 and all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.norm()>0 for v in stages)
    assert all(any(p.grad is not None and p.grad.norm()>0 for p in core.parameters()) for core in model.prior)
    assert all(p.grad is None and not p.requires_grad for p in model.frozen.estimator.parameters())
    coarse=list(model.frozen.coarse.parameters())
    if candidate=='joint_coarse':assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in coarse)
    else:assert all(not p.requires_grad and p.grad is None for p in coarse)
    optimizer=module('training').build_optimizer(model)
    module('training').set_learning_rate(optimizer,246)
    module('training').clip_groups(optimizer)
    optimizer.step()
    after=model.state_dict()
    changed={key for key in before if not torch.equal(before[key],after[key])}
    assert any(key.startswith('prior.') for key in changed)
    assert not any(key.startswith('frozen.estimator.') for key in changed)
    assert any(key.startswith('frozen.coarse.') for key in changed)==(candidate=='joint_coarse')


def test_joint_gradient_does_not_use_parent_estimate_no_grad_boundary():
    model=build('joint_coarse').train()
    def forbidden(*args):raise AssertionError('Old parent estimate detaches coarse')
    model.frozen.estimate=forbidden
    output=model(packets());output.square().mean().backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.frozen.coarse.parameters())


@pytest.mark.parametrize('candidate',['prior_only','joint_coarse'])
def test_single_packet_and_eval_have_no_cross_packet_or_state_update(candidate):
    model=build(candidate).eval();x=packets(3);before=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        whole=model(x);single=torch.cat([model(v[None]) for v in x])
    assert torch.allclose(whole,single,atol=2e-6,rtol=1e-5)
    assert all(torch.equal(v,model.state_dict()[key]) for key,v in before.items())


def test_optimizer_groups_cover_all_and_only_trainable_parameters_with_fixed_LR_ratio():
    t=module('training')
    for candidate in ('prior_only','joint_coarse'):
        model=build(candidate);opt=t.build_optimizer(model)
        actual=[id(p) for group in opt.param_groups for p in group['params']]
        assert len(actual)==len(set(actual))
        assert set(actual)=={id(p) for p in model.parameters() if p.requires_grad}
        for update in (0,246,247,4939):
            t.set_learning_rate(opt,update)
            groups={g['name']:g for g in opt.param_groups}
            assert groups['prior']['lr']==t.learning_rate(update)
            if candidate=='joint_coarse':assert groups['coarse']['lr']==pytest.approx(.1*groups['prior']['lr'])
            else:assert set(groups)=={'prior'}
    with pytest.raises(ValueError):t.learning_rate(4940)


def test_clean_target_is_detached_and_loss_is_exact_original_IQ_NMSE():
    target=packets().requires_grad_();estimate=(packets()+.03).requires_grad_()
    value,metrics=module('training').iq_loss(estimate,target)
    expected=(estimate-target.detach()).square().mean((1,2))/target.detach().square().mean((1,2)).clamp_min(1e-4)
    assert torch.equal(value,expected) and set(metrics)=={'iq_nmse'}
    value.mean().backward();assert target.grad is None and torch.isfinite(estimate.grad).all()


@pytest.mark.parametrize('bad',[torch.ones(2,256),torch.ones(2,2,128),torch.ones(2,2,256,dtype=torch.int64),torch.full((2,2,256),float('nan'))])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):build('joint_coarse')(bad)


def test_unknown_candidate_rejected():
    with pytest.raises(ValueError):build('unknown')


def payload(candidate='joint_coarse'):
    d=module('design')
    from experiments.pamf_uncertainty_restore import design as parent
    return dict(schema='joint_coarse_checkpoint_v1',config=d.configuration(),kind=d.KIND,candidate=candidate,
        prior_initialization='scratch',inherited_prior_epochs=0,training_TX_labels_read=0,training_F_access=False,target_training=False,
        epoch=20,updates=4940,component_updates=module('runtime').update_counts(candidate,4940),
        parent=dict(coarse_run=parent.COARSE_RUN,coarse_commit=parent.COARSE_COMMIT,action_run=parent.ACTION_RUN,
            action_commit=parent.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False))


@pytest.mark.parametrize('mutation',['schema','TX','F','target','parent','budget','coarse','estimator'])
def test_checkpoint_rejects_wrong_permissions_inheritance_or_updates(mutation):
    r=module('runtime');value=payload();r.validate_payload(value)
    if mutation=='schema':value['schema']='analytic_unrolled_checkpoint_v1'
    elif mutation=='TX':value['training_TX_labels_read']=1
    elif mutation=='F':value['training_F_access']=True
    elif mutation=='target':value['target_training']=True
    elif mutation=='parent':value['parent']['action_commit']='unknown'
    elif mutation=='budget':value['updates']=4941
    elif mutation=='coarse':value['component_updates']['coarse_updates']=0
    elif mutation=='estimator':value['component_updates']['action_estimator_updates']=1
    with pytest.raises(ValueError):r.validate_payload(value)


@pytest.mark.parametrize('candidate',['prior_only','joint_coarse'])
def test_real_architecture_checkpoint_roundtrip_and_component_flags(tmp_path,candidate):
    from experiments.pamf_action_restore.models import ActionRestorer
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    r=module('runtime');m=module('models');torch.manual_seed(27)
    model=m.JointCoarseRestorer(candidate,ActionRestorer(ComplexTFUNet())).eval();path=tmp_path/'checkpoint.pt'
    r.save_checkpoint(path,model,None,0,0,payload(candidate)['parent']);fresh,value=r.load_checkpoint(path,'cpu')
    assert value['component_updates']==dict(prior_updates=0,coarse_updates=0,action_estimator_updates=0)
    assert all(p.requires_grad==(candidate=='joint_coarse') for p in fresh.frozen.coarse.parameters())
    assert not any(p.requires_grad for p in fresh.frozen.estimator.parameters())
    with torch.no_grad():assert torch.equal(model(packets()),fresh(packets()))


@pytest.mark.parametrize('decision,row',[
    (dict(status='SOURCE_DECISION_FROZEN',qualified=False,selected=None,target_rows=[]),'prior_only'),
    (dict(status='SOURCE_DECISION_FROZEN',qualified=True,selected='joint_coarse',target_rows=['joint_coarse']),'prior_only'),
    (dict(status='SOURCE_DECISION_FROZEN',qualified=True,selected='joint_coarse',target_rows=['joint_coarse','prior_only']),'joint_coarse'),
])
def test_target_is_rejected_before_any_target_path_access(tmp_path,decision,row):
    import json
    (tmp_path/'source_decision.json').write_text(json.dumps(decision),encoding='utf-8')
    args=SimpleNamespace(output=tmp_path,row=row,formal_reference=tmp_path/'must_not_be_read')
    with pytest.raises(ValueError):module('evaluation').prepare_target(args)
    assert not (tmp_path/row).exists()


def test_original_five_source_guards_and_no_analytic_calibration_are_preserved():
    d=module('design');cfg=d.configuration()
    from experiments.pamf_analytic_unrolled_restore import design as old
    assert cfg['source_thresholds']==old.configuration()['source_thresholds']
    assert cfg['training_F_access'] is False and cfg['target_training'] is False
    assert cfg['calibration_count']==0 and cfg['coarse_LR_multiplier']==.1
    assert set(cfg['rows'])=={'prior_only','joint_coarse'}


def test_controller_waits_for_exact_previous_owned_process_and_ignores_other_owner(tmp_path):
    import json
    c=module('controller');d=module('design');root=tmp_path/d.PRECEDING;root.mkdir();proc=tmp_path/'proc';proc.mkdir()
    (root/'launch.json').write_text(json.dumps(dict(commit=d.PRECEDING_COMMIT,pid=123)),encoding='utf-8')
    (root/'complete.json').write_text(json.dumps(dict(status='ANALYZED')),encoding='utf-8')
    (root/'workers.json').write_text(json.dumps(dict(rows=[])),encoding='utf-8')
    process=proc/'123';process.mkdir();(process/'status').write_text('State:\tS (sleeping)\n')
    (process/'cmdline').write_bytes(('python\0-m\0experiments.pamf_analytic_unrolled_restore.runner\0--output\0'+str(root)).encode())
    with pytest.raises(ValueError):c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc_root=proc)
    (process/'cmdline').write_bytes(b'python\0other-owner-job')
    c.require_preceding_finished(SimpleNamespace(preceding_run=root),proc_root=proc)


def test_real_64_tap_estimator_and_complex_CG_return_finite_coarse_gradients():
    from experiments.pamf_action_restore.models import ActionRestorer
    from experiments.pamf_reception_restore.models import ComplexTFUNet
    torch.manual_seed(23)
    model=module('models').JointCoarseRestorer('joint_coarse',ActionRestorer(ComplexTFUNet())).train()
    # Nonzero fixed head weights exercise the estimator input derivative,
    # including its actual 64-tap action and complex CG, without trained F.
    with torch.no_grad():
        model.frozen.estimator.global_head.weight.uniform_(-.001,.001)
        model.frozen.estimator.core.out.weight.uniform_(-.001,.001)
    stages,_,action,base=model.stage_outputs(packets());assert action.kernel.shape[-1]==64
    assert action.kernel.requires_grad
    loss,_=module('training').iq_loss(stages[-1],.95*packets()+.03);loss.mean().backward()
    gradients=[p.grad for p in model.frozen.coarse.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and sum(float(g.norm()) for g in gradients)>0
    assert not any(p.grad is not None for p in model.frozen.estimator.parameters())
