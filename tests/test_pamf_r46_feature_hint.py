import copy
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as HumanParent
from experiments.pamf_r46_feature_hint import design as d
from experiments.pamf_r46_feature_hint.models import CompactSuccessor,compact_from_parent,objective
from experiments.pamf_r46_feature_hint.hints import FeatureHints

torch.set_num_threads(2)

def pair(arm='all_hint'):
    torch.manual_seed(60)
    teacher=HumanParent('prune_refine').eval().requires_grad_(False)
    child=compact_from_parent(teacher,arm)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in values] for k,values in child.initialization_report['channel_maps'].items()}
    return teacher,child,maps

def forward_pair(hints,teacher,child,x):
    hints.begin()
    with torch.no_grad():wanted=teacher.stage_outputs(x)[0]
    hints.teacher_done();actual=child.stage_outputs(x)[0]
    return wanted,actual,hints.finish()

@pytest.mark.parametrize('arm',d.ARMS)
def test_existing_pruned_graph_and_joint_gradient_without_teacher_update(arm):
    teacher,child,maps=pair(arm);x=torch.randn(2,2,256)*.1
    original=copy.deepcopy(teacher.state_dict());keys=tuple(child.state_dict())
    with FeatureHints(teacher,child,maps,arm) as hints:
        assert len(hints.links)==(84 if arm=='context_hint' else 168)
        wanted,actual,features=forward_pair(hints,teacher,child,x)
        loss,terms=objective(actual,wanted,x,1,arm,features)
        assert loss.shape==(2,) and torch.isfinite(loss).all()
        assert terms['teacher_context_feature_nmse'].min()>0
        assert bool(terms['teacher_prior_feature_nmse'].count_nonzero())==(arm=='all_hint')
        loss.mean().backward()
        for module in (child.frozen.coarse,child.frozen.estimator,*child.prior):
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
        assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v,original[k]) for k,v in teacher.state_dict().items())
    assert sum(p.numel() for p in child.parameters())==273765 and tuple(child.state_dict())==keys
    assert all(not m._forward_hooks for model in (teacher,child) for m in model.modules())
    assert child.initialization_report['IQ_examples']==child.initialization_report['new_calibration_IQ']==0

@pytest.mark.parametrize('arm',d.ARMS)
def test_per_IQ_losses_are_batch_independent_and_hooks_do_not_change_outputs(arm):
    teacher,child,maps=pair(arm);child.eval();x=torch.randn(2,2,256)*.1
    with torch.no_grad():plain=child(x)
    with FeatureHints(teacher,child,maps,arm) as hints:
        with torch.no_grad():
            _,actual,together=forward_pair(hints,teacher,child,x)
            separate=[forward_pair(hints,teacher,child,v[None])[2] for v in x]
        assert torch.equal(plain,actual[-1])
        for key in together:
            assert torch.allclose(together[key],torch.cat([v[key] for v in separate]),rtol=1e-5,atol=1e-7)

@pytest.mark.parametrize('fault',['student_first','repeat_teacher','missing_teacher','missing_student','unfinished_pair','closed'])
def test_incomplete_or_out_of_order_pairs_reject_and_close_hooks(fault):
    teacher,child,maps=pair();x=torch.randn(1,2,256)*.1
    hints=FeatureHints(teacher,child,maps,'all_hint')
    try:
        with pytest.raises(ValueError):
            hints.begin()
            if fault=='student_first':child(x)
            elif fault=='repeat_teacher':teacher(x);teacher(x)
            elif fault=='missing_teacher':hints.teacher_done()
            elif fault=='unfinished_pair':hints.begin()
            elif fault=='closed':hints.close();hints.finish()
            else:teacher(x);hints.teacher_done();hints.finish()
    finally:hints.close()
    assert all(not m._forward_hooks for model in (teacher,child) for m in model.modules())

@pytest.mark.parametrize('fault',['unknown','train_mode','trainable_teacher'])
def test_bad_scope_or_teacher_contract_reject_before_hook_registration(fault):
    teacher,child,maps=pair();arm='all_hint'
    if fault=='unknown':arm='target_selected'
    elif fault=='train_mode':teacher.train()
    else:next(teacher.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):FeatureHints(teacher,child,maps,arm)
    assert all(not m._forward_hooks for model in (teacher,child) for m in model.modules())

def test_fixed_weight_closes_and_prior_addition_keeps_same_context_coefficient():
    x=torch.randn(2,2,256)*.1;outputs=[x+.01,x+.02,x+.03];wanted=[x,x,x]
    context=torch.tensor([2.,3.],requires_grad=True);prior=torch.tensor([5.,7.],requires_grad=True)
    zero=torch.zeros_like(context)
    a,ta=objective(outputs,wanted,x,10,'context_hint',dict(teacher_context_feature_nmse=context,teacher_prior_feature_nmse=zero))
    b,tb=objective(outputs,wanted,x,10,'all_hint',dict(teacher_context_feature_nmse=context,teacher_prior_feature_nmse=prior))
    assert torch.allclose(b-a,.02*prior)
    assert torch.allclose(a,ta['iq_nmse']+.05*ta['spectral_error']+d.teacher_weight(10,'context_hint')*ta['teacher_stage_nmse']+.02*context)
    b.sum().backward();assert torch.allclose(context.grad,torch.full_like(context,.02)) and torch.allclose(prior.grad,torch.full_like(prior,.02))

@pytest.mark.parametrize('fault',['scope','keys','shape','prior_in_context'])
def test_unregistered_loss_components_reject(fault):
    x=torch.zeros(2,2,256);features=dict(teacher_context_feature_nmse=torch.ones(2),teacher_prior_feature_nmse=torch.zeros(2));arm='context_hint'
    if fault=='scope':arm='target_selected'
    elif fault=='keys':features['F_embedding']=torch.ones(2)
    elif fault=='shape':features['teacher_context_feature_nmse']=torch.ones(1)
    else:features['teacher_prior_feature_nmse']=torch.ones(2)
    with pytest.raises(ValueError):objective([x]*3,[x]*3,x,1,arm,features)

def test_matching_widths_state_keys_and_fixed_source_only_plan():
    config=d.configuration();a=CompactSuccessor('context_hint');b=CompactSuccessor('all_hint')
    assert tuple(a.state_dict())==tuple(b.state_dict()) and sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==273765
    assert config['hint']['context_modules']==config['hint']['prior_modules']==84
    assert config['hint']['new_parameters']==config['hint']['extra_teacher_or_student_G_forwards']==config['hint']['extra_calibration_IQ']==0
    assert config['training']['same_context_weight_both_rows'] and config['training']['source_clean_pairs']
    assert all(config['training'][k]==0 for k in ('TX_labels_read','F_weights_read','F_outputs_read','V_reads','target_reads'))
    assert config['evaluation']['target168000x7'] and config['evaluation']['all_rows_regardless_source']
    assert config['prototype_only'] and not config['full_pipeline_implemented']
