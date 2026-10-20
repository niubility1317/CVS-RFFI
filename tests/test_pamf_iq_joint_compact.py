import copy
import torch
from torch import nn
import pytest
from experiments.pamf_iq_joint_compact import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,channel_indices,transplant_core,compact_from_teacher,objective,convolution_linear_macs
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet

torch.set_num_threads(2)


@pytest.mark.parametrize('arm',d.ARMS)
def test_all_components_are_compact_and_jointly_trainable_with_shared_long_only(arm):
    model=JointCompactRestorer(arm)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    assert all(p.requires_grad for p in model.parameters())
    maps=[dict(m.named_modules()) for m in model.prior]
    for name,m in maps[0].items():
        if isinstance(m,nn.Conv1d) and m.kernel_size[0]>1:
            assert maps[0][name].weight is maps[1][name].weight is maps[2][name].weight
            assert maps[0][name].bias is not maps[1][name].bias
        elif isinstance(m,nn.GroupNorm):assert maps[0][name].weight is not maps[1][name].weight


def test_channels_preserve_all_four_group_partitions():
    assert channel_indices(16,8).tolist()==[0,1,4,5,8,9,12,13]


@pytest.mark.parametrize('old,new',[(16,0),(16,9),(12,16),(15,8),(True,4)])
def test_illegal_channel_mapping_rejected(old,new):
    with pytest.raises(ValueError):channel_indices(old,new)


def test_decoder_input_channel_maps_keep_skip_after_original_full_width():
    source=Core(incoming=9);target=Core(incoming=9,widths=(16,32,48,64))
    for block in source.decode:
        with torch.no_grad():
            weight=block[0].net[0].weight
            weight.copy_(torch.arange(weight.shape[1])[None,:,None].expand_as(weight))
    maps=transplant_core(source,target)
    for i,block in enumerate(target.decode):
        level=3-i;old=(32,64,96,128)
        expected=torch.cat((maps[level],old[level]+maps[level-1]))
        assert torch.equal(block[0].net[0].weight[0,:,0],expected.float())


@pytest.mark.parametrize('arm',d.ARMS)
def test_teacher_is_unchanged_and_joint_loss_reaches_coarse_action_and_prior(arm):
    torch.manual_seed(2701)
    teacher=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for name,m in teacher.named_modules():
            if isinstance(m,nn.Conv1d) and (name.endswith('.out')):m.weight.normal_(0,.002)
        teacher.frozen.estimator.global_head.weight.normal_(0,.0002)
    before={k:v.clone() for k,v in teacher.state_dict().items()}
    model=compact_from_teacher(teacher,arm)
    x=torch.randn(2,2,256)*.1;r=.8*x+torch.randn_like(x)*.01
    with torch.no_grad():desired=teacher.stage_outputs(r)[0]
    values=model.stage_outputs(r)[0];loss,terms=objective(values,desired,x,1);loss.mean().backward()
    for part in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads)>0
    assert all(torch.equal(v,before[k]) for k,v in teacher.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    assert model(r).shape==r.shape and torch.isfinite(model(r)).all()
    assert model.initialization_report['F_reads']==model.initialization_report['IQ_examples']==0


def test_paired_clean_reconstruction_is_optimized_and_teacher_is_detached():
    clean=torch.ones(2,2,256,requires_grad=True)
    output=[torch.zeros_like(clean,requires_grad=True) for _ in range(3)]
    teacher=[torch.zeros_like(clean,requires_grad=True) for _ in range(3)]
    loss,terms=objective(output,teacher,clean,20)
    assert torch.equal(loss,torch.ones(2))
    loss.sum().backward()
    assert output[-1].grad.abs().sum()>0
    assert all(v.grad is None for v in teacher) and clean.grad is None
    assert d.teacher_weight(1)==10 and d.teacher_weight(20)==1


@pytest.mark.parametrize('epoch',[0,21,1.0,True])
def test_unregistered_weight_schedule_rejected(epoch):
    with pytest.raises(ValueError):d.teacher_weight(epoch)


def test_source_contract_is_F_free_and_both_new_versions_always_formally_tested():
    config=d.configuration()
    assert config['F_weights_read']==config['F_outputs_read']==config['TX_labels_read']==0
    assert not config['V_training'] and not config['target_training'] and not config['source_performance_gate']
    assert config['target_acceptance_allowance_pp']==1 and config['target_count_per_view']==168000
    assert len(config['views'])==7 and len(config['arms'])==2
    assert config['context'].startswith('Jointly train') and config['paired_clean_optimized']


def test_MAC_counter_includes_actual_convolution_and_linear_calls_and_removes_hooks():
    model=nn.Sequential(nn.Conv1d(2,3,3,padding=1),nn.Flatten(),nn.Linear(24,4))
    assert convolution_linear_macs(model,torch.ones(1,2,8))==144+96
    assert all(not m._forward_hooks for m in model.modules())
