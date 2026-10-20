import copy
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_anchor_refine.models import compact_from_parent as static_prune
from experiments.pamf_r46_delayed_prune import design as d
from experiments.pamf_r46_delayed_prune.models import CompactSuccessor,dense_from_parent,fixed_maps,transition,validate_maps


@pytest.fixture
def parent():
    torch.set_num_threads(2);torch.manual_seed(65)
    return Parent('prune_refine').cpu().eval()


def test_fixed_budget_and_source_permissions():
    cfg=d.configuration()
    for arm in d.ARMS:
        assert len(cfg['schedule'][arm])==20
        assert cfg['schedule'][arm].count('dense')==d.WARMUP_EPOCHS[arm]
        assert cfg['schedule'][arm].count('compact')==20-d.WARMUP_EPOCHS[arm]
    assert cfg['new_total_epochs']==20 and d.UPDATES==4940 and d.MICRO==d.BATCH==256
    assert cfg['new_calibration_IQ']==cfg['new_trainable_parameters']==cfg['TX_labels_read']==0
    assert not any(cfg[k] for k in ('F_training_access','V_training','target_training','target_feedback'))
    assert cfg['teacher_IQ_examples_per_row']==cfg['source_IQ_examples_per_row']==1260000


@pytest.mark.parametrize('arm',d.ARMS)
def test_dense_exact_parent_and_compact_exact_static_at_zero_updates(parent,arm):
    before={k:v.clone() for k,v in parent.state_dict().items()}
    model=dense_from_parent(parent,arm)
    assert sum(p.numel() for p in model.parameters())==350521
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    final=transition(model,d.WARMUP_EPOCHS[arm]);control=static_prune(parent,'prune_decay')
    assert sum(p.numel() for p in final.parameters())==273765
    assert sum(p.numel() for p in final.frozen.parameters())==185615
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in final.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items())
    assert final.initialization_report['channel_maps']==model.initialization_report['channel_maps']
    assert all(p.requires_grad for p in final.parameters())
    # Final function, alias layout, and portable strict state do not need the warmup model.
    fresh=CompactSuccessor(arm);fresh.load_state_dict(final.state_dict(),strict=True)
    assert final.prior[0].encoder.blocks[0][0].net[0].weight is final.prior[2].encoder.blocks[0][0].net[0].weight
    with torch.no_grad():
        x=torch.randn(1,2,256)
        assert torch.equal(model(x),parent(x))
        assert torch.equal(final(x),control(x))
        assert torch.equal(fresh(x),final(x))


def test_transition_uses_trained_student_weights_without_rechoosing_maps(parent):
    model=dense_from_parent(parent,d.ARMS[0]);original=copy.deepcopy(model.initialization_report['channel_maps'])
    with torch.no_grad():
        for p in model.parameters():p.add_(.003)
    final=transition(model,5)
    selected=torch.tensor(original['action'][-1])
    assert torch.equal(final.frozen.estimator.global_head.weight,model.frozen.estimator.global_head.weight[:,selected])
    assert torch.equal(final.frozen.estimator.global_head.bias,model.frozen.estimator.global_head.bias)
    assert final.initialization_report['channel_maps']==original
    assert final.initialization_report['teacher']=='original_M35_only'
    assert final.initialization_report['optimizer_resets']==1
    assert not final.initialization_report['global_schedule_restarted']


@pytest.mark.parametrize('epoch',[0,1,4,6,10,True,5.0])
def test_transition_rejects_wrong_epoch_before_weight_access(parent,epoch):
    model=dense_from_parent(parent,d.ARMS[0])
    with pytest.raises(ValueError):transition(model,epoch)


def test_no_second_transition(parent):
    final=transition(dense_from_parent(parent,d.ARMS[0]),5)
    with pytest.raises(ValueError):transition(final,5)


@pytest.mark.parametrize('bad',['unknown','R64',None])
def test_only_fixed_rows(parent,bad):
    with pytest.raises(ValueError):dense_from_parent(parent,bad)
    with pytest.raises(ValueError):CompactSuccessor(bad)


@pytest.mark.parametrize('bad',['duplicate','range','group','prior','missing','float'])
def test_map_negatives(parent,bad):
    maps=fixed_maps(parent);target=CompactSuccessor(d.ARMS[0])
    if bad=='duplicate':maps['action'][3][1]=maps['action'][3][0]
    elif bad=='range':maps['action'][3][-1]=10000
    elif bad=='group':maps['action'][3]=list(range(40))
    elif bad=='prior':maps['prior1'][3]=list(range(40))
    elif bad=='missing':maps.pop('action')
    elif bad=='float':maps['action'][3][0]=float(maps['action'][3][0])
    with pytest.raises(ValueError):validate_maps(parent,target,maps)


@pytest.mark.parametrize('epoch',[0,21,True,1.0])
def test_schedule_negatives(epoch):
    with pytest.raises(ValueError):d.phase(epoch,d.ARMS[0])


def test_rows_freeze_same_original_maps(parent):
    early=dense_from_parent(parent,d.ARMS[0]);late=dense_from_parent(parent,d.ARMS[1])
    assert early.initialization_report['channel_maps']==late.initialization_report['channel_maps']
    assert d.phase(5,d.ARMS[0])=='dense' and d.phase(6,d.ARMS[0])=='compact'
    assert d.phase(10,d.ARMS[1])=='dense' and d.phase(11,d.ARMS[1])=='compact'
