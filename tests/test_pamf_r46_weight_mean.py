import copy
import json
import pytest
import torch
from torch import nn
from experiments.pamf_r46_weight_mean import design as d
from experiments.pamf_r46_weight_mean.averaging import SourceParameterMeans
from experiments.pamf_r46_weight_mean.models import CompactSuccessor,source_from_parent
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_anchor_refine.models import compact_from_parent as static_prune


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.left=nn.Linear(2,2);self.right=nn.Linear(2,2)
        self.right.weight=self.left.weight;self.register_buffer('fixed',torch.tensor([2,3],dtype=torch.int64))


def path(model,means,*,vary=False):
    expected=None
    for update in range(1,4941):
        if vary:
            with torch.no_grad():
                for p in model.parameters():p.fill_(update*1e-5)
        means.after_step(model,update)
        if update>=2471:
            value=torch.tensor(update*1e-5)
            expected=value.clone() if expected is None else expected.lerp(value,1-d.EMA_DECAY)
        if update%247==0:means.after_epoch(model,update//247)
    return expected


def test_scalar_closed_form_mean_ema_aliases_and_unchanged_student():
    model=Tiny();means=SourceParameterMeans(model);expected=path(model,means,vary=True)
    before=copy.deepcopy(model.state_dict())
    for readout,want in [('final',torch.tensor(.0494)),('late_mean',torch.tensor(18*247*1e-5)),('step_ema',expected)]:
        state=means.state_for(model,readout)
        assert torch.allclose(state['left.weight'],torch.full_like(state['left.weight'],want),atol=1e-7,rtol=1e-6)
        assert torch.equal(state['left.weight'],state['right.weight'])
        assert torch.equal(state['fixed'],before['fixed'])
        fresh=Tiny();fresh.load_state_dict(state,strict=True)
        assert fresh.left.weight is fresh.right.weight
        assert means.metadata(readout)['shared_source_updates']==4940
    assert all(torch.equal(v,before[n]) for n,v in model.state_dict().items())
    assert means.ema_count==2470 and means.mean_count==5


def test_constant_trajectory_is_exact_for_all_three_readouts():
    model=Tiny();before=copy.deepcopy(model.state_dict());means=SourceParameterMeans(model);path(model,means)
    for arm in d.ARMS:
        got=means.state_for(model,arm)
        assert all(torch.equal(v,before[n]) for n,v in got.items())


@pytest.mark.parametrize('readout',d.ARMS)
def test_incomplete_path_never_exports_source_candidate(readout):
    model=Tiny();means=SourceParameterMeans(model)
    with pytest.raises(ValueError):means.state_for(model,readout)


@pytest.mark.parametrize('update',[0,2,2471,4941,True,1.0])
def test_skipped_duplicated_or_noninteger_updates_rejected(update):
    model=Tiny();means=SourceParameterMeans(model)
    with pytest.raises(ValueError):means.after_step(model,update)


def test_duplicate_step_and_partial_epoch_rejected():
    model=Tiny();means=SourceParameterMeans(model);means.after_step(model,1)
    with pytest.raises(ValueError):means.after_step(model,1)
    with pytest.raises(ValueError):means.after_epoch(model,1)
    with pytest.raises(ValueError):means.after_step(Tiny(),2)


def test_buffer_updates_and_nonfinite_parameters_reject_final_export():
    model=Tiny();means=SourceParameterMeans(model);path(model,means)
    model.fixed[0]=9
    with pytest.raises(ValueError,match='fixed buffers'):means.state_for(model,'late_mean')
    model.fixed[0]=2
    with torch.no_grad():model.left.weight.fill_(float('nan'))
    with pytest.raises(ValueError,match='Finite'):means.state_for(model,'final')


def test_replaced_graph_or_precision_rejected():
    with pytest.raises(ValueError):SourceParameterMeans(Tiny().double())
    model=Tiny();means=SourceParameterMeans(model);model.left.bias=nn.Parameter(model.left.bias.clone())
    with pytest.raises(ValueError):means.after_step(model,1)


def test_exact_existing_static_compression_and_no_added_inference_state():
    torch.set_num_threads(2);torch.manual_seed(66);parent=Parent('prune_refine').eval().requires_grad_(False)
    before=copy.deepcopy(parent.state_dict());source=source_from_parent(parent);control=static_prune(parent,'prune_decay')
    assert sum(p.numel() for p in source.parameters())==273765
    assert sum(p.numel() for p in source.frozen.parameters())==185615
    assert all(torch.equal(v,control.state_dict()[n]) for n,v in source.state_dict().items())
    for arm in d.ARMS:
        final=CompactSuccessor(arm);final.load_state_dict(source.state_dict(),strict=True)
        assert sum(p.numel() for p in final.parameters())==273765
        assert final.prior[0].encoder.blocks[0][0].net[0].weight is final.prior[2].encoder.blocks[0][0].net[0].weight
        assert final.prior[0].encoder.blocks[0][0].net[0].bias is not final.prior[1].encoder.blocks[0][0].net[0].bias
    assert all(torch.equal(v,before[n]) for n,v in parent.state_dict().items())


def test_three_readouts_share_one_budget_and_fixed_source_permissions():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert cfg['source_training_paths']==1 and cfg['shared_source_epochs']==20 and cfg['shared_source_updates']==4940
    assert cfg['source_IQ_examples_total']==cfg['teacher_IQ_examples_total']==1260000
    assert cfg['L_exposures_total']==126000 and cfg['U_exposures_total']==1134000
    assert cfg['new_calibration_IQ']==cfg['new_trainable_parameters']==cfg['extra_neural_forwards']==0
    assert not any(cfg[k] for k in ('F_training_access','V_training','target_training','source_score_selection','target_feedback'))
    assert d.EMA_DECAY==.99 and d.MEAN_EPOCHS==(16,17,18,19,20)
