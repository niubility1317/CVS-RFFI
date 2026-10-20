"""Synthetic algebra/protocol checks; no source IQ, labels or recognition claims."""
import copy
import json
import pytest
import torch
from torch import nn
from experiments.pamf_r46_group_fold import design as d,models as m
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_group_fold import calibration as c

torch.set_num_threads(2)


def inputs(correlation=0.):
    torch.manual_seed(64);parent=Parent('prune_refine').eval().requires_grad_(False)
    model=m.group.compact_from_parent(parent,'group_preserve')
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in levels] for k,levels in model.initialization_report['channel_maps'].items()}
    groups=[g for g in m.grouped_links(m.context_links(parent,model,maps)) if len(g[0].inputs)<g[0].source.weight.shape[1]]
    moments={}
    for group in groups:
        link=group[0];n=link.source.weight.shape[1];count=256*len(group)*(1 if isinstance(link.source,nn.Linear) else d.CAL_POSITIONS)
        gram=torch.eye(n,dtype=torch.float64)+float(correlation)*torch.ones(n,n,dtype=torch.float64)
        moments[link.name]=dict(members=[v.name for v in group],second_moment=gram.tolist(),sampled_channel_vectors=count)
    calibration=dict(c.metadata(),moments=moments)
    return parent,calibration,moments


def test_fixed_same_graph_full_prior_source_only_design():
    cfg=d.configuration();assert json.loads(json.dumps(cfg))==cfg
    assert d.FOLD_STRENGTH==dict(fold_full=1.,fold_half=.5)
    assert all(v==248828 for v in cfg['parameters'].values())
    assert set(cfg['context_widths'])==set(d.ARMS) and all(v==3 for v in cfg['context_GN_groups'].values())
    assert cfg['new_calibration_IQ']==cfg['new_G_forwards']==256 and cfg['new_trainable_parameters']==0
    assert cfg['shared_calibration']['parent_G_inputs']==256 and cfg['matched_control']['row']=='group_preserve'


@pytest.mark.parametrize('arm',d.ARMS)
def test_diagonal_second_moments_reduce_exactly_to_R62_weight_only(arm):
    parent,calibration,moments=inputs();reference=m.group.compact_from_parent(parent,'group_preserve')
    model=m.compact_from_parent(parent,arm,calibration)
    assert all(torch.equal(value,reference.state_dict()[name]) for name,value in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters())==248828
    assert {v.num_groups for v in model.frozen.modules() if isinstance(v,nn.GroupNorm)}=={3}
    assert {v.num_groups for v in model.prior.modules() if isinstance(v,nn.GroupNorm)}=={4}


def test_half_strength_is_midpoint_and_only_context_weights_change():
    parent,calibration,moments=inputs(.2);before=copy.deepcopy(parent.state_dict());stats=copy.deepcopy((calibration,moments))
    control=m.group.compact_from_parent(parent,'group_preserve');full=m.compact_from_parent(parent,'fold_full',calibration)
    half=m.compact_from_parent(parent,'fold_half',calibration)
    for name,value in control.state_dict().items():
        a=full.state_dict()[name];b=half.state_dict()[name]
        assert torch.allclose(b,(value+a)*.5,rtol=1e-6,atol=1e-7)
        if name.startswith('prior.') or name.endswith('.bias') or '.net.1.' in name or '.net.4.' in name:assert torch.equal(a,value)
    assert any(not torch.equal(v,control.state_dict()[k]) for k,v in full.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items()) and (calibration,moments)==stats
    report=full.initialization_report
    assert report['prior_copied_exactly'] and report['retained_normalization_partition_preserved']
    assert report['folded_unique_weights']>0 and all(not name.startswith('prior') for name in report['projection_fits'])
    assert all(v['reconstructed_channel_error']<=v['uncorrected_channel_error'] for v in report['projection_fits'].values())
    with torch.no_grad():assert torch.isfinite(full(torch.randn(2,2,256)*.1)).all()


@pytest.mark.parametrize('field,value',[('F_reads',1),('TX_labels_read',1),('V_reads',1),('target_reads',1),('clean_IQ_reads',1),('parent_commit','wrong'),('source_unique_IQ',64)])
def test_reused_source_permission_and_parent_mismatch_fail(field,value):
    parent,calibration,moments=inputs();calibration[field]=value
    with pytest.raises(ValueError):m.compact_from_parent(parent,'fold_full',calibration)


@pytest.mark.parametrize('bad',['missing','exposure','nonsymmetric','nonfinite','dimension'])
def test_original_moment_contract_and_actual_input_algebra_are_required(bad):
    parent,calibration,moments=inputs();name=next(k for k in moments if k.startswith('coarse_spectral'))
    if bad=='missing':moments.pop(name)
    elif bad=='exposure':moments[name]['sampled_channel_vectors']=1
    elif bad=='nonsymmetric':moments[name]['second_moment'][0][1]=1.
    elif bad=='nonfinite':moments[name]['second_moment'][0][0]=float('nan')
    elif bad=='dimension':moments[name]['second_moment']=[[1.]]
    with pytest.raises(ValueError):m.compact_from_parent(parent,'fold_full',calibration)


def test_unregistered_arm_rejects_before_statistics_access():
    with pytest.raises(ValueError):m.compact_from_parent(None,'unknown',None)


def test_all103_moments_observed_from_frozen_parent_and_hooks_removed():
    parent,_,_=inputs();before=copy.deepcopy(parent.state_dict())
    value=c.collect(parent,[torch.randn(2,2,256)*.1],batch=1)
    assert value['frozen_parent_IQ_inputs']==2 and len(value['moments'])==103
    for name,item in value['moments'].items():
        assert item['sampled_channel_vectors']==(2 if name=='action.global_head' else 32)
        gram=torch.tensor(item['second_moment']);assert torch.isfinite(gram).all() and torch.allclose(gram,gram.T)
    assert all(not module._forward_pre_hooks for module in parent.modules())
    assert all(torch.equal(v,before[k]) for k,v in parent.state_dict().items()) and all(p.grad is None for p in parent.parameters())
