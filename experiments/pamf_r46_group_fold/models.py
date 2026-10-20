"""Reconstruct dropped context inputs inside existing dense weights, preserving GN3."""
import torch
from torch import nn
from . import design as d
from experiments.pamf_r46_group_drop import models as group
from experiments.pamf_r46_fold_compensate.compensation import Link,grouped_links,reconstruction,folded_weight
from experiments.pamf_r46_multipoint_fast.models import objective,spectral_error


def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered complete-group source-fold row required')
    model=group.CompactSuccessor('group_preserve');model.arm=arm;return model


def core_links(source,target,maps,prefix):
    old=[b[0].net[0].out_channels for b in source.encoder.blocks]
    new=[b[0].net[0].out_channels for b in target.encoder.blocks]
    if len(maps)!=len(old):raise ValueError('All four whole-group maps required')
    for keep,before,after in zip(maps,old,new):
        size=before//4
        if (before%4 or after!=3*size or keep.dtype!=torch.int64 or keep.device.type!='cpu' or keep.shape!=(after,)
                or keep.tolist()!=sorted(set(keep.tolist())) or (keep<0).any() or (keep>=before).any()
                or sorted(int(((keep>=g*size)&(keep<(g+1)*size)).sum()) for g in range(4))!=[0,size,size,size]):
            raise ValueError('Exactly three intact original groups required')
    links=[]
    def block(src,dst,inputs,outputs,name):
        for index,selected in ((0,inputs),(3,outputs)):
            links.append(Link(name+f'.net.{index}',src.net[index],dst.net[index],selected,outputs))
        if isinstance(src.skip,nn.Conv1d) and isinstance(dst.skip,nn.Conv1d):links.append(Link(name+'.skip',src.skip,dst.skip,inputs,outputs))
        elif not isinstance(src.skip,nn.Identity) or not isinstance(dst.skip,nn.Identity):raise ValueError('Residual topology differs')
    inputs=torch.arange(source.encoder.blocks[0][0].net[0].in_channels)
    for level,(src,dst,outputs) in enumerate(zip(source.encoder.blocks,target.encoder.blocks,maps)):
        block(src[0],dst[0],inputs,outputs,prefix+f'.encoder.blocks.{level}.0')
        block(src[1],dst[1],outputs,outputs,prefix+f'.encoder.blocks.{level}.1');inputs=outputs
    for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
        level=len(old)-1-index;outputs=maps[level-1];inputs=torch.cat((maps[level],old[level]+maps[level-1]))
        block(src[0],dst[0],inputs,outputs,prefix+f'.decode.{index}.0')
        block(src[1],dst[1],outputs,outputs,prefix+f'.decode.{index}.1')
    links.append(Link(prefix+'.out',source.out,target.out,maps[0],torch.arange(source.out.out_channels)))
    return links


def context_links(parent,model,maps):
    result=[]
    for name,src,dst in (
        ('coarse_spectral',parent.frozen.coarse.spectral,model.frozen.coarse.spectral),
        ('coarse_temporal',parent.frozen.coarse.temporal,model.frozen.coarse.temporal),
        ('action',parent.frozen.estimator.core,model.frozen.estimator.core)):
        result+=core_links(src,dst,maps[name],name)
    result.append(Link('action.global_head',parent.frozen.estimator.global_head,model.frozen.estimator.global_head,maps['action'][-1],torch.arange(129)))
    return result


@torch.no_grad()
def compact_from_parent(parent,arm,calibration):
    if arm not in d.ARMS:raise ValueError('Registered source-fold row required before data access')
    from .calibration import validate_calibration
    validate_calibration(calibration);moments=calibration['moments']
    before={name:value.detach().clone() for name,value in parent.state_dict().items()}
    model=group.compact_from_parent(parent,'group_preserve')
    maps={name:[torch.tensor(v,dtype=torch.int64) for v in levels] for name,levels in model.initialization_report['channel_maps'].items()}
    links=[members for members in grouped_links(context_links(parent,model,maps)) if len(members[0].inputs)<members[0].source.weight.shape[1]];fits={}
    if {v[0].name for v in links}!=set(moments):raise ValueError('All complete context input moment groups required')
    for members in links:
        link=members[0]
        if moments[link.name]['members']!=[v.name for v in members]:raise ValueError('Context source module uses differ')
        gram=torch.tensor(moments[link.name]['second_moment'],dtype=torch.float64)
        if gram.shape!=(link.source.weight.shape[1],)*2:raise ValueError('Original full-parent input dimensions differ')
        projection,fit=reconstruction(gram,link.inputs,d.RIDGE_FRACTION)
        link.target.weight.copy_(folded_weight(link,projection,d.FOLD_STRENGTH[arm]))
        fits[link.name]=dict(members=[v.name for v in members],inputs=link.inputs.tolist(),outputs=link.outputs.tolist(),**fit)
    if not fits or sum(p.numel() for p in model.parameters())!=d.PARAMETERS[arm]:raise ValueError('Complete context-only folding required')
    if any(not torch.equal(v,before[k]) for k,v in parent.state_dict().items()):raise ValueError('Designated teacher changed')
    if any(not torch.equal(v,parent.prior.state_dict()[k]) for k,v in model.prior.state_dict().items()):raise ValueError('Full prior must remain exactly copied at initialization')
    model.arm=arm
    model.initialization_report.update(mode='whole_context_group_source_moment_fold',fold_strength=d.FOLD_STRENGTH[arm],
        context_group_count=3,prior_group_count=4,prior_copied_exactly=True,retained_normalization_partition_preserved=True,
        IQ_examples=256,source_unique_IQ=128,calibration_L_exposures=128,calibration_U_exposures=128,calibration_seed=d.CAL_SEED,
        calibration_shared_once=True,shared_calibration_G_inputs=256,initialization_G_forwards=0,new_trainable_parameters=0,no_intercept=True,ridge_fraction=d.RIDGE_FRACTION,
        folded_unique_weights=len(fits),folded_module_uses=sum(len(v['members']) for v in fits.values()),projection_fits=fits)
    return model
