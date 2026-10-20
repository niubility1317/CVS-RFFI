"""Two weight-only compression operators on the fixed R38 restorer."""
import copy

import torch
from torch import nn

from . import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,_conv,_block,objective,convolution_linear_macs
from experiments.pamf_r17_compact.models import FactorConv
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r38_successor.models import CompactSuccessor as NarrowSkeleton


def factor_long_convolutions(model,*,initialize):
    """Decompose each unique shared weight once; keep each step's own bias."""
    cache={};plan=[];original_weights=[]
    def visit(module,prefix=''):
        for name,child in list(module.named_children()):
            path=prefix+name
            if isinstance(child,nn.Conv1d) and child.kernel_size[0]>1:
                rank=max(1,min(child.out_channels,child.in_channels*child.kernel_size[0])//2)
                if rank*(child.in_channels*child.kernel_size[0]+child.out_channels)>=child.weight.numel():continue
                key=id(child.weight)
                # Keep the original objects alive until traversal ends so an id
                # cannot be reused for an unrelated convolution later on.
                original_weights.append(child.weight)
                value=FactorConv.from_conv(child,.5,initialize=initialize and key not in cache)
                if key in cache:
                    value.left.weight,value.right.weight=cache[key]
                    if initialize and value.right.bias is not None:
                        with torch.no_grad():value.right.bias.copy_(child.bias)
                else:cache[key]=(value.left.weight,value.right.weight)
                setattr(module,name,value)
                plan.append(dict(path=path,input=child.in_channels,output=child.out_channels,kernel=child.kernel_size[0],rank=rank))
            else:visit(child,path+'.')
    visit(model)
    return plan


def important_channels(blocks,new):
    old=blocks[0].net[0].out_channels
    if not 0<new<=old or old%4 or new%4:raise ValueError('GN4-compatible pruning widths required')
    score=torch.zeros(old,dtype=torch.float64)
    for block in blocks:
        for index in (0,3):
            weight=block.net[index].weight.detach().double()
            part=weight.square().mean((1,2));score+=part/part.mean().clamp_min(1e-30)
    chosen=[]
    for group in range(4):
        start=group*(old//4)
        order=torch.argsort(score[start:start+old//4],descending=True,stable=True)[:new//4]
        chosen.append(order.sort().values+start)
    return torch.cat(chosen)


def prune_core(source,target):
    old=[blocks[0].net[0].out_channels for blocks in source.encoder.blocks]
    widths=[blocks[0].net[0].out_channels for blocks in target.encoder.blocks]
    maps=[important_channels(blocks,width) for blocks,width in zip(source.encoder.blocks,widths)]
    inputs=torch.arange(source.encoder.blocks[0][0].net[0].in_channels)
    for src,dst,outputs in zip(source.encoder.blocks,target.encoder.blocks,maps):
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs);inputs=outputs
    for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
        level=len(old)-1-index;outputs=maps[level-1]
        inputs=torch.cat((maps[level],old[level]+maps[level-1]))
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs)
    _conv(source.out,target.out,maps[0],torch.arange(source.out.out_channels))
    return maps


def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Only registered R38 compression methods allowed')
    if arm=='low_rank':
        model=JointCompactRestorer('dense_half');factor_long_convolutions(model,initialize=False)
    else:model=NarrowSkeleton('three_quarter_iq')
    model.arm=arm;model.requires_grad_(True);model.initialization_report=None
    return model


def compact_from_parent(parent,arm):
    if arm not in d.ARMS or getattr(parent,'arm',None)!='dense_half' or sum(p.numel() for p in parent.parameters())!=794529:
        raise ValueError('Only user-designated R38 794529 parent required')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Weight-only compression must initialize on CPU')
    if arm=='low_rank':
        model=copy.deepcopy(parent);plan=factor_long_convolutions(model,initialize=True)
        detail=dict(factored_convolutions=plan,unique_factored_weights=len({id(v.left.weight) for v in model.modules() if isinstance(v,FactorConv)}))
    else:
        model=CompactSuccessor(arm);maps={}
        maps['coarse_spectral']=prune_core(parent.frozen.coarse.spectral,model.frozen.coarse.spectral)
        maps['coarse_temporal']=prune_core(parent.frozen.coarse.temporal,model.frozen.coarse.temporal)
        action_maps=prune_core(parent.frozen.estimator.core,model.frozen.estimator.core);maps['action']=action_maps
        with torch.no_grad():
            model.frozen.estimator.global_head.weight.copy_(parent.frozen.estimator.global_head.weight[:,action_maps[-1]])
            model.frozen.estimator.global_head.bias.copy_(parent.frozen.estimator.global_head.bias)
            model.frozen.estimator.initial_kernel.copy_(parent.frozen.estimator.initial_kernel)
        for index,(source,target) in enumerate(zip(parent.prior,model.prior)):maps['prior'+str(index)]=prune_core(source,target)
        share_long_weights(model.prior)
        detail=dict(channel_maps={key:[v.tolist() for v in values] for key,values in maps.items()})
    model.arm=arm;model.requires_grad_(True)
    model.initialization_report=dict(mode=d.MODE[arm],group_count=4,IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,**detail)
    return model
