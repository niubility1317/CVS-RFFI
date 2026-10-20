"""Collapse prior-only linear factor pairs, preserving shared kernels and bias states."""
import copy
import time
import torch
from torch import nn
from experiments.pamf_r17_compact.models import FactorConv


def _validate_pair(value):
    if not isinstance(value,FactorConv) or not isinstance(value.left,nn.Conv1d) or not isinstance(value.right,nn.Conv1d):
        raise ValueError('Existing FactorConv pair required')
    left,right=value.left,value.right
    if (left.groups!=1 or right.groups!=1 or left.bias is not None or left.padding_mode!='zeros'
            or right.padding_mode!='zeros' or right.kernel_size!=(1,) or right.stride!=(1,)
            or right.padding!=(0,) or right.dilation!=(1,) or left.out_channels!=right.in_channels
            or left.weight.device!=right.weight.device or left.weight.dtype!=right.weight.dtype
            or left.weight.dtype not in (torch.float32,torch.float64)):
        raise ValueError('Linear ungrouped zero-padded spatial-then-pointwise pair required')
    if any(m.training for m in value.modules()) or any(p.requires_grad for p in value.parameters()):
        raise ValueError('Frozen eval pair required')
    if any(not torch.isfinite(p).all() for p in value.parameters()):
        raise ValueError('Finite factor weights and bias required')


@torch.no_grad()
def dense_pair(value):
    _validate_pair(value);left,right=value.left,value.right
    dense=(right.weight[:,:,0]@left.weight.flatten(1)).reshape(right.out_channels,left.in_channels,left.kernel_size[0])
    with torch.random.fork_rng(devices=[]):
        result=nn.Conv1d(left.in_channels,right.out_channels,left.kernel_size,stride=left.stride,
            padding=left.padding,dilation=left.dilation,bias=right.bias is not None,device='cpu',dtype=dense.dtype)
    result.weight=nn.Parameter(dense.detach(),requires_grad=False)
    if right.bias is not None:result.bias=nn.Parameter(right.bias.detach().clone(),requires_grad=False)
    return result.eval()


@torch.no_grad()
def collapse_prior(model):
    if not isinstance(model,nn.Module) or any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Complete frozen eval restorer required')
    if not isinstance(getattr(model,'prior',None),nn.ModuleList) or len(model.prior)!=3 or len({id(p) for p in model.prior})!=3:
        raise ValueError('Three distinct prior modules required')
    found=[(i,name,value) for i,core in enumerate(model.prior) for name,value in core.named_modules() if isinstance(value,FactorConv)]
    if not found:raise ValueError('Existing prior factor pairs required')
    for _,_,value in found:_validate_pair(value)
    started=time.perf_counter();before=sum(p.numel() for p in model.parameters());result=copy.deepcopy(model)
    kernels={};biases={};audit=[]
    pending=[(i,name,value) for i,core in enumerate(result.prior) for name,value in core.named_modules() if isinstance(value,FactorConv)]
    for index,name,value in pending:
        key=(id(value.left.weight),id(value.right.weight));merged=dense_pair(value)
        if key in kernels:merged.weight=kernels[key]
        else:kernels[key]=merged.weight
        if value.right.bias is not None:
            bias_key=id(value.right.bias)
            if bias_key in biases:merged.bias=biases[bias_key]
            else:biases[bias_key]=merged.bias
        container,_,leaf=name.rpartition('.')
        parent=dict(result.prior[index].named_modules())[container]
        setattr(parent,leaf,merged)
        audit.append(dict(module='prior.'+str(index)+'.'+name,
            factor_weight_elements=value.left.weight.numel()+value.right.weight.numel(),dense_weight_elements=merged.weight.numel(),
            stride=list(merged.stride),padding=list(merged.padding),dilation=list(merged.dilation),bias=merged.bias is not None))
    result.eval().requires_grad_(False)
    report=dict(mode='prior_only_linear_pair_collapse',before_parameters=before,
        after_parameters=sum(p.numel() for p in result.parameters()),factor_modules_replaced=len(pending),
        unique_kernel_pairs=len(kernels),expanded_kernel_elements=sum(p.numel() for p in kernels.values()),
        bias_alias_groups=len(biases),context_unchanged=True,normalization_unchanged=True,
        source_model_unchanged=True,new_training_updates=0,IQ_reads=0,F_reads=0,TX_labels_read=0,
        seconds=time.perf_counter()-started,modules=audit,
        claim='Algebraic linear composition;floating-point execution order differs;requires independent seven-scene tests')
    return result,report
