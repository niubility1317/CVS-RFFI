"""Fold source-only channel reconstruction into ordinary Conv/Linear weights."""
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass
class Link:
    name:str
    source:nn.Module
    target:nn.Module
    inputs:torch.Tensor
    outputs:torch.Tensor

def core_links(source,target,maps,prefix):
    from experiments.pamf_r46_source_prune.models import validate_maps
    old=validate_maps(source,target,maps);links=[]
    def block(src,dst,inputs,outputs,name):
        for index,selected in ((0,inputs),(3,outputs)):
            links.append(Link(name+f'.net.{index}',src.net[index],dst.net[index],selected,outputs))
        if isinstance(src.skip,nn.Conv1d) and isinstance(dst.skip,nn.Conv1d):links.append(Link(name+'.skip',src.skip,dst.skip,inputs,outputs))
        elif not isinstance(src.skip,nn.Identity) or not isinstance(dst.skip,nn.Identity):raise ValueError('Matching residual skip required')
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

def make_links(parent,model,maps):
    values=[]
    for name,src,dst in (
        ('coarse_spectral',parent.frozen.coarse.spectral,model.frozen.coarse.spectral),
        ('coarse_temporal',parent.frozen.coarse.temporal,model.frozen.coarse.temporal),
        ('action',parent.frozen.estimator.core,model.frozen.estimator.core),
        *((f'prior{i}',src,dst) for i,(src,dst) in enumerate(zip(parent.prior,model.prior)))):
        values+=core_links(src,dst,maps[name],name)
    values.append(Link('action.global_head',parent.frozen.estimator.global_head,model.frozen.estimator.global_head,maps['action'][-1],torch.arange(129)))
    return values

def grouped_links(links):
    groups={};targets={}
    for link in links:
        if not isinstance(link.source,(nn.Conv1d,nn.Linear)) or type(link.source)!=type(link.target):raise ValueError('Matching ordinary Conv/Linear required')
        if isinstance(link.source,nn.Conv1d) and (link.source.groups!=1 or link.target.groups!=1):raise ValueError('Dense convolution required')
        old=link.source.weight.shape;new=link.target.weight.shape
        for indices,count,bound in ((link.inputs,new[1],old[1]),(link.outputs,new[0],old[0])):
            if indices.dtype!=torch.int64 or indices.device.type!='cpu' or indices.shape!=(count,) or indices.tolist()!=sorted(set(indices.tolist())) or (indices<0).any() or (indices>=bound).any():raise ValueError('Sorted unique in-range transplant maps required')
        if old[2:]!=new[2:]:raise ValueError('Kernel shape differs')
        key=id(link.source.weight);target=id(link.target.weight)
        if target in targets and targets[target]!=key:raise ValueError('Different source weights became tied')
        targets[target]=key
        if key in groups:
            first=groups[key][0]
            if id(first.target.weight)!=target or not torch.equal(first.inputs,link.inputs) or not torch.equal(first.outputs,link.outputs):raise ValueError('Tied weights require identical channel maps and target aliases')
        groups.setdefault(key,[]).append(link)
    return list(groups.values())

def reconstruction(gram,keep,ridge_fraction):
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or gram.dtype!=torch.float64 or not torch.isfinite(gram).all():raise ValueError('Finite float64 channel second moment required')
    n=gram.shape[0]
    if keep.dtype!=torch.int64 or keep.device.type!='cpu' or keep.ndim!=1 or not len(keep) or keep.tolist()!=sorted(set(keep.tolist())) or (keep<0).any() or (keep>=n).any():raise ValueError('Valid retained channels required')
    if not math.isfinite(ridge_fraction) or ridge_fraction<=0:raise ValueError('Positive fixed ridge required')
    if not torch.allclose(gram,gram.T,rtol=1e-10,atol=1e-12) or (gram.diag()<0).any():raise ValueError('Symmetric nonnegative second moment required')
    m=len(keep);projection=torch.zeros(n,m,dtype=torch.float64);projection[keep]=torch.eye(m,dtype=torch.float64)
    dropped=torch.tensor([i for i in range(n) if i not in keep.tolist()],dtype=torch.int64)
    covariance=gram.index_select(0,keep).index_select(1,keep)
    ridge=ridge_fraction*max(float(covariance.diag().mean()),1e-12)
    if len(dropped):
        coefficients=torch.linalg.solve(covariance+ridge*torch.eye(m,dtype=torch.float64),gram.index_select(0,keep).index_select(1,dropped)).T
        projection[dropped]=coefficients
    if not torch.isfinite(projection).all():raise ValueError('Finite reconstruction required')
    error=float(torch.trace(gram)-2*torch.trace(projection@gram[keep,:])+torch.trace(projection@covariance@projection.T))
    baseline=float(gram.diag()[dropped].sum()) if len(dropped) else 0.
    scale=max(float(torch.trace(gram)),1e-30)
    if error < -1e-8*scale or error>baseline+1e-8*scale:raise ValueError('Ridge channel fit exceeds the zero-reconstruction control')
    return projection,dict(ridge=ridge,uncorrected_channel_error=max(0.,baseline)/scale,reconstructed_channel_error=max(0.,error)/scale,
        coefficient_frobenius=float(projection.norm()),retained_identity=True,no_intercept=True)

@torch.no_grad()
def folded_weight(link,projection,strength):
    if type(strength) not in (float,int) or not math.isfinite(strength) or not 0<strength<=1:raise ValueError('Fixed positive folding strength<=1 required')
    full=link.source.weight.detach().double().index_select(0,link.outputs)
    if projection.dtype!=torch.float64 or projection.shape!=(full.shape[1],len(link.inputs)) or not torch.isfinite(projection).all():raise ValueError('Matching finite channel reconstruction required')
    if not torch.equal(projection[link.inputs],torch.eye(len(link.inputs),dtype=torch.float64)):raise ValueError('Retained channels must reconstruct identically')
    compact=full.index_select(1,link.inputs)
    folded=full@projection if full.ndim==2 else torch.einsum('oik,ij->ojk',full,projection)
    result=(compact+float(strength)*(folded-compact)).to(link.target.weight.dtype)
    if result.shape!=link.target.weight.shape or not torch.isfinite(result).all():raise ValueError('Invalid folded dense weight')
    return result
