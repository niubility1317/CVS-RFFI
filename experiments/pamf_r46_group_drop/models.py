"""Drop complete context GN4 groups; keep the full shared waveform prior."""
import torch
from torch import nn
from . import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,_block,_conv
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import Core,Encoder,ComplexTFUNet
from experiments.pamf_restoration.models import Block,zero_output
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r46_multipoint_fast.models import objective,spectral_error


def select_groups(score):
    if (score.ndim!=1 or not score.is_floating_point() or len(score)==0 or len(score)%4
            or not torch.isfinite(score).all() or (score<0).any()):
        raise ValueError('Finite nonnegative channel scores in four complete groups required')
    size=len(score)//4
    chosen=torch.argsort(score.reshape(4,size).sum(1),descending=True,stable=True)[:3].sort().values
    return torch.cat([torch.arange(int(group)*size,(int(group)+1)*size,device=score.device) for group in chosen])


class GroupBlock(Block):
    """Same original Block forward/kernels/skip, with explicit GN grouping."""
    def __init__(self,a,b,groups):
        nn.Module.__init__(self)
        self.net=nn.Sequential(nn.Conv1d(a,b,5,padding=2),nn.GroupNorm(groups,b),nn.SiLU(),
            nn.Conv1d(b,b,3,padding=1),nn.GroupNorm(groups,b),nn.SiLU())
        self.skip=nn.Conv1d(a,b,1) if a!=b else nn.Identity()


class GroupEncoder(Encoder):
    def __init__(self,incoming,widths,groups):
        nn.Module.__init__(self)
        self.blocks=nn.ModuleList(nn.Sequential(GroupBlock(a,b,groups),GroupBlock(b,b,groups))
            for a,b in zip((incoming,*widths[:-1]),widths))


class GroupCore(Core):
    def __init__(self,incoming,outgoing,widths,groups):
        nn.Module.__init__(self)
        self.encoder=GroupEncoder(incoming,widths,groups);self.conditional=False
        self.decode=nn.ModuleList(nn.Sequential(GroupBlock(a+b,b,groups),GroupBlock(b,b,groups))
            for a,b in zip(widths[:0:-1],widths[-2::-1]))
        self.out=zero_output(nn.Conv1d(widths[0],outgoing,1))


class CompactSuccessor(JointCompactRestorer):
    def __init__(self,arm):
        nn.Module.__init__(self)
        if arm not in d.ARMS:raise ValueError('Registered whole-group comparison row required')
        self.arm=arm
        coarse=ComplexTFUNet()
        groups=d.CONTEXT_GROUPS[arm]
        coarse.spectral=GroupCore(incoming=4,outgoing=4,widths=d.SPECTRAL_WIDTHS[arm],groups=groups)
        coarse.temporal=GroupCore(incoming=4,outgoing=2,widths=d.CONTEXT_WIDTHS[arm],groups=groups)
        self.frozen=ActionRestorer(coarse)
        self.frozen.estimator.core=GroupCore(incoming=4,outgoing=4,widths=d.CONTEXT_WIDTHS[arm],groups=groups)
        self.frozen.estimator.global_head=nn.Linear(d.CONTEXT_WIDTHS[arm][-1],129)
        self.prior=nn.ModuleList(Core(incoming=9,widths=d.WIDTHS[arm]) for _ in range(3))
        share_long_weights(self.prior)
        self.requires_grad_(True);self.initialization_report=None


def group_maps(source):
    result=[]
    for blocks in source.encoder.blocks:
        old=blocks[0].net[0].out_channels;score=torch.zeros(old,dtype=torch.float64)
        for block in blocks:
            for index in (0,3):
                part=block.net[index].weight.detach().double().square().mean((1,2))
                score+=part/part.mean().clamp_min(1e-30)
        result.append(select_groups(score))
    return result


def prune_core(source,target,maps):
    old=[block[0].net[0].out_channels for block in source.encoder.blocks]
    widths=[block[0].net[0].out_channels for block in target.encoder.blocks]
    if len(maps)!=len(old):raise ValueError('Complete core channel maps required')
    for indices,original,narrowed in zip(maps,old,widths):
        size=original//4
        if original%4 or narrowed!=3*size or indices.dtype!=torch.int64 or indices.shape!=(narrowed,):
            raise ValueError('Exactly three complete original groups required')
        if indices.tolist()!=sorted(set(indices.tolist())) or (indices<0).any() or (indices>=original).any():
            raise ValueError('Sorted unique original channel indices required')
        if sorted(int(((indices>=g*size)&(indices<(g+1)*size)).sum()) for g in range(4))!=[0,size,size,size]:
            raise ValueError('Partial original groups are not allowed')
    inputs=torch.arange(source.encoder.blocks[0][0].net[0].in_channels)
    for src,dst,outputs in zip(source.encoder.blocks,target.encoder.blocks,maps):
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs);inputs=outputs
    for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
        level=len(old)-1-index;outputs=maps[level-1]
        inputs=torch.cat((maps[level],old[level]+maps[level-1]))
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs)
    _conv(source.out,target.out,maps[0],torch.arange(source.out.out_channels))


def compact_from_parent(parent,arm):
    if (arm not in d.ARMS or getattr(parent,'arm',None)!='prune_refine'
            or sum(p.numel() for p in parent.parameters())!=350521):
        raise ValueError('Only designated source-final350521 parent and fixed comparison rows allowed')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Data-free transplant requires CPU parent')
    sources=dict(coarse_spectral=parent.frozen.coarse.spectral,coarse_temporal=parent.frozen.coarse.temporal,
        action=parent.frozen.estimator.core)
    if any(module.num_groups!=4 for core in sources.values() for module in core.modules() if isinstance(module,nn.GroupNorm)):
        raise ValueError('Original GN4 parent context required')
    model=CompactSuccessor(arm)
    targets=dict(coarse_spectral=model.frozen.coarse.spectral,coarse_temporal=model.frozen.coarse.temporal,
        action=model.frozen.estimator.core)
    maps={name:group_maps(core) for name,core in sources.items()}
    for name,source in sources.items():prune_core(source,targets[name],maps[name])
    with torch.no_grad():
        model.frozen.estimator.global_head.weight.copy_(parent.frozen.estimator.global_head.weight[:,maps['action'][-1]])
        model.frozen.estimator.global_head.bias.copy_(parent.frozen.estimator.global_head.bias)
        model.frozen.estimator.initial_kernel.copy_(parent.frozen.estimator.initial_kernel)
    model.prior.load_state_dict(parent.prior.state_dict(),strict=True)
    model.initialization_report=dict(mode='whole_context_groups_weight_only',context_group_count=d.CONTEXT_GROUPS[arm],
        original_group_count=4,retained_original_groups=3,prior_group_count=4,prior_copied_exactly=True,
        retained_normalization_partition_preserved=arm=='group_preserve',
        channel_maps={name:[indices.tolist() for indices in values] for name,values in maps.items()},
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
        parent_unchanged=True,exact_function_equivalence=False,new_trainable_parameters=0)
    return model
