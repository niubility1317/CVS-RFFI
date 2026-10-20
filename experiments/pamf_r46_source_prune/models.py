import torch
from . import design as d
from experiments.pamf_r46_compact_refine.models import CompactSuccessor as Existing273765
from experiments.pamf_r46_multipoint_fast.models import objective as original_objective,spectral_error
from experiments.pamf_iq_joint_compact.models import _block,_conv
from experiments.pamf_r17_fidelity.models import share_long_weights

def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered source-selection arm required')
    model=Existing273765('prune_refine');model.arm=arm;return model

def selected_channels(score,new):
    if score.ndim!=1 or not score.is_floating_point() or not torch.isfinite(score).all() or (score<0).any():raise ValueError('Finite nonnegative source scores required')
    old=len(score)
    if type(new) is not int or not 0<new<=old or old%4 or new%4:raise ValueError('GN4-compatible narrowing required')
    return torch.cat([torch.argsort(score[g*old//4:(g+1)*old//4],descending=True,stable=True)[:new//4].sort().values+g*old//4 for g in range(4)])

def validate_maps(source,target,maps):
    old=[b[0].net[0].out_channels for b in source.encoder.blocks];new=[b[0].net[0].out_channels for b in target.encoder.blocks]
    if len(maps)!=len(old):raise ValueError('One map per encoder level required')
    for indices,n,m in zip(maps,old,new):
        if indices.dtype!=torch.int64 or indices.shape!=(m,) or indices.tolist()!=sorted(set(indices.tolist())) or (indices<0).any() or (indices>=n).any():raise ValueError('Sorted unique in-range maps required')
        if any(int(((indices>=g*n//4)&(indices<(g+1)*n//4)).sum())!=m//4 for g in range(4)):raise ValueError('Each GN4 group keeps its fixed quota')
    return old

def prune_with_maps(source,target,maps):
    old=validate_maps(source,target,maps);inputs=torch.arange(source.encoder.blocks[0][0].net[0].in_channels)
    for src,dst,outputs in zip(source.encoder.blocks,target.encoder.blocks,maps):
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs);inputs=outputs
    for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
        level=len(old)-1-index;outputs=maps[level-1];inputs=torch.cat((maps[level],old[level]+maps[level-1]))
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs)
    _conv(source.out,target.out,maps[0],torch.arange(source.out.out_channels))

def compact_from_parent(parent,arm,calibration):
    if arm not in d.ARMS or getattr(parent,'arm',None)!='prune_refine' or sum(p.numel() for p in parent.parameters())!=350521:raise ValueError('Only designated350521parent allowed')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Source-calibrated transplant is CPU-only')
    from .calibration import validate_calibration
    validate_calibration(calibration)
    maps={k:[torch.tensor(v,dtype=torch.int64) for v in values] for k,values in calibration['maps'][arm].items()}
    if any(any(not torch.equal(a,b) for a,b in zip(maps['prior0'],maps['prior'+str(i)])) for i in (1,2)):raise ValueError('Shared prior maps must be identical')
    model=CompactSuccessor(arm)
    prune_with_maps(parent.frozen.coarse.spectral,model.frozen.coarse.spectral,maps['coarse_spectral'])
    prune_with_maps(parent.frozen.coarse.temporal,model.frozen.coarse.temporal,maps['coarse_temporal'])
    prune_with_maps(parent.frozen.estimator.core,model.frozen.estimator.core,maps['action'])
    with torch.no_grad():
        model.frozen.estimator.global_head.weight.copy_(parent.frozen.estimator.global_head.weight[:,maps['action'][-1]])
        model.frozen.estimator.global_head.bias.copy_(parent.frozen.estimator.global_head.bias)
        model.frozen.estimator.initial_kernel.copy_(parent.frozen.estimator.initial_kernel)
    for i,(src,dst) in enumerate(zip(parent.prior,model.prior)):prune_with_maps(src,dst,maps['prior'+str(i)])
    share_long_weights(model.prior);model.requires_grad_(True)
    model.initialization_report=dict(mode=d.MODE[arm],group_count=4,IQ_examples=256,source_unique_IQ=128,calibration_L_exposures=128,calibration_U_exposures=128,
        TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,
        shared_prior_maps=True,channel_maps=calibration['maps'][arm],calibration_seed=d.CAL_SEED,calibration_projection_seed=d.CAL_PROJECTION_SEED)
    return model

def objective(student,teacher,clean,epoch,arm):
    if arm not in d.ARMS:raise ValueError('Registered source-selection arm required')
    return original_objective(student,teacher,clean,epoch)
