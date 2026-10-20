"""Keep M35 channel choices fixed while changing the time of compression."""
import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,_block,_conv
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r38_compression_methods.models import important_channels
from experiments.pamf_r46_multipoint_fast.models import objective,spectral_error


class CompactSuccessor(JointCompactRestorer):
    def __init__(self,arm):
        nn.Module.__init__(self)
        if arm not in d.ARMS:raise ValueError('Fixed delayed-pruning row required')
        self.arm=arm;coarse=ComplexTFUNet()
        coarse.spectral=Core(incoming=4,outgoing=4,widths=d.SPECTRAL_WIDTHS[arm])
        coarse.temporal=Core(incoming=4,widths=d.CONTEXT_WIDTHS[arm])
        self.frozen=ActionRestorer(coarse)
        self.frozen.estimator.core=Core(incoming=4,outgoing=4,widths=d.CONTEXT_WIDTHS[arm])
        self.frozen.estimator.global_head=nn.Linear(d.CONTEXT_WIDTHS[arm][-1],129)
        self.prior=nn.ModuleList(Core(incoming=9,widths=d.WIDTHS[arm]) for _ in range(3))
        share_long_weights(self.prior);self.requires_grad_(True);self.initialization_report=None


def cores(model):
    return dict(coarse_spectral=model.frozen.coarse.spectral,coarse_temporal=model.frozen.coarse.temporal,
        action=model.frozen.estimator.core,**{'prior'+str(i):m for i,m in enumerate(model.prior)})


def validate_maps(source,target,maps):
    if not isinstance(maps,dict) or set(maps)!=set(cores(source)):
        raise ValueError('Exactly the six fixed context/prior map families required')
    for name,original in cores(source).items():
        narrowed=cores(target)[name];levels=maps[name]
        if not isinstance(levels,list) or len(levels)!=4:raise ValueError('Four complete encoder level maps required')
        for src,dst,indices in zip(original.encoder.blocks,narrowed.encoder.blocks,levels):
            old=src[0].net[0].out_channels;new=dst[0].net[0].out_channels
            if not isinstance(indices,list) or any(type(v) is not int for v in indices):
                raise ValueError('Integer fixed channel indices required')
            if indices!=sorted(set(indices)) or len(indices)!=new or not indices or min(indices)<0 or max(indices)>=old:
                raise ValueError('Sorted unique original channel indices with exact final width required')
            if old%4 or new%4 or any(sum(g*(old//4)<=v<(g+1)*(old//4) for v in indices)!=new//4 for g in range(4)):
                raise ValueError('Equal numbers retained in each original GN4 group required')
    if any(maps['prior0']!=maps['prior'+str(i)] for i in (1,2)):
        raise ValueError('Tied prior weights require one identical channel map')


def fixed_maps(parent):
    if getattr(parent,'arm',None)!='prune_refine' or sum(p.numel() for p in parent.parameters())!=d.INITIAL_PARAMETERS:
        raise ValueError('Only the designated full M35 graph may freeze channel choices')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Weight-only map choice requires CPU parent')
    target=CompactSuccessor(d.ARMS[0]);maps={}
    for name,source in cores(parent).items():
        maps[name]=[important_channels(s,t[0].net[0].out_channels).tolist()
            for s,t in zip(source.encoder.blocks,cores(target)[name].encoder.blocks)]
    validate_maps(parent,target,maps)
    return maps


def dense_from_parent(parent,arm):
    if arm not in d.ARMS:raise ValueError('Registered row required')
    maps=fixed_maps(parent)
    model=copy.deepcopy(parent);model.arm=arm;model.requires_grad_(True)
    model.initialization_report=dict(mode='dense_M35_then_fixed_map_pruning',phase='dense',
        channel_maps=maps,warmup_epochs=d.WARMUP_EPOCHS[arm],initial_parameters=d.INITIAL_PARAMETERS,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
        parent_unchanged=True,exact_initial_function_equivalence=True,map_recomputed_after_initialization=False,
        teacher='original_M35_only')
    return model


def _slice_core(source,target,levels):
    old=[b[0].net[0].out_channels for b in source.encoder.blocks]
    maps=[torch.tensor(v,dtype=torch.int64) for v in levels]
    inputs=torch.arange(source.encoder.blocks[0][0].net[0].in_channels)
    for src,dst,outputs in zip(source.encoder.blocks,target.encoder.blocks,maps):
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs);inputs=outputs
    for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
        level=len(old)-1-index;outputs=maps[level-1]
        inputs=torch.cat((maps[level],old[level]+maps[level-1]))
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs)
    _conv(source.out,target.out,maps[0],torch.arange(source.out.out_channels))


def transition(model,completed_epoch):
    arm=getattr(model,'arm',None);initial=getattr(model,'initialization_report',None)
    if arm not in d.ARMS or type(completed_epoch) is not int or completed_epoch!=d.WARMUP_EPOCHS[arm]:
        raise ValueError('Compression only after this row fixed complete warmup epoch')
    if sum(p.numel() for p in model.parameters())!=d.INITIAL_PARAMETERS or not isinstance(initial,dict) or initial.get('phase')!='dense':
        raise ValueError('Only the same row original dense source student may transition once')
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Perform deterministic structural transition on CPU')
    result=CompactSuccessor(arm);maps=initial['channel_maps'];validate_maps(model,result,maps)
    for name,source in cores(model).items():_slice_core(source,cores(result)[name],maps[name])
    with torch.no_grad():
        indices=torch.tensor(maps['action'][-1],dtype=torch.int64)
        result.frozen.estimator.global_head.weight.copy_(model.frozen.estimator.global_head.weight[:,indices])
        result.frozen.estimator.global_head.bias.copy_(model.frozen.estimator.global_head.bias)
        result.frozen.estimator.initial_kernel.copy_(model.frozen.estimator.initial_kernel)
    share_long_weights(result.prior)
    result.initialization_report=copy.deepcopy(initial)
    result.initialization_report.update(phase='compact',transition_completed_epoch=completed_epoch,
        transition_completed_updates=completed_epoch*247,optimizer_resets=1,global_schedule_restarted=False,
        channel_maps_unchanged=True,new_calibration_IQ=0,external_student_checkpoint_reads=0)
    return result
