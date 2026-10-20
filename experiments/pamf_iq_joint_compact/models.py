import torch
from torch import nn
from . import design as d
from experiments.pamf_action_restore.models import ActionRestorer,ActionEstimator
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r17_compact.models import stages


def channel_indices(old,new):
    if not (type(old) is type(new) is int and 0<new<=old and old%4==new%4==0):
        raise ValueError('Positive four-group-compatible channel narrowing required')
    return torch.cat([torch.arange(group*(old//4),group*(old//4)+new//4) for group in range(4)])


def _conv(source,target,inputs,outputs):
    if not isinstance(source,nn.Conv1d) or not isinstance(target,nn.Conv1d) or source.groups!=target.groups or source.groups!=1:
        raise ValueError('Ordinary dense matched Conv1d required')
    selected=source.weight.detach().index_select(0,outputs).index_select(1,inputs)
    if selected.shape!=target.weight.shape or source.kernel_size!=target.kernel_size or source.stride!=target.stride or source.padding!=target.padding:
        raise ValueError('Channel-sliced convolution dimensions differ')
    with torch.no_grad():
        target.weight.copy_(selected)
        if target.bias is not None:target.bias.copy_(source.bias.detach().index_select(0,outputs))


def _block(source,target,inputs,outputs):
    _conv(source.net[0],target.net[0],inputs,outputs)
    _conv(source.net[3],target.net[3],outputs,outputs)
    with torch.no_grad():
        for index in (1,4):
            target.net[index].weight.copy_(source.net[index].weight.detach().index_select(0,outputs))
            target.net[index].bias.copy_(source.net[index].bias.detach().index_select(0,outputs))
    if isinstance(target.skip,nn.Conv1d):_conv(source.skip,target.skip,inputs,outputs)
    elif not isinstance(source.skip,nn.Identity):raise ValueError('Residual skip topology differs')


def transplant_core(source,target):
    old=[v[0].net[0].out_channels for v in source.encoder.blocks]
    new=[v[0].net[0].out_channels for v in target.encoder.blocks]
    maps=[channel_indices(a,b) for a,b in zip(old,new)]
    inputs=torch.arange(source.encoder.blocks[0][0].net[0].in_channels)
    for src,dst,outputs in zip(source.encoder.blocks,target.encoder.blocks,maps):
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs);inputs=outputs
    for index,(src,dst) in enumerate(zip(source.decode,target.decode)):
        level=len(old)-1-index;outputs=maps[level-1]
        # Decoder concatenates the upsampled value and the skip. The skip starts
        # after the OLD full upstream width, not after the narrowed width.
        inputs=torch.cat((maps[level],old[level]+maps[level-1]))
        _block(src[0],dst[0],inputs,outputs);_block(src[1],dst[1],outputs,outputs)
    _conv(source.out,target.out,maps[0],torch.arange(source.out.out_channels))
    return maps


class NarrowCoarse(ComplexTFUNet):
    def __init__(self,arm):
        nn.Module.__init__(self)
        self.spectral=Core(incoming=4,outgoing=4,widths=d.SPECTRAL_WIDTHS[arm])
        self.temporal=Core(incoming=4,widths=d.WIDTHS[arm])


class JointCompactRestorer(nn.Module):
    def __init__(self,arm,*,share=True):
        super().__init__()
        if arm not in d.ARMS:raise ValueError('Registered dense compact arm required')
        self.arm=arm
        self.frozen=ActionRestorer(NarrowCoarse(arm))
        self.frozen.estimator.core=Core(incoming=4,outgoing=4,widths=d.WIDTHS[arm])
        self.frozen.estimator.global_head=nn.Linear(d.WIDTHS[arm][-1],129)
        self.prior=nn.ModuleList([Core(incoming=9,widths=d.WIDTHS[arm]) for _ in range(3)])
        if share:share_long_weights(self.prior)
        self.frozen.requires_grad_(True)
        self.initialization_report=None

    def estimate_context(self,received):
        # ActionRestorer.estimate intentionally disables coarse gradients for the
        # historical method. Joint optimization must call both modules directly.
        base=self.frozen.coarse(received)
        action,native=self.frozen.estimator(received,base)
        return base,action,native

    def stage_outputs(self,received):
        return stages(self.prior,received,self.estimate_context(received))

    def training_outputs(self,received):
        output,variance,action,base=self.stage_outputs(received)
        return output[-1],variance,action,base

    def forward(self,received):return self.training_outputs(received)[0]


def compact_from_teacher(teacher,arm):
    if getattr(teacher,'candidate',None)!='unroll_unshared' or len(teacher.prior)!=3:
        raise ValueError('Only the original source-selected R17 unshared teacher is allowed')
    if any(p.device.type!='cpu' for p in teacher.parameters()):raise ValueError('Initialize deterministic channel maps on CPU')
    model=JointCompactRestorer(arm,share=False)
    transplant_core(teacher.frozen.coarse.spectral,model.frozen.coarse.spectral)
    transplant_core(teacher.frozen.coarse.temporal,model.frozen.coarse.temporal)
    maps=transplant_core(teacher.frozen.estimator.core,model.frozen.estimator.core)
    with torch.no_grad():
        model.frozen.estimator.global_head.weight.copy_(teacher.frozen.estimator.global_head.weight[:,maps[-1]])
        model.frozen.estimator.global_head.bias.copy_(teacher.frozen.estimator.global_head.bias)
        model.frozen.estimator.initial_kernel.copy_(teacher.frozen.estimator.initial_kernel)
    for src,dst in zip(teacher.prior,model.prior):transplant_core(src,dst)
    share_long_weights(model.prior);model.requires_grad_(True)
    model.initialization_report=dict(mode='group_preserving_dense_channel_slices_shared_long',group_count=4,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
        original_teacher_unchanged=True,exact_function_equivalence=False,context_trainable=True)
    return model


def objective(student,teacher,clean,epoch):
    if len(student)!=3 or len(teacher)!=3 or any(v.shape!=clean.shape for v in (*student,*teacher)):
        raise ValueError('Three matched student/teacher stages and paired source IQ required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(student[-1]-clean.detach()).square().mean((1,2))/energy
    fidelity=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    return paired+d.teacher_weight(epoch)*fidelity,dict(iq_nmse=paired,teacher_stage_nmse=fidelity)


@torch.no_grad()
def convolution_linear_macs(model,value):
    total=0;handles=[]
    def observe(module,inputs,output):
        nonlocal total
        if isinstance(module,nn.Conv1d):total+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        elif isinstance(module,nn.Linear):total+=output.numel()*module.in_features
    try:
        for module in model.modules():
            if isinstance(module,(nn.Conv1d,nn.Linear)):handles.append(module.register_forward_hook(observe))
        model(value)
    finally:
        for handle in handles:handle.remove()
    return total
