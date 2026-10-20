import torch
from torch import nn
from . import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer,transplant_core
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_r17_fidelity.models import share_long_weights


class CompactSuccessor(JointCompactRestorer):
    def __init__(self,arm,*,share=True):
        nn.Module.__init__(self)
        if arm not in d.ARMS:raise ValueError('Registered R38 successor arm required')
        self.arm=arm;coarse=ComplexTFUNet()
        coarse.spectral=Core(incoming=4,outgoing=4,widths=d.SPECTRAL_WIDTHS[arm]);coarse.temporal=Core(incoming=4,widths=d.WIDTHS[arm])
        self.frozen=ActionRestorer(coarse)
        self.frozen.estimator.core=Core(incoming=4,outgoing=4,widths=d.WIDTHS[arm]);self.frozen.estimator.global_head=nn.Linear(d.WIDTHS[arm][-1],129)
        self.prior=nn.ModuleList([Core(incoming=9,widths=d.WIDTHS[arm]) for _ in range(3)])
        if share:share_long_weights(self.prior)
        self.requires_grad_(True);self.initialization_report=None


def compact_from_parent(parent,arm):
    if arm not in d.ARMS or getattr(parent,'arm',None)!='dense_half' or sum(p.numel() for p in parent.parameters())!=794529:
        raise ValueError('Only user-designated R38 dense_half parent required')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('CPU channel initialization required')
    model=CompactSuccessor(arm,share=d.LEVEL[arm]=='keep')
    if d.LEVEL[arm]=='keep':model.load_state_dict(parent.state_dict(),strict=True)
    else:
        transplant_core(parent.frozen.coarse.spectral,model.frozen.coarse.spectral)
        transplant_core(parent.frozen.coarse.temporal,model.frozen.coarse.temporal)
        maps=transplant_core(parent.frozen.estimator.core,model.frozen.estimator.core)
        with torch.no_grad():
            model.frozen.estimator.global_head.weight.copy_(parent.frozen.estimator.global_head.weight[:,maps[-1]])
            model.frozen.estimator.global_head.bias.copy_(parent.frozen.estimator.global_head.bias)
            model.frozen.estimator.initial_kernel.copy_(parent.frozen.estimator.initial_kernel)
        for old,new in zip(parent.prior,model.prior):transplant_core(old,new)
        share_long_weights(model.prior)
    model.requires_grad_(True)
    model.initialization_report=dict(mode='R38_exact_or_group4_dense_slice',level=d.LEVEL[arm],group_count=4,
        IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
        exact_function_equivalence=d.LEVEL[arm]=='keep')
    return model
