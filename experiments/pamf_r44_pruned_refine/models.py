import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r38_compression_methods.models import prune_core,important_channels


class CompactSuccessor(JointCompactRestorer):
    def __init__(self,arm):
        nn.Module.__init__(self)
        if arm not in d.ARMS:raise ValueError('Only registered R44 refinement rows allowed')
        self.arm=arm;coarse=ComplexTFUNet()
        coarse.spectral=Core(incoming=4,outgoing=4,widths=d.SPECTRAL_WIDTHS[arm]);coarse.temporal=Core(incoming=4,widths=d.WIDTHS[arm])
        self.frozen=ActionRestorer(coarse)
        self.frozen.estimator.core=Core(incoming=4,outgoing=4,widths=d.WIDTHS[arm]);self.frozen.estimator.global_head=nn.Linear(d.WIDTHS[arm][-1],129)
        self.prior=nn.ModuleList([Core(incoming=9,widths=d.WIDTHS[arm]) for _ in range(3)])
        share_long_weights(self.prior);self.requires_grad_(True);self.initialization_report=None


def compact_from_parent(parent,arm):
    if arm not in d.ARMS or getattr(parent,'arm',None)!='group_pruned' or sum(p.numel() for p in parent.parameters())!=442805:
        raise ValueError('Only human-designated source-final R44 group_pruned442805 parent allowed')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Data-free compression must initialize on CPU')
    if arm=='retain_refine':model=copy.deepcopy(parent);detail=dict(channel_maps=None)
    else:
        model=CompactSuccessor(arm);maps={}
        maps['coarse_spectral']=prune_core(parent.frozen.coarse.spectral,model.frozen.coarse.spectral)
        maps['coarse_temporal']=prune_core(parent.frozen.coarse.temporal,model.frozen.coarse.temporal)
        action=prune_core(parent.frozen.estimator.core,model.frozen.estimator.core);maps['action']=action
        with torch.no_grad():
            model.frozen.estimator.global_head.weight.copy_(parent.frozen.estimator.global_head.weight[:,action[-1]])
            model.frozen.estimator.global_head.bias.copy_(parent.frozen.estimator.global_head.bias)
            model.frozen.estimator.initial_kernel.copy_(parent.frozen.estimator.initial_kernel)
        for index,(source,target) in enumerate(zip(parent.prior,model.prior)):maps['prior'+str(index)]=prune_core(source,target)
        share_long_weights(model.prior);detail=dict(channel_maps={k:[v.tolist() for v in values] for k,values in maps.items()})
    model.arm=arm;model.requires_grad_(True)
    model.initialization_report=dict(mode=d.MODE[arm],group_count=4,IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=arm=='retain_refine',**detail)
    return model


def spectral_error(restored,clean):
    if restored.shape!=clean.shape or restored.ndim!=3 or restored.shape[1:]!=(2,256):raise ValueError('Paired IQ2x256 required')
    target=clean.detach();pred=torch.complex(restored[:,0],restored[:,1]);ref=torch.complex(target[:,0],target[:,1]);errors=[]
    for size in (64,128,256):
        window=torch.hann_window(size,periodic=True,device=restored.device,dtype=restored.dtype)
        def power(x):return torch.fft.fft(x.unfold(-1,size,size//2)*window,norm='ortho').abs().square()
        desired=power(ref);scale=desired.mean((1,2),keepdim=True).clamp_min(1e-4)
        errors.append((torch.log1p(power(pred)/scale)-torch.log1p(desired/scale)).square().mean((1,2)))
    return torch.stack(errors).mean(0)


def objective(student,teacher,clean,epoch):
    if len(student)!=3 or len(teacher)!=3 or any(v.shape!=clean.shape for v in (*student,*teacher)):
        raise ValueError('Three matched stages and paired sourceclean required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(student[-1]-clean.detach()).square().mean((1,2))/energy
    fidelity=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    spectral=spectral_error(student[-1],clean)
    return paired+.05*spectral+d.teacher_weight(epoch)*fidelity,dict(iq_nmse=paired,spectral_error=spectral,teacher_stage_nmse=fidelity)
