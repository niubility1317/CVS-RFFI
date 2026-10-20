import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_compact_refine.models import spectral_error
from experiments.pamf_r38_factor_prior.models import factor_modules

def apply_factors(model,initialize):
    audit={}
    for name,core in (('coarse_spectral',model.frozen.coarse.spectral),('coarse_temporal',model.frozen.coarse.temporal),('action',model.frozen.estimator.core)):
        audit[name]=factor_modules([core],.625,initialize=initialize)
    audit['prior']=factor_modules(model.prior,.625,initialize=initialize)
    return audit

class CompactSuccessor(Parent):
    def __init__(self,arm):
        if arm not in d.ARMS:raise ValueError('Registered low-rank arm required')
        super().__init__('prune_refine');self.arm=arm;apply_factors(self,False)

def compact_from_parent(parent,arm):
    if arm not in d.ARMS or getattr(parent,'arm',None)!='prune_refine' or sum(p.numel() for p in parent.parameters())!=350521:
        raise ValueError('Only human-designated R46 source-final350521 parent')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Data-free CPU SVD initialization required')
    model=copy.deepcopy(parent);audit=apply_factors(model,True);model.arm=arm;model.requires_grad_(True)
    model.initialization_report=dict(mode=d.MODE[arm],group_count=4,IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,channel_dimensions_unchanged=True,normalization_values_exact=True,factor_layers=audit)
    return model

def paired_outputs(model,teacher,received):
    with torch.no_grad():desired=teacher.stage_outputs(received)[0]
    if d.CONDITIONAL_WEIGHT[model.arm]==0:
        return model.stage_outputs(received)[0],desired,None
    captured=[];handles=[]
    def capture(module,inputs,output):captured.append((inputs[0],output))
    try:
        handles=[core.register_forward_hook(capture) for core in model.prior]
        outputs=model.stage_outputs(received)[0]
    finally:
        for handle in handles:handle.remove()
    if len(captured)!=3:raise ValueError('Exactly three actual student prior calls required')
    errors=[]
    for index,(inputs,output) in enumerate(captured):
        with torch.no_grad():target=teacher.prior[index](inputs.detach())
        scale=target.square().mean((1,2)).clamp_min(1e-4)
        errors.append((output-target).square().mean((1,2))/scale)
    return outputs,desired,torch.stack(errors).mean(0)

def objective(student,teacher,clean,epoch,arm,conditional=None):
    if arm not in d.ARMS or len(student)!=3 or len(teacher)!=3 or any(v.shape!=clean.shape for v in (*student,*teacher)):
        raise ValueError('Registered three-stage matched sourceIQ required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(student[-1]-clean.detach()).square().mean((1,2))/energy
    fidelity=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    spectral=spectral_error(student[-1],clean)
    if d.CONDITIONAL_WEIGHT[arm]:
        if conditional is None or conditional.shape!=paired.shape:raise ValueError('Actual conditional prior error required')
    elif conditional is not None:raise ValueError('Control must not execute conditional branch')
    conditional=paired.new_zeros(paired.shape) if conditional is None else conditional
    return paired+.05*spectral+d.teacher_weight(epoch)*fidelity+d.CONDITIONAL_WEIGHT[arm]*conditional,dict(iq_nmse=paired,spectral_error=spectral,teacher_stage_nmse=fidelity,conditional_prior_nmse=conditional)
