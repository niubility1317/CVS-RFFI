import copy
import torch
from . import design as d
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r38_factor_prior.models import factor_modules
from experiments.pamf_r46_compact_refine.models import spectral_error
from experiments.pamf_r17_compact.models import stages

class CompactSuccessor(Parent):
    def __init__(self,arm):
        if arm not in d.ARMS:raise ValueError('Registered prior-preserve arm required')
        super().__init__('prune_refine');self.arm=arm;factor_modules(self.prior,d.FRACTIONS[arm],initialize=False)
        self.frozen.eval().requires_grad_(False);self.prior.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode);self.frozen.eval();return self
    @torch.no_grad()
    def estimate_context(self,received):return super().estimate_context(received)

def compact_from_parent(parent,arm):
    if arm not in d.ARMS or getattr(parent,'arm',None)!='prune_refine' or sum(p.numel() for p in parent.parameters())!=350521:raise ValueError('Only original R46source-final350521 parent')
    if any(p.device.type!='cpu' for p in parent.parameters()):raise ValueError('Data-free CPU SVD required')
    model=CompactSuccessor(arm)
    model.frozen.load_state_dict(parent.frozen.state_dict(),strict=True)
    priors=copy.deepcopy(parent.prior);audit=factor_modules(priors,d.FRACTIONS[arm],initialize=True);model.prior=priors.requires_grad_(True)
    model.initialization_report=dict(mode=d.MODE[arm],group_count=4,IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,
        exact_function_equivalence=False,channel_dimensions_unchanged=True,normalization_values_exact=True,context_exact=True,context_trainable=False,factor_layers=audit)
    return model

def paired_outputs(model,teacher,received):
    with torch.no_grad():
        context=teacher.estimate_context(received);desired=stages(teacher.prior,received,context)[0]
    captured=[]
    def capture(module,inputs,output):captured.append((inputs[0],output))
    handles=[p.register_forward_hook(capture) for p in model.prior]
    try:outputs=stages(model.prior,received,context)[0]
    finally:
        for handle in handles:handle.remove()
    if len(captured)!=3:raise ValueError('Three actual student prior calls required')
    errors=[]
    for index,(inputs,output) in enumerate(captured):
        with torch.no_grad():target=teacher.prior[index](inputs.detach())
        errors.append((output-target).square().mean((1,2))/target.square().mean((1,2)).clamp_min(1e-4))
    return outputs,desired,torch.stack(errors).mean(0)

def objective(student,teacher,clean,epoch,arm,conditional=None):
    if arm not in d.ARMS or len(student)!=3 or len(teacher)!=3 or any(v.shape!=clean.shape for v in (*student,*teacher)):raise ValueError('Three matched sourceIQ stages required')
    d.teacher_weight(epoch)
    if conditional is None or conditional.shape!=(clean.shape[0],):raise ValueError('Actual conditional prior error required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(student[-1]-clean.detach()).square().mean((1,2))/energy
    fidelity=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,teacher)]).mean(0)
    spectral=spectral_error(student[-1],clean)
    return paired+.05*spectral+fidelity+.05*conditional,dict(iq_nmse=paired,spectral_error=spectral,teacher_stage_nmse=fidelity,conditional_prior_nmse=conditional)
