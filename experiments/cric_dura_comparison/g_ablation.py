"""Registered source-trained G mechanism variants; actual native computations."""
from copy import deepcopy
import torch
from torch import nn
from .g_lineage import objective

ARMS=('coarse_only','prior_no_CG','no_image_action','no_innovation_input',
      'copy_no_KD','random_with_KD','random_no_KD')


class CoarseOnly(nn.Module):
    def __init__(self,parent):
        super().__init__();self.coarse=deepcopy(parent.frozen.coarse)
    def stage_outputs(self,x):
        y=self.coarse(x)
        return [y],torch.full_like(x[:,0],1e-8),None,y
    def forward(self,x):return self.coarse(x)


class MechanismG(nn.Module):
    def __init__(self,model,arm):
        super().__init__();self.model=model;self.arm=arm
    def stage_outputs(self,received):
        from experiments.pamf_restoration.models import complex_iq,pair
        from experiments.pamf_action_restore.models import SCALE
        from experiments.pamf_uncertainty_restore.models import weighted_map
        from experiments.pamf_action_restore.operators import Action
        base,action,noise=self.model.estimate_context(received)
        if self.arm=='no_image_action':
            action=Action(action.kernel,action.main,torch.zeros_like(action.image))
        variance=noise[:,None].expand(-1,256)
        observed=complex_iq(received);current=base;outputs=[]
        logvar=noise.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        for prior_core in self.model.prior:
            if self.arm=='no_innovation_input':innovation=torch.zeros_like(received)
            else:
                innovation=pair(action.adjoint(observed-action(complex_iq(current))))
                innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
            prior=base+SCALE*prior_core(torch.cat((received,base,current,innovation,logvar),1))
            current=prior if self.arm=='prior_no_CG' else pair(weighted_map(action,observed,complex_iq(prior),variance,steps=2,initial=complex_iq(current)))
            outputs.append(current)
        return outputs,variance,action,base
    def forward(self,x):return self.stage_outputs(x)[0][-1]


def build(parent,arm):
    from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
    if arm not in ARMS:raise ValueError('Registered G mechanism required')
    if arm=='coarse_only':return CoarseOnly(parent).requires_grad_(True)
    model=StageDistilled() if arm.startswith('random') else StageDistilled(parent)
    model.requires_grad_(True)
    return MechanismG(model,arm) if arm in ('prior_no_CG','no_image_action','no_innovation_input') else model


def loss(student,teacher,arm,received,clean,epoch,desired=None):
    from experiments.pamf_r44_pruned_refine.models import spectral_error
    values=student.stage_outputs(received)[0]
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(values[-1]-clean.detach()).square().mean((1,2))/energy
    total=paired+.05*spectral_error(values[-1],clean)
    if arm not in ('copy_no_KD','random_no_KD'):
        if desired is None:
            with torch.no_grad():desired=teacher.stage_outputs(received)[0]
        desired=desired[-len(values):]
        if len(desired)!=len(values) or any(a.shape!=b.shape or b.requires_grad for a,b in zip(values,desired)):
            raise ValueError('Detached, aligned teacher targets required')
        fidelity=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(values,desired)]).mean(0)
        total=total+(1.-.9*(epoch-1)/19.)*fidelity
    return total
