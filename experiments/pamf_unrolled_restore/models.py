import copy
import torch
from torch import nn
from . import design as d
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map

class UnrolledRestorer(nn.Module):
    def __init__(self,candidate,frozen):
        super().__init__()
        if candidate not in d.ROWS:raise ValueError('Registered three-step prior required')
        self.candidate=candidate;self.frozen=frozen.requires_grad_(False).eval()
        core=Core(incoming=9)
        self.prior=nn.ModuleList([core] if candidate=='unroll_shared' else [copy.deepcopy(core) for _ in range(3)])

    def train(self,mode=True):super().train(mode);self.frozen.eval();return self

    def stage_outputs(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):raise ValueError('Only current received IQ2x256 required')
        with torch.no_grad():base,action,native=self.frozen.estimate(received)
        variance=native[:,None].expand(-1,256);observed=complex_iq(received);current=base;stages=[]
        logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
        for k in range(3):
            innovation=pair(action.adjoint(observed-action(complex_iq(current))))
            innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
            inputs=torch.cat((received,base,current,innovation,logvar),1)
            prior=base+SCALE*self.prior[0 if len(self.prior)==1 else k](inputs)
            current=pair(weighted_map(action,observed,complex_iq(prior),variance,steps=2,initial=complex_iq(current)))
            stages.append(current)
        return stages,variance,action,base

    def training_outputs(self,received):
        stages,variance,action,base=self.stage_outputs(received);return stages[-1],variance,action,base

    def forward(self,received):return self.training_outputs(received)[0]
