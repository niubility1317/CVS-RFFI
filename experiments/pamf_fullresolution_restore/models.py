import torch
from torch import nn
from . import design as d
from experiments.pamf_restoration.models import Block,zero_output,complex_iq,pair
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_uncertainty_restore.models import weighted_map


class FullResolutionTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(4,128,5,padding=2)
        self.blocks=nn.ModuleList(Block(128,128,dilation) for dilation in (1,2,4,8,16,32))
        self.out=zero_output(nn.Conv1d(128,2,1))

    def forward(self,value):
        if value.ndim!=3 or value.shape[1:]!=(4,256):raise ValueError('Current received/coarse IQ4x256 required')
        value=self.stem(value)
        for block in self.blocks:value=block(value)
        return self.out(value)


class ResolutionRestorer(nn.Module):
    def __init__(self,candidate,frozen):
        super().__init__()
        if candidate not in d.ROWS:raise ValueError('Registered scratch prior required')
        self.candidate=candidate
        self.frozen=frozen.requires_grad_(False).eval()
        self.prior=Core(incoming=4) if candidate=='unet_control' else FullResolutionTCN()

    def train(self,mode=True):
        super().train(mode);self.frozen.eval();return self

    def training_outputs(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):raise ValueError('Current received IQ2x256 only')
        with torch.no_grad():base,action,variance=self.frozen.estimate(received)
        prior=base+SCALE*self.prior(torch.cat((received,base),1))
        variance=variance[:,None].expand(-1,256)
        output=pair(weighted_map(action,complex_iq(received),complex_iq(prior),variance))
        return output,variance,action,base

    def forward(self,received):return self.training_outputs(received)[0]
