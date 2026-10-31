"""Per-record global prior context; received-derived inputs only, no routing."""
import torch
from torch import nn
from torch.nn import functional as F

BOUND=.1
HIDDEN=32


def descriptor(value):
    if value.ndim!=3 or value.shape[1:]!=(9,256):
        raise ValueError('Actual nine-channel prior input Bx9x256 required')
    # The three statistics retain phase-sensitive means, energy and temporal
    # variation. No across-record statistic, physical state or F input is used.
    return torch.cat((value.mean(-1),value.square().mean(-1),
                      (value[:,:,1:]-value[:,:,:-1]).square().mean(-1)),1)


class ContextPrior(nn.Module):
    def __init__(self,core,seed):
        super().__init__()
        if getattr(core,'conditional',False) or len(core.encoder.blocks)!=4:
            raise ValueError('Original unconditional four-level prior Core required')
        self.encoder,self.decode,self.out=core.encoder,core.decode,core.out
        self.widths=tuple(b[0].net[0].out_channels for b in self.encoder.blocks)
        # Preserve the paired control's CPU state and CUDA probe stream. The
        # zero final layer makes the complete prior exactly unmodulated at E0.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.conditioner=nn.Sequential(nn.Linear(27,HIDDEN),nn.SiLU(),nn.Linear(HIDDEN,2*sum(self.widths)))
            nn.init.zeros_(self.conditioner[-1].weight);nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self,value):
        parameters=self.conditioner(descriptor(value));offset=0;features=[]
        for index,(block,width) in enumerate(zip(self.encoder.blocks,self.widths)):
            value=block(F.avg_pool1d(value,2) if index else value)
            scale=1+BOUND*parameters[:,offset:offset+width].tanh();offset+=width
            shift=BOUND*parameters[:,offset:offset+width].tanh();offset+=width
            value=value*scale[:,:,None]+shift[:,:,None];features.append(value)
        for block,skip in zip(self.decode,features[-2::-1]):
            value=block(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        return self.out(value)


def attach(model,seed):
    if len(model.prior)!=2 or any(isinstance(p,ContextPrior) for p in model.prior):
        raise ValueError('One original two-prior graph required')
    model.prior=nn.ModuleList(ContextPrior(core,int(seed)+2026103110+i) for i,core in enumerate(model.prior))
    added=sum(v.numel() for p in model.prior for v in p.conditioner.parameters())
    model.initialization_report=dict(model.initialization_report,context_film_parameters=added,
        context_film_input='per-recordactualprior9channels/mean-energy-adjacent-difference-energy/27D',
        context_film_hidden=HIDDEN,context_film_bound=BOUND,context_film_levels=4,
        context_film_final_layer_zero=True,context_film_CPU_CUDA_rng_preserved=True,
        scene_or_RX_or_F_inputs=False)
    return model
