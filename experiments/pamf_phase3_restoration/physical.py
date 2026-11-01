"""Scratch jointly learned action plus two unshared data-consistent MSAT priors."""
import math
import torch
from torch import nn
from .multiscale import MultiscaleAttentionRestorer
from .physical_operators import Action,soft_map

PRIOR_VARIANCE=.01
CG_STEPS=8
PRIOR_COUNT=2
MIN_NOISE=1e-7
TOTAL_CG=PRIOR_COUNT*CG_STEPS
COMPLEX_FFT_COUNT=1+PRIOR_COUNT*(8+4*CG_STEPS)


def complex_iq(value):return torch.complex(value[:,0],value[:,1])
def pair(value):return torch.stack((value.real,value.imag),dim=1)


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('coordinates',torch.linspace(-1,1,256)[None,None])
        self.stem=nn.Conv1d(3,64,7,padding=3)
        self.blocks=nn.ModuleList(nn.Sequential(nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv1d(64,64,3,padding=d,dilation=d),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv1d(64,64,3,padding=1)) for d in (1,2,4,8))
        self.kernel_head=nn.Linear(64,128)
        self.coefficient_head=nn.Conv1d(64,4,3,padding=1)
        self.noise_head=nn.Linear(64,1)
        for head in (self.kernel_head,self.coefficient_head,self.noise_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        impulse=torch.zeros(64);impulse[0]=1
        self.register_buffer('impulse',impulse)

    def forward(self,received):
        x=self.stem(torch.cat((received,self.coordinates.expand(len(received),-1,-1)),1))
        for block in self.blocks:x=x+block(x)
        pooled=x.mean(-1);h=self.kernel_head(pooled)
        kernel=torch.complex(self.impulse+h[:,:64],h[:,64:])
        kernel=kernel/kernel.abs().square().sum(-1,keepdim=True).sqrt().clamp_min(1e-6)
        c=self.coefficient_head(x)
        main=1+torch.complex(c[:,0],c[:,1]);image=torch.complex(c[:,2],c[:,3])
        log_variance=(math.log(.0005)+self.noise_head(pooled).squeeze(1)).clamp(math.log(MIN_NOISE),0.)
        return Action(kernel,main,image),log_variance.exp()


class PhysicalMSAT(nn.Module):
    def __init__(self):
        super().__init__();self.estimator=ActionEncoder()
        # Context is original received IQ, real adjoint residual, and log noise.
        self.priors=nn.ModuleList(MultiscaleAttentionRestorer(context_channels=5) for _ in range(PRIOR_COUNT))

    def forward(self,received,*,return_aux=False):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Only current per-record FP32 IQ2x256 required')
        action,variance=self.estimator(received);observed=complex_iq(received);current=observed
        condition=variance.log()[:,None,None].expand(-1,1,256)
        for prior in self.priors:
            back=action.adjoint(observed-action(current))
            context=torch.cat((received,pair(back),condition),1)
            mean=complex_iq(prior(pair(current),context))
            current=soft_map(action,observed,mean,variance,PRIOR_VARIANCE,steps=CG_STEPS)
        restored=pair(current)
        return (restored,dict(action=action,variance=variance)) if return_aux else restored


def action_supervision(aux,clean,kernel,coefficients,variance):
    n=len(clean)
    if (clean.shape!=(n,2,256) or kernel.shape!=(n,2,64) or coefficients.shape!=(n,4,256)
            or variance.shape!=(n,) or not all(torch.isfinite(t).all() for t in (kernel,coefficients,variance))
            or (variance<0).any()):raise ValueError('Matched finite source simulator action/noise required')
    with torch.no_grad():
        teacher=Action(complex_iq(kernel),torch.complex(coefficients[:,0],coefficients[:,1]),
            torch.complex(coefficients[:,2],coefficients[:,3]))
        expected=teacher(complex_iq(clean.detach()))
    actual=aux['action'](complex_iq(clean.detach()))
    error=(actual-expected).abs().square().mean(-1)/expected.abs().square().mean(-1).clamp_min(1e-4)
    noise_error=(aux['variance'].log()-variance.detach().clamp_min(MIN_NOISE).log()).square()
    return .1*error+.001*noise_error,dict(action_forward_nmse=error,noise_log_mse=noise_error)


def scratch_physical(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=PhysicalMSAT()
    model.initialization_report=dict(mode='scratch_joint_single30',model_seed=seed,ancestor_checkpoints=[],
        training_stages=1,prior_count=PRIOR_COUNT,CG_steps_each=CG_STEPS,TX_labels_read=0,target_reads=0)
    return model
