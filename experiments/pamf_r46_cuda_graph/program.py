"""Original R46/R18 arithmetic with value checks outside graph capture.

The original Action/weighted_map routines convert finite/positive CUDA scalars
to Python booleans. Here only those checks are moved out of the captured work;
the kernels, reflect boundary, conjugate adjoint, warm start and two CG steps
per stage keep the original operation order and original module parameters.
"""
import math
import torch
from torch.nn import functional as F
from experiments.pamf_action_restore.models import KERNEL_LENGTH, SCALE
from experiments.pamf_action_restore.operators import Action, real_dot
from experiments.pamf_restoration.models import complex_iq, pair


class StaticAction(Action):
    def __init__(self, kernel, main, image):
        if kernel.ndim != 2 or main.ndim != 2 or main.shape != image.shape or len(kernel) != len(main):
            raise ValueError('Batched complex FIR/main/image required')
        if not all(t.is_complex() for t in (kernel,main,image)):
            raise ValueError('Complex action required')
        self.kernel,self.main,self.image=kernel,main,image
        self.count,self.length=main.shape[-1],kernel.shape[-1]
        if not 1 <= self.length <= self.count:
            raise ValueError('Reflect FIR support must fit within the current window')
        self.index=torch.cat((torch.arange(self.length-1,0,-1,device=main.device),torch.arange(self.count,device=main.device)))
        self.nfft=1 << (self.count+2*self.length-3).bit_length()
        self.response=torch.fft.fft(kernel,n=self.nfft)


def estimate(estimator, received, coarse):
    features=estimator.core.encoder(torch.cat((received,coarse),1))
    value=features[-1]
    global_value=estimator.global_head(value.mean(-1))
    kernel=torch.complex(global_value[:,:KERNEL_LENGTH]+estimator.initial_kernel,
                         global_value[:,KERNEL_LENGTH:2*KERNEL_LENGTH])
    kernel=kernel/kernel.abs().square().sum(-1,keepdim=True).sqrt().clamp_min(1e-6)
    for block,skip in zip(estimator.core.decode,features[-2::-1]):
        value=block(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
    coefficients=estimator.core.out(value)
    main=1+torch.complex(coefficients[:,0],coefficients[:,1])
    image=torch.complex(coefficients[:,2],coefficients[:,3])
    log_variance=math.log(1e-8)+(math.log(1.)-math.log(1e-8))*torch.sigmoid(global_value[:,-1]+.351)
    return StaticAction(kernel,main,image),log_variance.exp()


def weighted_map(action, observed, prior, variance, *, initial):
    # This experiment always uses expanded Bx256 variance, a matched complex
    # warm start, fixed SCALE**2 and exactly two iterations.
    if variance.shape != observed.shape or variance.is_complex() or initial.shape != prior.shape or not initial.is_complex():
        raise ValueError('Static matched variance and complex warm start required')
    prior_variance=SCALE**2
    weight=variance.reciprocal()
    rhs=action.adjoint(weight*(observed-action(prior)))
    delta=initial-prior
    residual=rhs-action.adjoint(weight*action(delta))-delta/prior_variance
    direction=residual
    squared=real_dot(residual,residual)
    for _ in range(2):
        normal=action.adjoint(weight*action(direction))+direction/prior_variance
        alpha=squared/real_dot(direction,normal).clamp_min(1e-20)
        delta=delta+alpha[:,None]*direction
        residual=residual-alpha[:,None]*normal
        following=real_dot(residual,residual)
        direction=residual+(following/squared.clamp_min(1e-20))[:,None]*direction
        squared=following
    return prior+delta


def outputs(model, received):
    if received.ndim != 3 or received.shape[1:] != (2,256) or len(model.prior) != 3:
        raise ValueError('Original three-stage R46/R18 current IQ required')
    base=model.frozen.coarse(received)
    action,native=estimate(model.frozen.estimator,received,base)
    variance=native[:,None].expand(-1,256)
    observed,current=complex_iq(received),base
    logvar=native.log().clamp(-16,4)[:,None,None].expand(-1,1,256)/8
    for index in range(3):
        innovation=pair(action.adjoint(observed-action(complex_iq(current))))
        innovation=innovation/(innovation.square().mean((1,2),keepdim=True)+1e-6).sqrt()
        inputs=torch.cat((received,base,current,innovation,logvar),1)
        prior=base+SCALE*model.prior[index](inputs)
        current=pair(weighted_map(action,observed,complex_iq(prior),variance,initial=complex_iq(current)))
    return current,variance,action,base
