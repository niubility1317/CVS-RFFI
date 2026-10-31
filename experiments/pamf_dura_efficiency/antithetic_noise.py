"""Source-only residual sign augmentation with consistent RMS/action/noise."""
import torch


class ScaledAction:
    def __init__(self,action,gain):self.action,self.gain=action,gain
    def __call__(self,value):return self.gain[:,None]*self.action(value)
    def adjoint(self,value):return self.action.adjoint(self.gain[:,None]*value)


@torch.no_grad()
def augment(received,clean,action,variance,requested):
    if (received.ndim!=3 or received.shape[1:]!=(2,256) or clean.shape!=received.shape
            or variance.shape!=(len(received),) or requested.shape!=(len(received),) or requested.dtype!=torch.bool):
        raise ValueError('Matched source B2x256 IQ, positive variance and bool mask required')
    if not all(torch.isfinite(v).all() for v in [received,clean,variance]) or (variance<=0).any():
        raise ValueError('Finite source IQ and positive variance required')
    observed=torch.complex(received[:,0],received[:,1]);x=torch.complex(clean[:,0],clean[:,1])
    noiseless=action(x);residual=observed-noiseless;power=observed.abs().square().mean(1)
    if not torch.isfinite(noiseless).all() or (power<=1e-12).any():raise ValueError('Finite positive-power source observation required')
    # Skip numerical replay residue of exact clean examples; do not invent noise.
    effective=requested&(residual.abs().square().mean(1)>1e-12*power)
    proposal=torch.where(effective[:,None],noiseless-residual,observed)
    proposed_power=proposal.abs().square().mean(1)
    if not torch.isfinite(proposal).all() or (proposed_power<=1e-12).any():raise ValueError('Antithetic source observation has invalid power')
    gain=torch.where(effective,(power/proposed_power).sqrt(),torch.ones_like(power))
    output=gain[:,None]*proposal;new_variance=gain.square()*variance
    if not torch.isfinite(output).all() or not torch.isfinite(new_variance).all() or (new_variance<=0).any():
        raise FloatingPointError('Nonfinite source sign augmentation')
    ratio=output.abs().square().mean(1)/power
    if (ratio-1).abs().max()>3e-6:raise FloatingPointError('Received source power changed')
    return torch.stack((output.real,output.imag),1),ScaledAction(action,gain),new_variance,dict(
        requested=requested.float(),effective=effective.float(),gain=gain,power_ratio=ratio)
