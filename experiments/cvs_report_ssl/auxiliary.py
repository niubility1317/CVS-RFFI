"""Independent physical readouts; none predicts TX labels or scalar R margins."""
import torch
from torch import nn
from torch.nn import functional as F
from .physical import iq,aligned_cycles,repetition_quality,known_linear
from .objectives import physical_weights

class PhysicalHeads(nn.Module):
 def __init__(self,linear=False,repetition=False):
  super().__init__()
  self.linear=nn.Sequential(nn.Linear(698,96),nn.SiLU(),nn.Linear(96,4)) if linear else None
  self.repetition=nn.Sequential(nn.Linear(349,96),nn.SiLU(),nn.Linear(96,36)) if repetition else None

def observable_weights(x,task):
 w=physical_weights(repetition_quality(x),task)
 return w*(iq(x).abs().square().mean(1).detach()>1e-8)

@torch.no_grad()
def repetition_plan(model,x,cycle):
 """Context excludes one cycle; existing packet RMS/alignment is shared.

 This is an ordinary residual/reliability task, NOT a strict blind spot.
 Public-template routing and normalization are from the other three cycles.
 """
 if cycle not in range(4):raise ValueError('Expected one of four cycles')
 cycles=aligned_cycles(x);context_mean=cycles[:,[j for j in range(4) if j!=cycle]].mean(1)
 context=x.clone();context[:,0,80+cycle*20:100+cycle*20]=context_mean.real
 context[:,1,80+cycle*20:100+cycle*20]=context_mean.imag
 frontend=next(m.response for m in model.id_backbone.modules() if hasattr(m,'response_projection'))
 scale=context_mean.abs().square().mean(1).clamp_min(1e-10).sqrt()
 unit=context_mean/scale[:,None];bank=torch.complex(frontend.bank_real.to(x.dtype),frontend.bank_imag.to(x.dtype))
 corr=unit@bank.conj().T/20;chosen=corr.abs().square().argmax(1)
 reference=bank[chosen];gain=corr.gather(1,chosen[:,None]).squeeze(1)
 ideal=torch.fft.fft(reference)[:,frontend.tone_bins]
 def residual(value):
  transfer=torch.fft.fft(value/scale[:,None])[:,frontend.tone_bins]*ideal.conj()/ideal.abs().square().clamp_min(1e-10)
  r=transfer*gain.conj()[:,None]/(gain.abs().square()+1e-6)[:,None]-1
  r=r/torch.sqrt(1+r.abs().square()/16)
  return torch.cat((r.real,r.imag),1)
 target=residual(cycles[:,cycle]);baseline=residual(context_mean)
 return dict(x=context.detach(),target=target.detach(),period_mean_baseline=baseline.detach(),weights=observable_weights(x,'repetition'),cycle=cycle)

def linear_objective(model,heads,x,parameters,y=None,scale=.015):
 changed=known_linear(x,parameters,scale)
 clean_logits,_,h0=model.representation(x);changed_logits,_,h1=model.representation(changed)
 prediction=heads.linear(torch.cat((h0,h1-h0),1));w=observable_weights(x,'linear')
 per=F.smooth_l1_loss(prediction,parameters,reduction='none').mean(1)
 # Only legal L can use TX CE. U stability is a detached same-packet soft target.
 if y is not None:stable=F.cross_entropy(changed_logits,y)
 else:stable=(F.kl_div(F.log_softmax(changed_logits,1),clean_logits.detach().softmax(1),reduction='none').sum(1)*w).sum()/len(x)
 return (per*w).sum()/len(x),stable,dict(prediction_mse=float((prediction-parameters).detach().square().mean()),zero_baseline_mse=float(parameters.square().mean()),usable=float(w.mean()),actual_tap_rms=float((parameters*scale).square().mean().sqrt()),IQ_relative_delta=float((changed-x).flatten(1).norm(dim=1).div(x.flatten(1).norm(dim=1).clamp_min(1e-8)).mean()))

def repetition_objective(model,heads,plan):
 _,_,h=model.representation(plan['x']);value=heads.repetition(h)
 pred=value[:,:24];logvar=value[:,24:].clamp(-6,4)
 error=(pred-plan['target']).square();pair=error[:,:12]+error[:,12:]
 nll=.5*(pair*torch.exp(-logvar)+2*logvar).mean(1)
 loss=((nll+.1*error.mean(1))*plan['weights']).sum()/len(pred)
 return loss,dict(prediction_mse=float(error.detach().mean()),zero_baseline_mse=float(plan['target'].square().mean()),period_mean_baseline_mse=float((plan['target']-plan['period_mean_baseline']).square().mean()),diagonal_variance_mean=float(logvar.detach().exp().mean()),usable=float(plan['weights'].mean()),cycle=plan['cycle'],strict_blindspot=False)
