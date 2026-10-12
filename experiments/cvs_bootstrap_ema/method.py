import torch
from torch.nn import functional as F
def alpha(step):return max(0.,min(.8,.8*(4440-step)/(4440-1111))) if step>1110 else 0.
@torch.no_grad()
def ema_update(ema,student,step):
 decay=min(.99,1-1/(step+1));targets=dict(ema.named_parameters());sources=dict(student.named_parameters())
 if targets.keys()!=sources.keys():raise ValueError('EMA parameter schema differs')
 for name,p in targets.items():p.mul_(decay).add_(sources[name].detach(),alpha=1-decay)
 for name,b in ema.named_buffers():b.copy_(dict(student.named_buffers())[name])
 return decay
def target(ema_logits,anchor,step):
 q=ema_logits.detach().softmax(1)
 if anchor is not None:
  if anchor.shape!=q.shape or not torch.isfinite(anchor).all() or (anchor<0).any() or not torch.allclose(anchor.sum(1),torch.ones(len(anchor),device=anchor.device),atol=1e-5):raise ValueError('Finite source probability anchor required')
  a=alpha(step);q=(1-a)*q+a*anchor.detach()
 return q.detach()
def objective(student,x,y,ux,ema,anchor,ids,passes,bank,step,seed):
 logits=student(x);ce=F.cross_entropy(logits,y);kl=logits.sum()*0;detail=dict(U_packets=0,accepted_U=0,teacher_forwards=0,anchor_weight=0.,ramp=0.,L_accuracy=float((logits.argmax(1)==y).float().mean()))
 if ux is not None:
  if step<=1110 or bank is None:raise ValueError('Unexpected early U access')
  with torch.no_grad():ema.eval();q=target(ema(ux),anchor,step)
  conf,pseudo=q.max(1);mask,history=bank.update(ids,pseudo,conf,passes)
  generator=torch.Generator(device=ux.device).manual_seed(seed+step*9901);strong=ux+torch.randn(ux.shape,generator=generator,device=ux.device,dtype=ux.dtype)*.015
  devices=[ux.device.index or 0] if ux.is_cuda else []
  with torch.random.fork_rng(devices=devices):torch.manual_seed(seed+step*1979);z=student(strong)
  kl=(F.kl_div(z.log_softmax(1),q,reduction='none').sum(1)*mask).mean();ramp=min(1.,(step-1110)/1110)
  detail.update(U_packets=len(ux),accepted_U=int(mask.sum()),teacher_forwards=1,anchor_weight=alpha(step) if anchor is not None else 0.,ramp=ramp,U_confidence=float(conf.mean()),history=history)
 else:ramp=0.
 return ce+.5*ramp*kl,dict(CE=float(ce.detach()),KL=float(kl.detach())),detail
