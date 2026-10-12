import torch
from torch.nn import functional as F
from experiments.cvs_cross_rx_feedback.method import parameter_selection,preconditioner
from experiments.cvs_finite_margin.objective import deterministic

@torch.no_grad()
def reward(model,x,params,gm,opt,epsilon=.001):
 P=preconditioner(params,opt);direction=[torch.zeros_like(p) if g is None else g.detach()*a for p,g,a in zip(params,gm,P)];norm=sum(v.square().sum() for v in direction).sqrt().clamp_min(1e-20);direction=[v/norm for v in direction];saved=[p.detach().clone() for p in params]
 try:
  with deterministic(model):
   for p,s,v in zip(params,saved,direction):p.copy_(s+epsilon*v)
   plus=model(x).log_softmax(1)
   for p,s,v in zip(params,saved,direction):p.copy_(s-epsilon*v)
   minus=model(x).log_softmax(1)
 finally:
  for p,s in zip(params,saved):p.copy_(s)
 if any(not torch.equal(p,s) for p,s in zip(params,saved)):raise ValueError('Student virtualreward restore differs')
 raw=(plus-minus)/(2*epsilon);raw=raw-raw.mean(1,keepdim=True);scale=raw.square().mean().sqrt().clamp_min(1e-4);r=(raw/scale).clamp(-2,2).detach()
 if not torch.isfinite(r).all():raise FloatingPointError('Finite-difference teacherreward')
 return r,dict(reward_raw_RMS=float(scale),reward_abs_mean=float(r.abs().mean()),virtual_restore_exact=True,virtual_direction_norm=float(norm))

def gradients(student,teacher,x,y,mx,my,ux,ids,passes,bank,sopt,sp,tp,c,step):
 # No student main graph exists during its virtual parameter perturbations.
 r=detail=None
 if ux is not None:
  if step<=1110 or len(ux)!=128 or len(mx) not in (25,26) or len(ids)!=128 or bank is None:raise ValueError('Illegal sourcefeedback U batch/timing')
  with deterministic(student):mloss=F.cross_entropy(student(mx),my)
  gm=torch.autograd.grad(mloss,sp,allow_unused=True);r,detail=reward(student,ux,sp,gm,sopt,c['feedback_teacher']['finite_difference_epsilon'])
 devices=[ux.device.index or 0] if ux is not None and ux.is_cuda else [x.device.index or 0] if x.is_cuda else []
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(c['feedback_teacher']['teacher_init_seed']+step*2719);tx=teacher(x)
 tce=F.cross_entropy(tx,y);teacherparams=[p for p in teacher.parameters() if p.requires_grad];tcg=torch.autograd.grad(tce,teacherparams,allow_unused=True)
 q=None;aux=tce.detach()*0;auxnorm=0.;mask=None;ramp=0.
 if ux is not None:
  with deterministic(teacher):q=teacher(ux).softmax(1)
  aux=(q*r).sum(1).mean();tag=torch.autograd.grad(aux,tp,allow_unused=True);auxnorm=sum(float(g.detach().square().sum()) for g in tag if g is not None)**.5;ramp=min(1.,max(0.,(step-1110)/1110));extra={id(p):g for p,g in zip(tp,tag)};weight=.1*ramp if c['arm']=='FEEDBACK_TEACHER' else 0.
 else:extra={};weight=0.
 tg=[]
 for p,g in zip(teacherparams,tcg):
  a=extra.get(id(p));g=torch.zeros_like(p) if g is None else g.detach();tg.append(g if a is None else g+weight*a.detach())
 logits=student(x);sce=F.cross_entropy(logits,y);kl=sce.detach()*0;accepted=0;hist=[0]*6
 if ux is not None:
  q=q.detach();confidence,pseudo=q.max(1);mask,history=bank.update(ids,pseudo,confidence,passes);accepted=int(mask.sum());hist=torch.bincount(pseudo[mask],minlength=6).tolist()
  gen=torch.Generator(device=ux.device).manual_seed(c['model_seed']+step*9901);strong=ux+torch.randn(ux.shape,generator=gen,device=ux.device)*.015
  with torch.random.fork_rng(devices=[ux.device.index or 0] if ux.is_cuda else []):
   torch.manual_seed(c['model_seed']+step*1979);uz=student(strong)
  kl=(F.kl_div(uz.log_softmax(1),q,reduction='none').sum(1)*mask).mean()
  detail.update(accepted_mask=mask.cpu().tolist(),pseudo_prediction_classes=hist,history=history)
 else:detail=dict(reward_raw_RMS=0.,reward_abs_mean=0.,virtual_restore_exact=True,virtual_direction_norm=0.)
 loss=sce+.25*ramp*kl;loss.backward()
 for p,g in zip(teacherparams,tg):p.grad=g
 metrics=dict(student_CE=float(sce.detach()),teacher_CE=float(tce.detach()),student_KL=float(kl.detach()),weighted_KL=float(.25*ramp*kl.detach()),teacher_aux_raw=float(aux.detach()),teacher_aux_weighted=float(weight*aux.detach()),teacher_aux_gradient_norm=auxnorm,student_total=float(loss.detach()),teacher_total=float(tce.detach()+weight*aux.detach()),student_train_accuracy=float((logits.argmax(1)==y).float().mean()),teacher_train_accuracy=float((tx.argmax(1)==y).float().mean()),U_packets=128 if ux is not None else 0,accepted_U=accepted,teacher_aux_active=int(ux is not None),ramp=ramp)
 return metrics,detail
