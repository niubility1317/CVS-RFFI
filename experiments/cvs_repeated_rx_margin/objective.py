import math
import torch
from torch.nn import functional as F
from experiments.cvs_finite_margin.objective import deterministic,robust_loss
from .drift import challenges

def supervised(model,out,y,delta,kind,lam):
 logits=out['tx_logits'];ce=F.cross_entropy(logits,y)
 if delta.shape!=(len(y),4,160) or delta.requires_grad or not torch.isfinite(delta).all() or (delta.norm(dim=-1)>.200001).any():raise ValueError('Detached bounded four source drift candidates required')
 if not lam or kind=='base' or not bool(delta.any()):return ce,ce,ce*0,dict(candidate_count=5,margin_threat=0.,aux_minus_CE=0.,zero_strength_exact=True)
 u=F.normalize(out['id_feat_joint'],dim=1,eps=1e-4);shifted=F.normalize(u[:,None,:]+delta,dim=2,eps=1e-4)
 extra=30*F.linear(shifted,F.normalize(model.head.weight,dim=1,eps=1e-4));extra=torch.where((delta.norm(dim=-1)==0)[:,:,None],logits[:,None,:],extra)
 candidates=torch.cat((logits[None],extra.transpose(0,1)),0);env,worst=robust_loss(candidates,y)
 aux=F.cross_entropy(candidates.flatten(0,1),y.repeat(5)) if kind=='mean CE' else env
 original=logits.gather(1,y[:,None])-logits;mask=F.one_hot(y,6).bool();om=original.masked_fill(mask,torch.inf).min(1).values;wm=worst.masked_fill(mask,torch.inf).min(1).values
 net=lam*(aux-ce)
 return ce,aux,net,dict(candidate_count=5,margin_threat=float((om-wm).detach().mean()),original_min_margin=float(om.detach().mean()),worst_min_margin=float(wm.detach().mean()),env_minus_maxCE=float((env-torch.stack([F.cross_entropy(v,y) for v in candidates]).max()).detach()),aux_minus_CE=float((aux-ce).detach()),zero_strength_exact=False)

def objective(model,x,y,rx,ux,ids,passes,bank,c,step,epoch,stats,folds,day):
 out=model.forward_identity_only(x);delta,q=challenges(stats,y,rx,day,folds,epoch);estimated_norm=float(delta.norm(dim=-1).mean())
 if c['arm']=='MATCHED_BASE':delta=torch.zeros_like(delta)
 ce,aux,net,diag=supervised(model,out,y,delta,c['drift']['objective'],c['drift']['lambda_weight']);kl=ce.new_zeros(());ramp=0.;accepted=0
 if ux is not None:
  if step<=1110 or len(ux)!=128 or bank is None:raise ValueError('Illegal sourceU timing/batch')
  with deterministic(model),torch.no_grad():prob=model(ux).softmax(1).detach()
  confidence,pseudo=prob.max(1);mask,history=bank.update(ids,pseudo,confidence,passes);accepted=int(mask.sum());gen=torch.Generator(device=ux.device).manual_seed(c['model_seed']+step*9901);strong=ux+torch.randn(ux.shape,generator=gen,device=ux.device)*.015
  with torch.random.fork_rng(devices=[ux.device.index or 0] if ux.is_cuda else []):
   torch.manual_seed(c['model_seed']+step*1979);uz=model(strong)
  kl=(F.kl_div(uz.log_softmax(1),prob,reduction='none').sum(1)*mask).reshape(4,32).mean(1).mean();ramp=min(1.,max(0.,(step-1110)/1110))
 else:history=None
 loss=ce+.25*ramp*kl+net
 metrics=dict(CE=float(ce.detach()),L_accuracy=float((out['tx_logits'].argmax(1)==y).float().mean()),auxiliary_supervised=float(aux.detach()),net_added=float(net.detach()),drift_strength=min(1.,max(0.,(epoch-5)/10)),estimated_drift_mean_norm=estimated_norm,effective_drift_mean_norm=float(delta.norm(dim=-1).mean()),q_mean=float(q.mean()),q_zero_fraction=float((q==0).float().mean()),margin_threat=diag['margin_threat'],KL=float(kl.detach()),weighted_KL=float(.25*ramp*kl.detach()),ramp=ramp,U_packets=128 if ux is not None else 0,accepted_U=accepted,teacher_forwards=int(ux is not None),total_loss=float(loss.detach()))
 return loss,metrics,dict(drift=diag,history=history,_net_tensor=net)

def gradient_diagnostics(loss,net,params,head):
 base=loss-net;bg=torch.autograd.grad(base,params,allow_unused=True,retain_graph=True);ng=torch.autograd.grad(net,params,allow_unused=True,retain_graph=True);result={}
 for name,ishead in [('encoder',False),('cosine_head',True)]:
  bn=nn=dot=loss.new_zeros(())
  for p,b,n in zip(params,bg,ng):
   if (p is head)!=ishead:continue
   if b is not None:bn=bn+b.detach().square().sum()
   if n is not None:nn=nn+n.detach().square().sum()
   if b is not None and n is not None:dot=dot+(b.detach()*n.detach()).sum()
  result[name]=dict(base_norm=float(bn.sqrt()),net_added_norm=float(nn.sqrt()),ratio=float(nn.sqrt()/bn.sqrt().clamp_min(1e-12)),cosine=float((dot/(bn*nn).sqrt().clamp_min(1e-20)).clamp(-1,1)) if nn>0 and bn>0 else None)
  result[name]['angle_degrees']=math.degrees(math.acos(result[name]['cosine'])) if result[name]['cosine'] is not None else None
 return result
