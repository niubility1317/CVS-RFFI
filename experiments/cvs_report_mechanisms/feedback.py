import torch
from torch.nn import functional as F
from experiments.cvs_finite_margin.objective import deterministic

def parameter_selection(model,pg):
 owned={id(p) for g in pg for p in g['params']};items=[(n,p) for n,p in model.named_parameters() if id(p) in owned and p is not model.head.weight]
 return [n for n,p in items],[p for n,p in items]

def preconditioner(params,opt):
 groups={id(p):g for g in opt.param_groups for p in g['params']};result=[]
 for p in params:
  state=opt.state.get(p,{});g=groups[id(p)];step=float(state.get('step',0))
  if not step or 'exp_avg_sq' not in state:result.append(torch.ones_like(p));continue
  correction=1-g['betas'][1]**step
  result.append((state['exp_avg_sq'].detach()/correction).sqrt().add(g['eps']).reciprocal())
 return result

def scores(gm,gu,pre):
 dot=normm=normu=None
 for a,b,p in zip(gm,gu,pre):
  if a is None or b is None:continue
  x=(a.detach()*p*b.detach()).sum();m=(a.detach()*p*a.detach()).sum();u=(b.detach()*p*b.detach()).sum()
  dot=x if dot is None else dot+x;normm=m if normm is None else normm+m;normu=u if normu is None else normu+u
 if dot is None:return 0.,0.,0.
 cosine=dot/(normm*normu).sqrt().clamp_min(1e-20)
 return float(cosine.clamp(-1.,1.)),float(normm.sqrt()),float(normu.sqrt())

def bounded_weights(values,device):
 v=torch.tensor(values,device=device).clamp_min(0);return (v/v.mean().clamp_min(1e-12)).clamp_max(2.).detach()

def objective(model,x,y,mx,my,ux,ids,passes,bank,opt,params,c,step):
 logits=model(x);ce=F.cross_entropy(logits,y);zero=logits.new_zeros(());kl=zero;weighted=zero
 detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean().detach()),U_packets=0,accepted_U=0,teacher_forwards=0,feedback_forwards=0,feedback_gradients=0,U_gradient_groups=0,mean_weight=0.,positive_groups=0,zero_U_step=1,P_gradient_M_norm=0.,P_gradient_U_norm=0.,weighted_U_L_gradient_ratio=0.,gradient_cosines=[],group_weights=[],U_prediction_histogram=[0]*6)
 if ux is None:return ce,dict(CE=float(ce.detach()),KL=0.,weighted_KL=0.,ramp=0.),detail
 if not c['source_U'] or step<=1110 or len(ux)!=128 or len(mx) not in (25,26):raise ValueError('Illegal feedback/U timing or batch')
 with deterministic(model),torch.no_grad():q=model(ux).softmax(1).detach()
 confidence,pseudo=q.max(1);mask,history=bank.update(ids,pseudo,confidence,passes)
 gen=torch.Generator(device=ux.device).manual_seed(c['model_seed']+step*9901);strong=ux+torch.randn(ux.shape,generator=gen,device=ux.device,dtype=ux.dtype)*.015
 with deterministic(model):mloss=F.cross_entropy(model(mx),my)
 gm=torch.autograd.grad(mloss,params,allow_unused=True);pre=preconditioner(params,opt)
 devices=[ux.device.index or 0] if ux.is_cuda else []
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(c['model_seed']+step*1979);uz=model(strong)
 losses=F.kl_div(uz.log_softmax(1),q,reduction='none').sum(1)*mask
 micro=losses.reshape(4,32).mean(1);cosines=[];unorm=[];ug=[];mn=0.
 for loss in micro:
  gu=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True);value,mn,un=scores(gm,gu,pre);cosines.append(value);unorm.append(un);ug.append([v.detach() if v is not None else None for v in gu])
 weights=bounded_weights(cosines,ux.device) if c['arm']=='META_SSL' else torch.ones(4,device=ux.device)
 kl=micro.mean();weighted=(micro*weights).mean();ramp=min(1.,max(0.,(step-1110)/1110))
 gl=torch.autograd.grad(ce,params,allow_unused=True,retain_graph=True);ln=zero.clone();un=zero.clone()
 for i,g in enumerate(gl):
  if g is not None:ln=ln+g.detach().square().sum()
  combined=sum((weights[j]*ug[j][i]/4 for j in range(4) if ug[j][i] is not None),start=torch.zeros_like(params[i]))
  un=un+combined.square().sum()
 ratio=.25*ramp*un.sqrt()/ln.sqrt().clamp_min(1e-20)
 detail.update(U_packets=128,accepted_U=int(mask.sum()),teacher_forwards=1,feedback_forwards=1,feedback_gradients=1,U_gradient_groups=4,mean_weight=float(weights.mean()),positive_groups=sum(v>0 for v in cosines),zero_U_step=int(not (weights>0).any() or not mask.any()),P_gradient_M_norm=mn,P_gradient_U_norm=sum(unorm)/4,weighted_U_L_gradient_ratio=float(ratio),gradient_cosines=cosines,group_weights=weights.tolist(),U_prediction_histogram=torch.bincount(pseudo[mask],minlength=6).tolist(),accepted_mask=mask.cpu().tolist(),history=history,U_confidence=float(confidence.mean()),U_entropy=float(-(q*q.clamp_min(1e-12).log()).sum(1).mean()))
 return ce+.25*ramp*weighted,dict(CE=float(ce.detach()),KL=float(kl.detach()),weighted_KL=float(weighted.detach()),ramp=ramp),detail
