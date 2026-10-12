from contextlib import contextmanager
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_crossrx_relation.core import capture
from experiments.cvs_finite_action.operators import apply
@contextmanager
def evaluation_mode(model):
 modes=[(m,m.training) for m in model.modules()];before={n:b.detach().clone() for n,b in model.named_buffers()}
 devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
 try:
  with torch.random.fork_rng(devices=devices):
   model.eval();yield
  if any(not torch.equal(b,before[n]) for n,b in model.named_buffers()):raise ValueError('Eval response changed buffer')
 finally:
  for m,mode in modes:m.training=mode
def eval_forward(model,x,features=False,gradient=False):
 with evaluation_mode(model),torch.set_grad_enabled(gradient):
  if features:
   with capture(model,['time_down','t2.pool']) as h:logits=model(x)
  else:logits=model(x);h={}
 return logits,h
def KL(logits,q):return F.kl_div(logits.log_softmax(1),q,reduction='none').sum(1).mean()
def objective(c,model,peer,response,L,li,U,ui,step,device):
 x=L['x'][li].to(device);y=L['y'][li].to(device);logits=model(x);zero=logits.sum()*0
 parts={'ce':F.cross_entropy(logits,y),'KL':zero,'extra':zero};details=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),L_margin=float((logits.gather(1,y[:,None]).squeeze(1)-logits.masked_fill(F.one_hot(y,6).bool(),float('-inf')).max(1).values).mean().detach()),L_packets=len(li),U_packets=0,forward_calls=1,extra_forward_calls=0,extra_L_packets=0,extra_U_packets=0,response_readout_calls=0,response_targets_L=0,response_targets_U=0,teacher_forwards=0,EMA_updates=0,satellite_calls=0,U_accuracy=None,U_accuracy_reason='U TX truth hidden',role_task_losses={})
 ramp=min(1.,step/c['semi']['ramp_updates']);q=None;ux=None;weak=None;uh=None
 if c['unlabeled']:
  if U is None or ui is None or peer is None or any(k in U for k in ('y','label','labels','tx','TX','truth')):raise ValueError('Legal U/peer required')
  if any(peer['ids'][i]!=U['ids'][i] for i in ui.tolist()):raise ValueError('Peer U PID differs')
  ux=U['x'][ui].to(device);q=peer['q'][ui].to(device).detach();needsU=c['response']['extra_eval'] or 'U_s' in c['response']['roles']
  weak,uh=eval_forward(model,ux,features='U_s' in c['response']['roles'],gradient=needsU)
  gen=torch.Generator().manual_seed(c['model_seed']+step*9901);noise=torch.randn(ux.shape,generator=gen,dtype=ux.dtype).to(device)*c['semi']['noise_std'];ul=model(ux+noise)
  parts['KL']=KL(ul,q);details.update(U_packets=len(ui),forward_calls=3,accepted_U=len(ui),peer_JS=float(peer['peer_JS'][ui].mean()),teacher_live_disagreement=float((weak.detach().argmax(1)!=q.argmax(1)).float().mean()),teacher_entropy=float(-(q*q.clamp_min(1e-12).log()).sum(1).mean()),U_confidence=float(q.max(1).values.mean()))
 elif any(v is not None for v in (U,ui,peer,response)):raise ValueError('CE path must omit U/peer/response')
 extras=[];rc=c['response'];extraL=rc['extra_eval'] or 'L_s' in rc['roles']
 if extraL:
  el,lh=eval_forward(model,x,features='L_s' in rc['roles'],gradient=True);details.update(extra_forward_calls=1,extra_L_packets=len(li));details['forward_calls']+=1
 if rc['extra_eval']:
  extras=[F.cross_entropy(el,y),KL(weak,q)];details['extra_U_packets']=len(ui)
 if rc['enabled']:
  if response is None:raise ValueError('Response readers unavailable')
  for role,idx,h in [('L_s',li,lh if extraL else None),('U_s',ui,uh)]:
   if role not in rc['roles']:continue
   loss,info=response.loss(h,role,idx,device);extras.append(loss);details['role_task_losses'][role]=info;details['response_readout_calls']+=4;details['response_targets_L' if role=='L_s' else 'response_targets_U']+=len(idx)
   if role=='U_s':details['extra_U_packets']=len(ui)
 if rc['IQ_action']:
  kind='spectral' if step%2 else 'timing';a=rc['known_parameters'][kind];ax=apply(x,kind,a,c['model_seed']+step);au=apply(ux,kind,a,c['model_seed']+step+700000003)
  joined=torch.cat((ax,au));acted,_=eval_forward(model,joined,gradient=True);al,apu=acted.split((len(li),len(ui)));extras=[F.cross_entropy(al,y),KL(apu,q)];details.update(action_kind=kind,action_a=a,actual_IQ_operation=True,extra_forward_calls=1,extra_L_packets=len(li),extra_U_packets=len(ui));details['forward_calls']+=1
 if extras:parts['extra']=torch.stack(extras).mean()
 extra_ramp=min(1.,step/rc['ramp_updates']);weighted={'ce':parts['ce']*c['losses']['ce'],'KL':parts['KL']*c['losses']['KL']*ramp,'extra':parts['extra']*rc['weight']*extra_ramp}
 details.update(ramp=ramp,extra_ramp=extra_ramp,auxiliary_optimizer_steps=0,known_response_not_TX_label=True)
 return sum(weighted.values()),parts,weighted,details
