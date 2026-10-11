import torch
from torch.nn import functional as F
from experiments.cvs_report_ssl.objectives import soft_identity_kl,identity_weights

def objective(c,model,heads,L,indices,U,uindices,teacher,conditional,step,device):
 x=L['x'][indices].to(device);y=L['y'][indices].to(device);logits=model(x);zero=logits.sum()*0
 clean_ce=F.cross_entropy(logits,y);parts=dict(ce=clean_ce,soft_identity=zero,linear=zero,linear_identity=zero,repetition=zero,critic=zero)
 details=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),accepted_U=0,U_confidence=None,RX_L_packets=0,RX_U_packets=0,critic_packets=0,clean_CE=float(clean_ce.detach()))
 generator=torch.Generator().manual_seed(c['model_seed']+step*9901);action_generator=torch.Generator().manual_seed(c['model_seed']+step*29011)
 active=c['rx_action']['enabled'] and step>888;ramp=max(0.,min((step-888)/888.,1.))
 if active:
  changed,_,info=heads.choose(x,L,indices,True,step,action_generator);aug_ce=F.cross_entropy(model(changed),y);fraction=c['rx_action']['L_augmented_CE_fraction']*ramp
  parts['ce']=(1-fraction)*clean_ce+fraction*aug_ce;details.update(RX_L_packets=len(x),augmented_L_CE=float(aug_ce.detach()),augmented_L_fraction=fraction,L_action=info)
 if teacher is not None:
  ux=U['x'][uindices].to(device)
  with torch.no_grad():q=teacher.probabilities(ux);w=identity_weights(q,c['semi']['confidence_threshold'])
  if active:
   changed,_,info=heads.choose(ux,U,uindices,False,step,action_generator);details.update(RX_U_packets=len(ux),U_action=info)
  else:changed=ux
  noise=torch.randn(ux.shape,generator=generator,dtype=ux.dtype).to(device)*c['semi']['strong_noise_std']
  parts['soft_identity']=soft_identity_kl(model(changed+noise),q,w)
  details.update(accepted_U=int(w.sum()),U_confidence=float(q.max(1).values.mean()),U_class_accepted=torch.bincount(q.argmax(1)[w.bool()],minlength=6).cpu().tolist(),U_predictions=q.argmax(1).cpu().tolist(),U_accepted_mask=w.bool().cpu().tolist(),conditional_EMA_disagreement=0.,U_accuracy=None)
 if active and heads.critic is not None and step%c['rx_action']['critic_every']==0:
  ci=indices[-32:];loss,info=heads.critic_objective(model,x[-32:],y[-32:],L,ci,step,action_generator)
  parts['critic']=loss;details.update(critic_packets=len(ci),critic=info)
 weighted={k:v*c['losses'][k]*(ramp if k=='soft_identity' else 1.) for k,v in parts.items()}
 return sum(weighted.values()),parts,weighted,details
