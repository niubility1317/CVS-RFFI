import torch
from torch.nn import functional as F
from experiments.cvs_u_geometry_audit.geometry import terms

def output(model,x):
 out=model.forward_identity_only(x);return out['tx_logits'],out['id_feat_joint']

@torch.no_grad()
def weak(model,x):
 modes=[(m,m.training) for m in model.modules()]
 try:model.eval();return tuple(v.detach() for v in output(model,x))
 finally:
  for m,mode in modes:m.training=mode

def objective(c,model,bank,L,li,U,ui,step,device):
 x=L['x'][li].to(device);y=L['y'][li].to(device);logits=model(x);zero=logits.sum()*0
 parts=dict(ce=F.cross_entropy(logits,y),KL=zero,feature=zero,margin=zero)
 ramp=max(0.,min(1.,(step-c['semi']['warmup_updates'])/c['semi']['ramp_updates']))
 active=step>c['semi']['warmup_updates'];physical=active and c['rx_action']['enabled']
 gen=torch.Generator().manual_seed(c['model_seed']+step*29011)
 detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),U_packets=0,accepted_U=0,physical_L=0,physical_U=0,feature_U=0,margin_U=0,ramp=ramp,weak_forwards=0)
 if physical:
  changed,_,_=bank.choose(x,L,li,True,step,gen,uniform=True)
  parts['ce']=(1-.5*ramp)*parts['ce']+.5*ramp*F.cross_entropy(model(changed),y)
  detail['physical_L']=len(x)
 if ui is not None:
  if not active or not c['unlabeled']:raise ValueError('Unexpected U forward')
  ux=U['x'][ui].to(device);a,f=weak(model,ux);changed=ux
  if physical:changed,_,_=bank.choose(ux,U,ui,False,step,gen,uniform=True);detail['physical_U']=len(ux)
  noisegen=torch.Generator().manual_seed(c['model_seed']+step*9901)
  noise=torch.randn(ux.shape,generator=noisegen,dtype=ux.dtype).to(device)*c['semi']['noise_std']
  b,g=output(model,changed+noise)
  losses=terms(b,g,a,f)
  for key in ('KL','feature','margin'):
   if c['losses'][key]>0:parts[key]=losses[key].mean()
  detail.update(U_packets=len(ux),accepted_U=int((a.softmax(1).max(1).values>=c['semi']['confidence']).sum()),weak_forwards=1,
   feature_U=len(ux) if c['losses']['feature'] else 0,margin_U=len(ux) if c['losses']['margin'] else 0,
   U_confidence=float(a.softmax(1).max(1).values.mean()),
   unit_feature_std=float(F.normalize(g.detach(),dim=1,eps=1e-4).std(0,unbiased=False).mean()),
   raw_margin_mse=float(losses['margin'].detach().mean()),raw_feature_distance=float(losses['feature'].detach().mean()))
 weighted={k:v*c['losses'][k]*(1. if k=='ce' else ramp) for k,v in parts.items()}
 detail.update(auxiliary_steps=0,auxiliary_L=0,auxiliary_forwards=0,fishr_steps=0,mixstyle_steps=0)
 if c['multilevel']['enabled'] and active and step%c['multilevel']['interval']==0:
  aux_ce,fishr,info=model.multilevel.loss(model,L,step,device,c['multilevel'])
  parts.update(aux_ce=aux_ce,fishr=fishr)
  weighted.update(aux_ce=aux_ce*c['multilevel']['aux_ce_weight']*ramp,fishr=fishr*c['multilevel']['fishr_weight']*ramp)
  detail.update(multilevel=info,auxiliary_steps=1,auxiliary_L=c['multilevel']['batch_size'],
   auxiliary_forwards=2 if c['multilevel']['mixstyle'] else 1,fishr_steps=int(c['multilevel']['fishr']),mixstyle_steps=int(c['multilevel']['mixstyle']))
 else:
  parts.update(aux_ce=zero,fishr=zero);weighted.update(aux_ce=zero,fishr=zero)
 return sum(weighted.values()),parts,weighted,detail
