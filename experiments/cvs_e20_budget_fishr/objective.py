"""Live-student SSL plus shallow/middle source-only objectives."""
import torch
from torch.nn import functional as F
from experiments.cvs_early_geometry.objective import output,weak
from .multilevel import budget_fishr


def objective(c,model,bank,L,li,U,ui,step,device):
 x=L['x'][li].to(device);y=L['y'][li].to(device);logits=model(x);zero=logits.sum()*0
 parts=dict(ce=F.cross_entropy(logits,y),KL=zero,feature=zero,margin=zero)
 ramp=max(0.,min(1.,(step-c['semi']['warmup_updates'])/c['semi']['ramp_updates']))
 active=step>c['semi']['warmup_updates'];mc=c['multilevel']
 detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),U_packets=0,accepted_U=0,physical_L=0,physical_U=0,feature_U=0,margin_U=0,ramp=ramp,weak_forwards=0,
  auxiliary_steps=0,auxiliary_L=0,auxiliary_forwards=0,fishr_steps=0,mixstyle_steps=0,mixstyle_U_steps=0,mixed_U_packets=0)
 if ui is not None:
  if not active or not c['unlabeled']:raise ValueError('Unexpected U forward')
  ux=U['x'][ui].to(device);a,_=weak(model,ux);q=a.softmax(1).detach()
  noisegen=torch.Generator().manual_seed(c['model_seed']+step*9901)
  noise=torch.randn(ux.shape,generator=noisegen,dtype=ux.dtype).to(device)*c['semi']['noise_std']
  if mc['mixstyle_u']:
   ids=[U['ids'][i] for i in ui.tolist()]
   with model.multilevel.main_style(q,U['rx'][ui].to(device),U['day'][ui].to(device),ids,step,mc) as style:
    b,g=output(model,ux+noise)
   detail.update(U_style=style,mixstyle_U_steps=1)
   # Layer counts are reported separately; sum is explicitly packet-layer uses.
   detail['mixed_U_packets']=sum(v.get('changed_packets',0) for v in style['levels'].values())
  else:b,g=output(model,ux+noise)
  mask=q.max(1).values>=c['semi']['confidence']
  parts['KL']=(F.kl_div(b.log_softmax(1),q,reduction='none').sum(1)*mask).mean()
  detail.update(U_packets=len(ux),accepted_U=int(mask.sum()),weak_forwards=1,U_confidence=float(q.max(1).values.mean()),unit_feature_std=float(F.normalize(g.detach(),dim=1,eps=1e-4).std(0,unbiased=False).mean()))
 weighted={k:v*c['losses'][k]*(1. if k=='ce' else ramp) for k,v in parts.items()}
 if mc['enabled'] and active and step%mc['interval']==0:
  aux_ce,terms,info=model.multilevel.loss(model,L,step,device,mc)
  fishr=torch.stack(list(terms.values())).mean() if terms else zero
  wf,budget=budget_fishr(model,parts['ce'],terms,mc,ramp)
  parts.update(aux_ce=aux_ce,fishr=fishr)
  weighted.update(aux_ce=aux_ce*mc['aux_ce_weight']*ramp,fishr=wf)
  detail.update(multilevel=info,fishr_budget=budget,auxiliary_steps=1,auxiliary_L=mc['batch_size'],auxiliary_forwards=2 if mc['mixstyle_l'] else 1,fishr_steps=int(mc['fishr_mode']!='none'),mixstyle_steps=int(mc['mixstyle_l']))
 else:
  parts.update(aux_ce=zero,fishr=zero);weighted.update(aux_ce=zero,fishr=zero)
 return sum(weighted.values()),parts,weighted,detail
