import torch
from torch.nn import functional as F

@torch.no_grad()
def fit(features,y,rx,settings,mode):
 if features.ndim!=2 or y.shape!=(len(features),) or rx.shape!=y.shape:raise ValueError('L shape mismatch')
 if y.dtype!=torch.long or set(y.tolist())!=set(range(settings['classes'])):raise ValueError('All registered L classes required')
 if not torch.isfinite(features).all():raise ValueError('Nonfinite features')
 dims=features.shape[1];transform=torch.eye(dims,device=features.device,dtype=features.dtype)
 eigvals=None
 if mode=='within_class_whitened':
  means=torch.stack([features[y==c].double().mean(0) for c in range(settings['classes'])])
  residual=features.double()-means[y];cov=residual.T@residual/len(residual);scale=cov.trace()/dims
  if scale<=0:raise ValueError('Degenerate within-class covariance')
  alpha=settings['covariance_shrinkage'];cov=(1-alpha)*cov+alpha*scale*torch.eye(dims,device=cov.device,dtype=cov.dtype)
  eigvals,vectors=torch.linalg.eigh(cov);transform=((vectors*eigvals.clamp_min(1e-12).rsqrt())@vectors.T).to(features.dtype)
 elif mode!='cosine':raise ValueError('Unknown fixed metric')
 bank=F.normalize(features@transform,dim=1);domains=rx.unique(sorted=True)
 if any(int((rx==r).sum())<settings['k_per_RX'] for r in domains):raise ValueError('Insufficient L anchors')
 return dict(bank=bank.clone(),y=y.clone(),rx=rx.clone(),domains=domains,transform=transform,settings=dict(settings),mode=mode,
  diagnostics=dict(fit_role='L_s',fit_count=len(features),classes=settings['classes'],domain_counts={str(int(r)):int((rx==r).sum()) for r in domains},metric_mode=mode,within_class_eigen_min=float(eigvals.min()) if eigvals is not None else None,within_class_eigen_max=float(eigvals.max()) if eigvals is not None else None,U_or_V_fit=False))

@torch.no_grad()
def predict(state,features,rx,base):
 if features.shape[0]!=len(rx) or base.shape!=(len(rx),state['settings']['classes']):raise ValueError('Query shape mismatch')
 if not torch.isfinite(base).all() or not torch.allclose(base.sum(1),torch.ones(len(base),device=base.device),atol=1e-5):raise ValueError('Invalid base probabilities')
 domains=state['domains'];known=(rx[:,None]==domains[None,:])
 if not known.any(1).all():raise ValueError('This diagnostic expects known source RX')
 query=F.normalize(features@state['transform'],dim=1);all_q=[]
 for start in range(0,len(query),256):
  values=[]
  for domain in domains:
   mask=state['rx']==domain;bank=state['bank'][mask];labels=state['y'][mask]
   sim,idx=(query[start:start+256]@bank.T).topk(state['settings']['k_per_RX'],dim=1)
   w=(sim/state['settings']['temperature']).softmax(1)
   values.append((F.one_hot(labels[idx],state['settings']['classes']).to(w.dtype)*w[:,:,None]).sum(1))
  all_q.append(torch.stack(values,1))
 q=torch.cat(all_q);other=~known;denom=other.sum(1)
 cross=(q*other[:,:,None]).sum(1)/denom[:,None];same=(q*known[:,:,None]).sum(1)
 all_equal=q.mean(1);votes=q.argmax(2);pred=cross.argmax(1)
 unanimous=((votes==pred[:,None])|known).all(1)
 gate=unanimous&(cross.max(1).values>=state['settings']['consensus_min_confidence'])
 mixed=(base+cross)/2;gated=torch.where(gate[:,None],mixed,base)
 probs=dict(model=base,same_RX=same,all_RX_equal=all_equal,cross_RX=cross,mixed_cross_RX=mixed,consensus_mixed=gated)
 for p in probs.values():
  if not torch.isfinite(p).all() or not torch.allclose(p.sum(1),torch.ones(len(p),device=p.device),atol=1e-5):raise ValueError('Readout probabilities differ')
 diagnostics=dict(count=len(features),cross_RX_count=int(denom[0]),same_RX_used_in_cross=False,all_RX_unanimous_fraction=float(unanimous.float().mean()),correction_gate_fraction=float(gate.float().mean()),cross_model_disagreement=float((pred!=base.argmax(1)).float().mean()),gate_disagreement_count=int((gate&(pred!=base.argmax(1))).sum()),U_accuracy=None)
 return probs,diagnostics
