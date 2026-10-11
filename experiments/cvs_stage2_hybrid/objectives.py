import torch
from torch.nn import functional as F


def head_logits(model,z):
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 if head.scale!=30. or z.shape[1]!=160:raise ValueError('Actual joint160 cosine head required')
 return 30*F.linear(F.normalize(z,dim=1,eps=1e-4),F.normalize(head.weight,dim=1,eps=1e-4))


def cross_rx_contrastive(z,y,rx,ids,temperature=.1):
 if len(y)!=len(z) or ids.shape!=y.shape or rx.shape!=y.shape:raise ValueError('Invalid labeled source batch')
 distinct=ids[:,None]!=ids[None,:]
 same=y[:,None]==y[None,:]
 positive=distinct&same&(rx[:,None]!=rx[None,:])
 allowed=positive|(distinct&~same)
 active=positive.any(1)&allowed.any(1)
 if not active.any():return z.sum()*0.,dict(active_anchors=0,positive_pairs=0)
 scores=F.normalize(z,dim=1,eps=1e-4)@F.normalize(z,dim=1,eps=1e-4).T/temperature
 s=scores[active];p=positive[active];denom=torch.logsumexp(s.masked_fill(~allowed[active],-torch.inf),dim=1)
 loss=((denom[:,None]-s)*p).sum(1)/p.sum(1)
 return loss.mean(),dict(active_anchors=int(active.sum()),positive_pairs=int(p.sum()),
  excluded_same_physical_pairs=int((~distinct).sum()),same_rx_same_tx_not_negative=True)


def cross_rx_feature_mix(model,z,y,rx,indices,relation,generator):
 if relation is None:return z.sum()*0.,dict(pairs=0)
 mixes,labels=[],[]
 for r in relation['relations']:
  if not r['valid']:continue
  left,right=(torch.tensor(p,device=z.device)+96 for p in r['positions'])
  if len(left)!=8 or len(right)!=8 or not torch.equal(y[left],y[right]):raise ValueError('Expected source same-TX bags')
  if torch.any(indices[left]==indices[right]) or torch.any(rx[left]==rx[right]):raise ValueError('Expected distinct physical packets from different RX')
  lam=(.25+.5*torch.rand(8,1,generator=generator)).to(z)
  mixes.append(lam*z[left]+(1-lam)*z[right]);labels.append(y[left])
 if not mixes:return z.sum()*0.,dict(pairs=0)
 mixed=torch.cat(mixes)
 return F.cross_entropy(head_logits(model,mixed),torch.cat(labels)),dict(pairs=len(mixed),unit_sphere_interpolation=True)


def manifold_mix(model,z,y,permutation,lam):
 if not 0<=lam<=1:raise ValueError('Mix coefficient outside unit interval')
 scores=head_logits(model,lam*z+(1-lam)*z[permutation])
 return lam*F.cross_entropy(scores,y)+(1-lam)*F.cross_entropy(scores,y[permutation])
