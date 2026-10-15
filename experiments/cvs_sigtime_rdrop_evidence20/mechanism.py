"""Separate self confidence from same-identity cross-RX confusion evidence."""
import torch
from torch.nn import functional as F
ARMS=('FULL','LOCAL','SAME_RX','CROSS_RX');FIT=(3,4,6,8)
def new_audit():return dict(calls=0,packets=0,selected_anchors=0,transported_anchors=0,max_chain_error=0.,max_mass_error=0.,invalid_identity_edges=0,invalid_RX_edges=0)
def transport(y,rx,arm,dtype):
 if arm not in ARMS or y.ndim!=1 or y.dtype!=torch.long or rx.shape!=y.shape or rx.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not bool(torch.isin(rx,torch.tensor(FIT,device=rx.device)).all()):raise ValueError('Visible legal L labels and effective fit RX only')
 n=len(y);P=torch.eye(n,dtype=dtype,device=y.device);selected=torch.zeros(n,dtype=torch.bool,device=y.device)
 # List-only metadata operations do not consume native torch/numpy/Python RNG.
 yl=y.detach().cpu().tolist();rl=rx.detach().cpu().tolist()
 for label in sorted(set(yl)):
  groups=[[i for i,(a,b) in enumerate(zip(yl,rl)) if a==label and b==r][:2] for r in FIT]
  groups=sorted((g for g in groups if len(g)==2),key=lambda g:g[0])
  if len(groups)<2:continue
  chosen=[i for g in groups for i in g];opposite={i:chosen[(p+len(groups))%len(chosen)] for p,i in enumerate(chosen)}
  for group in groups:
   for i in group:
    selected[i]=True
    if arm in ('FULL','LOCAL'):continue
    peers=[j for j in group if j!=i] if arm=='SAME_RX' else [opposite[i]]
    P[i]=0;P[i,peers]=1/len(peers)
 tolerance=2e-6;mass=max(float((P.sum(0)-1).abs().max()),float((P.sum(1)-1).abs().max()))
 invalid_id=int(((y[:,None]!=y[None,:])&(P>0)).sum())
 off=(P>0)&~torch.eye(n,dtype=torch.bool,device=y.device)
 invalid_rx=int((off&((rx[:,None]!=rx[None,:]) if arm=='SAME_RX' else (rx[:,None]==rx[None,:]))).sum()) if arm in ('SAME_RX','CROSS_RX') else int(off.sum())
 if mass>tolerance or invalid_id or invalid_rx:raise ValueError('Unbalanced or illegal evidence transport')
 return P,dict(selected=int(selected.sum()),transported=int(off.any(1).sum()),mass_error=mass,invalid_identity_edges=invalid_id,invalid_RX_edges=invalid_rx)
def partition(logits,y):
 mask=~F.one_hot(y,num_classes=6).bool();other=logits.masked_select(mask).reshape(len(y),5)
 binary=torch.stack((logits.gather(1,y[:,None]).flatten(),other.logsumexp(1)),1).log_softmax(1)
 return binary,other.log_softmax(1)
def components(first,second,y,P,detach_mass):
 b1,q1=partition(first,y);b2,q2=partition(second,y)
 binary=.5*((b1.exp()-b2.exp())*(b1-b2)).sum(1).mean()
 w1,w2=b1[:,1].exp(),b2[:,1].exp()
 if detach_mass:w1,w2=w1.detach(),w2.detach()
 # Exact weighted mean of KL(q_i||q_j); P is label-preserving and doubly stochastic.
 k12=(q1.exp()*q1).sum(1)-(q1.exp()*(P@q2)).sum(1)
 k21=(q2.exp()*q2).sum(1)-(q2.exp()*(P@q1)).sum(1)
 relation=.5*(w1*k12+w2*k21).mean()
 return binary,relation
def consistency(first,second,y,rx,arm,audit):
 if first.shape!=second.shape or first.shape!=(len(y),6) or not torch.isfinite(first).all() or not torch.isfinite(second).all():raise ValueError('Finite paired native six-class L logits required')
 P,stats=transport(y,rx,arm,first.dtype);binary,relation=components(first,second,y,P,True)
 p,q=first.log_softmax(1),second.log_softmax(1);full=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
 eye=torch.eye(len(y),device=y.device,dtype=first.dtype);chain_parts=components(first,second,y,eye,False);chain=chain_parts[0]+chain_parts[1];error=float((chain-full).abs().detach())
 if error>2e-5:raise ValueError('Native KL chain-rule reconstruction differs')
 loss=full if arm=='FULL' else binary+relation
 audit['calls']+=1;audit['packets']+=len(y);audit['selected_anchors']+=stats['selected'];audit['transported_anchors']+=stats['transported'];audit['max_chain_error']=max(audit['max_chain_error'],error);audit['max_mass_error']=max(audit['max_mass_error'],stats['mass_error'])
 return loss,dict(binary_KL=binary.detach(),relation_KL=relation.detach(),native_symmetric_KL=full.detach(),selected_anchors=stats['selected'],transported_anchors=stats['transported'],chain_error=error)
