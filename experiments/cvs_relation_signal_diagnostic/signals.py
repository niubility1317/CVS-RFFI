import torch
import torch.nn.functional as F

def alignment(left,right):
 if len(left)!=len(right):raise ValueError('Gradient parameter alignment differs')
 dot=aa=bb=0.
 for a,b in zip(left,right):
  if any(v is not None and not torch.isfinite(v).all() for v in (a,b)):raise FloatingPointError('Nonfinite feature gradient')
  if a is not None:aa+=float(a.detach().double().square().sum())
  if b is not None:bb+=float(b.detach().double().square().sum())
  if a is not None and b is not None:
   if a.shape!=b.shape:raise ValueError('Gradient shape differs')
   dot+=float((a.detach().double()*b.detach().double()).sum())
 den=(aa*bb)**.5
 return dict(dot=dot,CE_norm=aa**.5,aux_norm=bb**.5,cosine=dot/den if den else None,local_descent_reduces_CE=dot>0)

def packet_alignment(a,b):
 if a.shape!=b.shape or a.ndim!=2:raise ValueError('Per-packet gradient shape differs')
 if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite per-packet gradient')
 dot=(a.detach().double()*b.detach().double()).sum(-1)
 den=a.detach().double().norm(dim=-1)*b.detach().double().norm(dim=-1)
 return dict(dot=dot,cosine=torch.where(den>0,dot/den.clamp_min(1e-30),torch.zeros_like(dot)),nonzero=den>0)

def margin(logits,y):
 if logits.shape!=(len(y),6) or y.ndim!=1 or not torch.all((y>=0)&(y<6)):raise ValueError('Only legal labeled six-TX source measurements')
 other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).amax(-1)
 return logits.gather(1,y[:,None]).squeeze(1)-other

def checks():
 a=torch.tensor([[1.,0.],[0.,1.]])
 assert alignment((a,),(a,))['cosine']==1.
 assert alignment((a,),(-a,))['cosine']==-1.
 assert alignment((a,),(None,))['cosine'] is None
 assert packet_alignment(a,-a)['dot'].tolist()==[-1.,-1.]
 assert margin(torch.tensor([[3.,2.,0.,0.,0.,0.]]),torch.tensor([0])).item()==1.
 for labels in (torch.tensor([-1]),torch.tensor([6])):
  try:margin(torch.zeros(1,6),labels)
  except ValueError:pass
  else:raise AssertionError('Hidden/out-of-range TX measurement accepted')
 try:alignment((a,),(torch.zeros(3,2),))
 except ValueError:pass
 else:raise AssertionError('Mismatched gradient shape accepted')
 return dict(status='PASS',positive_negative_zero_gradient_directions=True,hidden_TX_negative=True,gradient_shape_negative=True)
