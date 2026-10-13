import math
import torch
from torch.nn import functional as F

def fit(z,y,seed,epochs=20,batch_size=256):
 if z.ndim!=2 or z.shape[1]!=160 or y.shape!=(len(z),) or len(z)<6 or set(y.tolist())!=set(range(6)) or not 1<=epochs<=50:raise ValueError('Six-class frozen feature fit differs')
 if z.requires_grad or not torch.isfinite(z).all():raise ValueError('Frozen finite features required')
 g=torch.Generator(device='cpu').manual_seed(seed)
 w=torch.nn.Parameter(torch.empty(6,160).uniform_(-math.sqrt(6/166),math.sqrt(6/166),generator=g))
 opt=torch.optim.Adam([w],lr=.01,weight_decay=0.)
 losses=[];updates=0
 for epoch in range(epochs):
  order=torch.randperm(len(z),generator=g);total=0.
  for start in range(0,len(z),batch_size):
   take=order[start:start+batch_size];logits=30.*F.linear(z[take],F.normalize(w,dim=1,eps=1e-4));loss=F.cross_entropy(logits,y[take])
   if not torch.isfinite(loss):raise FloatingPointError('Cosine refit loss')
   opt.zero_grad(set_to_none=True);loss.backward()
   if not torch.isfinite(w.grad).all():raise FloatingPointError('Cosine refit gradient')
   opt.step();updates+=1;total+=float(loss.detach())*len(take)
  losses.append(total/len(z))
 return w.detach(),dict(epochs=epochs,updates=updates,losses=losses,feature_updates=0)
def logits(z,w):return 30.*F.linear(z,F.normalize(w,dim=1,eps=1e-4))
def support_indices(y,rx,held,k,seed):
 if k not in (5,10) or held not in (1,3,4,6,8):raise ValueError('Fixed source probe support differs')
 g=torch.Generator().manual_seed(seed+held*1000)
 selected=[]
 for cls in range(6):
  pool=((y==cls)&(rx!=held)).nonzero().flatten()
  if len(pool)<k:raise ValueError('Insufficient legal support')
  selected.extend(pool[torch.randperm(len(pool),generator=g)[:k]].tolist())
 result=torch.tensor(selected)
 if len(result.unique())!=6*k or (rx[result]==held).any():raise ValueError('Head-fit RX overlap')
 return result
