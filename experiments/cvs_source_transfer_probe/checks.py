import torch
from torch.nn import functional as F
from .readout import fit,logits,support_indices
from . import design as d

def checks():
 torch.set_num_threads(2)
 y=torch.arange(6).repeat_interleave(10);z=F.one_hot(y,num_classes=160).float()
 w,a=fit(z,y,901);v,b=fit(z,y,901)
 assert torch.equal(w,v) and a==b and a['updates']==20 and a['losses'][-1]<a['losses'][0] and torch.equal(logits(z,w).argmax(1),y) and not z.requires_grad
 yy=torch.arange(6).repeat_interleave(100);rx=torch.tensor([1,3,4,6,8]*120)
 negatives=0
 for held in d.SOURCE_RX:
  i=support_indices(yy,rx,held,5,77);j=support_indices(yy,rx,held,10,77)
  assert set(i.tolist()).issubset(j.tolist()) and (rx[i]!=held).all() and torch.equal(torch.bincount(yy[i],minlength=6),torch.full((6,),5))
 for call in [lambda:fit(z,y,1,epochs=51),lambda:fit(z[:10],y[:10],1),lambda:fit(z.requires_grad_(),y,1),lambda:support_indices(yy,rx,0,5,77),lambda:support_indices(yy,rx,1,7,77)]:
  try:call()
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid refit accepted')
 for row in d.rows():
  c=d.config(row);d.validate(c);c['target_access']=True
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Target probe accepted')
 print('PASS: exact matched refits, simple separability, nested supports, RX exclusion; '+str(negatives)+' negatives')
if __name__=='__main__':checks()
