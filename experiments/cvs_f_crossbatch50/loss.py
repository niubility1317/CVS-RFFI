"""Current-anchor gradients, with optional three previous actual L batches."""
import torch
from torch.nn import functional as F
LEVELS=('time_down','t2.pool')
def metadata(y,rx,ids):
 if y.dtype!=torch.long or rx.dtype!=torch.long or y.ndim!=1 or rx.shape!=y.shape or len(ids)!=len(y) or len(set(ids))!=len(ids) or not all(isinstance(i,str) and i for i in ids) or not bool(((y>=0)&(y<6)).all()) or not all(int(r) in (1,3,4,6,8) for r in rx.detach().cpu().tolist()):raise ValueError('Distinct visible current sourceL TX/RX/physical IDs only')
def describe(h):
 if h.ndim!=3 or not bool(torch.isfinite(h).all()):raise ValueError('Finite actual F map required')
 return torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1)
class Memory:
 def __init__(self):self.entries=[];self.updates=0
 def push(self,v,y,rx,ids,step):
  metadata(y,rx,ids)
  if v.ndim!=2 or len(v)!=len(y) or not bool(torch.isfinite(v).all()) or v.requires_grad or v.grad_fn is not None or (self.entries and step<=self.entries[-1]['step']):raise ValueError('Detached finite chronological source memory only')
  self.entries.append(dict(v=v.clone(),y=y.detach().clone(),rx=rx.detach().clone(),ids=tuple(ids),step=int(step)));self.entries=self.entries[-3:];self.updates+=1
 def previous(self,ids,step):
  seen=set(ids);values=[];ages=[];removed=0;duplicates=0
  for e in reversed(self.entries):
   age=step-e['step']
   if age<=0:raise ValueError('Current/future memory leakage')
   if age>3:continue
   keep=[]
   for i,pid in enumerate(e['ids']):
    if pid in seen:
     removed+=pid in ids;duplicates+=pid not in ids
    else:seen.add(pid);keep.append(i)
   if keep:
    values.append((e['v'][keep],e['y'][keep],e['rx'][keep]));ages.extend([age]*len(keep))
  return values,ages,removed,duplicates

def penalty(maps,y,rx,ids,banks,mode,step):
 if tuple(maps)!=LEVELS or tuple(banks)!=LEVELS or mode not in ('ANCHOR','MEM4'):raise ValueError('Fixed twolevel source memory modes only')
 metadata(y,rx,ids);values=[];detail=[]
 for level in LEVELS:
  v=describe(maps[level])
  if len(v)!=len(y):raise ValueError('Current L feature count mismatch')
  center=v.detach().mean(0);scale=v.detach().std(0,unbiased=False).clamp_min(1e-4);z=F.normalize((v-center)/scale,dim=1,eps=1e-4)
  prior,ages,removed,duplicates=banks[level].previous(ids,step) if mode=='MEM4' else ([],[],0,0)
  rawkeys=torch.cat([v.detach()]+[e[0] for e in prior]);ky=torch.cat([y]+[e[1] for e in prior]);kr=torch.cat([rx]+[e[2] for e in prior]);keys=F.normalize((rawkeys-center)/scale,dim=1,eps=1e-4).detach()
  pos=(y[:,None]==ky[None,:])&(rx[:,None]!=kr[None,:]);neg=y[:,None]!=ky[None,:];den=pos|neg;valid=pos.any(1)&neg.any(1);counts=pos.sum(1);logits=z@keys.T/.2
  value=(torch.logsumexp(logits[valid].masked_fill(~den[valid],-torch.inf),1)-(logits[valid]*pos[valid]).sum(1)/counts[valid]-counts[valid].to(v.dtype).log()).mean() if bool(valid.any()) else z.sum()*0.
  if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite memory contrastive objective')
  per=[]
  for c in range(6):
   mask=valid&(y==c);n=int(mask.sum());per.append(dict(TX=c,count=n,positive_mean=float(counts[mask].float().mean()) if n else None,negative_mean=float(neg[mask].sum(1).float().mean()) if n else None))
  values.append(value);detail.append(dict(level=level,mode=mode,shape=list(maps[level].shape),loss=float(value.detach()),valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(counts[valid].min()) if bool(valid.any()) else 0,positive_max=int(counts[valid].max()) if bool(valid.any()) else 0,key_count=len(ky),historical_keys=len(ky)-len(y),max_history_age=max(ages,default=0),mean_history_age=sum(ages)/len(ages) if ages else 0.,current_ID_duplicates_removed=removed,older_ID_duplicates_removed=duplicates,per_TX_keys=per))
 return torch.stack(values).mean(),detail
