"""Live current keys and detached three-step sourceL history."""
import torch
from torch.nn import functional as F
from experiments.cvs_f_crossbatch50.loss import Memory,describe,metadata,LEVELS
def penalty(maps,y,rx,ids,banks,mode,step):
 if tuple(maps)!=LEVELS or tuple(banks)!=LEVELS or mode!='SYM4':raise ValueError('Fixed twolevel source memory modes only')
 metadata(y,rx,ids);values=[];detail=[]
 for level in LEVELS:
  v=describe(maps[level])
  if len(v)!=len(y):raise ValueError('Current L feature count mismatch')
  center=v.detach().mean(0);scale=v.detach().std(0,unbiased=False).clamp_min(1e-4);z=F.normalize((v-center)/scale,dim=1,eps=1e-4)
  prior,ages,removed,duplicates=banks[level].previous(ids,step) if mode=='SYM4' else ([],[],0,0)
  rawkeys=torch.cat([v]+[e[0] for e in prior]);ky=torch.cat([y]+[e[1] for e in prior]);kr=torch.cat([rx]+[e[2] for e in prior]);keys=F.normalize((rawkeys-center)/scale,dim=1,eps=1e-4)
  pos=(y[:,None]==ky[None,:])&(rx[:,None]!=kr[None,:]);neg=y[:,None]!=ky[None,:];den=pos|neg;valid=pos.any(1)&neg.any(1);counts=pos.sum(1);logits=z@keys.T/.2
  value=(torch.logsumexp(logits[valid].masked_fill(~den[valid],-torch.inf),1)-(logits[valid]*pos[valid]).sum(1)/counts[valid]-counts[valid].to(v.dtype).log()).mean() if bool(valid.any()) else z.sum()*0.
  if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite memory contrastive objective')
  per=[]
  for c in range(6):
   mask=valid&(y==c);n=int(mask.sum());per.append(dict(TX=c,count=n,positive_mean=float(counts[mask].float().mean()) if n else None,negative_mean=float(neg[mask].sum(1).float().mean()) if n else None))
  values.append(value);detail.append(dict(level=level,mode=mode,shape=list(maps[level].shape),loss=float(value.detach()),valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(counts[valid].min()) if bool(valid.any()) else 0,positive_max=int(counts[valid].max()) if bool(valid.any()) else 0,key_count=len(ky),historical_keys=len(ky)-len(y),max_history_age=max(ages,default=0),mean_history_age=sum(ages)/len(ages) if ages else 0.,current_ID_duplicates_removed=removed,older_ID_duplicates_removed=duplicates,per_TX_keys=per))
 return torch.stack(values).mean(),detail
