"""Split definitions never use U TX; normalization is fitted downstream."""
import torch
def make(L,q,c):
 if L.get('role')!='L_s' or len(set(L['ids']))!=len(L['ids']):raise ValueError('Lawful unique L required')
 n=len(L['ids']);allids=torch.arange(n);held=torch.zeros(n,dtype=torch.bool);g=torch.Generator().manual_seed(c['source_split_seed'])
 for tx in range(6):
  for rx in (1,3,4,6,8):
   for day in (1,2,3):
    idx=torch.where((L['y']==tx)&(L['rx']==rx)&(L['day']==day))[0]
    idx=idx[torch.tensor(sorted(range(len(idx)),key=lambda j:L['ids'][int(idx[j])]))];idx=idx[torch.randperm(len(idx),generator=g)]
    if len(idx)!=70:raise ValueError('Source L cell differs')
    held[idx[:14]]=True
 result={'PID80_20':(allids[~held],allids[held])}
 for tx in range(6):mask=L['y']==tx;result[f'TX{tx}']=(allids[~mask],allids[mask])
 for day in (1,2,3):mask=L['day']==day;result[f'day{day}']=(allids[~mask],allids[mask])
 # Source-only group definition uses observed coherence, not TX labels.
 order=sorted(range(n),key=lambda i:(float(q[i,2]),L['ids'][i]));group=torch.empty(n,dtype=torch.long)
 for j,i in enumerate(order):group[i]=min(2,j*3//n)
 for k in range(3):mask=group==k;result[f'quality{k}']=(allids[~mask],allids[mask])
 for fit,ev in result.values():
  if not len(fit) or not len(ev) or set(fit.tolist())&set(ev.tolist()):raise ValueError('Independent nonempty fit/eval required')
 return result,dict(source_only=True,quality_boundaries=[float(q[order[n//3],2]),float(q[order[2*n//3],2])],quality_group_counts=[int((group==i).sum()) for i in range(3)],group_definition_uses_all_L_quality=True,predictor_normalization_fit_only=True,encoder_saw_source_L_RX=True)
