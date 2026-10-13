"""L-only current-coordinate, threefold TX/RX/day drift estimates."""
import time
import torch
from torch.nn import functional as F
from experiments.cvs_finite_margin.objective import deterministic
from . import design as d

def partition(L):
 n=len(L['ids']);y,rx,day=L['y'],L['rx'],L['day']
 if len(set(L['ids']))!=n or y.shape!=rx.shape or y.shape!=day.shape or len(y)!=n or set(y.tolist())!=set(range(6)) or set(rx.tolist())!=set(d.RX) or set(day.tolist())!=set(d.DAYS):raise ValueError('Legal L TX/RX/day cells required')
 folds=torch.full((n,),-1,dtype=torch.long);records=[]
 for tx in range(6):
  for r in d.RX:
   for dayvalue in d.DAYS:
    ix=((y==tx)&(rx==r)&(day==dayvalue)).nonzero().flatten().tolist();ix.sort(key=lambda i:L['ids'][i])
    if len(ix)<3:raise ValueError('Three disjoint nonempty cell folds required')
    for rank,i in enumerate(ix):folds[i]=rank%3
    records.append(dict(TX=tx,RX=r,day=dayvalue,count=len(ix),fold_counts=[int((folds[ix]==k).sum()) for k in range(3)],first_ID=L['ids'][ix[0]],last_ID=L['ids'][ix[-1]]))
 if (folds<0).any():raise ValueError('L packet missing fold')
 return folds,dict(sourceL_only=True,packets=n,cells=records,assignment='cell sorted physical IDs rank modulo3',all_L_train=True,U_or_target_truth_read=False,content_comparability='Same TX/day controlled; unpaired waveform windows/payload positions are observational and not guaranteed matched across RX. IDs retained; no pure hardware claim.')

def estimates(features,L,folds,c,epoch):
 if features.shape!=(len(L['ids']),160) or features.requires_grad or not torch.isfinite(features).all():raise ValueError('Detached current normalized features required')
 means=torch.empty(3,6,5,3,160);counts=torch.zeros(3,6,5,3,dtype=torch.long)
 for f in range(3):
  for tx in range(6):
   for r,rv in enumerate(d.RX):
    for day,dv in enumerate(d.DAYS):
     mask=(folds==f)&(L['y']==tx)&(L['rx']==rv)&(L['day']==dv);counts[f,tx,r,day]=int(mask.sum())
     if not mask.any():raise ValueError('Empty crossfit mean cell')
     means[f,tx,r,day]=features[mask].mean(0)
 delta=torch.empty(6,5,3,3,4,160);qq=torch.empty(6,5,3,3,4);records=[]
 gen=torch.Generator().manual_seed(c['model_seed']+epoch*104729)
 for tx in range(6):
  for r in range(5):
   for day in range(3):
    for f in range(3):
     a,b=[k for k in range(3) if k!=f]
     for k,dest in enumerate(v for v in range(5) if v!=r):
      da=means[a,tx,dest,day]-means[a,tx,r,day];db=means[b,tx,dest,day]-means[b,tx,r,day]
      q=(2*(da*db).sum()/(da.square().sum()+db.square().sum()+1e-12)).clamp(0,1);mean=(da+db)/2;norm=mean.norm();bounded=mean/torch.maximum(norm/.2,norm.new_ones(()));change=q*bounded
      if c['arm']=='RANDOM_ROBUST':
       random=torch.randn(160,generator=gen);change=F.normalize(random,dim=0,eps=1e-12)*change.norm()
      delta[tx,r,day,f,k]=change;qq[tx,r,day,f,k]=q
      records.append(dict(TX=tx,RX=d.RX[r],destination_RX=d.RX[dest],day=d.DAYS[day],anchor_fold=f,estimator_folds=[a,b],q=float(q),mean_norm=float(norm),bounded_norm=float(bounded.norm()),effective_norm=float(change.norm()),clipped=bool(norm>.2)))
 return dict(delta=delta.detach(),q=qq.detach()),dict(epoch=epoch,sourceL_only=True,fit_packets=len(features),fold_cell_counts=counts.tolist(),mean_bank_bytes=means.numel()*means.element_size(),q_zero_fraction=float((qq==0).float().mean()),q_quantiles=torch.quantile(qq.flatten(),torch.tensor([0.,.25,.5,.75,1.])).tolist(),effective_norm_quantiles=torch.quantile(delta.norm(dim=-1).flatten(),torch.tensor([0.,.25,.5,.75,1.])).tolist(),directions=records,U_or_target_truth_read=False,query_runtime_state=False,pure_RX_causal_claim=False)

@torch.no_grad()
def refresh(model,L,folds,device,c,epoch):
 start=time.perf_counter();buffers={k:v.clone() for k,v in model.named_buffers()};features=[];devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
 with torch.random.fork_rng(devices=devices),deterministic(model):
  for begin in range(0,len(L['ids']),256):features.append(F.normalize(model.forward_identity_only(L['x'][begin:begin+256].to(device))['id_feat_joint'],dim=1,eps=1e-4).cpu())
 if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Statistic refresh mutated model buffers')
 stats,diag=estimates(torch.cat(features),L,folds,c,epoch);diag['elapsed_seconds']=time.perf_counter()-start;stats={k:v.to(device) for k,v in stats.items()};return stats,diag

def challenges(stats,y,rx,day,folds,epoch):
 if not set(rx.tolist())<=set(d.RX) or not set(day.tolist())<=set(d.DAYS) or (folds<0).any() or (folds>=3).any():raise ValueError('Legal labeled source metadata required')
 ri=(rx[:,None]==torch.tensor(d.RX,device=rx.device)).long().argmax(1);di=(day[:,None]==torch.tensor(d.DAYS,device=day.device)).long().argmax(1)
 return d.strength(epoch)*stats['delta'][y,ri,di,folds],stats['q'][y,ri,di,folds]
