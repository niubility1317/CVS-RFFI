"""Readonly Phase1 L_s geometry; descriptive diagnostics, no training or target.

No rank threshold declares a bottleneck or changes an already running plan.
The source feature cache is never a Phase2 input.
"""
import torch
import torch.nn.functional as F

def spectrum(z):
 if z.ndim!=2 or len(z)<2 or not torch.isfinite(z).all():raise ValueError('Finite multiple feature rows required')
 z=z.double();centered=z-z.mean(0);cov=centered.T@centered/len(z)
 eigen=torch.linalg.eigvalsh(cov).clamp_min(0).flip(0);total=eigen.sum()
 if float(total)<=1e-14:
  return dict(trace=float(total),effective_rank=0.,participation_rank=0.,top1_fraction=0.,top5_fraction=0.,top10_fraction=0.,positive_eigenvalues=0)
 p=eigen/total;positive=p>1e-14
 return dict(trace=float(total),effective_rank=float(torch.exp(-(p[positive]*p[positive].log()).sum())),participation_rank=float(1/p.square().sum()),top1_fraction=float(p[0]),top5_fraction=float(p[:5].sum()),top10_fraction=float(p[:10].sum()),positive_eigenvalues=int((eigen>total*1e-9).sum()))

def analyze(z,tx,rx,role='L_s'):
 if role!='L_s':raise ValueError('This descriptor uses legal L_s only')
 if z.ndim!=2 or len(z)!=len(tx) or len(z)!=len(rx) or not torch.isfinite(z).all() or (z.norm(dim=1)<=1e-12).any():raise ValueError('Aligned finite nonzero source features required')
 z=F.normalize(z.double(),dim=1);tx=tx.long();rx=rx.long();mean=z.mean(0);N=len(z)
 total=float((z-mean).square().sum()/N);between=0.;conditional_rx=0.;within_cell=0.;classes=[];receivers=[]
 for t in sorted(tx.unique().tolist()):
  mask=tx==t;zt=z[mask];rt=rx[mask];mt=zt.mean(0);between+=float(len(zt)*(mt-mean).square().sum()/N)
  classes.append(dict(TX=t,count=len(zt),spectrum=spectrum(zt)))
  for r in sorted(rt.unique().tolist()):
   cell=zt[rt==r];mc=cell.mean(0);conditional_rx+=float(len(cell)*(mc-mt).square().sum()/N);within_cell+=float((cell-mc).square().sum()/N)
 for r in sorted(rx.unique().tolist()):
  zr=z[rx==r];receivers.append(dict(RX=r,count=len(zr),spectrum=spectrum(zr)))
 error=abs(total-between-conditional_rx-within_cell)
 if error>1e-10*max(1.,total):raise ValueError('Source nested variance decomposition failed')
 fraction=lambda v:v/total if total>1e-14 else 0.
 return dict(role='L_s',count=N,embedding=z.shape[1],unit_normalized=True,global_spectrum=spectrum(z),variance=dict(total=total,between_TX=between,RX_within_TX=conditional_rx,within_TX_RX=within_cell,between_TX_fraction=fraction(between),RX_within_TX_fraction=fraction(conditional_rx),within_TX_RX_fraction=fraction(within_cell),decomposition_error=error),by_TX=classes,by_RX=receivers,inference='descriptive source geometry only; low rank or receiver information does not prove causal transfer failure; no target evidence',model_updates=0,query_read=False)

def check():
 torch.set_num_threads(2)
 # Known rank-one geometry and exact class/RX variance partition.
 z=torch.tensor([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
 tx=torch.tensor([0,0,1,1]);rx=torch.tensor([0,1,0,1]);v=analyze(z,tx,rx)
 assert abs(v['variance']['between_TX_fraction']-.5)<1e-12 and abs(v['variance']['RX_within_TX_fraction']-.5)<1e-12 and abs(v['variance']['within_TX_RX_fraction'])<1e-12
 assert abs(v['global_spectrum']['effective_rank']-2)<1e-12
 rankone=spectrum(torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]]));assert abs(rankone['effective_rank']-1)<1e-12
 for role in ('U_s','V','query','target'):
  try:analyze(z,tx,rx,role)
  except ValueError:pass
  else:raise AssertionError('Unregistered feature role accepted')
 return dict(status='PASS',known_rank_and_variance_decomposition=True,source_L_only=True,no_model_or_target_access=True)
if __name__=='__main__':
 import json
 print(json.dumps(check()))
