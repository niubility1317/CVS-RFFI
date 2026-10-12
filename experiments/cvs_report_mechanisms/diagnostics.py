import torch
from .frontends import DIMS,features,fragments,matrix_loops
def content(L):
 values={k:[] for k in DIMS};matrix_valid=[]
 with torch.no_grad():
  for begin in range(0,len(L['ids']),256):
   x=L['x'][begin:begin+256]
   for arm in DIMS:values[arm].append(features(arm,x))
   a=fragments(x);mirror=(-torch.arange(32))%32;v=torch.stack((a,a[:,:,mirror].conj()),3).permute(0,2,1,3);matrix_valid.append(matrix_loops(v)[1])
 values={k:torch.cat(v).double() for k,v in values.items()};summary=[];centered={}
 for arm,z in values.items():
  mean=z.mean(0);centered[arm]=z-mean;cells=torch.stack([torch.stack([z[(L['y']==tx)&(L['rx']==rx)].mean(0) for rx in (1,3,4,6,8)]) for tx in range(6)]);txmean=cells.mean(1);rxres=cells-txmean[:,None];total=float(centered[arm].square().mean());identity=float((txmean-txmean.mean(0)).square().mean());nuisance=float(rxres.square().mean())
  summary.append(dict(arm=arm,dimension=z.shape[1],coordinate_std=z.std(0).tolist(),zero_packet_fraction=float((z.abs().sum(1)==0).double().mean()),total_variation=total,TX_mean_variation=identity,conditional_RX_mean_variation=nuisance,mean_rank=int(torch.linalg.matrix_rank(txmean-txmean.mean(0),rtol=1e-6)),scope='descriptive trainingL statistics,not performance/gate'))
 redundancy=[]
 for i,aa in enumerate(DIMS):
  for bb in list(DIMS)[i+1:]:
   a,b=centered[aa],centered[bb];num=(a.T@b).square().sum();den=((a.T@a).square().sum()*(b.T@b).square().sum()).sqrt().clamp_min(1e-20);redundancy.append(dict(arms=[aa,bb],linear_CKA=float(num/den)))
 return dict(status='VERIFIED',source_L=len(L['ids']),sourceV_or_target_access=False,selection_gate=False,statistics=summary,linear_redundancy=redundancy,matrix_valid_fraction=torch.cat(matrix_valid).double().mean(0).tolist(),shared_fragment_RX_chain='hypothesis;not established fromthese statistics',joint_invariance='separate idealactions do not prove simultaneous generalRX/CFO/channel invariance')
