"""Source weak-pair separation in disjoint samples of each physical RX/day."""
import torch
from torch.nn import functional as F
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
MODES=('SELF_GAP','CROSS_GAP');MARGIN=.4

def estimates(a,b):
 return (a*b).sum(-1),.5*(a.square().sum(-1)+b.square().sum(-1))

def penalty(first,second,y,domain,physical_ids,mode):
 if mode not in MODES or first.ndim!=2 or first.shape!=second.shape or first.dtype!=second.dtype or first.device!=second.device or first.shape[0]!=len(y) or first.shape[1]<2 or y.shape!=domain.shape or y.ndim!=1 or y.dtype!=torch.long or domain.dtype!=torch.long or y.device!=first.device or domain.device!=first.device or not bool(((y>=0)&(y<6)).all()) or not bool(((domain>=0)&(domain<15)).all()) or not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):raise ValueError('Finite paired visible-source features and original RX/day labels only')
 if not isinstance(physical_ids,(tuple,list)) or len(physical_ids)!=len(y) or any(not isinstance(v,str) or not v for v in physical_ids) or len(set(physical_ids))!=len(y):raise ValueError('Unique current physical source L packets required')
 # Views share IQ; statistical halves use distinct packets, never RDrop views.
 z=.5*(F.normalize(first,dim=1,eps=1e-4)+F.normalize(second,dim=1,eps=1e-4));weak=(y==1)|(y==3);group=2*domain+(y==3).long();membership=F.one_hot(group,30).to(z.dtype)*weak[:,None];ordinal=membership.cumsum(0).gather(1,group[:,None]).squeeze(1).long();A=membership*(ordinal%2==1)[:,None];B=membership*(ordinal%2==0)[:,None];ca=A.sum(0);cb=B.sum(0);counts=ca+cb;valid=(counts.reshape(15,2)>=2).all(1)
 if not bool(valid.any()):return (first.sum()+second.sum())*0.,dict(eligible=False,valid_cells=0,active_cells=0,used_packets=0,cells=[])
 ma=A.T@z/ca.clamp_min(1)[:,None];mb=B.T@z/cb.clamp_min(1)[:,None];da=ma[0::2]-ma[1::2];db=mb[0::2]-mb[1::2];cross,selfgap=estimates(da,db);selected=cross if mode=='CROSS_GAP' else selfgap;violations=F.relu(MARGIN*MARGIN-selected)/(MARGIN*MARGIN);value=violations[valid].mean()
 if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite disjoint source class-gap loss')
 # Store only aggregated geometry/counts, no packet IDs or feature vectors.
 detail=torch.stack((counts[0::2],counts[1::2],ca[0::2],ca[1::2],cb[0::2],cb[1::2],cross,selfgap,violations,valid.to(z.dtype)),1).detach().cpu().tolist();cells=[]
 for i,v in enumerate(detail):
  if not v[9]:continue
  cells.append(dict(domain=i,RX=RX_BY_DOMAIN[i],day=i%3+1,counts=[int(v[0]),int(v[1])],half_A_counts=[int(v[2]),int(v[3])],half_B_counts=[int(v[4]),int(v[5])],cross_gap_squared=v[6],self_gap_squared=v[7],half_variation=v[7]-v[6],violation=v[8]))
 return value,dict(eligible=True,valid_cells=len(cells),active_cells=sum(v['violation']>0 for v in cells),used_packets=sum(sum(v['counts']) for v in cells),cells=cells)
