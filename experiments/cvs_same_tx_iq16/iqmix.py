"""Small coherent same-identity observed-IQ interpolation, not a receiver simulator."""
import torch
EPS=1e-6

def mix(anchor,donor,coefficient=.1):
 if coefficient not in (0.,.1) or anchor.ndim!=3 or anchor.shape[1:]!=(2,256) or donor.shape!=anchor.shape or anchor.dtype!=torch.float32 or donor.dtype!=anchor.dtype or anchor.device!=donor.device or not bool(torch.isfinite(anchor).all() and torch.isfinite(donor).all()):raise ValueError('Fixed finite float32 matched IQ[N,2,256] and coefficient required')
 if coefficient==0:return anchor.clone(),dict(relative_IQ_change=[0.]*len(anchor),relative_power_error=[0.]*len(anchor))
 a=torch.complex(anchor[:,0].double(),anchor[:,1].double());b=torch.complex(donor[:,0].double(),donor[:,1].double());pa=a.abs().square().mean(1);pb=b.abs().square().mean(1)
 b=b*torch.sqrt(pa/pb.clamp_min(EPS**2))[:,None]
 def omega(v):
  w=v[:,80:160];cross=(w[:,20:]*w[:,:-20].conj()).sum(1);return torch.atan2(cross.imag,cross.real+EPS**2)/20
 delta=omega(b)-omega(a);n=torch.arange(256,device=a.device,dtype=torch.float64);b=b*torch.exp(-1j*delta[:,None]*n)
 cross=(a[:,80:160].conj()*b[:,80:160]).sum(1);phase=torch.angle(cross);b=b*torch.exp(-1j*phase[:,None])
 v=(1-coefficient)*a+coefficient*b;power=v.abs().square().mean(1);v=v*torch.sqrt(pa/power.clamp_min(EPS**2))[:,None]
 y=torch.stack((v.real,v.imag),dim=1).to(anchor.dtype)
 change=torch.sqrt((y.double()-anchor.double()).square().sum(1).mean(1)/pa.clamp_min(EPS**2));error=(y.double().square().sum(1).mean(1)-pa).abs()/pa.clamp_min(EPS**2)
 if not bool(torch.isfinite(y).all()) or bool((change>.25001).any()):raise ValueError('Bounded power-restored IQmix failed')
 return y,dict(relative_IQ_change=change.tolist(),relative_power_error=error.tolist(),relative_frequency_alignment=delta.tolist(),common_phase_alignment=phase.tolist())

def donor_indices(labels,rx,day,ids):
 if len(ids)!=5040 or len(set(ids))!=5040 or any(len(v)!=5040 for v in (labels,rx,day)) or set(rx)!={3,4,6,8} or set(day)!={1,2,3} or set(labels)!=set(range(6)):raise ValueError('Only complete visible fit L5040 roles allowed')
 cells={}
 for i,key in enumerate(zip(labels,rx,day)):cells.setdefault(key,[]).append(i)
 if len(cells)!=72 or any(len(v)!=70 for v in cells.values()):raise ValueError('All original70packet TX/RX/day cells required')
 for key in cells:cells[key].sort(key=lambda i:ids[i])
 rank={i:j for members in cells.values() for j,i in enumerate(members)};same=[];cross=[]
 for i,(y,r,d) in enumerate(zip(labels,rx,day)):
  j=(rank[i]+1)%70;s=cells[(y,r,d)][j];others=[cells[(y,q,d)][j] for q in sorted(set(rx)) if q!=r]
  if s==i or len(set(others))!=3 or any(labels[k]!=y or day[k]!=d or rx[k]==r or ids[k]==ids[i] for k in others):raise ValueError('Donors must be distinct visible sameTX/day physical records')
  same.append(s);cross.append(others)
 return same,cross

def qualify(raw,same,cross,changes,power_errors):
 conditions=dict(overall_accuracy=100*(cross['accuracy']-raw['accuracy'])>=-.5,overall_macroF1=100*(cross['macro_F1']-raw['macro_F1'])>=-.5,cross_vs_sameRX=100*(cross['accuracy']-same['accuracy'])>=-.5,raw_correct_retention=cross['raw_correct_retention']>=.99,nonzero_IQ_change=float(torch.tensor(changes,dtype=torch.float64).quantile(.5))>=.01,bounded_IQ_change=max(changes)<=.25001,power_preserved=max(power_errors)<=1e-5)
 for key in ('RX','TX','day'):conditions['every_'+key]=min(100*(cross['by_'+key][v]['accuracy']-raw['by_'+key][v]['accuracy']) for v in raw['by_'+key])>=-1.
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,accuracy_gain_pp=100*(cross['accuracy']-raw['accuracy']),macroF1_gain_pp=100*(cross['macro_F1']-raw['macro_F1']),cross_vs_sameRX_accuracy_gain_pp=100*(cross['accuracy']-same['accuracy']),raw_correct_retention=cross['raw_correct_retention'],median_relative_IQ_change=float(torch.tensor(changes,dtype=torch.float64).quantile(.5)),max_relative_IQ_change=max(changes),max_relative_power_error=max(power_errors),identity_guard_only=True,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False)
