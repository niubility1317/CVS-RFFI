"""Non-fitted source waveform statistics;no classifier,model or augmentation."""
import numpy as np
from . import design as d

def observe(x):
 x=np.asarray(x)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype.kind!='f' or not np.isfinite(x).all():raise ValueError('Finite nonempty realfloat IQ[N,2,256] required')
 z=(x[:,0].astype(np.float64)+1j*x[:,1].astype(np.float64))[:,80:160]
 with np.errstate(over='raise',invalid='raise',divide='raise'):
  c=z[:,1:-1]**2*np.conj(z[:,:-2])*np.conj(z[:,2:]);mag=np.abs(c);q=c/(mag+d.EPS)
 if q.shape!=(len(x),78) or not np.isfinite(q).all():raise ValueError('Complete finite78 triplets required,no filtering')
 curve=np.stack((q.real,q.imag),axis=-1).reshape(len(x),156);mean=q.mean(1);soft=np.abs(q)
 return dict(curve=curve,zero_triplets=np.sum(mag==0,axis=1),soft_modulus_min=soft.min(1),soft_modulus_mean=soft.mean(1),mean_real=mean.real,mean_imag=mean.imag,dispersion=1-np.abs(mean))

def _gain(b,m):return None if b==0 else 1.-m/b

def decide(b,m):
 b,m=(np.asarray(v,dtype=np.float64) for v in (b,m))
 if b.shape!=(18,4,156) or m.shape!=b.shape or not np.isfinite(b).all() or not np.isfinite(m).all() or np.any(b<0) or np.any(m<0):raise ValueError('Complete finite nonnegative fixedrisk arrays required')
 baseline=float(b.mean());risk=float(m.mean());gain=_gain(baseline,risk);per_rx={str(r):_gain(float(b[:,i].mean()),float(m[:,i].mean())) for i,r in enumerate(d.FIT)};foldg=[_gain(float(bb.mean()),float(mm.mean())) for bb,mm in zip(b,m)];positive=sum(v is not None and v>0 for v in foldg);c=d.config()['criteria'];conditions=dict(pooled_transfer=gain is not None and gain>=c['min_pooled_risk_gain'],all_RX_nonnegative=all(v is not None and v>=c['min_each_RX_risk_gain'] for v in per_rx.values()),positive_TXday_groups=positive>=c['min_positive_TXday_groups']);return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,criteria=c,baseline_MSE=baseline,predicted_MSE=risk,pooled_risk_gain=gain,per_RX_risk_gain=per_rx,positive_groups=positive,group_risk_gain=foldg,total_groups=18)

def analyze(observation,tx,rx,day):
 values=np.asarray(observation['curve'],dtype=np.float64)
 if values.shape!=(5040,156) or not np.isfinite(values).all() or np.any(np.linalg.norm(values.reshape(5040,78,2),axis=2)>1+1e-10):raise ValueError('All5040 finite softunit-complex observations required')
 tx,rx,day=(np.asarray(v) for v in (tx,rx,day))
 if any(v.shape!=(5040,) or v.dtype.kind not in 'iu' for v in (tx,rx,day)) or set(tx)!=set(range(6)) or set(rx)!=set(d.FIT) or set(day)!={1,2,3}:raise ValueError('Only balancedvisibleL TX/RX/day metadata permitted')
 cells=np.zeros((4,3,6,156));summaries=[]
 for ri,r in enumerate(d.FIT):
  for di,dd in enumerate((1,2,3)):
   for t in range(6):
    take=(rx==r)&(day==dd)&(tx==t)
    if int(take.sum())!=70:raise ValueError('Exactly72cells/70originalL each required')
    v=values[take];cells[ri,di,t]=v.mean(0);summaries.append(dict(RX=r,day=dd,TX=t,count=70,mean=cells[ri,di,t].tolist(),population_variance=v.var(0).tolist(),mean_vector_RMS=float(np.sqrt(np.mean(cells[ri,di,t]**2))),population_dispersion_RMS=float(np.sqrt(np.mean(v.var(0))))))
 base=[];error=[];folds=[]
 for hd in range(3):
  for ht in range(6):
   ds=[i for i in range(3) if i!=hd];ts=[i for i in range(6) if i!=ht];train=cells[:,ds][:,:,ts].copy();train-=train.mean(0,keepdims=True);prediction=train.mean((1,2));actual=cells[:,hd,ht].copy();actual-=actual.mean(0,keepdims=True);b=actual**2;m=(actual-prediction)**2;base.append(b);error.append(m);folds.append(dict(held_TX=ht,held_day=hd+1,train_TX=ts,train_days=[i+1 for i in ds],train_group_count=10,actual_centered_RX=actual.tolist(),predicted_RX=prediction.tolist(),baseline_MSE=float(b.mean()),predicted_MSE=float(m.mean())))
 decision=decide(base,error)
 for f,gain in zip(folds,decision['group_risk_gain']):f['risk_gain']=gain
 return dict(**decision,cells=summaries,folds=folds,fit_L_packets=5040,source_cells=72,triplets=393120,coordinates=156,source_only=True,checkpoint_loads=0,network_forwards=0,optimizer_steps=0,head_fit_updates=0,IQ_views_generated=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,scope='KnownsourceRX-associated waveformshape only;not oscillatoridentification,physicalRXreplacement,IQlabelpreservation or recognitiongain')
