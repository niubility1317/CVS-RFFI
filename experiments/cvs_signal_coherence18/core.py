"""Bounded signal projector and L-only cross-TX/day contrast; no classifier."""
import numpy as np
from . import design as d
def observe(x):
 x=np.asarray(x)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype.kind!='f' or not np.isfinite(x).all():raise ValueError('Finite nonempty original IQ[N,2,256] required')
 z=x[:,0,80:160].astype(np.float64)+1j*x[:,1,80:160].astype(np.float64)
 cross=np.sum(z[:,20:]*z[:,:-20].conj(),axis=1);omega=np.angle(cross)/20
 corrected=z*np.exp(-1j*omega[:,None]*np.arange(80));cycles=corrected.reshape(-1,4,20);mean=cycles.mean(1)
 e=np.mean(np.sum(np.abs(cycles)**2,axis=2),axis=1);q=np.sum(np.abs(mean)**2,axis=1);den=np.sqrt(q+1e-12*e)
 factor=np.divide(mean,den[:,None],out=np.zeros_like(mean),where=den[:,None]>0)
 a=np.sum(np.abs(z[:,:-20])**2,axis=1);b=np.sum(np.abs(z[:,20:])**2,axis=1);coherence=np.divide(np.abs(cross),np.sqrt(a*b),out=np.zeros(len(x)),where=a*b>0)
 fraction=np.divide(q,e,out=np.zeros(len(x)),where=e>0)
 if not np.isfinite(factor).all() or np.any(np.sum(abs(factor)**2,axis=1)>1+1e-12) or np.any(fraction>1+1e-12) or np.any(coherence>1+1e-12):raise ValueError('Bounded finite shape/quality required')
 return dict(factor_real=factor.real,factor_imag=factor.imag,lag20_coherence=coherence,coherent_fraction=fraction,window_energy=e,zero_mean=q==0)
def cube(factor,tx,rx,day,per_cell):
 factor=np.asarray(factor,dtype=np.complex128);tx,rx,day=(np.asarray(a) for a in (tx,rx,day));n=90*per_cell
 if factor.shape!=(n,20) or not np.isfinite(factor).all() or any(a.shape!=(n,) or a.dtype.kind not in 'iu' for a in (tx,rx,day)) or set(tx)!=set(range(6)) or set(rx)!=set(d.RX) or set(day)!={1,2,3} or np.any(np.sum(abs(factor)**2,axis=1)>1+1e-12):raise ValueError('Complete balanced original source factors required')
 matrices=np.empty((5,3,6,20,20),dtype=np.complex128);records=[]
 for i,r in enumerate(d.RX):
  for j,date in enumerate((1,2,3)):
   for t in range(6):
    v=factor[(rx==r)&(day==date)&(tx==t)]
    if len(v)!=per_cell:raise ValueError('Exact original cell membership required')
    m=np.einsum('ni,nj->ij',v,v.conj())/per_cell;matrices[i,j,t]=m;tr=np.sum(abs(v)**2,axis=1);dispersion=float(np.mean(tr**2)-np.sum(abs(m)**2))
    records.append(dict(RX=r,day=date,TX=t,count=per_cell,mean_trace=float(np.trace(m).real),Frobenius_norm=float(np.linalg.norm(m)),projector_population_Frobenius_dispersion=max(0.,dispersion)))
 return matrices,records
def evaluate(L,V):
 L,V=(np.asarray(v,dtype=np.complex128) for v in (L,V))
 if any(v.shape!=(5,3,6,20,20) or not np.isfinite(v).all() for v in (L,V)):raise ValueError('All180 finite complex20x20 source cellmeans required')
 if any(np.max(abs(v-v.conj().swapaxes(-1,-2)))>1e-12 for v in (L,V)):raise ValueError('Hermitian source matrices required')
 lc=L-L.mean(axis=0,keepdims=True);vc=V-V.mean(axis=0,keepdims=True);base=[];err=[];conditional=[];folds=[]
 for hd in range(3):
  for ht in range(6):
   days=[j for j in range(3) if j!=hd];tx=[j for j in range(6) if j!=ht];prediction=lc[:,days][:,:,tx].mean((1,2));txprediction=lc[:,days,ht].mean(1);actual=vc[:,hd,ht]
   b=np.sum(abs(actual)**2,axis=(-2,-1));e=np.sum(abs(actual-prediction)**2,axis=(-2,-1));c=np.sum(abs(actual-txprediction)**2,axis=(-2,-1));base.append(b);err.append(e);conditional.append(c)
   bv,ev,cv=map(lambda a:float(a.sum()),(b,e,c));folds.append(dict(TX=ht,day=hd+1,fit_L_cells=50,evaluation_V_cells=5,baseline_SSE=bv,shared_RX_SSE=ev,TXconditional_SSE=cv,risk_gain=1-ev/bv if bv>1e-20 else None))
 b,e,c=(np.array(v) for v in (base,err,conditional));B,E,C=map(lambda a:float(a.sum()),(b,e,c));gain=1-E/B if B>1e-20 else None;positive=sum(f['risk_gain'] is not None and f['risk_gain']>0 for f in folds);thresholds=d.config()['criteria'];tol=thresholds['numerical_SSE_tolerance'];rms=float(np.sqrt(b.mean()));perrx={str(r):dict(baseline_SSE=float(b[:,i].sum()),shared_RX_SSE=float(e[:,i].sum()),risk_gain=1-float(e[:,i].sum()/b[:,i].sum()) if b[:,i].sum()>1e-20 else None) for i,r in enumerate(d.RX)}
 conditions=dict(nontrivial_RX_shape=rms>=thresholds['min_baseline_Frobenius_RMS'],relative_risk_gain=gain is not None and gain>=thresholds['min_relative_risk_gain'],positive_folds=positive>=thresholds['min_positive_folds'],each_RX_nonnegative=all(a['shared_RX_SSE']<=a['baseline_SSE']+tol for a in perrx.values()),not_strong_TX_dependence=E<=thresholds['max_shared_to_TXconditional_SSE_ratio']*C+tol)
 return dict(decision='SOURCE_PERIODIC_SHAPE_ASSOCIATION_ONLY' if all(conditions.values()) else 'CLOSED_FIXED_PERIODIC_SIGNAL_SHAPE',conditions=conditions,criteria=thresholds,baseline_Frobenius_RMS=rms,baseline_SSE=B,shared_RX_SSE=E,TXconditional_SSE=C,risk_gain=gain,positive_folds=positive,per_RX=perrx,folds=folds,source_RX_holdout=None,L_only_fit=True,V_used_for_fit=False,shared_coefficient=1,IQ_operator_implemented=False,Ag_proved=False,joint_goal_complete=False,target_access=False)
