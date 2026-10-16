"""Non-fitted source metadata statistics;no classification,IQ modification or ML."""
import numpy as np
RX=(3,4,6,8)
def observe(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty IQ[N,2,256] required')
 z=x[:,0]+1j*x[:,1];a=z[:,20:160];b=z[:,40:180]
 pa=np.mean(np.abs(a)**2,axis=1);pb=np.mean(np.abs(b)**2,axis=1)
 if np.any(pa<=1e-20) or np.any(pb<=1e-20):raise ValueError('Nonzero fixedpreamble IQ required;no packet filtering')
 dot=np.mean(np.conj(a)*b,axis=1);coh=np.abs(dot)/np.sqrt(pa*pb);omega=np.angle(dot)/20.
 blocks=(z[:,20:180]*np.exp(-1j*omega[:,None]*np.arange(20,180))).reshape(len(x),8,20)
 residual=blocks-blocks.mean(1,keepdims=True);r0=np.mean(np.abs(residual)**2,axis=(1,2));power=np.mean(np.abs(blocks)**2,axis=(1,2));cols=[]
 for lag in range(1,5):
  left=residual[:,:,:-lag];right=residual[:,:,lag:];den=np.sqrt(np.mean(np.abs(left)**2,axis=(1,2))*np.mean(np.abs(right)**2,axis=(1,2)))
  q=np.divide(np.mean(np.conj(left)*right,axis=(1,2)),den,out=np.zeros(len(x),dtype=np.complex128),where=den>1e-20*power)
  cols.extend((q.real,q.imag))
 return dict(shape=np.stack(cols,axis=1),coherence=coh,residual_ratio=r0/power,zero_residual=r0<=1e-20*power)
def _gain(b,m):
 if b<=1e-20:return 0. if m<=1e-20 else None
 return 1.-m/b
def analyze(observation,tx,rx,day):
 from .design import config
 arrays={k:np.asarray(observation[k]) for k in ('shape','coherence','residual_ratio','zero_residual')}
 if arrays['shape'].shape!=(5040,8) or any(arrays[k].shape!=(5040,) for k in ('coherence','residual_ratio','zero_residual')) or any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError('Complete finite L5040 cycle observations required')
 if np.any(np.abs(arrays['shape'])>1+1e-10) or np.any(arrays['coherence']<0) or np.any(arrays['coherence']>1+1e-10) or np.any(arrays['residual_ratio']<0):raise ValueError('Invalid normalized correlation or residual power')
 tx,rx,day=(np.asarray(v) for v in (tx,rx,day))
 if any(v.shape!=(5040,) or v.dtype.kind not in 'iu' for v in (tx,rx,day)) or set(tx)!=set(range(6)) or set(rx)!=set(RX) or set(day)!={1,2,3}:raise ValueError('Only complete visiblefitL TX/RX/day metadata allowed')
 cells=np.zeros((4,3,6,8));summary=[]
 for i,r in enumerate(RX):
  for j,d in enumerate((1,2,3)):
   for t in range(6):
    take=(rx==r)&(day==d)&(tx==t)
    if int(take.sum())!=70:raise ValueError('Exactly70 originalL physical samples per cell required')
    cells[i,j,t]=np.median(arrays['shape'][take],axis=0)
    summary.append(dict(RX=r,day=d,TX=t,count=70,shape_median=cells[i,j,t].tolist(),coherence_median=float(np.median(arrays['coherence'][take])),coherence_q10=float(np.quantile(arrays['coherence'][take],.1)),residual_ratio_median=float(np.median(arrays['residual_ratio'][take])),zero_residual_count=int(arrays['zero_residual'][take].sum())))
 base=[];correct=[];wrong=[[],[],[]];folds=[]
 for hd in range(3):
  for ht in range(6):
   train=cells[:,[d for d in range(3) if d!=hd]][:,:,[t for t in range(6) if t!=ht]];prediction=train.mean((1,2));prediction-=prediction.mean(0,keepdims=True)
   actual=cells[:,hd,ht].copy();actual-=actual.mean(0,keepdims=True)
   b=np.square(actual);m=np.square(actual-prediction);base.append(b);correct.append(m)
   for s in range(1,4):wrong[s-1].append(np.square(actual-np.roll(prediction,s,axis=0)))
   bv=float(b.mean());mv=float(m.mean());gain=_gain(bv,mv)
   folds.append(dict(held_TX=ht,held_day=hd+1,baseline_shape_MSE=bv,correct_RX_shape_MSE=mv,risk_gain=gain))
 b=np.asarray(base);m=np.asarray(correct);baseline=float(b.mean());risk=float(m.mean());gain=_gain(baseline,risk);wrong_gains=[_gain(baseline,float(np.asarray(v).mean())) for v in wrong]
 per_rx={str(r):_gain(float(b[:,i].mean()),float(m[:,i].mean())) for i,r in enumerate(RX)};over=gain-max(wrong_gains) if gain is not None and all(v is not None for v in wrong_gains) else None;positive=sum(f['risk_gain'] is not None and f['risk_gain']>0 for f in folds);thresholds=config()['criteria']
 conditions=dict(periodic_cell_median=all(c['coherence_median']>=thresholds['min_cell_median_coherence'] for c in summary),periodic_cell_q10=all(c['coherence_q10']>=thresholds['min_cell_q10_coherence'] for c in summary),nontrivial_shape_contrast=np.sqrt(baseline)>=thresholds['min_shape_contrast_baseline_RMS'],pooled_shape_transfer=gain is not None and gain>=thresholds['min_pooled_shape_risk_gain'],positive_folds=positive>=thresholds['min_positive_folds'],all_RX_nonnegative=all(v is not None and v>=0 for v in per_rx.values()),correct_above_wrong_RX=over is not None and over>=thresholds['min_gain_over_best_wrong_RX'])
 conditions={k:bool(v) for k,v in conditions.items()}
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,thresholds=thresholds,baseline_shape_RMS=float(np.sqrt(baseline)),baseline_shape_MSE=baseline,correct_RX_shape_MSE=risk,pooled_shape_risk_gain=gain,positive_folds=positive,total_folds=18,per_RX_risk_gain=per_rx,wrong_RX_cyclic_risk_gains=wrong_gains,gain_over_best_wrong_RX=over,folds=folds,cells=summary,scope='KnownsourceRX conditionalshape contrast,not fitted classifier,physicalnoise identification or identitygain',fit_L_packets=5040,source_cells=72,network_forwards=0,optimizer_steps=0,head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
