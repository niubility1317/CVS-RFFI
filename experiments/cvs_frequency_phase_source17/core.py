"""Use unchanged observed-IQ measurement;source metadata only groups legal L."""
import numpy as np
from . import design as d
from experiments.cvs_frequency_phase17.estimator import observe,summarize

def metadata(tx,rx,day):
 tx,rx,day=(np.asarray(v) for v in (tx,rx,day))
 if any(v.shape!=(5040,) or v.dtype.kind not in 'iu' for v in (tx,rx,day)) or set(tx)!=set(range(6)) or set(rx)!=set(d.FIT) or set(day)!={1,2,3}:raise ValueError('Only complete visible fitL metadata permitted')
 cells=[]
 for t in range(6):
  for dd in (1,2,3):
   for r in d.FIT:
    indices=np.flatnonzero((tx==t)&(rx==r)&(day==dd))
    if len(indices)!=70:raise ValueError('All72 cells/70 original L packets required')
    cells.append((t,dd,r,indices))
 return cells

def analyze(closure,valid,quality,tx,rx,day):
 closure=np.asarray(closure);valid=np.asarray(valid);cells=metadata(tx,rx,day)
 if closure.shape!=(5040,8) or not np.iscomplexobj(closure) or not np.isfinite(closure).all() or np.any(np.abs(closure)>1+1e-9) or valid.shape!=(5040,) or valid.dtype.kind!='b':raise ValueError('All5040 observed finite softunit phaseclosures/boolean quality required')
 ratio=np.asarray(quality['min_carrier_ratio']);noise=np.asarray(quality['projection_noise_power']);cfo=np.asarray(quality['CFO_estimate_Hz'])
 if any(v.shape!=(5040,) or not np.isfinite(v).all() for v in (ratio,noise,cfo)) or np.any(ratio<0) or np.any(noise<=0) or not np.array_equal(valid,ratio>=25.):raise ValueError('Complete fixed measured quality required')
 mu=np.empty((6,3,4,8),dtype=np.complex128);summaries=[]
 for t,dd,r,idx in cells:
  v=closure[idx];av=v.mean(0);mu[t,dd-1,d.FIT.index(r)]=av
  summaries.append(dict(TX=t,day=dd,RX=r,count=70,valid_packets=int(valid[idx].sum()),closure_mean_real=av.real.tolist(),closure_mean_imag=av.imag.tolist(),closure_population_variance=(np.abs(v-av)**2).mean(0).tolist(),carrier_ratio_min=float(ratio[idx].min()),carrier_ratio_max=float(ratio[idx].max())))
 all_valid=bool(valid.all())
 if all_valid:estimate=summarize(mu)
 else:estimate=dict(identifiable=False,reason='Required carriers not identified in all original packets;none dropped;no shared-RX fit computed')
 return dict(decision='PASS' if estimate.get('detected_RX',False) else 'FAIL',criteria=d.config()['criteria'],all_packets_identifiable=all_valid,valid_packets=int(valid.sum()),rejected_packets=5040-int(valid.sum()),cells=summaries,estimate=estimate,source_only=True,fit_L_packets=5040,source_cells=72,observer_calls=1,IQ_views_generated=0,checkpoint_loads=0,network_forwards=0,optimizer_steps=0,head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,scope='Source RX-associated phase statistic only;not purehardware,identitypreservation orrecognitiongain')

def measure(x,tx,rx,day):
 x=np.asarray(x);metadata(tx,rx,day)
 if x.shape!=(5040,2,256) or x.dtype.kind!='f' or not np.isfinite(x).all():raise ValueError('All5040 finite original float IQ Nx2x256 required')
 z=x[:,0].astype(np.float64)+1j*x[:,1].astype(np.float64)
 c,valid,quality=observe(z)
 return analyze(c,valid,quality,tx,rx,day),dict(closure_real=c.real.tolist(),closure_imag=c.imag.tolist(),valid=valid.tolist(),quality={k:v.tolist() for k,v in quality.items()},TX=np.asarray(tx).tolist(),RX=np.asarray(rx).tolist(),day=np.asarray(day).tolist())
