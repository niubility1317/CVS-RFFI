"""Focused geometry/protocol negatives and generative counterexamples;0 realIQ/NN."""
import copy,json
import numpy as np
from . import design as d
from .core import observe,analyze

def main():
 negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except (ValueError,KeyError):negative+=1;return
  raise AssertionError('Invalid source input accepted')
 rng=np.random.default_rng(20261017201);x=rng.normal(size=(64,2,256));saved=x.copy();a=observe(x);assert np.array_equal(x,saved)
 z=x[:,0]+1j*x[:,1];z2=.8*z*np.exp(1j*(.37+.001*np.arange(256)));b=observe(np.stack((z2.real,z2.imag),axis=1))
 for k in ('shape','coherence','residual_ratio'):assert np.allclose(a[k],b[k],rtol=1e-10,atol=1e-12)
 carrier=rng.normal(size=20)+1j*rng.normal(size=20);z=np.tile(carrier,13)[:256];c=observe(np.tile(np.stack((z.real,z.imag))[None],(20,1,1)));assert np.max(c['residual_ratio'])<1e-25 and np.max(c['shape'])==0
 for v in (np.zeros((1,2,256)),np.zeros((1,2,255)),x*np.nan):reject(lambda v=v:observe(v))
 tx=[];rx=[];day=[];shape=[];domain=rng.normal(scale=.12,size=(4,8));common_tx=rng.normal(scale=.02,size=(6,8));common_day=rng.normal(scale=.02,size=(3,8))
 for i,r in enumerate(d.FIT):
  for j,dd in enumerate((1,2,3)):
   for t in range(6):
    for k in range(70):tx.append(t);rx.append(r);day.append(dd);shape.append(domain[i]+common_tx[t]+common_day[j]+rng.normal(scale=.002,size=8))
 tx,rx,day=(np.asarray(v,dtype=np.int64) for v in (tx,rx,day));obs=dict(shape=np.asarray(shape),coherence=np.full(5040,.96),residual_ratio=np.full(5040,.005),zero_residual=np.zeros(5040,dtype=bool));good=analyze(obs,tx,rx,day);assert good['decision']=='PASS' and good['positive_folds']==18
 weak=copy.deepcopy(obs);weak['shape']*=.001;assert analyze(weak,tx,rx,day)['decision']=='FAIL'
 impure=copy.deepcopy(obs);impure['coherence'][:70]=.79;assert analyze(impure,tx,rx,day)['decision']=='FAIL'
 txonly=copy.deepcopy(obs);txonly['shape']=np.array([common_tx[t]+common_day[dd-1] for t,dd in zip(tx,day)]);assert analyze(txonly,tx,rx,day)['decision']=='FAIL'
 drifting=copy.deepcopy(obs);drifting['shape']=np.asarray([domain[d.FIT.index(r)]*(1 if dd!=2 else -1)+common_tx[t] for r,dd,t in zip(rx,day,tx)]);assert analyze(drifting,tx,rx,day)['decision']=='FAIL'
 for arr,pos,val in [('tx',0,-1),('tx',0,6),('rx',0,1),('day',0,4)]:
  meta=dict(tx=tx.copy(),rx=rx.copy(),day=day.copy());meta[arr][pos]=val;reject(lambda meta=meta:analyze(obs,**meta))
 reject(lambda:analyze(obs,tx,rx.astype(float)+.01,day))
 for key,val in [('shape',np.zeros((5040,7))),('shape',np.full((5040,8),np.nan)),('shape',np.full((5040,8),2.)),('coherence',np.full(5040,1.1)),('residual_ratio',np.full(5040,-1.))]:
  bad=copy.deepcopy(obs);bad[key]=val;reject(lambda bad=bad:analyze(bad,tx,rx,day))
 bad=copy.deepcopy(obs);del bad['coherence'];reject(lambda:analyze(bad,tx,rx,day));badtx=tx.copy();badtx[0]=1;reject(lambda:analyze(obs,badtx,rx,day))
 for key,value in [('window',[40,200]),('cycle',16),('lags',[1,2,3]),('fit_RX',[1,3,4,6]),('target_access',True),('network_forwards',2),('checkpoint_loads',1)]:
  cfg=d.config();cfg[key]=value;reject(lambda cfg=cfg:d.validate(cfg))
 result=dict(status='PASS',negative_cases=negative,synthetic_IQ_observations=148,synthetic_metadata_rows=5040,synthetic_truth_examples=['persistentRXshape PASS','weakRXshape FAIL','poorperiodicity FAIL','TXonly FAIL','dayreversal FAIL'],global_gain_phase_linearphase_invariance=True,source_input_unmodified=True,real_IQ_reads=0,network_forwards=0,optimizer_steps=0,head_fit_updates=0,target_access=False)
 print(json.dumps(result));return result
if __name__=='__main__':main()
