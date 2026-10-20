"""Focused scatter algebra, partition independence, and negative cases; no data."""
import copy
import numpy as np
from .analysis import partition,covariance,precision,metrics
from . import design as d

def run():
 S=np.diag([9.,.1,.2]);P,summary=precision(S);assert .25-1e-10<=np.linalg.eigvalsh(P).min() and np.linalg.eigvalsh(P).max()<=4+1e-10 and abs(np.trace(P)/3-1)<1e-10
 signal=np.array([0.,1.,0.]);assert signal@P@signal/np.trace(P@S)>signal@signal/np.trace(S)
 assert np.allclose(precision(S*23)[0],P,atol=1e-12) and np.allclose(precision(np.eye(3))[0],np.eye(3)) and precision(np.zeros((3,3)))[1]['zero_trace_fallback']
 labels=np.array([(c,r,t) for c in range(6) for r in d.RX for t in d.DAYS for _ in range(6)]);y,rx,day=labels.T;part=partition(y,rx,day);assert np.array_equal(part,partition(y,rx,day)) and set(part)=={0,1,2}
 for c in range(6):
  for r in d.RX:
   for t in d.DAYS:assert [int(np.sum((y==c)&(rx==r)&(day==t)&(part==v))) for v in range(3)]==[2,2,2]
 rng=np.random.default_rng(84);z=rng.normal(size=(len(y),3));z[:,0]*=3;z[:,1]+=4*y;A=(part==0)&np.isin(y,[1,3]);first=precision(covariance(z[A],y[A]))[0];changed=z.copy();changed[part!=0]+=100*rng.normal(size=changed[part!=0].shape);assert np.array_equal(first,precision(covariance(changed[A],y[A]))[0])
 Q,_=np.linalg.qr(rng.normal(size=(3,3)));rotated=precision(covariance((z@Q)[A],y[A]))[0];assert np.allclose(rotated,Q.T@first@Q,atol=1e-11)
 old=metrics(z,y,rx,day,part,1,3,first);new=metrics(z@Q,y,rx,day,part,1,3,rotated);assert all(np.allclose(old[k],new[k],atol=1e-10) for k in old)
 groups=y[A]*15+np.searchsorted(d.RX,rx[A])*3+day[A]-1;shift=z[A]+groups[:,None]*np.array([.2,.4,.8]);assert np.allclose(covariance(z[A],groups),covariance(shift,groups),atol=1e-12)
 negative=0
 for bad in (np.array([[1.,1.],[0.,1.]]),np.diag([-1.,1.]),np.array([[float('nan')]])):
  try:precision(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid covariance accepted')
 bad_y=y.copy();bad_y[0]=-1
 try:partition(bad_y,rx,day)
 except ValueError:negative+=1
 else:raise AssertionError('Hidden/unknown TX accepted')
 rows=[dict(model_seed=s,pairs=[dict(indices=[1,3],metrics={'RAW':dict(SNR=1.,cross_RX_signal=1.,per_TX_RX_spread_to_signal=[1.,1.]),'TX':dict(SNR=1.2,cross_RX_signal=1.,per_TX_RX_spread_to_signal=[.9,.9])})]) for s in d.SEEDS]
 assert d.decision(rows)['PASS'];bad=copy.deepcopy(rows);bad[1]['pairs'][0]['metrics']['TX']['per_TX_RX_spread_to_signal'][1]=1.01;assert not d.decision(bad)['PASS'];bad[1]['pairs'][0]['metrics']['TX'].update(SNR=-1.,cross_RX_signal=-1.,per_TX_RX_spread_to_signal=[None,None]);assert not d.decision(bad)['PASS']
 return dict(status='PASS',bounded_mean1_precision=True,scale_and_rotation_invariance=True,isotropic_and_zero_covariance=True,disjoint_balanced_partitions=True,B_C_cannot_change_A_precision=True,conditional_group_shift_invariance=True,decision_positive_and_negative=True,negative_cases=negative,IQ_reads=0,model_forwards=0,head_fit_updates=0,optimizer_updates=0,target_reads=0)
if __name__=='__main__':
 import json
 print(json.dumps(run(),allow_nan=False))
