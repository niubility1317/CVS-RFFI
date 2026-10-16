"""Focused synthetic observer and role negatives; no model, real IQ or checkpoint."""
import numpy as np
from .core import observe,crossfit

def descriptors(mode):
 rng=np.random.default_rng(20261016161);rows=[];tx=[];rx=[];day=[]
 offsets=np.array([-.30,-.10,.10,.30])
 for ri,r in enumerate((3,4,6,8)):
  for d in (1,2,3):
   for t in range(6):
    if mode=='RX':mean=.5+offsets[ri]+.001*(t-2.5)
    elif mode=='TX':mean=.2+.1*t
    elif mode=='DAYFLIP':mean=.5+offsets[ri]*(d-2)
    elif mode=='TXFLIP':mean=.5+offsets[ri]*(1 if t%2 else -1)
    else:raise ValueError('Unknown known case')
    for _ in range(70):rows.append(mean+rng.normal(0,.0005));tx.append(t);rx.append(r);day.append(d)
 a=np.asarray(rows);return dict(primary=a,raw=a,odd=a+.00001,even=a-.00001,coherence=np.ones(len(a))),np.asarray(tx),np.asarray(rx),np.asarray(day)

def run():
 cases={name:crossfit(*descriptors(name)) for name in ('RX','TX','DAYFLIP','TXFLIP')}
 assert cases['RX']['decision']=='PASS' and all(cases[k]['decision']=='FAIL' for k in ('TX','DAYFLIP','TXFLIP'))
 obs,t,r,d=descriptors('RX');zero={k:np.zeros_like(v) for k,v in obs.items()};assert crossfit(zero,t,r,d)['decision']=='FAIL'
 for ptx,prx,pday in ((np.array([4,1,5,0,3,2]),{3:6,4:8,6:3,8:4},{1:3,2:1,3:2}),(np.array([5,4,3,2,1,0]),{3:8,4:6,6:4,8:3},{1:2,2:3,3:1})):
  again=crossfit(obs,ptx[t],np.asarray([prx[int(v)] for v in r]),np.asarray([pday[int(v)] for v in d]))
  assert again['conditions']==cases['RX']['conditions'] and abs(again['mean_risk_gain']-cases['RX']['mean_risk_gain'])<1e-12
 n=np.arange(256);base=np.exp(2j*np.pi*n/20);beta=.03;z=base+beta*base.conj();iq=np.stack((z.real,z.imag))[None];old=iq.copy();result=observe(iq);expected=2*beta/(1+beta**2)
 assert abs(result['primary'][0]-expected)<1e-11 and np.array_equal(iq,old)
 for factor,omega in ((1.,.007),(2.3*np.exp(.8j),-.011),(.6*np.exp(-1.4j),.003)):
  altered=z*factor*np.exp(1j*omega*n);got=observe(np.stack((altered.real,altered.imag))[None]);assert abs(got['primary'][0]-expected)<1e-10
 assert observe(np.zeros((2,2,256)))['primary'].tolist()==[0.,0.]
 outside=iq.copy();outside[:,:,:80]+=3;outside[:,:,160:]-=2;assert abs(observe(outside)['primary'][0]-expected)<1e-11
 negative=0
 bads=[(obs,t[:-1],r,d),(obs,np.full_like(t,-1),r,d),(obs,t,np.ones_like(r),d),(obs,t,r,np.zeros_like(d)),(dict(obs,primary=np.full(5040,np.nan)),t,r,d),(dict(obs,primary=np.ones(5039)),t,r,d),(dict(obs,primary=np.full(5040,1.1)),t,r,d)]
 tt=t.copy();tt[0]=1;bads.append((obs,tt,r,d))
 for args in bads:
  try:crossfit(*args)
  except ValueError:negative+=1
  else:raise AssertionError('Illegal source descriptor accepted')
 for bad in (np.zeros((2,2,255)),np.full((2,2,256),np.nan)):
  try:observe(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Illegal source IQ accepted')
 return dict(status='PASS',known_cases={k:v['decision'] for k,v in cases.items()},negative_cases=negative,allTX_and_RX_day_permutation_invariant=True,known_image_analytic_circularity_error=float(abs(result['primary'][0]-expected)),fixed_window_and_no_input_mutation=True,synthesized_descriptor_packets=20160,synthetic_IQ_packets_observed=7,network_forwards=0,optimizer_steps=0,source_IQ_reads=0,target_access=False)
if __name__=='__main__':
 import json
 print(json.dumps(run(),indent=2))
