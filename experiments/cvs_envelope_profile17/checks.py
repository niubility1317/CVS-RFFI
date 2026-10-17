"""Bounded synthetic basis/exclusion checks;no data,weights or network."""
import copy,json
import numpy as np
from .core import basis,observe,fit
def run():
 count=0
 def bad(fn):
  nonlocal count
  try:fn()
  except (ValueError,TypeError):count+=1
  else:raise AssertionError('Invalid input accepted')
 for s in [(2,256),(2,2,255),(2,3,256)]:bad(lambda s=s:observe(np.ones(s)))
 for x in [np.zeros((2,2,256)),np.full((2,2,256),np.nan),np.full((2,2,256),np.inf)]:bad(lambda x=x:observe(x))
 for x in [np.zeros((6,3,4,2)),np.full((6,3,5,2),np.nan)]:bad(lambda x=x:fit(x))
 rng=np.random.default_rng(20261017971);state=np.random.get_state();x=rng.normal(size=(12,2,256));log,coef,_=observe(x);ll,cc,_=observe(x*np.linspace(.2,4,12)[:,None,None])
 assert np.max(abs(log-ll))<1e-13 and np.max(abs(coef-cc))<1e-13
 assert np.max(abs(basis().T@basis()-np.eye(2)))<1e-14 and np.max(abs(basis().sum(0)))<1e-14
 temporal=np.exp(.03*np.cos(np.pi*(np.arange(256)+.5)/256));_,changed,_=observe(np.ones((1,2,256))*temporal);assert abs(changed[0,0])>.01
 rx=rng.normal(size=(5,2));tx=rng.normal(size=(6,1,1,2));day=rng.normal(size=(1,3,1,2));a=tx+day+rx[None,None]
 passed=fit(a);assert all(passed['conditions'].values()) and passed['relative_MSE']<1e-25
 for txi in range(6):
  for zi in range(3):
   changed=a.copy();changed[txi]*=17;changed[:,zi]*=23;b=fit(changed);idx=txi*3+zi
   assert np.array_equal(np.asarray(b['folds'][idx]['predicted']),np.asarray(passed['folds'][idx]['predicted']))
 interaction=np.asarray([-1,1,-1,1,-1,1])[:,None,None,None]*np.ones((1,3,1,1))*rx[None,None]
 assert fit(interaction)['decision'].startswith('FAIL')
 tx_only=np.broadcast_to(tx,(6,3,5,2)).copy();assert fit(tx_only)['decision'].startswith('FAIL')
 after=np.random.get_state();assert np.array_equal(state[1],after[1]) and state[2:]==after[2:]
 return dict(status='PASS',protocol_negatives=count,held_TX_day_exclusion_checks=18,synthetic_cases=6,constant_gain_cancelled=True,nonconstant_envelope_observable=True,TX_only_and_TX_RX_interaction_rejected=True,global_RNG_unchanged=True,real_IQ=0,CP=0,NN=0,updates=0,identity_or_Ag_gain_proved=False)
if __name__=='__main__':print(json.dumps(run()))
