"""Fixed scaling law,scientific counterexamples and source-role negative checks."""
import copy,json
import numpy as np
from .core import observe,evaluate,cube
from .permissions import select
from . import design as d

def role_fixture():
 class X:
  def __init__(self,n):self.shape=(n,2,256)
 class A(list):
  def tolist(self):return list(self)
 class Packed(dict):
  def __getitem__(self,key):
   if key not in ('contract','labeled','validation'):raise AssertionError('No U/query role access allowed')
   return super().__getitem__(key)
 p=Packed();ids={}
 for key,handle,n in [('labeled','L_s',70),('validation','V',300)]:
  rows=[(t,r,z) for t in range(6) for r in d.RX for z in (1,2,3) for _ in range(n)];v=dict(x=X(len(rows)),y=A(x[0] for x in rows),rx=A(x[1] for x in rows),day=A(x[2] for x in rows),ids=[handle+':'+str(i) for i in range(len(rows))]);p[key]=v;ids[handle]=v['ids'].copy()
 ids['U_s']=['U-hidden'];c=dict(role_ids=ids,native_role_comparison='EXACT_MATCH');p['contract']=copy.deepcopy(c);ready=dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE',target_access=False,source_contract=copy.deepcopy(c));return p,ready,c

def run():
 checks=[];negative=0
 def bad(call):
  nonlocal negative
  try:call()
  except (ValueError,KeyError,TypeError):negative+=1
  else:raise AssertionError('Negative accepted')
 rng=np.random.default_rng(2026101841);x=rng.normal(size=(12,2,256));z=x[:,0]+1j*x[:,1];a=1.07;y=np.abs(z)**(a-1)*z;y/=np.sqrt(np.mean(np.abs(y)**2,axis=1))[:,None];q=observe(x)['shape'];p=observe(np.stack((y.real,y.imag),axis=1))['shape'];error=float(np.max(np.abs(p-a*q)))
 if error>5e-14:raise ValueError('Unfloored radial quantile scaling law failed')
 u=observe(x*3.5)['shape']
 if np.max(np.abs(u-q))>5e-14:raise ValueError('Gain should cancel in centered quantiles')
 checks.append(dict(radial_scaling_max_error=error,gain_invariant=True));zero=observe(np.zeros((1,2,256)))
 if not zero['zero_RMS'][0] or not np.all(zero['shape']==0):raise ValueError('Zero packets must remain present and explicit')
 bad(lambda:observe(np.zeros((2,2,255))));bad(lambda:observe(np.full((1,2,256),np.nan)))
 shape=np.array([-2.,-1.3,-.6,.4,.8,1.2]);L=np.empty((6,5,3,6));V=L.copy();g=np.array([1.,1.08,.93,1.05,.96])
 for t in range(6):
  for r in range(5):
   for day in range(3):L[t,r,day]=shape*(1+.2*t+.03*day)*g[r];V[t,r,day]=shape*(1+.2*t+.03*day+.01)*g[r]
 value,arrays=evaluate(L,V)
 json.dumps(value,allow_nan=False)
 if not value['source_pass'] or value['SSE']['shared']>1e-25:raise ValueError('True common mild radial relation should qualify')
 checks.append(dict(common_mild_radial='PASS',positive_folds=value['positive_folds']))
 vv=V.copy();vv[:,2]+=.2;_,arrays2=evaluate(L,vv)
 if not np.array_equal(arrays['shared_slope'],arrays2['shared_slope']):raise ValueError('V influenced fitted source coefficients')
 checks.append(dict(V_does_not_fit_parameters=True))
 # Arbitrary additive shape biases must not be miscalled powerlaw evidence.
 A=L.copy();B=V.copy()
 for r in range(1,5):
  offset=np.array([-.1,.05,-.03,.08,-.02,.04])*r
  A[:,r]=A[:,0]+offset;B[:,r]=B[:,0]+offset
 v,_=evaluate(A,B)
 json.dumps(v,allow_nan=False)
 if v['conditions']['shared_vs_additive'] or v['source_pass']:raise ValueError('Additive counterexample wrongly qualified')
 checks.append(dict(additive_counterexample='CLOSED'))
 # TX-conditioned variation cannot be hidden by an overall source association.
 A=L.copy();B=V.copy()
 for t in range(6):
  for r in range(1,5):f=1+(.08 if t%2 else -.08)*(r/4);A[t,r]=A[t,0]*f;B[t,r]=B[t,0]*f
 v,_=evaluate(A,B)
 if v['conditions']['not_worse_TXconditional'] or v['source_pass']:raise ValueError('TX conditional counterexample wrongly qualified')
 checks.append(dict(TX_conditioned_counterexample='CLOSED'))
 A=L.copy();B=V.copy();A[:,1]=A[:,0]*1.4;B[:,1]=B[:,0]*1.4;v,_=evaluate(A,B)
 if v['conditions']['common_slope_mild'] or v['source_pass']:raise ValueError('Strong companding cannot qualify as mild')
 qv,_=evaluate(L,V,dict(max_floor_fraction_each_cell=.02,zero_RMS_packets=0))
 if qv['conditions']['valid_log_shape']:raise ValueError('Floor contamination qualification failed')
 checks.append(dict(strong_or_floor_contaminated='CLOSED'))
 p,ready,c=role_fixture();selected=select(p,ready,c)
 if set(selected)!={'L_s','V'}:raise ValueError('Only legal source roles expected')
 bad(lambda:select(p,dict(ready,target_access=True),c));wrong=copy.deepcopy(c);wrong['role_ids']['U_s']=[p['labeled']['ids'][0]];bad(lambda:select(p,ready,wrong));old=p['labeled']['ids'][0];p['labeled']['ids'][0]=p['labeled']['ids'][1];bad(lambda:select(p,ready,c));p['labeled']['ids'][0]=old;old=p['validation']['rx'][0];p['validation']['rx'][0]=2;bad(lambda:select(p,ready,c));p['validation']['rx'][0]=old;p['labeled']['x'].shape=(6300,2,255);bad(lambda:select(p,ready,c))
 bad(lambda:evaluate(L,np.zeros((6,4,3,6))));bad(lambda:cube(np.zeros((89,6)),dict(y=[],rx=[],day=[]),1))
 return dict(status='PASS',checks=checks,protocol_or_input_negative_cases=negative,synthetic_IQ_observations=37,real_IQ=0,CP=0,NN=0,updates=0,U_truth=0,target=0)
if __name__=='__main__':
 import json
 print(json.dumps(run()))
