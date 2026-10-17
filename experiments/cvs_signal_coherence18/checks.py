"""Protocol negatives, projector invariants and known statistical counterexamples."""
from copy import deepcopy
import numpy as np
from . import design as d
from .core import observe,cube,evaluate
from .permissions import select
class ShapeOnly:
 def __init__(self,n):self.shape=(n,2,256)
class SourceOnly(dict):
 def __getitem__(self,k):
  if k=='unlabeled':raise AssertionError('U container must remain unread')
  return super().__getitem__(k)
def synthetic_roles():
 p=SourceOnly();ids={'U_s':['opaque-hidden-U']}
 for key,handle,n in [('labeled','L_s',70),('validation','V',300)]:
  a=np.array([(t,r,date) for t in range(6) for r in d.RX for date in (1,2,3) for _ in range(n)]);unique=[handle+'-opaque-'+str(i) for i in range(len(a))];ids[handle]=unique;p[key]=dict(x=ShapeOnly(len(a)),y=a[:,0],rx=a[:,1],day=a[:,2],ids=unique)
 c=dict(role_ids=ids,native_role_comparison='EXACT_MATCH');p['contract']=deepcopy(c);ready=dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE',target_access=False,source_contract=deepcopy(c));return p,ready,c
def run():
 negatives=0
 def reject(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,TypeError):negatives+=1
  else:raise AssertionError('Protocol/shape negative accepted')
 p,ready,contract=synthetic_roles();assert set(select(p,ready,contract))=={'L_s','V'}
 for name in ('overlap','contract','RX','shape','ready','target'):
  q,r,c=map(deepcopy,(p,ready,contract))
  if name=='overlap':c['role_ids']['U_s'].append(c['role_ids']['L_s'][0]);q['contract']=deepcopy(c);r['source_contract']=deepcopy(c)
  elif name=='contract':q['contract']['role_ids']['V'][0]='wrong'
  elif name=='RX':q['validation']['rx'][:]=3
  elif name=='shape':q['labeled']['x'].shape=(6300,2,255)
  elif name=='ready':r['status']='UNKNOWN'
  else:r['target_access']=True
  reject(lambda:select(q,r,c))
 for key,value in [('RX',[3,4,6,8]),('U_used',True),('window',[0,80]),('period',16),('target_access',True)]:
  conf=d.config();conf[key]=value;reject(lambda:d.validate(conf))
 rng=np.random.default_rng(202610184101);base=rng.normal(size=(8,20))+1j*rng.normal(size=(8,20));n=np.arange(256);z=base[:,n%20]*np.exp(.012j*n);x=np.stack((z.real,z.imag),axis=1);before=x.copy();a=observe(x);u=a['factor_real']+1j*a['factor_imag'];P=u[:,:,None]*u[:,None,:].conj();maxerr=0.
 for gain,phase,delta in [(2.,.4,0.),(.3,-1.,0.),(1.,0.,.01),(1.7,2.,-.02)]:
  w=gain*z*np.exp(1j*(phase+delta*n));o=observe(np.stack((w.real,w.imag),axis=1));v=o['factor_real']+1j*o['factor_imag'];err=float(np.max(abs(P-v[:,:,None]*v[:,None,:].conj())));maxerr=max(maxerr,err)
  assert err<2e-13
 assert np.array_equal(x,before) and np.max(abs(P-P.conj().swapaxes(-1,-2)))<1e-14
 assert np.max(abs(np.trace(P,axis1=-2,axis2=-1)-1))<2e-12
 assert np.max(abs(observe(np.zeros((1,2,256)))['factor_real']))==0
 reject(lambda:observe(np.zeros((1,2,255))));reject(lambda:observe(np.full((1,2,256),np.nan)))
 reject(lambda:cube(np.zeros((90,20)),np.zeros(90,dtype=int),np.ones(90,dtype=int),np.ones(90,dtype=int),1))
 # Positive semidefinite,trace1 synthetic cellmeans with additive RX contrast.
 I=np.eye(20)/20;signal=np.stack([I+.04*(np.diag(np.eye(20)[i])-np.diag(np.eye(20)[(i+1)%5])) for i in range(5)]);stable=np.broadcast_to(signal[:,None,None],(5,3,6,20,20)).copy().astype(complex);zero=np.broadcast_to(I,stable.shape).copy();contrast=stable-stable.mean(0,keepdims=True)
 cases=dict(shared_RX=evaluate(stable,stable),TX_only=evaluate(zero,zero),L_V_reversal=evaluate(stable,2*stable.mean(0,keepdims=True)-stable),RX_day_reversal=evaluate(zero+contrast*np.array([-1,0,1])[None,:,None,None,None],zero+contrast*np.array([-1,0,1])[None,:,None,None,None]),RX_TX_reversal=evaluate(zero+contrast*np.array([-1,-1,-1,1,1,1])[None,None,:,None,None],zero+contrast*np.array([-1,-1,-1,1,1,1])[None,None,:,None,None]))
 for key,result in cases.items():assert all(result['conditions'].values())==(key=='shared_RX'),key
 changed=evaluate(stable,stable+.001*contrast)
 assert [f['shared_RX_SSE'] for f in cases['shared_RX']['folds']]!=[f['shared_RX_SSE'] for f in changed['folds']]
 reject(lambda:evaluate(stable[:4],stable));bad=stable.copy();bad[0,0,0,0,1]+=1j;reject(lambda:evaluate(bad,stable))
 return dict(status='PASS',negative_cases=negatives,synthetic_IQ_packets_observed=8*5+1,projector_invariance_max_error=maxerr,source_control_cases={k:v['decision'] for k,v in cases.items()},synthetic_role_packets=33300,real_IQ=0,CP=0,NN=0,updates=0,target=False)
if __name__=='__main__':
 import json
 print(json.dumps(run(),allow_nan=False))
