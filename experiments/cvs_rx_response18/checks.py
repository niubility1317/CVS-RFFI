"""Known response and scientific counterexamples; no raw IQ or model."""
from copy import deepcopy
import numpy as np
from . import design as d
from .core import F,fit,apply,evaluate
from .permissions import validate
def fixture(kind='shared'):
 rng=np.random.default_rng(202610184201);P=[]
 for role in range(2):
  a=np.empty((5,3,6,20,20),complex)
  for day in range(3):
   for tx in range(6):
    u=np.exp(rng.normal(0,.25,20)+1j*rng.uniform(-np.pi,np.pi,20));G=np.outer(u,u.conj())/np.vdot(u,u).real
    for r in range(5):
     h=np.exp(.08*r*np.cos(np.arange(20))+.15j*r*np.sin(np.arange(20)/3))
     if kind=='TX_only':h=np.ones(20)
     if kind=='reversal' and role==1:h=1/h
     if kind=='TX_response':h=np.exp((2*(tx%2)-1)*(.12*r*np.cos(np.arange(20))+.6j*r*np.sin(np.arange(20)/3)))
     a[r,day,tx]=F.conj().T@apply(h,G)@F
  P.append(a)
 return P
def run():
 count=0
 def reject(fn):
  nonlocal count
  try:fn()
  except (ValueError,KeyError,TypeError):count+=1
  else:raise AssertionError('Invalid source response evidence accepted')
 done=dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY',L=6300,V=27000,source_RX=list(d.RX),source_RX_holdout=None,unique_IQ_observations=33300,CP=0,NN=0,updates=0,U_used=False,U_truth_read=False,target_access=False)
 old=dict(source_contract=d.SOURCE,RX=list(d.RX),L=6300,V=27000,window=[80,160],period=20,IQ_points=256,target_access=False);resolved=dict(config=old,source_roles_exact_match=True);validate(done,resolved)
 for key,value in [('L',5580),('V',720),('source_RX',[3,4,6,8]),('U_truth_read',True),('target_access',True),('source_RX_holdout',1),('NN',1)]:
  bad=deepcopy(done);bad[key]=value;reject(lambda:validate(bad,resolved))
 for key,value in [('source_roles_exact_match',False),('config',dict(old,period=16))]:
  bad=deepcopy(resolved);bad[key]=value;reject(lambda:validate(done,bad))
 for key,value in [('reference_RX',3),('phase_anchor_bin',2),('epsilon',1e-10),('target_access',True)]:
  c=d.config();c[key]=value;reject(lambda:d.validate(c))
 cases={};coeffs=None
 for name in ('shared','TX_only','reversal','TX_response'):
  L,V=fixture(name);r,h=evaluate(L,V);cases[name]=dict(decision=r['decision'],response_SSE=r['response_SSE'],additive_SSE=r['additive_SSE'],conditions=r['conditions'])
  assert all(r['conditions'].values())==(name=='shared'),name
  if name=='shared':coeffs=h;shared=(L,V,r)
 L,V,r=shared;changed=V.copy();changed[[1,2]]=changed[[2,1]];rr,hh=evaluate(L,changed);assert np.array_equal(coeffs,hh),'V entered fitted response'
 G=F@L[0,0,0]@F.conj().T;h=coeffs[0,1];assert np.max(abs(apply(h,G)-apply((2+3j)*h,G)))<1e-12
 reject(lambda:fit(np.zeros((9,20,20)),np.zeros((9,20,20))));reject(lambda:evaluate(L[:4],V));bad=L.copy();bad[0,0,0,0,0]=np.nan;reject(lambda:evaluate(bad,V));bad=L.copy();bad[0,0,0]=-np.eye(20);reject(lambda:evaluate(bad,V))
 return dict(status='PASS',negative_cases=count,scientific_cases=cases,synthetic_LV_cell_matrices=4*180,closed_form_nonreference_fits=5*72,source_response_prediction_calls=5*90,real_IQ=0,CP=0,NN=0,updates=0,target=False)
if __name__=='__main__':
 import json
 print(json.dumps(run(),allow_nan=False))
