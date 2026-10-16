"""Focused algebraic/input checks only,not the registered noisy matrix."""
import json
import numpy as np
from . import design as d
from .estimator import observe,summarize,estimate
from .synthetic import generate,closure,centered_phase,response

def main():
 count=0
 def rejects(f):
  nonlocal count
  try:f()
  except (ValueError,TypeError,IndexError):count+=1;return
  raise AssertionError('Required invalidinput accepted')
 for k,v in [('IQ_length',255),('reference',[20,180]),('NN',1),('templates',['tone']),('draws',5),('data_seed',0),('allpass_a',[.1]),('output_root','other')]:
  c=d.config();c[k]=v;rejects(lambda:d.validate(c))
 for x in [np.zeros((1,255),complex),np.zeros((1,2,256),complex),np.ones((1,256)),np.full((1,256),complex(np.nan,0)),np.zeros((1,256),complex)]:rejects(lambda:observe(x))
 rejects(lambda:estimate(np.ones((1,256),complex)));rejects(lambda:summarize(np.ones((6,3,5,8),complex)))
 for t,s,c,n in [('bad',20.,'RX_LTI',1),('flat12',1.,'RX_LTI',1),('flat12',20.,'bad',1),('flat12',20.,'RX_LTI',0)]:rejects(lambda:generate(t,s,c,1,packets=n))
 x,truth=generate('qpsk12',None,'RX_LTI',d.DATA_SEED,packets=1);q,valid,_=observe(x);assert valid.all();base=q.copy();checks=1
 for phase,gain,cfo in [(1.7,.3,0.),(-2.1,4.,100000.),(.2,1.,-100000.)]:
  y=x*gain*np.exp(1j*(phase+2*np.pi*cfo*np.arange(256)/d.FS));z,v,_=observe(y);assert v.all() and np.max(np.abs(z-base))<2e-6;checks+=1
 y=x.copy();y[:,:80]=3+4j;y[:,160:]=-7j;z,v,_=observe(y);assert v.all() and np.array_equal(z,base);checks+=1
 z,truth=generate('flat12',None,'DEGENERATE',d.DATA_SEED,packets=1);_,v,_=observe(z);assert not v.any();checks+=1
 phaseRX=centered_phase(closure(response(np.array(d.ALLPASS),2*np.pi*np.array(d.HARMONICS)[None,:]/20)));mu=np.exp(1j*phaseRX)[None,None]*np.exp(1j*np.linspace(-2,2,6)[:,None,None,None]);mu=np.broadcast_to(mu,(6,3,4,8)).copy();s=summarize(mu);assert s['identifiable'] and s['detected_RX'] and np.max(np.abs(np.array(s['profile'])-phaseRX))<1e-12;checks+=1
 signs=np.where(np.arange(6)%2==0,1.,-1.);mu=np.exp(1j*signs[:,None,None,None]*phaseRX[None,None]);s=summarize(np.broadcast_to(mu,(6,3,4,8)));assert not s['detected_RX'] and s['crossfit_risk_improvement']<0;checks+=1
 return dict(status='PASS',negative_cases=count,algebraic_checks=checks,synthetic_packets_generated=144,synthetic_complex_points=36864,real_IQ=0,checkpoint=0,NN=0,updates=0,target_access=False,formal_matrix_executed=False)
if __name__=='__main__':print(json.dumps(main()))
