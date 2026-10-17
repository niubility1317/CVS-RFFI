"""Bounded current input permissions and estimator unit counterexamples."""
import copy,json
import numpy as np
from . import design as d
from .quantgrid import estimate
from .statistics import summarize

def reject(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError('Invalid input/config accepted')
def main():
 n=0
 for key,value in [('target_access',True),('U',1),('V',1),('held_RX1',1260),('checkpoint_reads',1),('network_forwards',2),('optimizer_updates',1),('automatic_training',True),('fit_RX',[1,3,4,6,8])]:
  c=copy.deepcopy(d.config());c[key]=value;reject(lambda:d.validate(c));n+=1
 for a in (np.zeros((2,256)),np.zeros((2,255)),np.full((2,256),np.nan)):reject(lambda:estimate(a));n+=1
 reject(lambda:summarize([]));n+=1
 rng=np.random.default_rng(d.SEED);pos=neg=0
 for q in (.125,.03125,.0078125,.001953125):
  for gain in (.3,1.,3.7):
   for offset in (0.,.37):
    raw=rng.normal(size=(2,256));x=((np.rint(raw/q)*q+offset)*gain).astype(np.float32);scale=float(np.sqrt(np.mean(x.astype(np.float64)**2)));z=estimate(x/scale);assert z['observable'] and abs(z['q']-q*gain/scale)/(q*gain/scale)<.001;pos+=1
 for dtype in (np.float32,np.float64):
  for i in range(12):assert not estimate(rng.normal(size=(2,256)).astype(dtype))['observable'];neg+=1
 for i in range(12):
  x=np.stack([np.sin(2*np.pi*np.arange(256)/20),np.cos(2*np.pi*np.arange(256)/20)]).astype(np.float32);assert not estimate(x)['observable'];neg+=1
 print(json.dumps(dict(status='PASS',protocol_negatives=n,synthetic_positive=pos,synthetic_negative=neg,real_IQ=0,checkpoint_reads=0,NN=0,updates=0)))
if __name__=='__main__':main()
