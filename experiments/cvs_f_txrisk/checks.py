"""Focused source-only calibration shape, label equivariance and limits."""
import copy,json
from .risk import calibrate

def run():
 counts=[(99-i,100) for i in range(6)];rx=[copy.deepcopy(counts) for _ in range(5)];base=calibrate(counts,rx)
 if abs(sum(base['weights'])-6)>1e-12 or not all(.5<=v<=1.5 for v in base['weights']) or base['weights'][-1]!=1.5:raise ValueError('Fixed bounded centered weighting differs')
 perm=[4,1,5,2,0,3];p=calibrate([counts[i] for i in perm],[[r[i] for i in perm] for r in rx])
 if any(abs(v-base['weights'][i])>1e-12 for v,i in zip(p['weights'],perm)):raise ValueError('TX label permutation changed algorithm')
 if calibrate([(90,100)]*6,[[(90,100)]*6 for _ in range(5)])['weights']!=[1.]*6:raise ValueError('Equal source difficulty must preserve baseline')
 negative=0
 for cc,rr,b in ((counts[:5],rx,.5),(counts,rx[:4],.5),(counts,rx,1.),([(0,0)]+counts[1:],rx,.5),([(101,100)]+counts[1:],rx,.5),(counts,[r[:5] for r in rx],.5)):
  try:calibrate(cc,rr,b)
  except ValueError:negative+=1
  else:raise ValueError('Invalid source calibration accepted')
 return dict(status='PASS',protocol_negative_cases=negative,label_permutation_equivariant=True,source_equal_risk_baseline_exact=True,bounded_weights_mean_one=True,source_or_target_IQ_reads=0,optimizer_updates=0,head_fit_updates=0)
if __name__=='__main__':print(json.dumps(run()))
