"""Focused source wrapper/protocol checks;no physical IQ/cache/model access."""
import copy,json
import numpy as np
from . import design as d
from .core import metadata,measure,analyze

def run():
 tx=np.repeat(np.arange(6),840);day=np.tile(np.repeat(np.arange(1,4),280),6);rx=np.tile(np.repeat(d.FIT,70),18)
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid source contract accepted')
 for meta in ((tx[:-1],rx[:-1],day[:-1]),(tx.astype(float),rx,day),(tx,rx.astype(float),day),(tx,rx,day.astype(float)),(tx+1,rx,day),(tx,np.where(rx==3,1,rx),day),(tx,rx,day-1),(np.concatenate(([1],tx[1:])),rx,day)):
  bad(lambda meta=meta:metadata(*meta))
 for key,value in [('fit_L',6300),('fit_RX',[1,3,4,6,8]),('days',[0,1,2]),('window',[64,144]),('carrier_noise_ratio_min',24.),('all_5040_packets_required',False),('network_forwards',1),('target_access',True),('U_truth_read',True)]:
  c=d.config();c[key]=value;bad(lambda c=c:d.validate(c))
 q=np.array((-6,-5,-4,-3,-2,-1,1,2,3,4,5,6));n=np.arange(256);a=np.array([-.45,-.15,.15,.45]);w=2*np.pi*q/20;h=(a[:,None]+np.exp(-1j*w))/(1+a[:,None]*np.exp(-1j*w));basis=np.exp(1j*n[:,None]*w)
 z=np.array([basis@(h[d.FIT.index(int(r))]*np.exp(1j*.01*int(t)*q*q)) for t,r in zip(tx,rx)])
 x=np.stack((z.real,z.imag),axis=1).astype(np.float32);saved=x.copy();r,o=measure(x,tx,rx,day);assert r['decision']=='PASS' and r['valid_packets']==5040 and len(r['estimate']['folds'])==18 and np.array_equal(x,saved)
 closure=np.array(o['closure_real'])+1j*np.array(o['closure_imag']);valid=np.array(o['valid']);quality={k:np.array(v) for k,v in o['quality'].items()}
 bad(lambda:measure(x[:-1],tx,rx,day));bad(lambda:analyze(closure[:,:7],valid,quality,tx,rx,day));bad(lambda:analyze(closure,valid.astype(int),quality,tx,rx,day))
 broken=copy.deepcopy(quality);broken['min_carrier_ratio'][0]=24.;bad(lambda:analyze(closure,valid,broken,tx,rx,day))
 rejected=valid.copy();rejected[0]=False;z0=analyze(closure,rejected,broken,tx,rx,day);assert z0['decision']=='FAIL' and z0['valid_packets']==5039 and 'folds' not in z0['estimate'] and sum(c['count'] for c in z0['cells'])==5040
 null=analyze(np.ones((5040,8),complex),valid,quality,tx,rx,day);assert null['decision']=='FAIL' and not null['estimate']['detected_RX']
 dispersed=closure.copy();dispersed[::2]*=-1;low=analyze(dispersed,valid,quality,tx,rx,day);assert low['decision']=='FAIL' and not low['estimate']['identifiable']
 return dict(status='PASS',protocol_negatives=negative,scientific_cases=4,synthetic_metadata_packets=5040,synthetic_IQ_generated=5040,synthetic_complex_points=1290240,source_wrapper_observer_calls=1,direct_summary_calls=3,real_IQ=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
