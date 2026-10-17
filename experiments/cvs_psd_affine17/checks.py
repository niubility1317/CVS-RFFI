"""Falsifiable synthetic PSD relations and protocol negatives;no real aggregate read."""
from copy import deepcopy
import json,math
import numpy as np
from .core import fit,simplex,analyze,probabilities
from .data import metadata
from . import design as d

def run():
 rng=np.random.default_rng(202610178101);n=0
 def reject(fn):
  nonlocal n
  try:fn()
  except (ValueError,TypeError,KeyError):n+=1
  else:raise AssertionError('Counterexample accepted')
 s=rng.dirichlet(np.ones(256),size=(6,3));noise=rng.dirichlet(np.ones(256));p=np.empty((2,6,3,4,256))
 for r,beta in enumerate((0.,.1,.2,.3)):p[:,:,:,r]=256*((1-beta)*s+beta*noise)
 positive,folds=analyze(p);assert positive['qualified'] and len(folds)==432
 assert max(row['model_SSE'] for row in folds if row['source_RX']<row['destination_RX'])<1e-16
 null=np.repeat(p[:,:,:,0:1],4,axis=3);z,_=analyze(null);assert not z['qualified']
 swapped=p.copy()
 for r in range(4):swapped[:,:,:,r]=np.roll(p[:,:,:,r],r,axis=1)
 z,_=analyze(swapped);assert not z['qualified']
 x=rng.dirichlet(np.ones(256),size=10);y=.7*x+.3*noise;f=fit(x,y);assert abs(f['a']-.7)<1e-6 and np.max(np.abs(f['b']-.3*noise))<1e-7
 z=simplex(np.linspace(-1,1,256),.2);assert np.min(z)>=0 and abs(z.sum()-.2)<1e-12
 q=np.linspace(-1,1,256);active=z>0;theta=np.mean(q[active]-z[active]);assert np.max(np.abs(q[active]-z[active]-theta))<1e-12 and np.max(q[~active]-theta)<1e-12
 for value in (-.1,1.1,float('nan')):reject(lambda:simplex(np.zeros(256),value))
 reject(lambda:simplex(np.zeros(255),.2));reject(lambda:fit(x[:9],y[:9]));reject(lambda:fit(x*2,y))
 for bad in (p[:1],p[:,:,:,:3],p[:,:,:,:,:255]):reject(lambda:analyze(bad))
 for value in (-1.,float('nan'),float('inf')):
  bad=p.copy();bad.flat[0]=value;reject(lambda:analyze(bad))
 config=dict(run_id=d.config()['source_origin_run'],source_RX=[3,4,6,8],source_TX=list(range(6)),source_days=[1,2,3],L_packets=5040,packets_per_cell=70,source_only=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,new_models=0,observation=dict(windows=['rectangular','hann'],FFT_bins=256));summary=dict(physical_packet_counts={'L':5040,'U':0,'V':0,'held_RX1':0,'target':0},source_cells=72);metadata(config,summary)
 for k,value in [('source_RX',[1,3,4,6]),('source_TX',list(range(7))),('source_days',[1,2]),('L_packets',6300),('packets_per_cell',69),('U_used',True),('V_used',True),('held_RX1_used',True),('target_access',True),('source_only',False),('model_forwards',1)]:
  bad=deepcopy(config);bad[k]=value;reject(lambda:metadata(bad,summary))
 bad=deepcopy(config);bad['observation']['windows']=['hann','rectangular'];reject(lambda:metadata(bad,summary))
 bad=deepcopy(summary);bad['physical_packet_counts']['U']=1;reject(lambda:metadata(config,bad))
 return dict(status='PASS',protocol_and_input_counterexamples=n,synthetic_full_PSD_cases=3,synthetic_PSD_fits=1296,additional_closed_form_fit=1,simplex_KKT_verified=True,source_aggregate_reads=0,real_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
