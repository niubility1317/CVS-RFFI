"""Independent full source-observation recount;no raw IQ,checkpoint or inference."""
import json,math
from pathlib import Path
import numpy as np
import sys
P=Path(sys.argv[1])
r=json.loads((P/'result.json').read_text());done=json.loads((P/'completion.json').read_text())
if done['status']!='ANALYZED_SOURCE_STATISTICS_ONLY' or r['source_IQ_unique']!=6300 or r['source_cells']!=90 or r['CP'] or r['NN'] or r['training_updates'] or r['target_access']:raise ValueError('Original completed source-only row required')
with np.load(P/'source_observations.npz',allow_pickle=False) as a:
 ids=a['ids'];y=a['labels'];rx=a['RX'];day=a['day'];profiles=a['log_profiles'];coef=a['coefficients'];power=a['packet_power']
 if ids.shape!=(6300,) or len(set(ids.tolist()))!=6300 or profiles.shape!=(6300,16) or coef.shape!=(6300,2) or not np.isfinite(profiles).all() or not np.isfinite(coef).all() or not np.isfinite(power).all() or (power<=0).any():raise ValueError('Complete finite observations required')
 if not np.array_equal(np.unique(y),np.arange(6)) or not np.array_equal(np.unique(rx),[1,3,4,6,8]) or not np.array_equal(np.unique(day),[1,2,3]):raise ValueError('Full originalsource groups required')
 D=[[math.sqrt(2/16)*math.cos(math.pi*(n+.5)*k/16) for k in (1,2)] for n in range(16)]
 direct=np.asarray([[math.fsum(float(profiles[i,n])*D[n][k] for n in range(16)) for k in range(2)] for i in range(6300)])
 coeff_error=float(abs(direct-coef).max());center_error=float(abs(profiles.sum(1)).max())
 if coeff_error>1e-12 or center_error>1e-12:raise ValueError('Fixed centering/DCT observation mismatch')
 cells={};max_cell_error=0.
 for tx in range(6):
  for z in range(1,4):
   for rr in (1,3,4,6,8):
    ix=np.flatnonzero((y==tx)&(day==z)&(rx==rr))
    if len(ix)!=70:raise ValueError('All90 cells must have70,no sample filtering')
    mean=[math.fsum(float(coef[i,k]) for i in ix)/70 for k in range(2)];cells[(tx,z,rr)]=mean
    saved=next(v for v in r['cells'] if (v['TX'],v['day'],v['RX'])==(tx,z,rr));var=[math.fsum((float(coef[i,k])-mean[k])**2 for i in ix)/70 for k in range(2)]
    error=max(abs(mean[k]-saved['mean_coeff'][k]) for k in range(2));error=max(error,max(abs(var[k]-saved['variance_coeff'][k]) for k in range(2)));max_cell_error=max(max_cell_error,error)
    if error>1e-12:raise ValueError('Cell sufficient statistic mismatch')
 totals=dict(prediction_SSE=0.,zero_SSE=0.,wrong_RX_SSE=0.);per={rr:[0.,0.] for rr in (1,3,4,6,8)};better=0;max_fold_error=0.;rxs=[1,3,4,6,8];folds=[]
 for tx in range(6):
  for z in range(1,4):
   raw=[[math.fsum(cells[(t,d,rr)][k] for t in range(6) if t!=tx for d in range(1,4) if d!=z)/10 for k in range(2)] for rr in rxs]
   pred=[[raw[i][k]-math.fsum(v[k] for v in raw)/5 for k in range(2)] for i in range(5)]
   obs=[[cells[(tx,z,rr)][k]-math.fsum(cells[(tx,z,u)][k] for u in rxs)/5 for k in range(2)] for rr in rxs]
   values=dict(prediction_SSE=0.,zero_SSE=0.,wrong_RX_SSE=0.)
   for i,rr in enumerate(rxs):
    err=math.fsum((obs[i][k]-pred[i][k])**2 for k in range(2));zero=math.fsum(obs[i][k]**2 for k in range(2));wrong=math.fsum((obs[i][k]-pred[(i-1)%5][k])**2 for k in range(2))
    values['prediction_SSE']+=err;values['zero_SSE']+=zero;values['wrong_RX_SSE']+=wrong;per[rr][0]+=err;per[rr][1]+=zero
   saved=r['folds'][tx*3+z-1]
   if saved['fit_TX']!=[t for t in range(6) if t!=tx] or saved['fit_days']!=[d for d in range(1,4) if d!=z] or saved['fit_RX']!=rxs:raise ValueError('Fold permission mismatch')
   for k,v in values.items():totals[k]+=v;max_fold_error=max(max_fold_error,abs(v-saved[k]))
   for i in range(5):
    for k in range(2):max_fold_error=max(max_fold_error,abs(pred[i][k]-saved['predicted'][i][k]),abs(obs[i][k]-saved['held_centered'][i][k]))
   if max_fold_error>1e-11:raise ValueError('Independent fold mismatch')
   better+=int(values['prediction_SSE']<values['zero_SSE']);folds.append(dict(TX=tx,day=z,**values))
 rel=totals['prediction_SSE']/totals['zero_SSE'];per_rx={str(rr):v[0]/v[1] for rr,v in per.items()}
 cond=dict(relative_MSE_at_most_0_9=rel<=.9,at_least15_of18_folds_better=better>=15,better_than_wrong_RX=totals['prediction_SSE']<totals['wrong_RX_SSE'],each_RX_better_than_zero=all(v<1 for v in per_rx.values()))
 if cond!=r['conditions'] or abs(rel-r['relative_MSE'])>1e-12 or better!=r['better_folds'] or any(abs(per_rx[k]-r['per_RX_relative_MSE'][k])>1e-12 for k in per_rx):raise ValueError('Decision/aggregate mismatch')
 out=dict(status='VERIFIED',source_observation_recounts=6300,DCT_coefficients_recounted=12600,cell_recounts=90,fold_recounts=18,RX_recounts=5,max_DCT_error=coeff_error,max_centering_error=center_error,max_cell_error=max_cell_error,max_fold_error=max_fold_error,relative_MSE=rel,better_folds=better,per_RX_relative_MSE=per_rx,conditions=cond,decision=r['decision'],IQ_reads=0,CP=0,NN=0,training_updates=0,target_access=False)
 with (P/'independent_recount_r01.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out,allow_nan=False))
