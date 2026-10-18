"""Independent full saved-quantile/cell/fit recount;never imports core or opens IQ/CP."""
import argparse,json,time
from pathlib import Path
import numpy as np
RX=(1,3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def run(base):
 start=time.perf_counter();B=Path(base);r=read(B/'result.json');meta=read(B/'source_ids.json');saved={};obs={};means={};counts={};maxerror=0.;floor=0.;zero=0
 with np.load(B/'observations.npz',allow_pickle=False) as a:
  for role,n in [('L_s',6300),('V',27000)]:
   for key in ('shape','amplitude_floor_fraction','zero_RMS'):obs[role+'_'+key]=a[role+'_'+key].copy()
   q=obs[role+'_shape'];m=meta[role]
   if q.shape!=(n,6) or not np.isfinite(q).all() or len(m['ids'])!=n or len(set(m['ids']))!=n or any(len(m[k])!=n for k in ('y','rx','day')) or np.any(np.diff(q,axis=1)<0) or np.any(q[:,2]>1e-14) or np.any(q[:,3]<-1e-14):raise ValueError('Complete valid source quantiles/metadata required')
   out=np.empty((6,5,3,6));allcounts=[]
   for t in range(6):
    for j,rx in enumerate(RX):
     for d in range(3):
      idx=[i for i in range(n) if (m['y'][i],m['rx'][i],m['day'][i])==(t,rx,d+1)]
      if len(idx)!=(70 if role=='L_s' else 300):raise ValueError('All90source cells required')
      out[t,j,d]=np.mean(q[idx],axis=0);floor=max(floor,float(np.mean(obs[role+'_amplitude_floor_fraction'][idx])));allcounts.append(len(idx))
   zero+=int(np.count_nonzero(obs[role+'_zero_RMS']));means[role]=out;counts[role]=dict(packets=n,cells=90,quantile_values=n*6,all_cell_counts_equal=len(set(allcounts))==1)
 if set(meta['L_s']['ids'])&set(meta['V']['ids']):raise ValueError('Source physical roles overlap')
 with np.load(B/'cell_quantiles.npz',allow_pickle=False) as a:
  for role in means:maxerror=max(maxerror,float(np.max(np.abs(a[role]-means[role]))))
 with np.load(B/'fitted_readout.npz',allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
 L=means['L_s'];V=means['V'];pred={key:V.copy() for key in ('shared','identity','additive','TXconditional')};pa=np.ones((6,5,3));pc=pa.copy();offset=np.zeros_like(L);fold=[]
 for t in range(6):
  for d in range(3):
   td=[(u,z) for u in range(6) if u!=t for z in range(3) if z!=d];same=[z for z in range(3) if z!=d]
   for j in range(1,5):
    x=np.array([L[u,0,z] for u,z in td]);y=np.array([L[u,j,z] for u,z in td]);xx=L[t,0,same];yy=L[t,j,same];alpha=float(x.ravel()@y.ravel())/float(x.ravel()@x.ravel());beta=float(xx.ravel()@yy.ravel())/float(xx.ravel()@xx.ravel());shift=np.mean(y-x,axis=0);pa[t,j,d]=alpha;pc[t,j,d]=beta;offset[t,j,d]=shift;ref=V[t,0,d];pred['shared'][t,j,d]=alpha*ref;pred['identity'][t,j,d]=ref;pred['additive'][t,j,d]=ref+shift;pred['TXconditional'][t,j,d]=beta*ref
   fold.append(dict(TX=t,day=d+1,SSE={k:float(np.sum((v[t,1:,d]-V[t,1:,d])**2)) for k,v in pred.items()}))
 for key,a in [('shared_slope',pa),('TXconditional_slope',pc),('additive_offset',offset),*[(('prediction_'+k),v) for k,v in pred.items()]]:maxerror=max(maxerror,float(np.max(np.abs(a-saved[key]))))
 SSE={k:float(np.sum((v[:,1:]-V[:,1:])**2)) for k,v in pred.items()};per={str(RX[j]):{k:float(np.sum((v[:,j]-V[:,j])**2)) for k,v in pred.items()} for j in range(1,5)};tol=1e-20;positive=sum(x['SSE']['shared']<x['SSE']['identity'] for x in fold);p=pa[:,1:];conditions=dict(nonzero_identity_error=np.sqrt(SSE['identity']/432)>=.01,shared_vs_identity=SSE['shared']<=.8*SSE['identity']+tol,shared_vs_additive=SSE['shared']<=.9*SSE['additive']+tol,positive_folds=positive>=15,each_nonreference_RX=all(v['shared']<=min(v['identity'],v['additive'])+tol for v in per.values()),not_worse_TXconditional=SSE['shared']<=1.05*SSE['TXconditional']+tol,common_slope_mild=bool(np.all((p>=.9)&(p<=1.1))),valid_log_shape=floor<=.01 and zero==0)
 maxmetric=max(abs(SSE[k]-r['SSE'][k]) for k in SSE)
 for x,y in zip(fold,r['folds']):
  if (x['TX'],x['day'])!=(y['TX'],y['day']):raise ValueError('Fold indexing differs')
  maxmetric=max(maxmetric,max(abs(x['SSE'][k]-y['SSE'][k]) for k in SSE))
 if maxerror>1e-10 or maxmetric>1e-10 or conditions!=r['conditions'] or all(conditions.values())!=r['source_pass'] or r['automatic_training'] or r['IQ_operator_implemented'] or r['label_preservation_proved'] or r['Ag_proved'] or r['joint_goal_complete']:raise ValueError('Independent scientific result mismatch')
 return dict(status='VERIFIED',decision=r['decision'],source_pass=r['source_pass'],conditions=conditions,SSE=SSE,per_RX_SSE=per,positive_folds=positive,baseline_RMS=float(np.sqrt(SSE['identity']/432)),slopes=dict(minimum=float(p.min()),maximum=float(p.max())),source_counts=counts,packet_quantile_values=199800,cell_quantile_values=1080,scalar_fits=144,additive_vectors=72,prediction_coordinates=4*432,reference_RX_excluded_from_positive_domain_count=True,max_array_error=maxerror,max_metric_error=maxmetric,quality=dict(max_floor_fraction_each_cell=floor,zero_RMS_packets=zero),IQ_reads=0,CP=0,NN=0,updates=0,U_used=False,target_access=False,label_preservation_proved=False,Ag_proved=False,joint_goal_complete=False,scope='Full savedquantile/cell/fit/prediction recount;not independent rawIQ quantile replay',wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();print(json.dumps(run(a.base),allow_nan=False))
