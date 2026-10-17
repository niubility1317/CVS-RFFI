"""Independent scalar/response recount using existing source cellmeans only."""
import argparse,json,time
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def run(path,source):
 start=time.perf_counter();out=Path(path);source=Path(source);done=read(out/'completion.json');r=read(out/'result.json');resolved=read(out/'resolved_config.json');c=resolved['config'];old=read(source/'completion.json');oldconf=read(source/'resolved_config.json')['config']
 if done['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or done['source_L']!=6300 or done['source_V']!=27000 or done['source_RX']!=[1,3,4,6,8] or done['source_RX_holdout'] is not None or any(done[k] for k in ('raw_IQ_reads','CP','NN','teacher_NN','updates','head_fit_updates','U_used','U_truth_read','target_access','IQ_views_generated','automatic_training')) or (c['reference_RX'],c['phase_anchor_bin'],c['epsilon'])!=(1,1,1e-12) or not resolved['source_handle_exact_match']:raise ValueError('Fixed source-only response scope differs')
 if old['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or (old['L'],old['V'],old['source_RX_holdout'])!=(6300,27000,None) or old['source_RX']!=[1,3,4,6,8] or oldconf['period']!=20 or oldconf['window']!=[80,160] or old['target_access']:raise ValueError('Original complete source handle differs')
 with np.load(source/'cell_matrices.npz',allow_pickle=False) as a:
  if set(a.files)!={'L_s','V'}:raise ValueError('Original complete source cells required')
  L,V=a['L_s'].copy(),a['V'].copy()
 with np.load(out/'responses.npz',allow_pickle=False) as a:
  if set(a.files)!={'responses'} or a['responses'].shape!=(18,5,20):raise ValueError('All90fixed responses required')
  saved=a['responses'].copy()
 if any(a.shape!=(5,3,6,20,20) or not np.isfinite(a).all() for a in (L,V)):raise ValueError('All180 finite source matrices required')
 # FFT(rows)/IFFT(columns) independently applies the unitary congruence.
 lf=np.fft.fft(np.fft.ifft(L,axis=-1),axis=-2);vf=np.fft.fft(np.fft.ifft(V,axis=-1),axis=-2);B=[];A=[];C=[];E=[];maxh=maxmetric=0.;identified=True;response_fits=0;fold=0
 for dd in range(3):
  for tt in range(6):
   pairs=[(d,t) for d in range(3) if d!=dd for t in range(6) if t!=tt];ref=np.array([lf[0,d,t] for d,t in pairs]);h=np.ones((5,20),complex)
   for rx in range(1,5):
    received=np.array([lf[rx,d,t] for d,t in pairs]);logs=[];ph=[]
    for k in range(20):
     logs.append(np.mean([np.log((max(received[i,k,k].real,0)+1e-12)/(max(ref[i,k,k].real,0)+1e-12)) for i in range(10)]));ph.append(np.vdot(ref[:,k,1],received[:,k,1])/10)
    amp=np.exp((np.array(logs)-np.mean(logs))/2);h[rx]=amp*np.exp(1j*np.angle(ph));h[rx]/=np.sqrt(np.vdot(h[rx],h[rx]).real/20);identified=identified and all(abs(x)>0 for x in ph);response_fits+=1
   maxh=max(maxh,float(np.max(abs(h-saved[fold]))));pred=[]
   for rx in range(5):
    v=h[rx,:,None]*vf[0,dd,tt]*h[rx,None,:].conj();v*=np.trace(vf[0,dd,tt]).real/np.trace(v).real;pred.append(np.fft.ifft(np.fft.fft(v,axis=-1),axis=-2))
   pred=np.array(pred);pred-=pred.mean(0);actual=V[:,dd,tt]-V[:,dd,tt].mean(0);pa=np.zeros_like(actual);pc=np.zeros_like(actual)
   for day,tx in pairs:pa+=(L[:,day,tx]-L[:,day,tx].mean(0))/10
   for day in range(3):
    if day!=dd:pc+=(L[:,day,tt]-L[:,day,tt].mean(0))/2
   values=[np.array([np.vdot(z,z).real for z in v]) for v in (actual,actual-pa,actual-pc,actual-pred)];b,a,cx,e=values;B.append(b);A.append(a);C.append(cx);E.append(e);f=r['folds'][fold]
   if (f['TX'],f['day'],f['reference_RX'],f['fit_L_cells'])!=(tt,dd+1,1,50):raise ValueError('Fixed statistical fold differs')
   for key,value in [('baseline_SSE',b.sum()),('additive_SSE',a.sum()),('TXconditional_SSE',cx.sum()),('response_SSE',e.sum())]:maxmetric=max(maxmetric,abs(f[key]-value))
   fold+=1
 B,A,C,E=(np.array(v) for v in (B,A,C,E));bs,aa,cc,ee=(float(v.sum()) for v in (B,A,C,E));gain=1-ee/aa if aa>1e-20 else None;positive=int(sum(a.sum()>1e-20 and e.sum()<a.sum() for a,e in zip(A,E)));checks=dict(nonzero_additive_error=aa>1e-20,additive_SSE_reduction=gain is not None and gain>=.20,positive_folds=positive>=15,each_RX_not_worse=bool(np.all(E.sum(0)<=A.sum(0)+1e-20)),not_worse_TXconditional=ee<=1.05*cc+1e-20,phase_products_nonzero=identified)
 for key,value in [('baseline_SSE',bs),('additive_SSE',aa),('TXconditional_SSE',cc),('response_SSE',ee),('response_vs_additive_gain',gain)]:
  if value is None:
   if r[key] is not None:raise ValueError('Null metric differs')
  else:maxmetric=max(maxmetric,abs(r[key]-value))
 if maxh>1e-7 or maxmetric>1e-9 or checks!=r['conditions'] or positive!=r['positive_folds'] or (r['decision']=='SOURCE_SHARED_RESPONSE_PREDICTION_ONLY')!=all(checks.values()):raise ValueError('Independent response/metric/source decision differs')
 for i,rx in enumerate((1,3,4,6,8)):
  for key,value in [('additive_SSE',A[:,i].sum()),('response_SSE',E[:,i].sum())]:
   if abs(r['per_RX'][str(rx)][key]-value)>1e-9:raise ValueError('RX partition differs')
 return dict(status='VERIFIED',run_id=r['run_id'],decision=r['decision'],conditions=checks,source_L=6300,source_V=27000,source_RX=[1,3,4,6,8],source_RX_holdout=None,source_cell_matrices=180,source_matrix_entries=72000,closed_form_response_fits=response_fits,response_entries_recounted=1800,registered_folds=18,predicted_V_cells=90,max_response_error=maxh,max_metric_error=maxmetric,raw_IQ_reads=0,CP=0,NN=0,updates=0,U_truth_reads=0,target_access=False,IQ_operator_for_training=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);a=p.parse_args();print(json.dumps(run(a.run,a.source),allow_nan=False))
