"""Independent outer-product recount; never refits or alters the saved response."""
import argparse,json,time
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def to_frequency(P):return np.fft.fft(np.fft.ifft(P,axis=-1),axis=-2)
def from_frequency(G):return np.fft.ifft(np.fft.fft(G,axis=-1),axis=-2)
def run(out,source,response):
 start=time.perf_counter();out,source,response=map(Path,(out,source,response));done=read(out/'completion.json');resolved=read(out/'resolved_config.json');r=read(out/'result.json');old=read(response/'result.json');c=resolved['config'];rxs=(1,3,4,6,8)
 if done['status']!='ARTIFACTS_COMPLETE_SOURCE_ONLY' or (done['source_L'],done['source_V'],done['source_RX_holdout'])!=(6300,27000,None) or done['source_RX']!=list(rxs) or any(done[k] for k in ('response_refits','raw_IQ_reads','CP','NN','teacher_NN','updates','head_fit_updates','U_used','U_truth_read','target_access','IQ_views_generated','automatic_training')) or not resolved['input_handles_exact_match'] or (c['reference_RX'],c['phase_anchor_bin'],c['epsilon'],c['window'],c['period'])!=(1,1,1e-12,[80,160],20):raise ValueError('Fixed no-refit source scope differs')
 with np.load(source/'cell_matrices.npz',allow_pickle=False) as a:L,V=a['L_s'].copy(),a['V'].copy()
 with np.load(source/'observations.npz',allow_pickle=False) as a:u=a['V_factor_real']+1j*a['V_factor_imag']
 with np.load(response/'responses.npz',allow_pickle=False) as a:H=a['responses'].copy()
 with np.load(out/'predicted_matrices.npz',allow_pickle=False) as a:
  if set(a.files)!={'centered_predictions'}:raise ValueError('Only complete saved predictions required')
  saved=a['centered_predictions'].copy()
 meta=read(source/'source_ids.json')['V'];rx,day,tx=(np.array(meta[k]) for k in ('rx','day','y'))
 if any(a.shape!=(5,3,6,20,20) or not np.isfinite(a).all() for a in (L,V)) or H.shape!=(18,5,20) or not np.isfinite(H).all() or np.any(abs(H)==0) or not np.array_equal(H[:,0],np.ones((18,20))) or u.shape!=(27000,20) or not np.isfinite(u).all() or saved.shape!=(18,5,20,20) or not np.isfinite(saved).all() or any(a.shape!=(27000,) or a.dtype.kind not in 'iu' for a in (rx,day,tx)):raise ValueError('Complete fixed saved source inputs required')
 B=[];A=[];C=[];M=[];E=[];max_prediction=max_metric=max_reference=0.;difference=0.;fold=0;transforms=0
 for dd in range(3):
  for tt in range(6):
   ref=u[(rx==1)&(day==dd+1)&(tx==tt)]
   if ref.shape!=(300,20):raise ValueError('Every original300reference packets required')
   P=ref[:,:,None]*ref[:,None,:].conj();G=to_frequency(P);traces=np.trace(P,axis1=-2,axis2=-1).real;max_reference=max(max_reference,float(np.max(abs(P.mean(0)-V[0,dd,tt]))));exact=[];mean=[]
   for h in H[fold]:
    raw=G*h[None,:,None]*h[None,None,:].conj();powers=np.trace(raw,axis1=-2,axis2=-1).real
    if np.any((traces>0)&(powers<=0)):raise ValueError('Positive packet response power required')
    scale=np.divide(traces,powers,out=np.zeros(300),where=powers>0);exact.append(from_frequency(np.mean(raw*scale[:,None,None],axis=0)));transforms+=300
    gm=to_frequency(V[0,dd,tt]);rawmean=np.diag(h)@gm@np.diag(h).conj().T;mean.append(from_frequency(rawmean*np.trace(gm).real/np.trace(rawmean).real))
   exact=np.array(exact);mean=np.array(mean);exact-=sum(exact)/5;mean-=sum(mean)/5;max_prediction=max(max_prediction,float(np.max(abs(exact-saved[fold]))))
   if max_prediction>1e-10:raise ValueError('Independent packet prediction differs')
   actual=V[:,dd,tt]-sum(V[:,dd,tt])/5;pa=np.zeros((5,20,20),complex);pc=pa.copy()
   for od in range(3):
    if od==dd:continue
    pc+=(L[:,od,tt]-sum(L[:,od,tt])/5)/2
    for ot in range(6):
     if ot!=tt:pa+=(L[:,od,ot]-sum(L[:,od,ot])/5)/10
   b,a,cc,m,e=(np.array([np.vdot(v,v).real for v in z]) for z in (actual,actual-pa,actual-pc,actual-mean,actual-exact));B.append(b);A.append(a);C.append(cc);M.append(m);E.append(e);diff=float(np.vdot(exact-mean,exact-mean).real);difference+=diff;f=r['folds'][fold];prior=old['folds'][fold]
   if (f['TX'],f['day'],f['reference_packets'])!=(tt,dd+1,300) or (prior['TX'],prior['day'])!=(tt,dd+1):raise ValueError('Fixed18fold identity differs')
   for key,value in [('baseline_SSE',b.sum()),('additive_SSE',a.sum()),('TXconditional_SSE',cc.sum()),('mean_proxy_SSE',m.sum()),('packetwise_SSE',e.sum()),('centered_predictions_difference_SSE',diff)]:max_metric=max(max_metric,abs(f[key]-value))
   for key,value in [('baseline_SSE',b.sum()),('additive_SSE',a.sum()),('TXconditional_SSE',cc.sum()),('response_SSE',m.sum())]:max_metric=max(max_metric,abs(prior[key]-value))
   fold+=1
 B,A,C,M,E=(np.array(a) for a in (B,A,C,M,E));bs,aa,cc,mm,ee=(float(a.sum()) for a in (B,A,C,M,E));gain=1-ee/aa if aa>1e-20 else None;positive=int(sum(a.sum()>1e-20 and e.sum()<a.sum() for a,e in zip(A,E)));checks=dict(nonzero_additive_error=aa>1e-20,additive_SSE_reduction=gain is not None and gain>=.20,positive_folds=positive>=15,each_RX_not_worse=bool(np.all(E.sum(0)<=A.sum(0)+1e-20)),not_worse_TXconditional=ee<=1.05*cc+1e-20,phase_products_nonzero=bool(old['conditions']['phase_products_nonzero']))
 for key,value in [('baseline_SSE',bs),('additive_SSE',aa),('TXconditional_SSE',cc),('mean_proxy_SSE',mm),('packetwise_SSE',ee),('prediction_difference_SSE',difference)]:max_metric=max(max_metric,abs(r[key]-value))
 for i,rxnum in enumerate(rxs):
  for key,value in [('additive_SSE',A[:,i].sum()),('TXconditional_SSE',C[:,i].sum()),('mean_proxy_SSE',M[:,i].sum()),('packetwise_SSE',E[:,i].sum())]:max_metric=max(max_metric,abs(r['per_RX'][str(rxnum)][key]-value))
 if max_reference>1e-12 or max_metric>1e-9 or checks!=r['conditions'] or positive!=r['positive_folds'] or (r['decision']=='SOURCE_NORMALIZATION_ORDER_ATTRIBUTION_ONLY')!=all(checks.values()) or done['decision']!=r['decision']:raise ValueError('Independent full source metrics/decision differ')
 return dict(status='VERIFIED',run_id=r['run_id'],decision=r['decision'],conditions=checks,source_L=6300,source_V=27000,source_RX=list(rxs),source_RX_holdout=None,source_complex_matrix_entries=72000,V_complex_factor_entries=540000,reference_V_packets=5400,complex_fixed_response_entries=1800,centered_prediction_entries_recounted=36000,packetwise_outer_response_matrices=transforms,folds=18,max_reference_matrix_error=max_reference,max_prediction_error=max_prediction,max_metric_error=max_metric,response_refits=0,raw_IQ_reads=0,CP=0,NN=0,updates=0,target_access=False,old_mean_proxy_decision_unchanged=True,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',required=True);p.add_argument('--response',required=True);a=p.parse_args();print(json.dumps(run(a.run,a.source,a.response),allow_nan=False))
