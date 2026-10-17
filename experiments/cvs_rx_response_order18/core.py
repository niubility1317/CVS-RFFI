"""Exact packetwise nonlinear readout of immutable source response parameters."""
import numpy as np
from . import design as d
F=np.fft.fft(np.eye(20),axis=0)/np.sqrt(20)
def mean_matrix(u):return np.einsum('ni,nj->ij',u,u.conj())/len(u)
def packet_prediction(u,h):
 u=np.asarray(u,dtype=np.complex128);h=np.asarray(h,dtype=np.complex128)
 if u.ndim!=2 or u.shape[1]!=20 or not len(u) or h.shape!=(20,) or not np.isfinite(u).all() or not np.isfinite(h).all() or np.any(abs(h)==0):raise ValueError('Finite complete factors and nonzero fixed response required')
 z=np.fft.fft(u,axis=1)/np.sqrt(20);v=z*h;trace=np.sum(abs(z)**2,axis=1);power=np.sum(abs(v)**2,axis=1)
 if np.any((trace>0)&(power<=0)) or not np.isfinite(power).all():raise ValueError('Finite nondegenerate response power required')
 scale=np.sqrt(np.divide(trace,power,out=np.zeros_like(trace),where=power>0));transformed=np.fft.ifft(v*scale[:,None],axis=1)*np.sqrt(20)
 return mean_matrix(transformed)
def matrix_prediction(P,h):
 G=F@P@F.conj().T;raw=h[:,None]*G*h[None,:].conj();power=np.trace(raw).real
 if power<=0:raise ValueError('Positive population power required')
 return F.conj().T@(raw*(np.trace(G).real/power))@F
def validate_inputs(L,V,H,u,meta):
 if any(a.shape!=(5,3,6,20,20) or not np.isfinite(a).all() or np.max(abs(a-a.conj().swapaxes(-1,-2)))>1e-12 for a in (L,V)):raise ValueError('Complete original180Hermitian matrices required')
 if H.shape!=(18,5,20) or not np.isfinite(H).all() or np.any(abs(H)==0) or not np.array_equal(H[:,0],np.ones((18,20))):raise ValueError('Exact18fixedresponses including reference identity required')
 if u.shape!=(27000,20) or not np.isfinite(u).all() or np.any(np.sum(abs(u)**2,axis=1)>1+1e-12):raise ValueError('Complete bounded27000V factors required')
 for key,allowed in [('rx',set(d.RX)),('day',{1,2,3}),('y',set(range(6)))]:
  a=np.asarray(meta[key])
  if a.shape!=(27000,) or a.dtype.kind not in 'iu' or set(a)!=allowed:raise ValueError('Original source V metadata required')
 for rx in d.RX:
  for day in (1,2,3):
   for tx in range(6):
    if np.sum((np.asarray(meta['rx'])==rx)&(np.asarray(meta['day'])==day)&(np.asarray(meta['y'])==tx))!=300:raise ValueError('All original300packet cells required')
def evaluate(L,V,H,u,meta,old):
 validate_inputs(L,V,H,u,meta);lc=L-L.mean(0,keepdims=True);vc=V-V.mean(0,keepdims=True);rows=[];predictions=[];B=[];A=[];C=[];M=[];E=[];differences=[];max_reference=0.
 for day in range(3):
  for tx in range(6):
   k=day*6+tx;mask=(np.asarray(meta['rx'])==1)&(np.asarray(meta['day'])==day+1)&(np.asarray(meta['y'])==tx);ref=u[mask];reference_error=float(np.max(abs(mean_matrix(ref)-V[0,day,tx])));max_reference=max(max_reference,reference_error)
   if reference_error>1e-12:raise ValueError('Saved factors do not reproduce original reference cell')
   exact=np.stack([packet_prediction(ref,h) for h in H[k]]);mean=np.stack([matrix_prediction(V[0,day,tx],h) for h in H[k]]);exact-=exact.mean(0,keepdims=True);mean-=mean.mean(0,keepdims=True);predictions.append(exact)
   actual=vc[:,day,tx];days=[j for j in range(3) if j!=day];txs=[j for j in range(6) if j!=tx];pa=lc[:,days][:,:,txs].mean((1,2));pc=lc[:,days,tx].mean(1)
   b,a,c,m,e=(np.sum(abs(v)**2,axis=(-2,-1)) for v in (actual,actual-pa,actual-pc,actual-mean,actual-exact));B.append(b);A.append(a);C.append(c);M.append(m);E.append(e);difference=float(np.sum(abs(exact-mean)**2));differences.append(difference)
   prior=old['folds'][k]
   if (prior['TX'],prior['day'])!=(tx,day+1) or max(abs(prior[key]-value) for key,value in [('baseline_SSE',b.sum()),('additive_SSE',a.sum()),('TXconditional_SSE',c.sum()),('response_SSE',m.sum())])>1e-9:raise ValueError('Old fixed controls/mean proxy changed')
   rows.append(dict(TX=tx,day=day+1,reference_packets=len(ref),baseline_SSE=float(b.sum()),additive_SSE=float(a.sum()),TXconditional_SSE=float(c.sum()),mean_proxy_SSE=float(m.sum()),packetwise_SSE=float(e.sum()),centered_predictions_difference_SSE=difference))
 B,A,C,M,E=(np.array(v) for v in (B,A,C,M,E));bs,aa,cc,mm,ee=(float(v.sum()) for v in (B,A,C,M,E));gain=1-ee/aa if aa>1e-20 else None;positive=int(sum(a.sum()>1e-20 and e.sum()<a.sum() for a,e in zip(A,E)));tol=1e-20
 checks=dict(nonzero_additive_error=aa>tol,additive_SSE_reduction=gain is not None and gain>=.20,positive_folds=positive>=15,each_RX_not_worse=bool(np.all(E.sum(0)<=A.sum(0)+tol)),not_worse_TXconditional=ee<=1.05*cc+tol,phase_products_nonzero=bool(old['conditions']['phase_products_nonzero']))
 return dict(decision='SOURCE_NORMALIZATION_ORDER_ATTRIBUTION_ONLY' if all(checks.values()) else 'CLOSED_NORMALIZATION_ORDER_EXPLANATION',conditions=checks,baseline_SSE=bs,additive_SSE=aa,TXconditional_SSE=cc,mean_proxy_SSE=mm,packetwise_SSE=ee,packetwise_vs_additive_gain=gain,packetwise_to_TXconditional_ratio=ee/cc if cc>tol else None,packetwise_vs_mean_proxy_gain=1-ee/mm if mm>tol else None,prediction_difference_SSE=float(sum(differences)),positive_folds=positive,folds=rows,per_RX={str(rx):dict(additive_SSE=float(A[:,i].sum()),TXconditional_SSE=float(C[:,i].sum()),mean_proxy_SSE=float(M[:,i].sum()),packetwise_SSE=float(E[:,i].sum())) for i,rx in enumerate(d.RX)},max_reference_matrix_error=max_reference,source_RX_holdout=None,response_refits=0,old_mean_proxy_decision_unchanged=True,IQ_operator_for_training=False,identity_preservation_proved=False,Ag_proved=False,joint_goal_complete=False,target_access=False),np.array(predictions)
