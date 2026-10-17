"""Closed-form response fit and held-source readout; no NN or IQ alteration."""
import numpy as np
from . import design as d
F=np.fft.fft(np.eye(20),axis=0)/np.sqrt(20)
def matrices(a):
 a=np.asarray(a,dtype=np.complex128)
 if a.shape!=(5,3,6,20,20) or not np.isfinite(a).all() or np.max(abs(a-a.conj().swapaxes(-1,-2)))>1e-12 or np.any(np.trace(a,axis1=-2,axis2=-1).real<=0) or np.min(np.linalg.eigvalsh(a))< -1e-12:raise ValueError('Complete positive-trace PSD source cell matrices required')
 return a
def fit(reference,received):
 reference,received=(np.asarray(a,dtype=np.complex128) for a in (reference,received))
 if reference.shape!=(10,20,20) or received.shape!=reference.shape or not np.isfinite(reference).all() or not np.isfinite(received).all():raise ValueError('Exactly10L cells per RX response required')
 a=np.maximum(np.diagonal(reference,axis1=-2,axis2=-1).real,0);b=np.maximum(np.diagonal(received,axis1=-2,axis2=-1).real,0);logs=np.log((b+1e-12)/(a+1e-12));logs-=logs.mean(1,keepdims=True);amplitude=np.exp(.5*logs.mean(0))
 products=received[:,:,1]*reference[:,:,1].conj();phase=np.mean(products,axis=0);h=amplitude*np.exp(1j*np.angle(phase));h/=np.sqrt(np.mean(abs(h)**2))
 if not np.isfinite(h).all() or np.any(abs(h)==0):raise ValueError('Finite nonzero response required')
 evidence=np.divide(abs(phase),np.mean(abs(products),axis=0),out=np.zeros(20),where=np.mean(abs(products),axis=0)>0)
 return h,dict(all_phase_products_nonzero=bool(np.all(abs(phase)>0)),phase_evidence_min=float(evidence.min()),amplitude_condition_number=float(abs(h).max()/abs(h).min()),amplitude_min=float(abs(h).min()),amplitude_max=float(abs(h).max()))
def apply(h,G):
 raw=h[:,None]*G*h[None,:].conj();den=float(np.trace(raw).real);trace=float(np.trace(G).real)
 if den<=0 or trace<=0 or not np.isfinite(den):raise ValueError('Positive finite response power required')
 return raw*(trace/den)
def evaluate(L,V):
 L,V=matrices(L),matrices(V);lf=F@L@F.conj().T;vf=F@V@F.conj().T;lc=L-L.mean(0,keepdims=True);vc=V-V.mean(0,keepdims=True);base=[];additive=[];conditional=[];response=[];folds=[];coefficients=[];evidence=[]
 for day in range(3):
  for tx in range(6):
   days=[j for j in range(3) if j!=day];txs=[j for j in range(6) if j!=tx];fitcells=lf[:,days][:,:,txs].reshape(5,10,20,20);hs=np.ones((5,20),complex);quality=[dict(all_phase_products_nonzero=True,phase_evidence_min=1.,amplitude_condition_number=1.,amplitude_min=1.,amplitude_max=1.)]
   for r in range(1,5):hs[r],q=fit(fitcells[0],fitcells[r]);quality.append(q)
   predicted=np.stack([F.conj().T@apply(h,vf[0,day,tx])@F for h in hs]);predicted-=predicted.mean(0,keepdims=True);actual=vc[:,day,tx];pa=lc[:,days][:,:,txs].mean((1,2));pc=lc[:,days,tx].mean(1)
   b=np.sum(abs(actual)**2,axis=(-2,-1));a=np.sum(abs(actual-pa)**2,axis=(-2,-1));c=np.sum(abs(actual-pc)**2,axis=(-2,-1));e=np.sum(abs(actual-predicted)**2,axis=(-2,-1));base.append(b);additive.append(a);conditional.append(c);response.append(e);coefficients.append(hs);evidence.append(quality)
   folds.append(dict(TX=tx,day=day+1,fit_L_cells=50,reference_RX=1,baseline_SSE=float(b.sum()),additive_SSE=float(a.sum()),TXconditional_SSE=float(c.sum()),response_SSE=float(e.sum()),response_vs_additive_gain=1-float(e.sum()/a.sum()) if a.sum()>1e-20 else None,fit_quality=quality))
 B,A,C,E=(np.array(v) for v in (base,additive,conditional,response));bs,aa,cc,ee=(float(v.sum()) for v in (B,A,C,E));gain=1-ee/aa if aa>1e-20 else None;positive=int(sum(a.sum()>1e-20 and e.sum()<a.sum() for a,e in zip(A,E)));tol=1e-20
 checks=dict(nonzero_additive_error=aa>1e-20,additive_SSE_reduction=gain is not None and gain>=.20,positive_folds=positive>=15,each_RX_not_worse=bool(np.all(E.sum(0)<=A.sum(0)+tol)),not_worse_TXconditional=ee<=1.05*cc+tol,phase_products_nonzero=all(q['all_phase_products_nonzero'] for fold in evidence for q in fold))
 result=dict(decision='SOURCE_SHARED_RESPONSE_PREDICTION_ONLY' if all(checks.values()) else 'CLOSED_FIXED_SHARED_RESPONSE',conditions=checks,baseline_SSE=bs,additive_SSE=aa,TXconditional_SSE=cc,response_SSE=ee,response_vs_additive_gain=gain,response_vs_zero_gain=1-ee/bs if bs>1e-20 else None,response_to_TXconditional_ratio=ee/cc if cc>1e-20 else None,positive_folds=positive,folds=folds,per_RX={str(r):dict(additive_SSE=float(A[:,i].sum()),response_SSE=float(E[:,i].sum()),response_vs_additive_gain=1-float(E[:,i].sum()/A[:,i].sum()) if A[:,i].sum()>1e-20 else None) for i,r in enumerate(d.RX)},source_RX_holdout=None,reference_RX=1,L_only_fit=True,V_fit=False,raw_IQ_reads=0,IQ_operator_for_training=False,Ag_proved=False,joint_goal_complete=False,target_access=False)
 return result,np.array(coefficients)
