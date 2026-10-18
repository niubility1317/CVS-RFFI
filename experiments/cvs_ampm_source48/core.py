import numpy as np
RX=(1,3,4,6,8)
U=np.asarray([-.5,0,.5],np.float64)
def iq_complex(x):
 x=np.asarray(x,np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('finite IQ256 required')
 z=x[:,0]+1j*x[:,1];p=np.mean(abs(z)**2,axis=1)
 if np.any(p<=1e-12) or np.max(abs(p-1))>1e-3:raise ValueError('existing RMS-normalized source IQ only')
 return z

def bound_from_L(z,rx):
 p=abs(z)**2;centered=abs(p/p.mean(1,keepdims=True)-1)
 q=np.asarray([np.quantile(centered[np.asarray(rx)==r],.99) for r in RX])
 if not np.isfinite(q).all() or (q<=0).any():raise ValueError('nondegenerate L power required')
 return min(.02,.05/float(max(q))),q

def curves(z,beta):
 # A constant packet phase and a linear phase ramp cancel in the magnitude.
 # T_b(x)=x*exp(i*b*(p-1)) translates this curve exactly: f_T(u)=f_x(u+b).
 z=np.asarray(z,np.complex128);beta=np.asarray(beta,np.float64)
 if z.ndim!=2 or z.shape[1]!=256 or beta.ndim!=1 or not np.isfinite(z).all() or not np.isfinite(beta).all():raise ValueError('invalid curve input')
 power=abs(z)**2;power/=power.mean(1,keepdims=True)
 dp=np.diff(power,axis=1);v=z[:,1:]*z[:,:-1].conj();den=abs(v).sum(1)
 if (den<=1e-12).any():raise ValueError('no usable adjacent observations')
 out=np.empty((len(z),len(beta),3),np.float64)
 for bi,b in enumerate(beta):
  for ui,u in enumerate(U):out[:,bi,ui]=abs((v*np.exp(1j*(u+b)*dp)).sum(1))/den
 if not np.isfinite(out).all():raise ValueError('nonfinite curves')
 return out

def cell_means(features,y,rx,day,expected_count):
 features=np.asarray(features,np.float64);out=np.empty((6,5,3)+features.shape[1:],np.float64)
 if features.ndim!=3 or features.shape[1:]!=(21,3) or not np.isfinite(features).all():raise ValueError('curve dimensions')
 for t in range(6):
  for ri,r in enumerate(RX):
   for di,d in enumerate((1,2,3)):
    mask=(y==t)&(rx==r)&(day==d)
    if int(mask.sum())!=expected_count:raise ValueError('complete legal90 cells required')
    out[t,ri,di]=features[mask].mean(0)
 return out

def evaluate(L,V,beta):
 if L.shape!=(6,5,3,21,3) or V.shape!=L.shape or not np.isfinite(L).all() or not np.isfinite(V).all() or len(beta)!=21 or beta[10]!=0:raise ValueError('fixed source cells/grid')
 rawL=L[:,:,:,10,:];rawV=V[:,:,:,10,:];truth=rawV-rawV.mean(1,keepdims=True)
 fit_indices=np.full((6,3,5),10,np.int64);pred=np.zeros_like(rawV);add=np.zeros_like(rawV);txcond=np.zeros_like(rawV)
 for t in range(6):
  for d in range(3):
   tt=[j for j in range(6) if j!=t];dd=[j for j in range(3) if j!=d]
   # Ten matching TX/day cells; independently sampled packets, no pairing claim.
   for r in range(1,5):
    errors=np.asarray([sum(float(np.square(L[j,0,k,b,:]-rawL[j,r,k,:]).sum()) for j in tt for k in dd) for b in range(21)])
    # Stable minimum: smaller absolute beta first only on exact equal SSE.
    b=min(range(21),key=lambda j:(errors[j],abs(float(beta[j])),float(beta[j])))
    fit_indices[t,d,r]=b;pred[t,r,d]=V[t,0,d,b]
   pred[t,0,d]=rawV[t,0,d]
   a=np.stack([rawL[j,:,k,:]-rawL[j,:,k,:].mean(0,keepdims=True) for j in tt for k in dd]).mean(0)
   c=np.stack([rawL[t,:,k,:]-rawL[t,:,k,:].mean(0,keepdims=True) for k in dd]).mean(0)
   add[t,:,d]=a;txcond[t,:,d]=c
 pred-=pred.mean(1,keepdims=True)
 arrays=dict(shared=(pred-truth)**2,identity=truth**2,additive=(add-truth)**2,TXconditional=(txcond-truth)**2)
 sse={k:float(v.sum()) for k,v in arrays.items()};folds={k:v.sum(axis=(1,3)) for k,v in arrays.items()};byrx={k:v.sum(axis=(0,2,3)) for k,v in arrays.items()}
 boundary=int(np.isin(fit_indices[:,:,1:],[0,20]).sum())
 checks=dict(nonzero_RX_difference=sse['identity']>1e-12,shared_beats_identity_20percent=sse['shared']<=.8*sse['identity'],shared_beats_additive_10percent=sse['shared']<=.9*sse['additive'],improves_15_of_18_folds=int((folds['shared']<folds['additive']).sum())>=15,all_RX_no_worse_additive=bool((byrx['shared']<=byrx['additive']+1e-20).all()),within_TXconditional_5percent=sse['shared']<=1.05*sse['TXconditional'],bounded_interior_fits=boundary==0)
 passed=all(checks.values())
 result=dict(source_qualified=passed,decision='SOURCE_PASS_NEXT_IDENTITY_TEST_ONLY' if passed else 'SOURCE_FAIL_FIXED_AMPM_TRANSLATION_CLOSED',criteria=checks,SSE=sse,folds_improving_additive=int((folds['shared']<folds['additive']).sum()),boundary_fits=boundary,fit_beta_min=float(beta[fit_indices[:,:,1:]].min()),fit_beta_max=float(beta[fit_indices[:,:,1:]].max()),per_RX={str(r):{k:float(v[i]) for k,v in byrx.items()} for i,r in enumerate(RX)},fold_SSE={k:v.tolist() for k,v in folds.items()})
 return result,dict(fit_indices=fit_indices,predicted=pred,truth=truth,additive=add,TXconditional=txcond)
