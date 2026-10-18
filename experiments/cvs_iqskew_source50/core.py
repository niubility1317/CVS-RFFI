import numpy as np
RX=(1,3,4,6,8)
OMEGA=2*np.pi*np.arange(1,128,dtype=np.float64)/256
DELAYS=np.linspace(-.5,.5,101);DELAYS[50]=0

def features(x):
 x=np.asarray(x,np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('finite source IQ256 only')
 power=np.mean(x[:,0]**2+x[:,1]**2,axis=1)
 if (power<=1e-12).any() or np.max(abs(power-1))>1e-3:raise ValueError('original RMS unit input only')
 f=np.fft.rfft(x,axis=-1)[...,1:128];I,Q=f[:,0],f[:,1];cross=I*Q.conj()
 return np.stack([abs(I)**2,abs(Q)**2,cross.real,cross.imag],axis=1)

def cells(f,y,rx,day,count,half=None):
 f=np.asarray(f,np.float64)
 if f.shape!=(len(y),4,127) or not np.isfinite(f).all():raise ValueError('complete finite periodogram features')
 if half not in (None,0,1):raise ValueError('Only fixed halves')
 out=np.empty((6,5,3,127),np.complex128)
 for t in range(6):
  for ri,r in enumerate(RX):
   for di,d in enumerate((1,2,3)):
    idx=np.flatnonzero((y==t)&(rx==r)&(day==d))
    if len(idx)!=count:raise ValueError('all90 original cells required')
    if half is not None:idx=idx[half::2]
    z=f[idx].mean(0);den=np.sqrt(z[0]*z[1])
    if np.any(den<=1e-12):raise ValueError('unsupported zero-power frequency;do not delete bins')
    out[t,ri,di]=(z[2]+1j*z[3])/den
 if (abs(out)>1+1e-10).any():raise ValueError('invalid spectral coherence')
 return out

def evaluate(L,V,L0,L1):
 for x in (L,V,L0,L1):
  if x.shape!=(6,5,3,127) or not np.isfinite(x).all() or (abs(x)>1+1e-10).any():raise ValueError('fixed all-cell coherence support')
 truth=V-V.mean(1,keepdims=True);pred=np.zeros_like(V);add=np.zeros_like(V);txcond=np.zeros_like(V);indices=np.full((6,3,5),50,np.int64)
 phase=np.exp(1j*DELAYS[:,None]*OMEGA)
 for t in range(6):
  for di in range(3):
   tt=[j for j in range(6) if j!=t];dd=[j for j in range(3) if j!=di]
   pred[t,0,di]=V[t,0,di]
   for r in range(1,5):
    err=np.asarray([sum(float(np.square(abs(L[j,0,k]*p-L[j,r,k])).sum()) for j in tt for k in dd) for p in phase])
    b=min(range(101),key=lambda j:(err[j],abs(float(DELAYS[j])),float(DELAYS[j])));indices[t,di,r]=b;pred[t,r,di]=V[t,0,di]*phase[b]
   add[t,:,di]=np.mean([L[j,:,k]-L[j,:,k].mean(0,keepdims=True) for j in tt for k in dd],axis=0)
   txcond[t,:,di]=np.mean([L[t,:,k]-L[t,:,k].mean(0,keepdims=True) for k in dd],axis=0)
 pred-=pred.mean(1,keepdims=True)
 arrays=dict(shared=abs(pred-truth)**2,identity=abs(truth)**2,additive=abs(add-truth)**2,TXconditional=abs(txcond-truth)**2)
 sse={k:float(v.sum()) for k,v in arrays.items()};folds={k:v.sum(axis=(1,3)) for k,v in arrays.items()};byrx={k:v.sum(axis=(0,2,3)) for k,v in arrays.items()}
 split=np.mean((L0*L1.conj()).real,axis=(0,2,3));boundary=int(np.isin(indices[:,:,1:],[0,100]).sum())
 checks=dict(nonzero_RX_difference=sse['identity']>1e-12,split_coherent_power=bool(split.mean()>=.0025 and (split>0).all()),shared_beats_identity_20percent=sse['shared']<=.8*sse['identity'],shared_beats_additive_10percent=sse['shared']<=.9*sse['additive'],improves_15_of_18_folds=int((folds['shared']<folds['additive']).sum())>=15,all_RX_no_worse_additive=bool((byrx['shared']<=byrx['additive']+1e-20).all()),within_TXconditional_5percent=sse['shared']<=1.05*sse['TXconditional'],bounded_interior_fits=boundary==0)
 result=dict(source_qualified=all(checks.values()),decision='SOURCE_PASS_NEXT_FINITE_WINDOW_IDENTITY_TEST_ONLY' if all(checks.values()) else 'SOURCE_FAIL_FIXED_IQ_SKEW_PHASE_LAW_CLOSED',criteria=checks,SSE=sse,L_split_coherent_power_by_RX={str(r):float(split[i]) for i,r in enumerate(RX)},folds_improving_additive=int((folds['shared']<folds['additive']).sum()),boundary_fits=boundary,relative_fit_delay_min=float(DELAYS[indices[:,:,1:]].min()),relative_fit_delay_max=float(DELAYS[indices[:,:,1:]].max()),per_RX={str(r):{k:float(v[i]) for k,v in byrx.items()} for i,r in enumerate(RX)},fold_SSE={k:v.tolist() for k,v in folds.items()})
 return result,dict(fit_indices=indices,predicted=pred,truth=truth,additive=add,TXconditional=txcond,L_coherence=L,V_coherence=V,L_half0=L0,L_half1=L1)
