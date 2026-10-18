import numpy as np
RX=(1,3,4,6,8)
BASIS=(np.arange(1,128,dtype=np.float64)/127)**2;BASIS-=BASIS.mean()
COEFFS=np.linspace(-1,1,101);COEFFS[50]=0

def cells(f,y,rx,day,count,half=None):
 f=np.asarray(f,np.float64)
 if f.shape!=(len(y),4,127) or not np.isfinite(f).all() or half not in (None,0,1):raise ValueError('full finite fixed127 features required')
 out=np.empty((6,5,3,127),np.float64)
 for t in range(6):
  for ri,r in enumerate(RX):
   for di,d in enumerate((1,2,3)):
    idx=np.flatnonzero((y==t)&(rx==r)&(day==d))
    if len(idx)!=count:raise ValueError('all90 original cells required')
    if half is not None:idx=idx[half::2]
    z=f[idx,:2].mean(0)
    if (z<=1e-12).any():raise ValueError('unsupported zero power;no bin deletion or clipping')
    a=10/np.log(10)*np.log(z[0]/z[1]);out[t,ri,di]=a-a.mean()
 return out

def load_observations(path,cfg):
 from .design import read
 c=read(cfg)
 if c.get('source_roles')!='EXACT_MATCH' or c['config'].get('L')!=6300 or c['config'].get('V')!=27000 or c['config'].get('U_used') is not False:raise ValueError('existing exact full source roles only')
 z=np.load(path,allow_pickle=False);out={}
 for key,n,count in [('L',6300,70),('V',27000,300)]:
  f=np.asarray(z[key+'_features'],np.float64);y=z[key+'_y'];rx=z[key+'_rx'];day=z[key+'_day'];ids=z[key+'_ids'].tolist()
  if f.shape!=(n,4,127) or len(set(ids))!=n or len(y)!=n or len(rx)!=n or len(day)!=n or not np.isfinite(f).all() or (f[:,:2]<0).any():raise ValueError('original full unique finite source data only')
  if not set(y)==set(range(6)) or not set(rx)==set(RX) or not set(day)=={1,2,3}:raise ValueError('source role metadata mismatch')
  if not np.allclose(f[:,2]**2+f[:,3]**2,f[:,0]*f[:,1],rtol=1e-10,atol=1e-10):raise ValueError('perpacket periodogram identity mismatch')
  out[key]=cells(f,y,rx,day,count);out[key+'_ids']=ids
  if key=='L':
   for h in (0,1):out['L'+str(h)]=cells(f,y,rx,day,count,h)
 if set(out['L_ids'])&set(out['V_ids']):raise ValueError('source L/V overlap')
 return out

def evaluate(L,V,L0,L1):
 for a in (L,V,L0,L1):
  if a.shape!=(6,5,3,127) or not np.isfinite(a).all() or abs(a.mean(-1)).max()>1e-10:raise ValueError('fixed centered logratio required')
 truth=V-V.mean(1,keepdims=True);pred=np.zeros_like(V);add=np.zeros_like(V);txcond=np.zeros_like(V);indices=np.full((6,3,5),50,np.int64)
 for t in range(6):
  for di in range(3):
   tt=[j for j in range(6) if j!=t];dd=[j for j in range(3) if j!=di]
   for r in range(1,5):
    contrast=np.asarray([L[j,r,k]-L[j,0,k] for j in tt for k in dd]);errors=np.square(contrast[None]-COEFFS[:,None,None]*BASIS).sum((1,2));b=min(range(101),key=lambda i:(errors[i],abs(float(COEFFS[i])),float(COEFFS[i])));indices[t,di,r]=b;pred[t,r,di]=COEFFS[b]*BASIS
   add[t,:,di]=np.mean([L[j,:,k]-L[j,:,k].mean(0,keepdims=True) for j in tt for k in dd],axis=0)
   txcond[t,:,di]=np.mean([L[t,:,k]-L[t,:,k].mean(0,keepdims=True) for k in dd],axis=0)
 pred-=pred.mean(1,keepdims=True)
 arr=dict(shared=(pred-truth)**2,identity=truth**2,additive=(add-truth)**2,TXconditional=(txcond-truth)**2);sse={k:float(v.sum()) for k,v in arr.items()};fold={k:v.sum((1,3)) for k,v in arr.items()};byrx={k:v.sum((0,2,3)) for k,v in arr.items()}
 a=L0-L0.mean(1,keepdims=True);b=L1-L1.mean(1,keepdims=True);split=(a*b).mean((0,2,3));boundary=int(np.isin(indices[:,:,1:],[0,100]).sum());improved=int((fold['shared']<fold['additive']).sum())
 checks=dict(nonzero_RX_difference=sse['identity']>1e-12,split_RX_contrast_power=bool(split.mean()>=.01 and (split>0).all()),shared_beats_identity_20percent=sse['shared']<=.8*sse['identity'],shared_beats_additive_10percent=sse['shared']<=.9*sse['additive'],improves_15_of_18_folds=improved>=15,all_RX_no_worse_additive=bool((byrx['shared']<=byrx['additive']+1e-20).all()),within_TXconditional_5percent=sse['shared']<=1.05*sse['TXconditional'],bounded_interior_fits=boundary==0)
 result=dict(source_qualified=all(checks.values()),decision='SOURCE_PASS_NEXT_FINITE_WINDOW_IDENTITY_TEST_ONLY' if all(checks.values()) else 'SOURCE_FAIL_FIXED_IQ_BAND_GAIN_LAW_CLOSED',criteria=checks,SSE=sse,L_split_RX_contrast_power_dB2={str(r):float(split[i]) for i,r in enumerate(RX)},folds_improving_additive=improved,boundary_fits=boundary,fit_coeff_min_dB=float(COEFFS[indices[:,:,1:]].min()),fit_coeff_max_dB=float(COEFFS[indices[:,:,1:]].max()),per_RX={str(r):{k:float(v[i]) for k,v in byrx.items()} for i,r in enumerate(RX)},fold_SSE={k:v.tolist() for k,v in fold.items()})
 return result,dict(fit_indices=indices,L=L,V=V,L0=L0,L1=L1,predicted=pred,truth=truth,additive=add,TXconditional=txcond,basis=BASIS,coefficient_grid=COEFFS)
