"""Independent saved-observable recount;no IQ,torch,model,or truth access."""
import sys,json,csv,time
from pathlib import Path
import numpy as np

def main(root):
 started=time.perf_counter();z=np.load(root/'observations.npz',allow_pickle=False);saved=json.loads((root/'result.json').read_text());rx=(1,3,4,6,8);omega=2*np.pi*np.arange(1,128)/256;grid=np.linspace(-.5,.5,101);grid[50]=0
 np.testing.assert_array_equal(z['delays'],grid);np.testing.assert_array_equal(z['omega'],omega);C={};feature_values=0;maxcell=0.
 for pre,n,count in [('L',6300,70),('V',27000,300)]:
  f=z[pre+'_features'];y=z[pre+'_y'];r=z[pre+'_rx'];day=z[pre+'_day'];ids=z[pre+'_ids'];feature_values+=f.size
  if f.shape!=(n,4,127) or len(set(ids.tolist()))!=n or not np.isfinite(f).all():raise ValueError('Incomplete real source observable bank')
  product=f[:,0]*f[:,1];error=abs(product-f[:,2]**2-f[:,3]**2)/np.maximum(product,1e-20)
  if float(error.max())>1e-10:raise ValueError('Invalid per-packet Fourier product relation')
  for half in ([None,0,1] if pre=='L' else [None]):
   c=np.empty((6,5,3,127),complex)
   for t in range(6):
    for ri,rr in enumerate(rx):
     for di,dd in enumerate((1,2,3)):
      idx=np.flatnonzero((y==t)&(r==rr)&(day==dd))
      if len(idx)!=count:raise ValueError('All90 source cells required')
      if half is not None:idx=idx[half::2]
      p=f[idx].sum(axis=0)/len(idx);c[t,ri,di]=(p[2]+1j*p[3])/np.sqrt(p[0]*p[1])
   key=pre if half is None else 'L_half'+str(half);stored=z[pre+'_coherence'] if half is None else z['L_half'+str(half)]
   maxcell=max(maxcell,float(abs(c-stored).max()));C[key]=c
 if maxcell>1e-12:raise ValueError('Cell coherence recount mismatch')
 L,V=C['L'],C['V'];phase=np.cos(grid[:,None]*omega)+1j*np.sin(grid[:,None]*omega);pred=np.empty_like(V);add=np.empty_like(V);tx=np.empty_like(V);fit=np.full((6,3,5),50,np.int64)
 for t in range(6):
  for day in range(3):
   tt=[j for j in range(6) if j!=t];dd=[j for j in range(3) if j!=day];pred[t,0,day]=V[t,0,day]
   for r in range(1,5):
    ref=L[tt,0][:,dd];target=L[tt,r][:,dd];errors=np.square(abs(ref[None]*phase[:,None,None,:]-target[None])).sum(axis=(1,2,3));b=int(np.lexsort((grid,abs(grid),errors))[0]);fit[t,day,r]=b;pred[t,r,day]=V[t,0,day]*phase[b]
   a=np.zeros((5,127),complex);b=np.zeros_like(a)
   for j in tt:
    for k in dd:a+=L[j,:,k]-L[j,:,k].mean(axis=0)
   for k in dd:b+=L[t,:,k]-L[t,:,k].mean(axis=0)
   add[t,:,day]=a/10;tx[t,:,day]=b/2
 pred-=pred.mean(axis=1,keepdims=True);truth=V-V.mean(axis=1,keepdims=True)
 np.testing.assert_array_equal(fit,z['fit_indices'])
 for key,a in [('predicted',pred),('truth',truth),('additive',add),('TXconditional',tx)]:np.testing.assert_allclose(a,z[key],rtol=1e-11,atol=1e-12)
 squared=dict(shared=abs(pred-truth)**2,identity=abs(truth)**2,additive=abs(add-truth)**2,TXconditional=abs(tx-truth)**2);sse={k:float(v.sum()) for k,v in squared.items()};fold={k:v.sum(axis=(1,3)) for k,v in squared.items()};byrx={k:v.sum(axis=(0,2,3)) for k,v in squared.items()};split=(C['L_half0']*C['L_half1'].conj()).real.mean(axis=(0,2,3));boundary=int(((fit[:,:,1:]==0)|(fit[:,:,1:]==100)).sum())
 checks=dict(nonzero_RX_difference=sse['identity']>1e-12,split_coherent_power=bool(split.mean()>=.0025 and (split>0).all()),shared_beats_identity_20percent=sse['shared']<=.8*sse['identity'],shared_beats_additive_10percent=sse['shared']<=.9*sse['additive'],improves_15_of_18_folds=int((fold['shared']<fold['additive']).sum())>=15,all_RX_no_worse_additive=bool((byrx['shared']<=byrx['additive']+1e-20).all()),within_TXconditional_5percent=sse['shared']<=1.05*sse['TXconditional'],bounded_interior_fits=boundary==0)
 if checks!=saved['criteria'] or all(checks.values())!=saved['source_qualified']:raise ValueError('Independent source verdict mismatch')
 for k in sse:np.testing.assert_allclose(sse[k],saved['SSE'][k],rtol=1e-11,atol=1e-12)
 out=root/'independent_analysis';out.mkdir()
 with (out/'per_RX_SSE.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['RX','shared','identity','additive','TXconditional','L_split_coherent_power'])
  for i,r in enumerate(rx):w.writerow([r,*[byrx[k][i] for k in ('shared','identity','additive','TXconditional')],split[i]])
 with (out/'held_TX_day_folds.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['held_TX','held_day','shared','identity','additive','TXconditional',*[str(r)+'_delay' for r in rx]])
  for t in range(6):
   for day in range(3):w.writerow([t,day+1,*[fold[k][t,day] for k in ('shared','identity','additive','TXconditional')],*grid[fit[t,day]]])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained');x=np.arange(5)
 for i,k in enumerate(('identity','additive','TXconditional','shared')):axes[0].bar(x+(i-1.5)*.18,byrx[k],.18,label=k)
 axes[0].set_xticks(x,rx);axes[0].set(xlabel='Source RX',ylabel='Held source V squared risk',title='Same all5RX / 18 TX-day folds');axes[0].legend()
 for r in range(1,5):axes[1].plot(np.arange(18),grid[fit[:,:,r]].ravel(),'.-',label='RX'+str(rx[r]))
 axes[1].axhline(.5,color='gray',ls=':');axes[1].axhline(-.5,color='gray',ls=':');axes[1].set(xlabel='Held TX/day fold',ylabel='Fitted relative phase-law delay (samples)',title='L-only fits;not identified hardware');axes[1].legend()
 axes[2].plot(np.arange(18),fold['shared'].ravel(),'.-',label='shared');axes[2].plot(np.arange(18),fold['additive'].ravel(),'.-',label='additive');axes[2].set(xlabel='Held TX/day fold',ylabel='V squared risk',title='No neural accuracy measured');axes[2].legend()
 for ax in axes:ax.grid(alpha=.2)
 for ext in ('png','pdf'):fig.savefig(out/('source_iqskew_effects.'+ext),dpi=170)
 plt.close(fig)
 result=dict(status='VERIFIED_INDEPENDENT_SOURCE_RECOUNT',criteria=checks,source_qualified=all(checks.values()),SSE=sse,per_RX={str(r):{k:float(v[i]) for k,v in byrx.items()} for i,r in enumerate(rx)},feature_real_values=feature_values,maximum_cell_coherence_delta=maxcell,all72_fit_indices_match=True,NN=0,IQ_reads=0,checkpoint_reads=0,training_updates=0,target_access=False,wall_seconds=time.perf_counter()-started)
 (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print('INDEPENDENT_SOURCE_RECOUNT PASS',flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
