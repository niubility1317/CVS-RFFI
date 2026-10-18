import csv,json,sys,time
from pathlib import Path
import numpy as np

def main(B):
 started=time.perf_counter();B=Path(B);meta=json.loads((B/'resolved_config.json').read_text());src=Path(meta['config']['input_statistics']);z=np.load(src,allow_pickle=False);saved=np.load(B/'fitted_statistics.npz',allow_pickle=False);result=json.loads((B/'result.json').read_text());rxs=[1,3,4,6,8]
 obs={};allids={}
 for role,n,count in [('L',6300,70),('V',27000,300)]:
  f=z[role+'_features'];ys=z[role+'_y'];rs=z[role+'_rx'];ds=z[role+'_day'];ids=z[role+'_ids'].tolist();allids[role]=set(ids)
  if f.shape!=(n,4,127) or len(ids)!=n or len(allids[role])!=n or not np.isfinite(f).all() or (f[:,:2]<0).any():raise ValueError('independent input support')
  if not np.allclose(f[:,2]**2+f[:,3]**2,f[:,0]*f[:,1],rtol=1e-10,atol=1e-10):raise ValueError('independent perpacket spectral product')
  for h in ([None,0,1] if role=='L' else [None]):
   out=np.empty((6,5,3,127))
   for ti in range(6):
    for ri,r in enumerate(rxs):
     for di,d in enumerate([1,2,3]):
      idx=np.flatnonzero((ys==ti)&(rs==r)&(ds==d))
      if len(idx)!=count:raise ValueError('independent full cell count')
      if h is not None:idx=idx[h::2]
      p=f[idx,:2].mean(0)
      if (p<=1e-12).any():raise ValueError('independent zero power')
      q=10*np.log10(p[0]/p[1]);out[ti,ri,di]=q-q.mean()
   obs[role if h is None else role+str(h)]=out
 if allids['L']&allids['V']:raise ValueError('independent role overlap')
 L,V,L0,L1=[obs[k] for k in ['L','V','L0','L1']];delta=max(float(abs(v-saved[k]).max()) for k,v in obs.items())
 if delta>1e-10:raise ValueError('independent cell mismatch')
 basis=np.square(np.arange(1,128)/127);basis-=basis.mean();grid=np.arange(-50,51)/50;pred=np.zeros_like(V);add=np.zeros_like(V);tx=np.zeros_like(V);fit=np.full((6,3,5),50,dtype=int)
 for t in range(6):
  for d in range(3):
   other_t=np.delete(np.arange(6),t);other_d=np.delete(np.arange(3),d)
   for r in range(1,5):
    diffs=L[other_t[:,None],r,other_d[None,:]]-L[other_t[:,None],0,other_d[None,:]]
    errors=np.array([np.square(diffs-b*basis).sum() for b in grid]);index=int(np.lexsort((grid,abs(grid),errors))[0]);fit[t,d,r]=index;pred[t,r,d]=grid[index]*basis
   train=L[other_t[:,None],:,other_d[None,:],:];train-=train.mean(2,keepdims=True);add[t,:,d]=train.mean((0,1))
   a=L[t,:,other_d,:];a-=a.mean(1,keepdims=True);tx[t,:,d]=a.mean(0)
 if not np.array_equal(fit,saved['fit_indices']):raise ValueError('independent 72fit mismatch')
 pred-=pred.mean(1,keepdims=True);truth=V-V.mean(1,keepdims=True);errors={'shared':np.square(pred-truth),'identity':truth**2,'additive':np.square(add-truth),'TXconditional':np.square(tx-truth)}
 sse={k:float(v.sum()) for k,v in errors.items()};fold={k:v.sum((1,3)) for k,v in errors.items()};byrx={k:v.sum((0,2,3)) for k,v in errors.items()};cross=((L0-L0.mean(1,keepdims=True))*(L1-L1.mean(1,keepdims=True))).mean((0,2,3));boundary=int(np.isin(fit[:,:,1:],[0,100]).sum());improved=int((fold['shared']<fold['additive']).sum())
 criteria=dict(nonzero_RX_difference=sse['identity']>1e-12,split_RX_contrast_power=bool(cross.mean()>=.01 and (cross>0).all()),shared_beats_identity_20percent=sse['shared']<=.8*sse['identity'],shared_beats_additive_10percent=sse['shared']<=.9*sse['additive'],improves_15_of_18_folds=improved>=15,all_RX_no_worse_additive=bool((byrx['shared']<=byrx['additive']+1e-20).all()),within_TXconditional_5percent=sse['shared']<=1.05*sse['TXconditional'],bounded_interior_fits=boundary==0)
 if criteria!=result['criteria']:raise ValueError('independent criteria mismatch')
 for k in sse:
  if not np.isclose(sse[k],result['SSE'][k],rtol=1e-11,atol=1e-11):raise ValueError('independent SSE mismatch')
  np.testing.assert_allclose(fold[k],result['fold_SSE'][k],rtol=1e-11,atol=1e-11)
 for i,r in enumerate(rxs):
  for k in byrx:
   if not np.isclose(byrx[k][i],result['per_RX'][str(r)][k],rtol=1e-11,atol=1e-11):raise ValueError('independent RX risk mismatch')
 O=B/'independent_analysis';O.mkdir()
 with (O/'per_RX_SSE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['RX',*errors]);w.writerows([[r,*[byrx[k][i] for k in errors]] for i,r in enumerate(rxs)])
 with (O/'held_TX_day_folds.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['TX','day',*errors,'RX3_gain_dB','RX4_gain_dB','RX6_gain_dB','RX8_gain_dB']);w.writerows([[t,d+1,*[fold[k][t,d] for k in errors],*grid[fit[t,d,1:]]] for t in range(6) for d in range(3)])
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(17,5.2));x=np.arange(5)
 for j,k in enumerate(errors):ax[0].bar(x+(j-1.5)*.19,byrx[k],.19,label=k)
 ax[0].set(xticks=x,xticklabels=rxs,xlabel='Source RX',ylabel='Held source V squared risk (dB squared)',title='All5 RX / 18 held TX-day folds');ax[0].legend(fontsize=9)
 for r in range(1,5):ax[1].plot(grid[fit[:,:,r]].ravel(),'.-',label='RX'+str(rxs[r]))
 ax[1].axhline(-1,c='gray',ls=':');ax[1].axhline(1,c='gray',ls=':');ax[1].set(xlabel='Held TX/day fold',ylabel='Differential edge-to-zero gain coefficient (dB)',title='L-only fits;not identified hardware');ax[1].legend(fontsize=9)
 for k in ['shared','additive']:ax[2].plot(fold[k].ravel(),'.-',label=k)
 ax[2].set(xlabel='Held TX/day fold',ylabel='Held source V squared risk (dB squared)',title='No neural accuracy measured');ax[2].legend()
 for a in ax:a.grid(alpha=.2)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(O/('source_iqband_effects.'+ext),dpi=150)
 plt.close(fig)
 q=dict(status='VERIFIED_INDEPENDENT_SOURCE_RECOUNT',criteria=criteria,source_qualified=all(criteria.values()),SSE=sse,maximum_cell_delta=delta,all72_fit_indices_match=True,source_derived_values_read=16916400,IQ_reads=0,checkpoint_reads=0,NN=0,training_updates=0,target_access=False,wall_seconds=time.perf_counter()-started)
 (O/'result.json').write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');print(json.dumps(q))
if __name__=='__main__':main(sys.argv[1])
