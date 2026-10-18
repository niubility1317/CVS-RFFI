"""Independent source-aggregate recount; no candidate imports, IQ or model access."""
import json,sys,time,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 started=time.perf_counter();base=Path(sys.argv[1]);out=base/'independent_analysis'
 if out.exists():raise FileExistsError('Immutable independent analysis')
 out.mkdir();a=np.load(base/'observations.npz',allow_pickle=False);beta=a['beta'];rx=[1,3,4,6,8];C={};max_error=0.
 for p,n in [('L',70),('V',300)]:
  x=np.empty((6,5,3,21,3))
  if len(set(a[p+'_ids'].tolist()))!=90*n:raise ValueError('Packet ID duplicate/count')
  for t in range(6):
   for ri,r in enumerate(rx):
    for di,d in enumerate([1,2,3]):
     take=np.flatnonzero((a[p+'_y']==t)&(a[p+'_rx']==r)&(a[p+'_day']==d))
     if len(take)!=n:raise ValueError('Cell count mismatch')
     x[t,ri,di]=np.sum(a[p+'_features'][take],axis=0,dtype=np.float64)/n
  max_error=max(max_error,float(np.max(abs(x-a[p+'_cells']))));C[p]=x
 L=C['L'];V=C['V'];rawL=L[:,:,:,10,:];rawV=V[:,:,:,10,:];allerr={k:np.zeros((6,5,3)) for k in ['shared','identity','additive','TXconditional']};fits=np.full((6,3,5),10,dtype=int)
 for t in range(6):
  for day in range(3):
   ti=np.flatnonzero(np.arange(6)!=t);di=np.flatnonzero(np.arange(3)!=day);predict=np.empty((5,3));predict[0]=V[t,0,day,10]
   for r in range(1,5):
    err=np.zeros(21)
    for j in ti:
     for k in di:err+=np.sum(np.square(L[j,0,k]-L[j,r,k,10][None,:]),axis=1)
    order=sorted(range(21),key=lambda i:(float(err[i]),abs(float(beta[i])),float(beta[i])))
    fits[t,day,r]=order[0];predict[r]=V[t,0,day,order[0]]
   truth=rawV[t,:,day]-rawV[t,:,day].mean(axis=0)
   predict-=predict.mean(axis=0)
   additive=np.mean([rawL[j,:,k]-rawL[j,:,k].mean(axis=0) for j in ti for k in di],axis=0)
   cond=np.mean([rawL[t,:,k]-rawL[t,:,k].mean(axis=0) for k in di],axis=0)
   for name,pr in [('shared',predict),('identity',np.zeros_like(truth)),('additive',additive),('TXconditional',cond)]:allerr[name][t,:,day]=np.sum((pr-truth)**2,axis=1)
 result=json.loads((base/'result.json').read_text());totals={k:float(v.sum()) for k,v in allerr.items()}
 for k,v in totals.items():max_error=max(max_error,abs(v-result['SSE'][k]))
 if not np.array_equal(fits,a['fit_indices']) or max_error>1e-12:raise ValueError('Independent numeric mismatch')
 for k,v in allerr.items():
  if not np.allclose(v.sum(axis=1),result['fold_SSE'][k],rtol=1e-11,atol=1e-14):raise ValueError('Fold SSE mismatch')
  for ri,r in enumerate(rx):
   if abs(v[:,ri,:].sum()-result['per_RX'][str(r)][k])>1e-12:raise ValueError('RX SSE mismatch')
 shared=allerr['shared'];add=allerr['additive'];ident=allerr['identity'];cond=allerr['TXconditional'];boundary=int(np.isin(fits[:,:,1:],[0,20]).sum())
 criteria=[totals['identity']>1e-12,totals['shared']<=.8*totals['identity'],totals['shared']<=.9*totals['additive'],int((shared.sum(1)<add.sum(1)).sum())>=15,bool((shared.sum((0,2))<=add.sum((0,2))+1e-20).all()),totals['shared']<=1.05*totals['TXconditional'],boundary==0]
 if criteria!=list(result['criteria'].values()) or all(criteria)!=result['source_qualified']:raise ValueError('Decision mismatch')
 with (out/'source_RX.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['RX']+list(allerr));w.writerows([[r]+[float(v[:,ri,:].sum()) for v in allerr.values()] for ri,r in enumerate(rx)])
 with (out/'source_folds.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['TX','day']+list(allerr));w.writerows([[t,d+1]+[float(v[t,:,d].sum()) for v in allerr.values()] for t in range(6) for d in range(3)])
 fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 labels=list(totals);ax[0].bar(labels,list(totals.values()),color=['#d95f02','#7570b3','#1b9e77','#666666']);ax[0].set(ylabel='Squared source RX difference error',title='L-only prediction of source V');ax[0].tick_params(axis='x',rotation=20)
 for ri,r in enumerate(rx):ax[1].bar(np.arange(4)+(ri-2)*.15,[result['per_RX'][str(r)][k] for k in labels],width=.14,label='RX '+str(r))
 ax[1].set_xticks(range(4),labels,rotation=20);ax[1].set(title='All five source RX',ylabel='Squared error');ax[1].legend(fontsize=8)
 ax[2].hist(beta[fits[:,:,1:]].ravel(),bins=21,color='#377eb8');ax[2].axvline(beta[0],ls='--',color='r');ax[2].axvline(beta[-1],ls='--',color='r');ax[2].set(title='72 L-only fits; bounds fixed by L',xlabel='Amplitude dependent phase coefficient',ylabel='Fit count')
 fig.suptitle(result['decision'],fontsize=11)
 for ext in ('png','pdf'):fig.savefig(out/('source_ampm_effects.'+ext),dpi=160)
 plt.close(fig)
 record=dict(status='PASS',source_feature_values=int(a['L_features'].size+a['V_features'].size),cell_means=180,fits=72,folds=18,RX=5,max_abs_numeric_difference=max_error,all_criteria_equal=True,SSE=totals,IQ_reads=0,NN=0,training_updates=0,target_access=False,wall_seconds=time.perf_counter()-started)
 (out/'independent_recount.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
