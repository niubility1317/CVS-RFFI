"""Independent frozen source predictions recount;zero NN,updates,truth access."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np

def write(p,obj):
 p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8')
def run(base,out,plots=True):
 tic=time.perf_counter();out.mkdir(exist_ok=False)
 z=np.load(base/'predictions.npz',allow_pickle=False)
 y=z['TX'];rx=z['RX'];day=z['day'];ids=z['ids'];views={v:z[v] for v in ('raw','conjugate')}
 assert y.shape==rx.shape==day.shape==(1800,) and len(np.unique(ids))==1800
 assert set(rx)=={1,3,4,6,8} and set(day)=={1,2,3} and set(y)==set(range(6))
 for r in sorted(set(rx)):
  for d in sorted(set(day)):
   for t in range(6):assert int(((rx==r)&(day==d)&(y==t)).sum())==20
 for logits in views.values():assert logits.shape==(1800,6) and np.isfinite(logits).all()
 native=json.loads((base/'source_metrics.json').read_text())['rows'];lookup={(r['view'],r['dimension'],r['stratum']):r for r in native};assert len(lookup)==30
 groups=[('overall','ALL',np.ones(1800,bool))]+[('RX',str(r),rx==r) for r in sorted(set(rx))]+[('day',str(d),day==d) for d in sorted(set(day))]+[('TX',str(t),y==t) for t in range(6)]
 rows=[];maxerr=0.
 for view,logits in views.items():
  pred=logits.argmax(1)
  for dim,label,mask in groups:
   cm=np.zeros((6,6),dtype=np.int64)
   for a,b in zip(y[mask],pred[mask]):cm[int(a),int(b)]+=1
   den=cm.sum(0)+cm.sum(1);f1=np.array([2*cm[c,c]/den[c] if den[c] else 0. for c in range(6)])
   sub=logits[mask];yy=y[mask];margin=np.array([(a[int(t)]-max(a[c] for c in range(6) if c!=t))/np.float32(30) for a,t in zip(sub,yy)],np.float32)
   row=dict(view=view,dimension=dim,stratum=label,count=int(mask.sum()),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/mask.sum()),macro_F1_all6=float(f1.mean()),mean_true_margin_over30=float(margin.mean()))
   old=lookup[(view,dim,label)];assert np.array_equal(cm,old['confusion']);assert row['count']==old['count'] and row['correct']==old['correct']
   for key in ('accuracy','macro_F1_all6','mean_true_margin_over30'):
    err=abs(row[key]-old[key]);maxerr=max(maxerr,err);assert err<1e-7,(view,dim,label,key,err)
   assert np.allclose(np.quantile(margin,[.05,.25,.5,.75,.95]),old['margin_quantiles_over30'],rtol=0,atol=1e-7)
   rows.append(row)
 a=views['raw'].argmax(1)==y;b=views['conjugate'].argmax(1)==y;paired=[]
 for dim,label,mask in groups:
  n=int(mask.sum());nraw=int((a&mask).sum());nb=int((b&mask).sum());cohort=int((a&b&mask).sum());rep=int((~a&b&mask).sum());loss=int((a&~b&mask).sum())
  assert nb-nraw==rep-loss and cohort+loss==nraw
  paired.append(dict(dimension=dim,stratum=label,count=n,raw_correct=nraw,conjugate_correct=nb,accuracy_raw=nraw/n,accuracy_conjugate=nb/n,delta_pp=100*(nb-nraw)/n,raw_correct_retention=cohort/nraw if nraw else 0.,repairs=rep,new_errors=loss))
 overall=paired[0];perrx=[v for v in paired if v['dimension']=='RX']
 checks=dict(global_accuracy_preserved=overall['delta_pp']>=-.5-1e-10,every_RX_accuracy_preserved=all(v['delta_pp']>=-1.5-1e-10 for v in perrx),fixed_raw_correct_retained=overall['raw_correct_retention']>=.99-1e-12)
 decision=json.loads((base/'source_decision.json').read_text());assert checks==decision['checks'] and decision['candidate_admissible']==all(checks.values())
 assert decision['status']==('SOURCE_GUARD_PASS' if all(checks.values()) else 'SOURCE_GUARD_FAIL')
 for k,key in [('accuracy_raw','accuracy_raw'),('accuracy_reflected','accuracy_conjugate'),('accuracy_delta_pp','delta_pp'),('fixed_raw_correct_retention','raw_correct_retention'),('repairs','repairs'),('new_errors','new_errors')]:assert abs(decision[k]-overall[key])<1e-10
 complete=json.loads((base/'complete.json').read_text());assert complete['NN']==3602 and complete['model_state_unchanged'] and complete['optimizer_updates']==0 and not complete['target_access'] and not complete['automatic_training']
 for name,data in [('source_metrics_recount.csv',rows),('paired_source.csv',paired)]:
  with (out/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 figfiles=[]
 if plots:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
  fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(5)
  ax[0].bar(x-.18,[100*r['accuracy_raw'] for r in perrx],.36,label='Raw',color='#2563eb');ax[0].bar(x+.18,[100*r['accuracy_conjugate'] for r in perrx],.36,label='Conjugate',color='#dc2626');ax[0].set_xticks(x,[r['stratum'] for r in perrx]);ax[0].set_xlabel('Physical source RX');ax[0].set_ylabel('Accuracy (%)');ax[0].set_ylim(0,105);ax[0].legend();ax[0].set_title('Frozen default head; same V1800')
  vals=[100*r['raw_correct_retention'] for r in perrx];ax[1].bar(x,vals,color='#7c3aed');ax[1].axhline(99,color='#475569',ls='--',label='Fixed 99% guard');ax[1].set_xticks(x,[r['stratum'] for r in perrx]);ax[1].set_xlabel('Physical source RX');ax[1].set_ylabel('Raw-correct retention (%)');ax[1].set_ylim(0,105);ax[1].legend();ax[1].set_title('No training; one model seed')
  for i,v in enumerate(vals):ax[1].text(i,v+2,f'{v:.1f}',ha='center')
  for ext in ('png','pdf'):p=out/('source_RX_guard.'+ext);fig.savefig(p,dpi=180);figfiles.append(str(p))
  plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
  for ax,view in zip(axes,views):
   cm=np.asarray(lookup[(view,'overall','ALL')]['confusion']);im=ax.imshow(cm/300*100,vmin=0,vmax=100,cmap='Blues')
   for i in range(6):
    for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>150 else 'black',fontsize=9)
   ax.set_xlabel('Default-head prediction');ax.set_ylabel('True source TX');ax.set_title(view.capitalize());ax.set_xticks(range(6));ax.set_yticks(range(6))
  fig.colorbar(im,ax=axes,label='Within true TX (%)',shrink=.8);fig.suptitle('Same 1800 source packets; 300 per TX; frozen F')
  for ext in ('png','pdf'):p=out/('source_confusion.'+ext);fig.savefig(p,dpi=180);figfiles.append(str(p))
  plt.close(fig)
 result=dict(status='VERIFIED_SOURCE_NEGATIVE',recounted_confusion_matrices=30,source_packets=1800,balanced_TX_RX_day=True,max_metric_error=maxerr,checks=checks,overall=overall,per_RX=perrx,new_NN=0,new_updates=0,target_access=False,U_truth_read=False,Ag_proved=False,crossseed_confirmed=False,automatic_training=False,goal_complete=False,figures=figfiles,analysis_wall_seconds=time.perf_counter()-tic)
 write(out/'verification.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('out',type=Path);p.add_argument('--no-plots',action='store_true');a=p.parse_args();run(a.base,a.out,not a.no_plots)
