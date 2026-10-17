"""Fixed source criteria;default-head scores,never feature similarity."""
import numpy as np
from . import design as d

def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=len(y),accuracy=float(np.trace(cm)/len(y)) if len(y) else None,macro_F1=float(f.mean()) if len(y) else None,confusion=cm.tolist())

def summarize(logits,y,rx,day,records):
 y=np.asarray(y);rx=np.asarray(rx);day=np.asarray(day);scores={};raw=np.asarray(logits['raw']).argmax(1);rawgood=raw==y
 for k,z in logits.items():
  a=np.asarray(z);p=a.argmax(1);s=metric(y,p);s['raw_correct_retention']=float(np.mean((p==y)[rawgood]));s['by_RX']={str(r):metric(y[rx==r],p[rx==r]) for r in d.FIT};s['by_day']={str(v):metric(y[day==v],p[day==v]) for v in (1,2,3)};s['by_TX']={str(t):dict(count=int(np.sum(y==t)),accuracy=float(np.mean(p[y==t]==t))) for t in range(6)}
  if k!='raw':
   scene=k.split('/')[0];rr=records[scene]
   for dim in ('state','lock'):
    v=np.asarray([r[dim] for r in rr]);s['by_'+dim]={str(t):metric(y[v==t],p[v==t]) for t in sorted(set(v.tolist()))}
  wrong=a.copy();wrong[np.arange(len(y)),y]=-np.inf;s['mean_true_margin_over30']=float(np.mean((a[np.arange(len(y)),y]-wrong.max(1))/30));scores[k]=s
 return scores

def decide(logits,y,records,scores):
 y=np.asarray(y);sub,mid,low=d.SCENES;delta=lambda scene:100*(scores[scene+'/MMSE24']['accuracy']-scores[scene+'/noEQ']['accuracy']);ref=scores[low+'/noEQ'];q=scores[low+'/MMSE24'];wrong=np.asarray(logits[low+'/noEQ']).argmax(1)!=y;affected=np.asarray([r['variants']['MMSE24']['relative_IQ_change']>1e-4 for r in records[low]])
 return dict(low_gain=delta(low)>=.5,mid_gain=delta(mid)>=.5,suburban_protection=delta(sub)>=-.25,all_RX_low_protection=all(100*(q['by_RX'][r]['accuracy']-ref['by_RX'][r]['accuracy'])>=-1. for r in q['by_RX']),low_original_errors_affected=bool(wrong.any()) and float(np.mean(affected[wrong]))>=.20)
