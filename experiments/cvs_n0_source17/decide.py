"""Expected single-draw identity risk;not averaged-logit inference."""
import numpy as np
from . import design as d

def metric(y,p):
 cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,p),1);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(count=len(y),accuracy=float(np.trace(cm)/len(y)) if len(y) else None,macro_F1=float(f.mean()) if len(y) else None,confusion=cm.tolist())
def summarize(logits,y,rx,day,records):
 y=np.asarray(y);rx=np.asarray(rx);day=np.asarray(day);scores={};rawgood=np.asarray(logits['raw']).argmax(1)==y
 for view,z in logits.items():
  a=np.asarray(z);p=a.argmax(1);s=metric(y,p);s['raw_correct_retention']=float(np.mean((p==y)[rawgood]));s['by_RX']={str(r):metric(y[rx==r],p[rx==r]) for r in d.FIT};s['by_day']={str(k):metric(y[day==k],p[day==k]) for k in (1,2,3)};s['by_TX']={str(t):dict(count=int(np.sum(y==t)),accuracy=float(np.mean(p[y==t]==t))) for t in range(6)}
  if view!='raw':
   scene,key=view.split('/');rr=records[scene]
   for dim in ('state','lock','nominal_lock'):
    group=np.asarray([r['variants'][key]['lock'] if dim=='lock' else r[dim] for r in rr]);s['by_'+dim]={str(k):metric(y[group==k],p[group==k]) for k in sorted(set(group.tolist()))}
  wrong=a.copy();wrong[np.arange(len(y)),y]=-np.inf;s['mean_true_margin_over30']=float(np.mean((a[np.arange(len(y)),y]-wrong.max(1))/30));scores[view]=s
 for scene in d.SCENES:
  pp=np.concatenate([np.asarray(logits[scene+'/'+v]).argmax(1) for v in ('N0_half','N0_1p5')]);yy=np.tile(y,2);rr=np.tile(rx,2);dd=np.tile(day,2);s=metric(yy,pp);s['unique_physical_L']=6300;s['interpretation']='Expected random single-view law from two equally weighted observations;no logit ensemble';s['by_RX']={str(r):metric(yy[rr==r],pp[rr==r]) for r in d.FIT};s['by_day']={str(k):metric(yy[dd==k],pp[dd==k]) for k in (1,2,3)};scores[scene+'/expected_mixture']=s
 return scores
def decide(logits,y,records,scores):
 sub,mid,low=d.SCENES;delta=lambda s:100*(scores[s+'/expected_mixture']['accuracy']-scores[s+'/nominal']['accuracy'])
 return dict(low_gain=delta(low)>=.5,mid_gain=delta(mid)>=.5,suburban_protection=delta(sub)>=-.25,all_RX_low_protection=all(100*(scores[low+'/expected_mixture']['by_RX'][str(r)]['accuracy']-scores[low+'/nominal']['by_RX'][str(r)]['accuracy'])>=-1. for r in d.FIT))
