"""Paired source effects: intervention benefit before versus after As training."""
import csv,json
import numpy as np
from . import design as d
def metric(z,y,cohort):
 z=np.asarray(z);y=np.asarray(y);cohort=np.asarray(cohort,bool)
 if z.shape!=(len(y),6) or cohort.shape!=y.shape or y.ndim!=1 or not len(y) or not np.isfinite(z).all() or not ((y>=0)&(y<6)).all():raise ValueError('Finite6class paired default logits required')
 p=z.argmax(-1);cm=np.bincount(y*6+p,minlength=36).reshape(6,6);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(6,float),where=den>0);other=z.copy()/30;other[np.arange(len(y)),y]=-np.inf;margin=z[np.arange(len(y)),y]/30-other.max(1)
 return dict(count=len(y),correct=int((p==y).sum()),accuracy=float((p==y).mean()),macro_f1=float(f.mean()),confusion=cm.tolist(),mean_margin_over30=float(margin.mean()),reference_clean_count=int(cohort.sum()),reference_clean_retention=float((p[cohort]==y[cohort]).mean()) if cohort.any() else None)
def effect(full,controlled,y):
 a=np.asarray(full).argmax(-1)==y;b=np.asarray(controlled).argmax(-1)==y
 if a.shape!=b.shape or a.shape!=np.asarray(y).shape:raise ValueError('Within-packet paired intervention required')
 return dict(removal_delta_pp=float(100*(b.mean()-a.mean())),repairs=int((~a&b).sum()),breaks=int((a&~b).sum()),unchanged=int((a==b).sum()))
def stable_reduction(values):return len(values)==2 and bool(np.isfinite(values).all()) and min(values)>=1.-1e-10
def masks(rx,physics):
 out=[('overall','ALL',np.ones(len(rx),bool))]
 out += [('RX',str(r),rx==r) for r in (1,3,4,6,8)]
 for si,s in enumerate(('LOS','PARTIAL','BLOCKED')):
  for li,l in enumerate(('locked','degraded','unlocked')):out.append(('state_lock',s+'/'+l,(physics['state']==si)&(physics['lock']==li)))
 ratio=physics['signal_to_added_noise_db']
 for lo,hi,name in [(-np.inf,0,'<0'),(0,3,'0:3'),(3,5,'3:5'),(5,10,'5:10'),(10,15,'10:15'),(15,np.inf,'>=15')]:out.append(('added_noise_db',name,(ratio>=lo)&(ratio<hi)))
 return [(a,b,c) for a,b,c in out if c.any()]
def collect(out,inputs,logits,physics):
 metrics=[];effects=[];reference=inputs['B00']['clean'].argmax(-1)==inputs['B00']['TX']
 for arm in d.ARMS:
  inp=inputs[arm];y0=inp['TX'];rx0=inp['RX'];z=logits[arm]
  for draw in (-1,0,1):
   draws=range(2) if draw==-1 else [draw]
   yy=np.tile(y0,len(draws));rx=np.tile(rx0,len(draws));co=np.tile(reference,len(draws))
   for si,scene in enumerate(d.SCENES):
    zz=z[list(draws),si].transpose(1,0,2,3).reshape(11,-1,6)
    ph={k:physics[k][list(draws),si,0].reshape(-1) for k in ('state','lock','signal_to_added_noise_db')}
    for dim,stratum,mask in masks(rx,ph):
     for ci,condition in enumerate(d.CONDITIONS):
      meta=dict(arm=arm,model_seed=2026092701,draw=draw,scene=scene,condition=condition,dimension=dim,stratum=stratum)
      metrics.append(dict(**meta,**metric(zz[ci,mask],yy[mask],co[mask])))
      if ci:effects.append(dict(**meta,count=int(mask.sum()),**effect(zz[0,mask],zz[ci,mask],yy[mask])))
 lookup={(v['arm'],v['draw'],v['scene'],v['condition'],v['dimension'],v['stratum']):v for v in effects};attenuation=[]
 for v in effects:
  if v['arm']!='B00':continue
  key=('B01',v['draw'],v['scene'],v['condition'],v['dimension'],v['stratum']);w=lookup[key]
  attenuation.append(dict(draw=v['draw'],scene=v['scene'],condition=v['condition'],dimension=v['dimension'],stratum=v['stratum'],count=v['count'],B00_removal_delta_pp=v['removal_delta_pp'],B01_removal_delta_pp=w['removal_delta_pp'],sensitivity_reduction_pp=v['removal_delta_pp']-w['removal_delta_pp']))
 fixed=[]
 for scene in d.SCENES:
  for condition in d.CONDITIONS[1:]:
   values=[next(v['sensitivity_reduction_pp'] for v in attenuation if v['scene']==scene and v['condition']==condition and v['dimension']=='overall' and v['draw']==draw) for draw in (0,1)]
   fixed.append(dict(scene=scene,condition=condition,reduction_each_draw_pp=values,at_least_1pp_both_draws=stable_reduction(values),claim='source-only descriptive;not crossseed/not promotion'))
 def savecsv(name,rows):
  keys=[k for k in rows[0] if k!='confusion']
  with (out/name).open('x',encoding='utf8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 savecsv('condition_metrics.csv',metrics);savecsv('paired_effects.csv',effects);savecsv('sensitivity_reduction.csv',attenuation)
 result=dict(status='SOURCE_ATTRIBUTION_ANALYZED',metrics=metrics,effects=effects,sensitivity_reduction=attenuation,descriptive_criteria=fixed,reference_clean_correct_count=int(reference.sum()),target_access=False,formal_scores_read=False,NN_analysis=0,training_updates=0,model_seed_count=1,crossseed_confirmed=False,goal_complete=False)
 d.write(out/'summary.json',result);plot(out,metrics,attenuation);return result
def plot(out,metrics,attenuation):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=list(d.CONDITIONS[1:]);fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
 for si,scene in enumerate(d.SCENES):
  ax=axes[0,si]
  for arm in d.ARMS:
   yy=[next(v['accuracy']*100 for v in metrics if v['arm']==arm and v['draw']==-1 and v['scene']==scene and v['condition']==c and v['dimension']=='overall') for c in d.CONDITIONS]
   ax.plot(range(11),yy,marker='o',label=arm)
  ax.set_xticks(range(11),['full']+names,rotation=70,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Source accuracy (%)');ax.set_title(scene);ax.legend();ax.grid(alpha=.2)
  ax=axes[1,si]
  for draw,offset in [(0,-.18),(1,.18)]:
   yy=[next(v['sensitivity_reduction_pp'] for v in attenuation if v['draw']==draw and v['scene']==scene and v['condition']==c and v['dimension']=='overall') for c in names]
   ax.bar(np.arange(10)+offset,yy,width=.36,label='channel draw '+str(draw))
  ax.axhline(0,color='black',lw=.8);ax.axhline(1,color='grey',ls='--',lw=.8);ax.set_xticks(range(10),names,rotation=70,ha='right');ax.set_ylabel('Sensitivity reduction B00 - B01 (pp)');ax.legend();ax.grid(axis='y',alpha=.2)
 fig.suptitle('Frozen matched E30 default heads; source V1800; identical cached interventions—no training or target access')
 for ext in ('png','pdf'):fig.savefig(out/('source_impairment_effects.'+ext),dpi=160)
 plt.close(fig)
