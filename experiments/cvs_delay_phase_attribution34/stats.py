"""Paired four-cell contrasts; no fitted head or target feedback."""
import csv
import numpy as np
from . import design as d
def contrast(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Four finite paired accuracy cells required')
 gap=b-a
 return dict(original_As_advantage_pp=float(100*gap[0]),B00_delay_gain_pp=float(100*(a[1]-a[0])),B01_delay_gain_pp=float(100*(b[1]-b[0])),As_advantage_reduction_after_delay_pp=float(100*(gap[0]-gap[1])),B00_phase_gain_pp=float(100*(a[2]-a[0])),B01_phase_gain_pp=float(100*(b[2]-b[0])),As_advantage_reduction_after_phase_pp=float(100*(gap[0]-gap[2])),B00_joint_gain_pp=float(100*(a[3]-a[0])),B01_joint_gain_pp=float(100*(b[3]-b[0])),B00_delay_phase_interaction_pp=float(100*(a[3]-a[1]-a[2]+a[0])),B01_delay_phase_interaction_pp=float(100*(b[3]-b[1]-b[2]+b[0])))
def criterion(rows,axis):
 if len(rows)!=4 or axis not in ('delay','phase'):return False
 for r in rows:
  g=r['B00_'+axis+'_gain_pp'];v=r['As_advantage_reduction_after_'+axis+'_pp'];base=r['original_As_advantage_pp']
  if not np.isfinite([g,v,base]).all() or g<5.-1e-10 or v<max(5.,.5*max(base,0))-1e-10:return False
 return True
def collect(out,inputs,logits):
 from experiments.cvs_practical_attribution33.stats import metric,effect
 rows=[];effects=[];contrasts=[];cohort=inputs['B00']['clean'].argmax(1)==inputs['B00']['TX'];y0=inputs['B00']['TX'];rx0=inputs['B00']['RX']
 for draw in (-1,0,1):
  ix=[0,1] if draw==-1 else [draw];y=np.tile(y0,len(ix));rx=np.tile(rx0,len(ix));co=np.tile(cohort,len(ix))
  for si,scene in enumerate(d.SCENES):
   packed={a:logits[a][ix,si].transpose(1,0,2,3).reshape(4,-1,6) for a in d.ARMS}
   for dim,stratum,mask in [('overall','ALL',np.ones(len(y),bool))]+[('RX',str(r),rx==r) for r in (1,3,4,6,8)]:
    accuracies={}
    for arm in d.ARMS:
     accuracies[arm]=[]
     for mi,mode in enumerate(d.MODES):
      meta=dict(arm=arm,model_seed=2026092701,draw=draw,scene=scene,mode=mode,dimension=dim,stratum=stratum)
      m=metric(packed[arm][mi,mask],y[mask],co[mask]);rows.append(dict(**meta,**m));accuracies[arm].append(m['accuracy'])
      if mi:effects.append(dict(**meta,**effect(packed[arm][0,mask],packed[arm][mi,mask],y[mask])))
    contrasts.append(dict(draw=draw,scene=scene,dimension=dim,stratum=stratum,count=int(mask.sum()),**contrast(accuracies['B00'],accuracies['B01'])))
 primary=[r for r in contrasts if r['draw'] in (0,1) and r['dimension']=='overall'];decision=dict(delay_major_contribution=criterion(primary,'delay'),phase_major_contribution=criterion(primary,'phase'),source_only=True,no_automatic_promotion=True,crossseed_confirmed=False,goal_complete=False)
 for name,values in [('metrics.csv',rows),('paired_effects.csv',effects),('contrasts.csv',contrasts)]:
  with (out/name).open('x',encoding='utf8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=[k for k in values[0] if k!='confusion'],extrasaction='ignore');w.writeheader();w.writerows(values)
 result=dict(status='SOURCE_DELAY_PHASE_ANALYZED',metrics=rows,effects=effects,contrasts=contrasts,source_decision=decision,reference_clean_correct_count=int(cohort.sum()),target_access=False,training_updates=0,model_seed_count=1)
 d.write(out/'summary.json',result);plot(out,rows);return result
def plot(out,rows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for si,scene in enumerate(d.SCENES):
  ax=axs[si]
  for ai,arm in enumerate(d.ARMS):
   yy=[next(v['accuracy']*100 for v in rows if v['arm']==arm and v['draw']==-1 and v['scene']==scene and v['mode']==m and v['dimension']=='overall') for m in d.MODES]
   bars=ax.bar(np.arange(4)+(ai-.5)*.34,yy,width=.34,label=arm)
   ax.bar_label(bars,fmt='%.1f',padding=3)
  ax.set_xticks(range(4),['delay24\noriginal phase','delay0\noriginal phase','delay24\nzero phase','delay0\nzero phase']);ax.set_ylim(0,105);ax.set_ylabel('Source accuracy (%)');ax.set_title(scene);ax.legend();ax.grid(axis='y',alpha=.2)
 fig.suptitle('Frozen E30 default heads; flat-kernel counterfactuals; same noise/receiver/AGC')
 for ext in ('png','pdf'):fig.savefig(out/('source_delay_phase.'+ext),dpi=160)
 plt.close(fig)
