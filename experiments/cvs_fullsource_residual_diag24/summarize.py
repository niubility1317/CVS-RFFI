"""Source labels only. Intervention evidence never selects a target result."""
import csv,json,math
from pathlib import Path
import numpy as np
from . import design as d
def metric(y,p):
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
 return dict(count=len(y),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/len(y)),macro_f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()),confusion=cm.tolist())
def collect(out):
 physics=dict(np.load(out/'observations/physics.npz',allow_pickle=False));data={}
 for arm in d.REFERENCES:
  with np.load(out/arm/'predictions.npz',allow_pickle=False) as f:data[arm]={k:f[k].copy() for k in f.files}
 ref=data['B00'];y=ref['TX'];cohort=ref['clean'].argmax(1)==y;records=[];allmetrics={};quality={}
 for arm,a in data.items():
  if any(not np.array_equal(a[k],ref[k]) for k in ['ids','TX','RX','day']):raise ValueError('Same original V physical panel required')
  if not np.isfinite(a['clean']).all() or not np.isfinite(a['satellite']).all():raise ValueError('Nonfinite source decisions')
  allmetrics[arm]={'clean':metric(y,a['clean'].argmax(1))};quality[arm]={}
  clean_pred=a['clean'].argmax(1)
  for si,scene in enumerate(d.SCENES):
   truth=np.broadcast_to(y,(2,d.N));full=a['satellite'][:,si,0].argmax(-1);baseline_quality=physics['signal_to_added_noise_db'][:,si,0]
   if not np.isfinite(baseline_quality).all():raise ValueError('Baseline added-noise ratio unavailable')
   qbin=np.digitize(baseline_quality,[3,8,15]);quality[arm][scene]=[]
   for ci,condition in enumerate(d.CONDITIONS):
    logits=a['satellite'][:,si,ci];p=logits.argmax(-1);competitor=logits.copy()
    for draw in d.DRAWS:competitor[draw,np.arange(d.N),y]=-np.inf
    margin=(np.take_along_axis(logits,truth[:,:,None],2).squeeze(2)-competitor.max(2))/30
    definitions=[('overall','ALL',np.ones((2,d.N),bool))]
    definitions += [('draw',str(draw),np.broadcast_to(np.arange(2)[:,None]==draw,(2,d.N))) for draw in d.DRAWS]
    definitions += [('RX',str(rx),np.broadcast_to(a['RX']==rx,(2,d.N))) for rx in d.RX]
    definitions += [('state_lock',s+'/'+l,(physics['state'][:,si,0]==sj)&(physics['lock'][:,si,0]==lj)) for sj,s in enumerate(['LOS','PARTIAL','BLOCKED']) for lj,l in enumerate(['locked','degraded','unlocked'])]
    definitions += [('quality',label,qbin==qi) for qi,label in enumerate(['<3','3-8','8-15','>=15'])]
    for dim,stratum,mask in definitions:
     if not mask.any():continue
     yy=truth[mask];pp=p[mask];r=metric(yy,pp);old=full[mask]==yy;now=pp==yy;reference=np.broadcast_to(cohort,(2,d.N))[mask]
     r.update(arm=arm,model_seed=d.SEED,scene=scene,condition=condition,dimension=dim,stratum=stratum,margin_over30=float(margin[mask].mean()),repairs_vs_full=int((now&~old).sum()),breaks_vs_full=int((~now&old).sum()),gain_vs_full_pp=float(100*(now.mean()-old.mean())),B00_clean_correct_count=int(reference.sum()),B00_clean_correct_retention=float(now[reference].mean()) if reference.any() else None,receiver_lock_changes=int(((physics['lock'][:,si,ci]!=physics['lock'][:,si,0])&mask).sum()),AGC_changes=int(((abs(physics['agc_gain_db'][:,si,ci]-physics['agc_gain_db'][:,si,0])>1e-12)&mask).sum()))
     records.append(r)
     if dim=='overall':allmetrics[arm][scene+'/'+condition]=r
     if dim=='quality' and ci==0:quality[arm][scene].append(r)
 del physics
 with (out/'per_row_source_strata.csv').open('x',encoding='utf8',newline='') as f:
  keys=[k for k in records[0] if k!='confusion'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in records])
 criteria={}
 for arm in ['B01','RAW_PRAC']:
  q={};urban=['practical_mid_urban','practical_low_urban']
  for control in ['no_noise','flat_H','no_carrier','no_RX_IQ']:
   q[control+'_fixed_gain5pp_both_urban_both_draws']=all(next(r['gain_vs_full_pp'] for r in records if r['arm']==arm and r['scene']==s and r['condition']==control+'/fixed' and r['dimension']=='draw' and r['stratum']==str(draw))>=5 for s in urban for draw in d.DRAWS)
  q['noise_state_coupled_gap3pp_both_urban']=all(allmetrics[arm][s+'/no_noise/total']['gain_vs_full_pp']-allmetrics[arm][s+'/no_noise/fixed']['gain_vs_full_pp']>=3 for s in urban)
  gradients=d.read(out/arm/'result.json')['gradients'];ug=[r for r in gradients if r['scene'] in urban]
  q['eval_CE_gradient_competition_both_urban_both_draws']=all(r['groups']['whole']['cosine'] is not None and r['groups']['whole']['cosine']<0 and r['groups']['whole']['satellite_norm']>=r['groups']['whole']['clean_norm'] for r in ug)
  criteria[arm]=q
 summary=dict(status='SOURCE_ANALYZED',model_seed_count=1,channel_draws=2,unique_source_V=d.N,source_V_read=27000,metrics=allmetrics,interpretation_checks=criteria,source_strata_records=len(records),B00_clean_correct_count=int(cohort.sum()),training_updates=0,head_fit_updates=0,target_access=False,automatic_training=False,goal_complete=False,limitations=['Only original source V; no target inference or selection','Off-distribution frozen interventions are not retraining causal effects','CE gradients use deterministic eval mode, not native RDrop/SupCon gradients','One model seed, two channel draws are not model-seed replication','Fixed H flattening preserves white-input kernel energy and numerical delay','Total no-added-noise receiver quality set to200dB; original recording noise remains unknown','Quality strata use original full signal-to-added-noise; never total SNR'])
 d.write(out/'summary.json',summary);plots(out,allmetrics,quality)
 return summary
def plots(out,metrics,quality):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 arms=list(d.REFERENCES);fig,axes=plt.subplots(3,3,figsize=(13,9),sharey=True);values=[]
 for ai,arm in enumerate(arms):
  for si,scene in enumerate(d.SCENES):
   ax=axes[ai,si];xx=np.arange(4)
   for off,effect,color in [(-.18,'fixed','#2563eb'),(.18,'total','#ea580c')]:
    v=[metrics[arm][scene+'/'+k+'/'+effect]['gain_vs_full_pp'] for k in ['no_noise','flat_H','no_carrier','no_RX_IQ']];values+=v
    ax.bar(xx+off,v,.36,label=effect,color=color)
   ax.set(xticks=xx,xticklabels=['No added noise','Flat H','No carrier','No RX IQ'],title=arm+' / '+scene.replace('practical_',''));ax.tick_params(axis='x',rotation=35);ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.2)
   if si==0:ax.set_ylabel('Accuracy gain vs full (pp)')
 axes[0,0].legend(frameon=False);axes[0,0].set_ylim(min(-2,min(values)-3),max(values)+5);fig.suptitle('FROZEN SOURCE V interventions; original test IQ and models unchanged');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('source_damage_effects.'+ext),dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
 for si,scene in enumerate(d.SCENES):
  for arm in arms:
   rr=quality[arm][scene];by={r['stratum']:r for r in rr};labels=['<3','3-8','8-15','>=15'];v=[100*by[q]['accuracy'] if q in by else np.nan for q in labels];axes[si].plot(range(4),v,marker='o',label=arm)
  axes[si].set(xticks=range(4),xticklabels=['<3','3-8','8-15','>=15'],xlabel='Full added-noise ratio bin (dB)',title=scene.replace('practical_',''),ylim=(0,102));axes[si].grid(alpha=.2)
 axes[0].set_ylabel('Source default-head accuracy (%)');axes[0].legend(frameon=False);fig.suptitle('Source V only; bin counts and state/lock strata are in CSV');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('source_quality_accuracy.'+ext),dpi=180)
 plt.close(fig)

 fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
 for ai,arm in enumerate(arms):
  rr=d.read(out/arm/'result.json')['gradients'];scenes=list(d.SCENES);keys=sorted(rr[0]['groups'])
  for si,scene in enumerate(scenes):
   vv=[np.mean([r['groups'][k]['cosine'] for r in rr if r['scene']==scene and r['groups'][k]['cosine'] is not None]) if any(r['groups'][k]['cosine'] is not None for r in rr if r['scene']==scene) else np.nan for k in keys]
   axes[ai].plot(range(len(keys)),vv,marker='o',label=scene.replace('practical_',''))
  axes[ai].set(xticks=range(len(keys)),xticklabels=keys,title=arm,ylim=(-1.05,1.05));axes[ai].tick_params(axis='x',rotation=60);axes[ai].axhline(0,color='gray');axes[ai].grid(alpha=.2)
 axes[0].set_ylabel('Clean/satellite CE gradient cosine');axes[0].legend(fontsize=7,frameon=False);fig.suptitle('Readonly eval-mode CE gradients on90 original V packets; not training gradients');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('source_gradient_cosine.'+ext),dpi=180)
 plt.close(fig)
