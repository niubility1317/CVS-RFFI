"""Static source-only figures from independently recomputed full outputs."""
import argparse,csv,json
from pathlib import Path
from .analyze import ARMS,VIEWS,read
SHORT=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban')
COLORS={'RAW10':'#4A7A5C','INDEP':'#3D6D9B','ANTI':'#C86A38'}
def draw(root,output):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 r=read(root/'analysis_summary.json');m=read(root/'mechanism_analysis.json');assert r['actual_execution']=='VERIFIED' and r['source_only'] and not r['target_access']
 with (root/'training_curves.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
 with (root/'source_strata.csv').open(encoding='utf8',newline='') as f:tx=list(csv.DictReader(f))
 names=[]
 def save(fig,name):
  fig.suptitle('Held source RX1 | seed '+str(r['model_seed'])+' | scratch E20 | default cosine head',fontsize=10)
  for ext in ('png','pdf'):fig.savefig(output/(name+'.'+ext),dpi=180)
  plt.close(fig);names.append(name+'.png')
 fig,axes=plt.subplots(1,2,figsize=(13.5,4.8),constrained_layout=True)
 for j,a in enumerate(ARMS):
  bars=axes[0].bar(np.arange(7)+(j-1)*.25,[100*r['scores'][a][v]['accuracy'] for v in VIEWS],width=.24,color=COLORS[a],label=a);axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=7,rotation=90)
 for j,ref in enumerate(ARMS[:2]):
  bars=axes[1].bar(np.arange(7)+(j-.5)*.34,[r['paired'][ref][v]['accuracy_gain_pp'] for v in VIEWS],width=.32,color=COLORS[ref],label='ANTI - '+ref);axes[1].bar_label(bars,fmt='%+.1f',padding=3,fontsize=8)
 for ax in axes:ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.legend(fontsize=8);ax.axhline(0,color='grey',lw=.6)
 axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[1].set_ylabel('Paired gain (percentage points)');save(fig,'source_accuracy')
 fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
 for a in ARMS:
  rows=[x for x in curves if x['arm']==a];e=[int(x['epoch']) for x in rows]
  for ax,key,title in zip(axes.flat,('base_total_L','extra_pair_loss','val_tx_acc','raw_CE','ordinary_CE','supcon'),('Common L objective','Mean of two extra same-IQ RDrop losses','Seen-source V accuracy (%)','Common raw CE','Common ordinary CE','Ordinary-view SupCon')):
   ax.plot(e,[float(x[key]) for x in rows],color=COLORS[a],label=a);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.axvline(13.5,color='grey',ls=':',lw=.7)
 for ax in axes.flat:ax.legend(fontsize=7);ax.set_xticks([1,5,10,13,15,20])
 save(fig,'training_curves')
 fig,axes=plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
 for row,ref in enumerate(ARMS[:2]):
  for ax,dim,order in zip(axes[row],('state','lock','quality_snr_db'),(['LOS','PARTIAL','BLOCKED'],['locked','degraded','unlocked'],['-inf:5','5:15','15:inf'])):
   cells=[x for x in m['strata'] if x['dimension']==dim and x['view'] in VIEWS[4:]];labels=[];vals=[]
   for view in VIEWS[4:]:
    for value in order:
     c=next((x for x in cells if x['view']==view and x['value']==value),None)
     if c is None:continue
     labels.append(SHORT[VIEWS.index(view)].replace(' urban','')+' / '+value+' (n='+str(c['count'])+')');vals.append(c['paired'][ref]['accuracy_gain_pp'])
   pos=np.arange(len(vals));ax.barh(pos,vals,color=[COLORS['ANTI'] if x>=0 else '#A84966' for x in vals]);ax.axvline(0,color='grey',lw=.7);ax.set_yticks(pos,labels,fontsize=7);ax.invert_yaxis();ax.set_title(dim);ax.set_xlabel('ANTI - '+ref+' (pp)')
 save(fig,'urban_strata')
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for a in ARMS:
  s=m['margins_and_retention'][a];axes[0].plot(range(7),[100*s[v]['fixed_raw_clean_correct_retention'] for v in VIEWS],marker='o',color=COLORS[a],label=a);axes[1].plot(range(7),[s[v]['scaled_margin_fixed']['mean'] for v in VIEWS],marker='o',color=COLORS[a],label=a)
 for ax in axes:ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axes[0].set_ylabel('Accuracy on fixed RAW10-clean-correct set (%)');axes[0].set_title('Fixed packet set, n='+str(m['fixed_raw_clean_correct_count']));axes[0].set_ylim(0,101);axes[1].set_title('Default head margin on the same fixed set');axes[1].set_ylabel('(True - best other logit) / 30');axes[1].axhline(0,color='grey',ls=':',lw=.8);save(fig,'fixed_clean_retention')
 fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
 for row,ref in enumerate(ARMS[:2]):
  for ax,dim,values in zip(axes[row],('TX','day'),(list(range(6)),[1,2,3])):
   arr=np.array([[float(next(x for x in tx if x['reference']==ref and x['view']==v and x['dimension']==dim and int(x['value'])==t)['gain_pp']) for v in VIEWS] for t in values]);bound=max(1,float(abs(arr).max()));im=ax.imshow(arr,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto');ax.set_xticks(range(7),SHORT,rotation=35,ha='right');ax.set_yticks(range(len(values)),[dim+' '+str(v) for v in values]);ax.set_title('ANTI - '+ref+': all '+dim+' groups')
   for i in range(len(values)):
    for j in range(7):ax.text(j,i,f'{arr[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='black' if abs(arr[i,j])<.65*bound else 'white')
   fig.colorbar(im,ax=ax,label='Percentage points',shrink=.8)
 save(fig,'error_redistribution')
 fig,axes=plt.subplots(2,2,figsize=(15,9),constrained_layout=True);fits=(3,4,6,8);cols=[(v,rx) for v in VIEWS[1:] for rx in fits];labels=[SHORT[VIEWS.index(v)].replace(' suburban','S').replace(' urban','U')+'/R'+str(rx) for v,rx in cols]
 for ax,a in zip(axes[0],ARMS[1:]):
  cov=read(root/(a+'-RX1-s'+str(r['model_seed']))/'source/satellite_supervision_coverage.json');cells=cov['cells'];arr=np.array([[sum(c['count'] for c in cells if c['TX']==t and c['RX']==rx and c['scene']==v) for v,rx in cols] for t in range(6)]);assert int(arr.sum())==1136640
  im=ax.imshow(arr,cmap='Blues',aspect='auto');ax.set_xticks(range(len(cols)),labels,rotation=90,fontsize=6);ax.set_yticks(range(6),['TX'+str(t) for t in range(6)]);ax.set_title(a+': actual TX x fit RX x scene observation counts');fig.colorbar(im,ax=ax,shrink=.65)
 for a in ARMS:
  rows=[x for x in curves if x['arm']==a];e=[int(x['epoch']) for x in rows];axes[1,0].plot(e,[float(x['extra_pair_CE']) for x in rows],color=COLORS[a],label=a)
  if a!='RAW10':axes[1,1].plot(e,[float(x['noise_view_pair_MSE']) for x in rows],color=COLORS[a],label=a)
 axes[1,0].set_title('Mean extra-view CE, all 4440 updates / arm');axes[1,0].set_ylabel('Label-smoothed CE');axes[1,1].set_title('Observed two-view IQ distance after receiver/AGC');axes[1,1].set_ylabel('Mean squared IQ difference')
 for ax in axes[1]:ax.set_xlabel('Epoch');ax.set_xticks([1,5,10,13,15,20]);ax.grid(alpha=.2);ax.legend(fontsize=8)
 save(fig,'noise_training_and_coverage')
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
 for a in ARMS:
  rows=[x for x in curves if x['arm']==a];e=[int(x['epoch']) for x in rows]
  axes[0].plot(e,[float(x['grad_before_clip']) for x in rows],color=COLORS[a],label=a);axes[1].plot(e,[float(x['grad_after_clip']) for x in rows],color=COLORS[a],label=a);axes[2].plot(e,[100*float(x['grad_pre_exceeds_limit']) for x in rows],color=COLORS[a],label=a)
 for ax,title in zip(axes,('Mean whole-model norm before clipping','Mean native total norm after clipping','Updates with pre-norm above original limit')):
  ax.set_title(title);ax.set_xlabel('Epoch');ax.set_xticks([1,5,10,13,15,20]);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.axvline(13.5,color='grey',ls=':',lw=.7)
 axes[0].set_ylabel('Gradient L2 norm');axes[1].set_ylabel('Gradient L2 norm');axes[2].set_ylabel('Updates (%)');save(fig,'gradient_health');return names
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(draw(a.input,a.output)))
