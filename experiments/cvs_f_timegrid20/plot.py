"""Offline full source figures;no model/IQ/checkpoint imports."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=('clean','high sub','mid sub','low sub','high urban','mid urban','low urban')
def draw(root):
 root=Path(root);s=json.loads((root/'analysis_summary.json').read_text());curves=json.loads((root/'training_curves.json').read_text());strata=json.loads((root/'source_strata.json').read_text());out=root/'figures';out.mkdir(exist_ok=False);artifacts=[]
 if s['status']!='ANALYZED' or s['target_access']:raise ValueError('Complete independent source analysis required')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});colors=['#28588b','#c27424'];arms=('D_raw','TIMEGRID')
 def save(fig,name):
  fig.suptitle('Source held RX1 / seed 2026092701 / own default cosine head / scratch E20',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
  for ext in ('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=180);artifacts.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(14,4))
 for ax,k,label in zip(axes,('accuracy','macro_F1'),('Accuracy (%)','Macro F1 (%)')):
  for i,a in enumerate(arms):ax.bar(np.arange(7)+(i-.5)*.34,[s['scores'][a][v][k]*100 for v in VIEWS],.34,label=a,color=colors[i])
  ax.set_xticks(range(7),LABELS,rotation=25,ha='right');ax.set_ylabel(label);ax.set_ylim(0,105);ax.legend()
 save(fig,'source_scene_results')
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for a,color in zip(arms,colors):
  r=[x for x in curves if x['arm']==a];e=[x['epoch'] for x in r]
  for ax,key,title in zip(axes.flat,('base_L','extra_CE','extra_KL','seenV_accuracy','grad_before_clip','lr'),('Shared base objective','Fourth pair CE','Fourth pair RDrop KL','Seen RX V health (%)','Native preclip gradient norm','Learning rate')):
   ax.plot(e,[x[key] for x in r],label=a,color=color);ax.set_title(title);ax.set_xlabel('Epoch');ax.axvline(13.5,color='#999999',ls=':',lw=.8)
 for ax in axes.flat:ax.legend(fontsize=8)
 save(fig,'complete_training_curves')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,dim,keys in zip(axes,('state','lock'),(('LOS','PARTIAL','BLOCKED'),('locked','degraded','unlocked'))):
  for i,a in enumerate(arms):
   val=[next(x['accuracy']*100 for x in strata if x['arm']==a and x['scene']=='practical_low_urban' and x['dimension']==dim and x['value']==v) for v in keys];ax.bar(np.arange(3)+(i-.5)*.34,val,.34,label=a,color=colors[i])
  ax.set_xticks(range(3),keys);ax.set_ylim(0,105);ax.set_ylabel('Low urban accuracy (%)');ax.legend()
 save(fig,'low_urban_strata')
 fig,axes=plt.subplots(1,2,figsize=(14,4))
 for i,a in enumerate(arms):
  axes[0].bar(np.arange(7)+(i-.5)*.34,[s['retention_on_fixed_D_raw_clean_correct'][v][a]['accuracy']*100 for v in VIEWS],.34,label=a,color=colors[i])
 axes[0].set_xticks(range(7),LABELS,rotation=25,ha='right');axes[0].set_ylabel('Accuracy on fixed D_raw clean-correct (%)');axes[0].legend()
 axes[1].bar(np.arange(7)-.18,[x['repairs'] for x in s['paired']],.36,label='repairs',color='#32836d');axes[1].bar(np.arange(7)+.18,[-x['breaks'] for x in s['paired']],.36,label='new errors',color='#b95454');axes[1].axhline(0,color='black',lw=.7);axes[1].set_xticks(range(7),LABELS,rotation=25,ha='right');axes[1].set_ylabel('Paired decisions relative to D_raw');axes[1].legend();save(fig,'fixed_clean_retention_and_errors')
 fig,axes=plt.subplots(1,2,figsize=(12,4));r=[x for x in curves if x['arm']=='TIMEGRID'];e=[x['epoch'] for x in r]
 axes[0].plot(e,[x['grid_relative_L2_mean']*100 for x in r],label='epoch mean');axes[0].plot(e,[x['grid_relative_L2_max']*100 for x in r],label='epoch max');axes[0].axhline(100*255/4096,color='#b95454',ls='--',label='universal bound');axes[0].set(xlabel='Epoch',ylabel='Relative L2 change (%)');axes[0].legend()
 axes[1].plot(e,[x['pseudo_selected'] for x in r],label='selected');axes[1].plot(e,[x['pseudo_total'] for x in r],label='observed');axes[1].set(xlabel='Epoch',ylabel='Original U observations/epoch');axes[1].legend();save(fig,'grid_exposure_and_original_U')
 (root/'plot_artifacts.json').write_text(json.dumps(dict(paths=artifacts,source_only=True,new_NN=0,IQ_or_CP_reads=0),indent=2)+'\n');return artifacts
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(draw(a.input)))
