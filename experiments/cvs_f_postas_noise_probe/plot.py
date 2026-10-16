"""Offline scientific figures from fixed source decisions only."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import design as d
def plot(root):
 root=Path(root);s=d.read(root/'analysis_summary.json');strata=d.read(root/'source_strata.json');m=d.read(root/'mechanism_strata.json');out=root/'plots';out.mkdir(exist_ok=False)
 if s['status']!='ANALYZED' or not s['source_only'] or s['target_access']:raise ValueError('Complete independent source analysis required')
 colors=['#28588b','#c27424','#32836d'];vl=['clean','full','no added noise / fixed RX'];arms=['RAW10','ANTI'];paths=[]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  fig.suptitle('Seen-source fitL diagnostic; frozen own default heads; seed 2026092701',fontsize=11);fig.tight_layout(rect=(0,0,1,.95))
  for ext in ('png','pdf'):p=out/(name+'.'+ext);fig.savefig(p,dpi=180);paths.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,sel in zip(axes,('all','BLOCKED','unlocked')):
  for i,v in enumerate(d.VIEWS):
   values=[100*(s['rows'][a]['scores'][v] if sel=='all' else s['rows'][a]['recoverability']['strata'][sel]['scores'][v])['accuracy'] for a in arms];bars=ax.bar(np.arange(2)+(i-1)*.24,values,.24,label=vl[i],color=colors[i]);ax.bar_label(bars,fmt='%.1f',fontsize=8)
  ax.set(xticks=range(2),xticklabels=arms,ylim=(0,108),title=sel,ylabel='Accuracy (%)')
 axes[0].legend(fontsize=8);save(fig,'accuracy_and_hard_states')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,dim in zip(axes,('state','lock')):
  keys=['LOS','PARTIAL','BLOCKED'] if dim=='state' else ['locked','degraded','unlocked']
  for i,v in enumerate(d.VIEWS):
   vals=[next(x['accuracy']*100 for x in strata if x['arm']=='ANTI' and x['view']==v and x['dimension']==dim and x['value']==k) for k in keys];ax.bar(np.arange(3)+(i-1)*.24,vals,.24,label=vl[i],color=colors[i])
  ax.set(xticks=range(3),xticklabels=keys,ylim=(0,105),ylabel='ANTI accuracy (%)',title=dim)
 axes[0].legend(fontsize=8);save(fig,'ANTI_state_and_lock')
 overall={(p['arm'],p['view']):p for p in m if p['dimension']=='all'}
 fig,axes=plt.subplots(2,2,figsize=(13,8))
 for arm,mark in zip(arms,('o','s')):
  axes[0,0].plot(range(3),[overall[arm,v]['mean_margin_over30'] for v in d.VIEWS],marker=mark,label=arm)
  axes[0,1].plot(range(3),[overall[arm,v]['mean_CE'] for v in d.VIEWS],marker=mark,label=arm)
 for i,v in enumerate(d.VIEWS):
  axes[1,0].plot(range(7),[overall['ANTI',v]['feature_cosine_to_clean'][k] for k in FEATURES],marker='o',color=colors[i],label=vl[i]);axes[1,1].plot(range(7),[overall['ANTI',v]['feature_gradient_L2'][k]['mean'] for k in FEATURES],marker='o',color=colors[i],label=vl[i])
 for ax in axes[0]:ax.set_xticks(range(3),vl);ax.legend(fontsize=8)
 axes[0,0].set_title('Mean default-head margin /30');axes[0,1].set_title('Mean unsmoothed CE')
 for ax in axes[1]:ax.set_xticks(range(7),FEATURES,rotation=35,ha='right');ax.legend(fontsize=8)
 axes[1,0].set_title('ANTI feature cosine to own clean');axes[1,1].set_title('ANTI intermediate CE derivative L2');axes[1,1].set_yscale('log');save(fig,'default_head_and_existing_branches')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,gate in zip(axes,('frequency','PA')):
  for arm,mark in zip(arms,('o','s')):ax.plot(range(3),[overall[arm,v]['gate'][gate]['mean'] for v in d.VIEWS],marker=mark,label=arm)
  ax.set(xticks=range(3),xticklabels=vl,title=gate+' existing sigmoid gate',ylabel='Mean gate activation');ax.legend()
 save(fig,'existing_gates');d.write(root/'plot_artifacts.json',dict(paths=paths,IQ_or_checkpoints_read=False,new_NN=0,source_only=True));return paths
FEATURES=('time_projection','frequency_projection','reference_projection','SIG_TIME','base_identity','PA_physical','joint160')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(plot(a.input)))
