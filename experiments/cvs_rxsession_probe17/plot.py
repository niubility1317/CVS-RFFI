"""Offline scientific figures from completed independent source aggregates only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
BANKS=('original','bank1','bank2','bank3');SCENES=('practical_mid_urban','practical_low_urban');COLORS=('#555555','#4477AA','#228833','#EE7733')
def draw(root):
 root=Path(root);s=json.loads((root/'analysis_summary.json').read_text());z=json.loads((root/'source_strata.json').read_text())
 if s['execution']!='VERIFIED' or s['predictions']!=45360 or not s['source_only'] or s['target_access']:raise ValueError('Completed independent full source readout first')
 out=root/'figures';out.mkdir(exist_ok=False);plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True});paths=[]
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,dpi=160,bbox_inches='tight');paths.append(p)
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.2))
 for ax,sc,title in zip(axes,SCENES,('Mid urban','Low urban')):
  vals=[100*s['groups'][sc]['banks'][b]['accuracy'] for b in BANKS];ax.bar(BANKS,vals,color=COLORS)
  for j,v in enumerate(vals):ax.text(j,v+.12,f'{v:.4f}',ha='center',fontsize=10)
  ax.set_ylim(0,max(vals)*1.12);ax.set_ylabel('Frozen default-head accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Parent-exposed fit L5040; one frozen model; RX banks are not model seeds',y=1.03);save(fig,'frozen_session_accuracy')
 fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(3)
 for sc,offset,title in zip(SCENES,(-.18,.18),('Mid urban','Low urban')):
  g=s['groups'][sc]['banks'];d=[100*(g[b]['accuracy']-g['original']['accuracy']) for b in BANKS[1:]];f=[100*g[b]['disagreement_vs_original'] for b in BANKS[1:]];iq=[100*s['input_relative_RMS'][sc][b]['median'] for b in BANKS[1:]];net=[next(v['net'] for v in s['paired'] if v['scene']==sc and v['bank']==b) for b in BANKS[1:]]
  for ax,v in zip(axes.flat,(d,f,iq,net)):ax.bar(x+offset,v,.35,label=title)
 for ax in axes.flat:ax.set_xticks(x,BANKS[1:]);ax.grid(axis='y',alpha=.2);ax.axhline(0,color='black',linewidth=.8)
 axes[0,0].set_ylabel('Accuracy change from original (pp)');axes[0,1].set_ylabel('Hard prediction disagreement (%)');axes[1,0].set_ylabel('Median relative IQ RMS change (%)');axes[1,1].set_ylabel('Repairs minus breaks (packets)');axes[0,0].legend()
 fig.suptitle('Same H/underlying noise draw; only injected RX session differs');save(fig,'session_perturbation_and_decisions')
 fig,axes=plt.subplots(1,3,figsize=(16,4.8));sc=SCENES[-1]
 for ax,dim,values in zip(axes,('state','lock','quality'),(('LOS','PARTIAL','BLOCKED'),('locked','degraded','unlocked'),('<0','[0,5)','[5,10)','[10,20)','>=20'))):
  values=[v for v in values if any(q['scene']==sc and q['dimension']==dim and q['value']==v for q in z)];x=np.arange(len(values));counts=[]
  for i,b in enumerate(BANKS):
   q=[next(q for q in z if q['scene']==sc and q['dimension']==dim and q['value']==v and q['bank']==b) for v in values];counts=[q['count'] for q in q];ax.bar(x+(i-1.5)*.2,[100*q['accuracy'] for q in q],.19,color=COLORS[i],label=b)
  ax.set_xticks(x,[v+'\nn='+str(n) for v,n in zip(values,counts)],rotation=20 if dim=='quality' else 0);ax.set_ylim(0,105);ax.set_title(dim);ax.set_ylabel('Low-urban fit accuracy (%)');ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=9,ncol=2);fig.suptitle('Shared physical states and quality strata; frozen trained-fit source, not new training gain');save(fig,'low_urban_session_strata')
 (out/'manifest.json').write_text(json.dumps(dict(status='VERIFIED',source_only=True,new_NN=0,figures=[p.name for p in paths]),indent=2)+'\n');return paths
