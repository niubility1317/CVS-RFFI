"""Offline scientific figures from the complete fixed source matrix only."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .analyze import OUT,ARMS,VIEWS
COLORS=('#555555','#4477AA','#EE7733','#228833')
def draw():
 s=json.loads((OUT/'analysis_summary.json').read_text(encoding='utf8'))
 if s['status']!='ANALYZED' or s['actual_execution']!='VERIFIED' or not s['source_only'] or s['target_access'] or set(s['scores'])!=set(ARMS):raise ValueError('Complete independent four-row source analysis first')
 out=OUT/'figures'
 if out.exists():raise FileExistsError('Preserve existing figures;no silent replacement')
 out.mkdir();paths=[];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,dpi=160,bbox_inches='tight');paths.append(p.name)
  plt.close(fig)
 labels=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban');x=np.arange(7)
 fig,ax=plt.subplots(figsize=(12,4.5))
 for a,color in zip(ARMS,COLORS):ax.plot(x,[100*s['scores'][a][v]['accuracy'] for v in VIEWS],marker='o',label=a,color=color)
 ax.set_xticks(x,labels,rotation=15);ax.set_ylabel('Frozen default-head accuracy (%)');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.legend(ncol=4);ax.set_title('Held source RX1 / L1260 / one model seed / final E20; formal target untouched');save(fig,'source_accuracy_all_views')
 fig,axes=plt.subplots(1,3,figsize=(15,4));x=np.arange(3)
 for ax,ref in zip(axes,ARMS[:3]):
  full=[s['paired'][ref][v]['repaired']-s['paired'][ref][v]['broken'] for v in VIEWS[4:]];common=[s['paired'][ref][v]['clean_conditioned']['both_clean_correct']['net_correct'] for v in VIEWS[4:]];counts=[s['paired'][ref][v]['clean_conditioned']['both_clean_correct']['count'] for v in VIEWS[4:]]
  ax.bar(x-.19,full,.36,label='All 1260');ax.bar(x+.19,common,.36,label='Both clean correct');ax.set_xticks(x,('High urban','Mid urban','Low urban'),rotation=15);ax.axhline(0,color='black',linewidth=.8);ax.grid(axis='y',alpha=.2);ax.set_title('Q_mix minus '+ref+'; common clean n='+str(counts[0]));ax.set_ylabel('Repaired minus broken (packets)')
 axes[0].legend(fontsize=9);fig.suptitle('Paired source identity decisions; descriptive subsets do not alter qualification');save(fig,'paired_urban_identity_errors')
 with (OUT/'training_curves.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=80:raise ValueError('All four complete E20 trajectories required')
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for a,color in zip(ARMS,COLORS):
  r=[v for v in rows if v['arm']==a]
  if [int(v['epoch']) for v in r]!=list(range(1,21)):raise ValueError('No truncated epoch curves')
  for ax,key in zip(axes.flat,('base_total_L','extra_pair_loss','supcon','val_tx_acc')):ax.plot(range(1,21),[float(v[key]) for v in r],label=a,color=color)
 for ax,key,title in zip(axes.flat,('base_total_L','extra_pair_loss','supcon','val_tx_acc'),('Common base objective','Extra pair CE + KL','Existing SupCon (unweighted)','Seen-source V accuracy (%)')):ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
 axes[0,0].legend(ncol=2);fig.suptitle('Full training trajectories; loss and seen-V health are not held-RX identity gains');save(fig,'full_training_curves')
 (out/'manifest.json').write_text(json.dumps(dict(status='VERIFIED',figures=paths,source_only=True,target_access=False,new_NN=0,new_updates=0),indent=2)+'\n',encoding='utf8');return paths
if __name__=='__main__':print(json.dumps(draw()))
