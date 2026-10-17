"""Offline scientific plots from complete fixed two-row source evidence."""
import csv,json
from pathlib import Path
from .analyze_e20 import OUT,ARMS,VIEWS

def draw():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 s=json.loads((OUT/'analysis_summary.json').read_text(encoding='utf8'))
 if s['status']!='ANALYZED' or s['actual_execution']!='VERIFIED' or not s['source_only'] or s['target_access'] or set(s['scores'])!=set(ARMS):raise ValueError('Complete independent source recount first')
 out=OUT/'figures'
 if out.exists():raise FileExistsError('Preserve existing figures')
 out.mkdir();paths=[];colors=('#666666','#3475a5');plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
 def save(fig,name):
  fig.tight_layout()
  for ext in ('png','pdf'):
   p=out/(name+'.'+ext);fig.savefig(p,dpi=160,bbox_inches='tight');paths.append(p.name)
  plt.close(fig)
 labels=('Clean','High suburban','Mid suburban','Low suburban','High urban','Mid urban','Low urban');x=list(range(7))
 fig,ax=plt.subplots(figsize=(12,4.5))
 for a,color in zip(ARMS,colors):ax.plot(x,[100*s['scores'][a][v]['accuracy'] for v in VIEWS],marker='o',label=a,color=color)
 ax.set_xticks(x,labels,rotation=15);ax.set_ylabel('Frozen default-head accuracy (%)');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.legend(ncol=3);ax.set_title('Held source RX1 / L1260 / fixed final E20 / one seed; no formal target');save(fig,'source_accuracy_all_views')
 fig,ax=plt.subplots(figsize=(8,4.3));full=[s['paired'][v]['all']['net_correct'] for v in VIEWS[4:]];common=[s['paired'][v]['both_clean_correct']['net_correct'] for v in VIEWS[4:]]
 ax.bar([i-.17 for i in range(3)],full,.32,label='All 1260 source packets',color='#3475a5');ax.bar([i+.17 for i in range(3)],common,.32,label='Both models clean-correct',color='#c66b31');ax.axhline(0,color='#444',lw=.7);ax.set_xticks(range(3),('High urban','Mid urban','Low urban'));ax.set_ylabel('Repair minus newly broken decisions (packets)');ax.set_title('O1 six-update L reorder versus matched O0');ax.legend(fontsize=8);save(fig,'urban_paired_identity_errors')
 with (OUT/'training_curves.csv').open(encoding='utf8',newline='') as stream:curves=list(csv.DictReader(stream))
 fig,axes=plt.subplots(1,3,figsize=(15,4.2))
 for a,color in zip(ARMS,colors):
  rows=[r for r in curves if r['arm']==a];assert len(rows)==20;epochs=[int(r['epoch']) for r in rows]
  axes[0].plot(epochs,[float(r['base_total_L']) for r in rows],label=a,color=color)
  axes[1].plot(epochs,[float(r['extra_pair_CE']) for r in rows],label=a+' CE',color=color);axes[1].plot(epochs,[float(r['extra_pair_KL']) for r in rows],ls='--',label=a+' KL applied',color=color)
  axes[2].plot(epochs,[float(r['val_tx_acc']) for r in rows],label=a,color=color)
 for ax in axes:ax.axvline(13.5,color='#aaa',ls=':');ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
 axes[0].set_ylabel('Original full base L objective');axes[1].set_ylabel('Extra-pair mean CE / symmetric KL');axes[2].set_ylabel('Native seen-source V accuracy (%)');fig.suptitle('Complete E20 curves; original U begins at E14; identical original CE / RDrop / SupCon in both rows');save(fig,'complete_training_curves')
 fig,axes=plt.subplots(1,2,figsize=(11,4.2));xx=list(range(3))
 for a,color,offset in zip(ARMS,colors,(-.16,.16)):
  axes[0].bar([i+offset for i in xx],[100*s['scores'][a][v]['accuracy'] for v in VIEWS[1:4]],.3,label=a,color=color)
  axes[1].bar([i+offset for i in xx],[s['suburban_gaps'][a][v]['clean_minus_suburban_pp'] for v in VIEWS[1:4]],.23,label=a,color=color)
 for ax in axes:ax.set_xticks(xx,('High','Mid','Low'));ax.set_xlabel('Suburban scenario');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Absolute accuracy (%)');axes[0].set_ylim(0,100);axes[1].set_ylabel('Own clean minus suburban (pp)');axes[1].axhline(2,ls='--',color='#888',label='Exploratory 2 pp goal');save(fig,'suburban_absolute_and_own_clean_gap')
 return dict(status='FIGURES_SAVED',files=paths,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(draw()))
