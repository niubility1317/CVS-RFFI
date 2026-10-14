"""Plot full scalar source curves and fixed default-head results from saved tables."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1]);v=json.loads((p/'results.json').read_text(encoding='utf-8'));a=v.get('combined_audit',v.get('audit'));c=v.get('combined_analysis',v.get('analysis'))
arms=['OLD_GATE_COS50','SIG_TIME','SIG_TIME_X3','SIG_TIME_MV_COV'];labels=['Old gate+cos','SIG_TIME','3 x SIG_TIME','SIG_TIME + MV + COV'];colors=['#7c8798','#3979b9','#7770a6','#c07348']
fig,axes=plt.subplots(2,2,figsize=(14,9))
for arm,label,color in zip(arms,labels,colors):
 rows=[r for r in a['rows'] if r['arm']==arm];assert rows
 for axis,key,title,ylabel in [(axes[0,0],'train_loss_tx_labeled','Source L loss','CE'),(axes[0,1],'val_tx_acc','Source V health; fixed final epoch','Accuracy (%)')]:
  data=np.array([[e[key] for e in r['source_epoch_curve']] for r in rows]);mean=data.mean(0);axis.plot(range(1,51),mean,label=label,color=color)
  if len(rows)>1:sd=data.std(0,ddof=1);axis.fill_between(range(1,51),mean-sd,mean+sd,color=color,alpha=.1)
  axis.set(title=title,xlabel='Epoch',ylabel=ylabel)
 data=np.array([[100*e['train_pseudo_selected']/e['train_pseudo_total'] for e in r['source_epoch_curve'] if e['epoch']>=34] for r in rows]);mean=data.mean(0);axes[1,0].plot(range(34,51),mean,label=label,color=color)
 if len(rows)>1:sd=data.std(0,ddof=1);axes[1,0].fill_between(range(34,51),mean-sd,mean+sd,color=color,alpha=.1)
for axis in axes[0]:axis.axvline(33.5,color='#9ca3af',ls='--')
axes[1,0].set(title='Hidden-U acceptance; no U truth',xlabel='Epoch',ylabel='Accepted (%)')
for i,arm in enumerate(arms):
 row=next(r for r in c['summary'] if r['arm']==arm);data=np.array(list(row['seed_accuracy'].values()))*100
 axes[1,1].scatter(np.full(len(data),i)+np.linspace(-.06,.06,len(data)),data,color=colors[i],s=45,zorder=3)
 axes[1,1].errorbar(i,data.mean(),yerr=data.std(ddof=1) if len(data)>1 else 0,fmt='D',color=colors[i],capsize=5)
 axes[1,1].text(i,data.mean()+.15,f'{data.mean():.4f}',ha='center',fontsize=9)
axes[1,1].axhline(84,color='#67707d',ls='--');axes[1,1].set(title=f'Clean default head; {len(data)} fixed model seed(s)',ylabel='Accuracy (%)',xticks=range(4),xticklabels=labels);axes[1,1].tick_params(axis='x',rotation=12)
for axis in axes.flat:axis.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('SIG_TIME statistics composition | scratch50 label33+pseudo17 | default jointly trained cosine',fontsize=13)
fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(p/'results.png',dpi=150);plt.close(fig)
