import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path(sys.argv[1]);v=json.loads((p/'results.json').read_text(encoding='utf-8'));a=v['combined_audit'];m=v['combined_source_style'];c=v['combined_analysis']
fig,ax=plt.subplots(2,3,figsize=(16,9));colors=['#7c8798','#3979b9','#c07348'];arms=['OLD_GATE_COS50','MATCHED_CONTROL','IDENTITY_MIXSTYLE'];labels=['Old gate+cos','Matched control','Identity MixStyle']
for arm,label,color in zip(arms,labels,colors):
 rows=[r for r in a['rows'] if r['arm']==arm];assert len(rows)==3
 for axis,key,title,ylabel in [(ax[0,0],'train_loss_tx_labeled','Source L loss; training views differ','CE'),(ax[0,1],'val_tx_acc','Source V health; no selection','Accuracy (%)')]:
  x=np.arange(1,51);values=np.array([[e[key] for e in r['source_epoch_curve']] for r in rows]);mean=values.mean(0);sd=values.std(0,ddof=1)
  axis.plot(x,mean,label=label,color=color);axis.fill_between(x,mean-sd,mean+sd,color=color,alpha=.1);axis.set(title=title,xlabel='Epoch',ylabel=ylabel)
 values=np.array([[100*e['train_pseudo_selected']/e['train_pseudo_total'] for e in r['source_epoch_curve'] if e['epoch']>=34] for r in rows]);x=np.arange(34,51);mean=values.mean(0);sd=values.std(0,ddof=1)
 ax[0,2].plot(x,mean,label=label,color=color);ax[0,2].fill_between(x,mean-sd,mean+sd,color=color,alpha=.1)
for axis in ax[0,:2]:axis.axvline(33.5,ls='--',color='#9ca3af')
ax[0,2].set(title='Hidden-U acceptance; no U truth',xlabel='Epoch',ylabel='Accepted (%)')
for i,arm in enumerate(arms):
 r=next(r for r in c['summary'] if r['arm']==arm);values=np.array(list(r['seed_accuracy'].values()))*100
 ax[1,0].scatter(np.full(3,i)+np.array([-.05,0,.05]),values,color=colors[i],s=38,zorder=3);ax[1,0].errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='D',color=colors[i],capsize=5)
ax[1,0].axhline(84,ls='--',color='#67707d');ax[1,0].set(title='Clean default head; 3 fixed model seeds',ylabel='Accuracy (%)',xticks=range(3),xticklabels=labels);ax[1,0].tick_params(axis='x',rotation=13)
x=np.arange(2);width=.34
for i,arm in enumerate(arms[1:]):
 rows=[r for r in m['rows'] if r['arm']==arm];assert len(rows)==3
 values=np.array([[r['moments'][n]['RX_within_TX_day_fraction']*100 for n in ('t1','t2')] for r in rows]);ax[1,1].bar(x+(i-.5)*width,values.mean(0),width,yerr=values.std(0,ddof=1),capsize=4,label=labels[i+1],color=colors[i+1])
ax[1,1].set(title='Source conditional RX fractions',ylabel='Centered mean/std SS (%)',xticks=x,xticklabels=['t1','t2']);ax[1,1].legend(fontsize=8)
for i,n in enumerate(('t1','t2')):
 values=[]
 for seed in (2026092701,2026092702,2026092703):
  candidate=next(r for r in m['rows'] if r['arm']==arms[2] and r['model_seed']==seed);control=next(r for r in m['rows'] if r['arm']==arms[1] and r['model_seed']==seed);values.append(candidate['moments'][n]['RX_within_TX_day_SS']-control['moments'][n]['RX_within_TX_day_SS'])
 ax[1,2].scatter(np.full(3,i)+np.array([-.05,0,.05]),values,color=colors[2],s=40);ax[1,2].hlines(np.mean(values),i-.2,i+.2,color=colors[2])
ax[1,2].axhline(0,color='#67707d',lw=1);ax[1,2].set(title='Absolute conditional RX SS difference',ylabel='MixStyle minus matched control',xticks=range(2),xticklabels=['t1','t2'])
for axis in ax.flat:axis.grid(alpha=.2)
ax[0,0].legend(fontsize=8);fig.suptitle('Identity-protected MixStyle | 3 predefined seeds | scratch50 label33+pseudo17 | default jointly trained cosine',fontsize=13)
fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(p/'results.png',dpi=150);plt.close(fig)
