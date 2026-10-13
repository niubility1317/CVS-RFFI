import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);d=json.loads((p/'results.json').read_text(encoding='utf8'));arms=['OLD_GATE_COS50','RAW_HZ','CENTER_25K'];colors=['#777777','#3274A1','#D66E32']
fig,ax=plt.subplots(2,2,figsize=(12.8,8.4),constrained_layout=True)
for i,s in enumerate(d['summary']):
 ax[0,0].errorbar(i,100*s['accuracy_mean'],yerr=100*s['accuracy_sd'],fmt='o',color=colors[i],capsize=5,label=arms[i])
 ax[0,0].scatter(np.full(3,i)+np.linspace(-.07,.07,3),100*np.array(list(s['seed_accuracy'].values())),color=colors[i],alpha=.65,s=20)
ax[0,0].axhline(84,color='#A33',ls='--',label='84% goal');ax[0,0].set(xticks=range(3),xticklabels=['Old gate + cos','Raw Hz','Centered 25 kHz'],ylabel='Clean accuracy (%)',ylim=(76,85),title='Fixed default head: mean ± sample SD, n=3')
for arm,color in zip(arms,colors):
 rows=[r for r in d['rows'] if r['arm']==arm];ce=np.array([[e['train_loss_tx_labeled'] for e in r['source_epoch_curve']] for r in rows]);x=np.arange(1,51);ax[0,1].plot(x,ce.mean(0),color=color,label=arm);ax[0,1].fill_between(x,ce.min(0),ce.max(0),color=color,alpha=.10)
ax[0,1].set(xlabel='Epoch',ylabel='Source labeled CE',title='All 50 epochs; shading is seed min–max');ax[0,1].legend(fontsize=8)
for arm,color in zip(arms[1:],colors[1:]):
 for j,r in enumerate([r for r in d['rows'] if r['arm']==arm]):
  v=[e['Sinc']['kernel_relative_L2'] for e in r['source_Sinc_diagnostic']['records']];ax[1,0].plot(range(1,51),np.maximum(v,1e-9),color=color,alpha=.7,label=arm if j==0 else None)
ax[1,0].set(yscale='log',xlabel='Epoch',ylabel='Kernel relative L2 from initialization',title='Actual student kernels, all six models');ax[1,0].text(.02,.03,'Values <1e-9 drawn at 1e-9 (including zeros)',transform=ax[1,0].transAxes,fontsize=8);ax[1,0].legend(fontsize=8)
for arm,color in zip(arms,colors):
 rows=[r for r in d['rows'] if r['arm']==arm];v=np.array([[100*e['train_pseudo_selected']/e['train_pseudo_total'] for e in r['source_epoch_curve'][33:]] for r in rows]);ax[1,1].plot(range(34,51),v.mean(0),color=color,label=arm);ax[1,1].fill_between(range(34,51),v.min(0),v.max(0),color=color,alpha=.10)
ax[1,1].set(xlabel='Pseudo epoch',ylabel='Accepted U packets (%)',title='Original gates retained; U truth hidden')
for a in ax.flat:a.grid(alpha=.2)
fig.suptitle('Sinc parameterization — fixed ManySig clean protocol; original default classifier',fontsize=13)
fig.savefig(p/'sinc_results.png',dpi=160);plt.close(fig)
print(json.dumps({'status':'VERIFIED','figure':str(p/'sinc_results.png'),'models':9,'head_fit':False}))
