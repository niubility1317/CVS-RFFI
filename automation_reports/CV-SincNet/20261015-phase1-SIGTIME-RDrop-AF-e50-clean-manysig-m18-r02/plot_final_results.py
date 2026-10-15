"""Plot all fixed seeds and complete source curves; no model fitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;s=json.loads((p/'analysis_summary.json').read_text(encoding='utf8'));a=json.loads((p/'final_source_audit.json').read_text(encoding='utf8'));arms=list('ABCDEF');colors=dict(A='#6b7280',B='#2563eb',C='#92400e',D='#9333ea',E='#ea580c',F='#059669');fig,axes=plt.subplots(1,2,figsize=(13,4.7))
x=np.arange(6);means=[s['summary'][k]['accuracy']['mean']*100 for k in arms];sd=[s['summary'][k]['accuracy']['sample_SD']*100 for k in arms]
axes[0].errorbar(x,means,yerr=sd,fmt='o',capsize=5,color='#111827',label='Mean ± sample SD')
for i,k in enumerate(arms):
 for j,r in enumerate(s['summary'][k]['per_seed']):axes[0].scatter(i+(j-1)*.08,r['accuracy']*100,color=colors[k],s=30)
axes[0].axhline(84,color='#999999',ls='--',lw=.9);axes[0].set_xticks(x,arms);axes[0].set_ylabel('Clean accuracy (%)');axes[0].grid(alpha=.2);axes[0].legend(fontsize=8);axes[0].set_title('All three seeds; no selected seed omitted')
names=list(s['comparisons']);m=[s['comparisons'][k]['mean_pp'] for k in names];d=[s['comparisons'][k]['sample_SD_pp'] for k in names];axes[1].errorbar(m,np.arange(6),xerr=d,fmt='o',capsize=4,color='#111827');axes[1].axvline(0,color='#999999',ls='--',lw=.9);axes[1].set_yticks(np.arange(6),names);axes[1].invert_yaxis();axes[1].set_xlabel('Paired difference / interaction (pp)');axes[1].grid(alpha=.2);axes[1].set_title('Mean ± sample SD of paired differences');fig.suptitle('SIG_TIME + RDrop A-F: fixed50, three model seeds');fig.tight_layout();fig.savefig(p/'final_performance.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
for row,seed in enumerate((2026092701,2026092702,2026092703)):
 for r in a['rows']:
  if r['model_seed']!=seed:continue
  curve=r['source_epoch_curve'];x=[v['epoch'] for v in curve];arm=r['arm']
  for col,key in enumerate(('train_loss_tx_labeled','val_tx_acc','train_pseudo_selected')):axes[row,col].plot(x,[v.get(key,0) for v in curve],label=arm,color=colors[arm],ls='--' if arm=='A' else '-',lw=1.1)
 for ax in axes[row]:ax.axvline(33.5,color='#999999',lw=.8,ls='--');ax.grid(alpha=.2);ax.set_xlim(1,50)
 axes[row,0].set_ylabel(f'Seed {seed}\nNative labeled CE')
for ax,title in zip(axes[0],('Native first-forward labeled CE','Source V accuracy (%)','Accepted hidden U per epoch')):ax.set_title(title)
for ax in axes[-1]:ax.set_xlabel('Epoch')
axes[0,0].legend(ncol=2,fontsize=8);fig.suptitle('Complete A-F source curves: 18 rows × 50 epochs',fontsize=14);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(p/'final_source_curves.png',dpi=160);plt.close(fig)
