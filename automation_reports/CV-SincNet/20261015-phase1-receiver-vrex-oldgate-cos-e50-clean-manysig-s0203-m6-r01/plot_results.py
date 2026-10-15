"""Regenerate figures from the committed aggregate results only."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
r=json.loads((a.report/'results.json').read_text(encoding='utf-8'))
c=r['combined_analysis'];rows=[x for audit in r['independent_audits'] for x in audit['rows']];source_obs=r['source_observations']
assert c['models']==9 and c['metric_records']==162 and len(rows)==9 and sum(len(x['source_epoch_curve']) for x in rows)==450
colors={'OLD_GATE_COS50':'#777777','RX_BALANCED':'#2673b8','RX_VREX':'#d46631'}
labels={'OLD_GATE_COS50':'Historical old gate + cosine','RX_BALANCED':'SIG_TIME + balanced CE','RX_VREX':'SIG_TIME + balanced CE + V-REx'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
for i,seed in enumerate(c['model_seeds']):
 for r in rows:
  if r['seed']!=seed:continue
  curve=r['source_epoch_curve'];ep=[x['epoch'] for x in curve];arm=r['arm']
  axs[i,0].plot(ep,[x['train_loss_tx_labeled'] for x in curve],color=colors[arm],label=labels[arm]);axs[i,1].plot(ep,[x['val_tx_acc'] for x in curve],color=colors[arm])
 axs[i,0].set(title=f'Seed {seed}: raw labelled CE',ylabel='CE',yscale='log');axs[i,1].set(title=f'Seed {seed}: source V health only',ylabel='Accuracy (%)')
for ax in axs.flat:ax.set(xlabel='Epoch',xlim=(1,50));ax.axvline(33.5,ls='--',color='#999999',lw=1);ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8);fig.suptitle('All 450 source epoch records | fixed final50 | V never selects models',fontsize=14);fig.savefig(a.report/'three_seed_source_curves.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
for i,s in enumerate(c['summary']):
 vals=[100*s['seed_accuracy'][str(seed)] for seed in c['model_seeds']];axs[0].bar(i,100*s['accuracy_mean'],yerr=100*s['accuracy_sd'],capsize=5,color=colors[s['arm']],alpha=.6);axs[0].scatter(np.arange(3)*.12+i-.12,vals,color=colors[s['arm']],edgecolor='black',s=35,zorder=3);axs[0].text(i,max(vals)+.35,f"{100*s['accuracy_mean']:.3f}",ha='center',fontsize=9)
axs[0].set_xticks([0,1,2],['Historical','Balanced','V-REx']);axs[0].axhline(84,ls='--',color='#aa5555',lw=1);axs[0].set(title='Clean accuracy: mean, sample SD, all seeds',ylabel='Accuracy (%)')
pairs=[x for x in c['paired'] if x['arm']=='RX_VREX']
for i,q in enumerate(pairs):
 vals=[100*x for x in q['seed_differences']];axs[1].bar(i,100*q['mean_difference'],yerr=100*q['sd_difference'],capsize=5,color=['#2673b8','#777777'][i],alpha=.5);axs[1].scatter(np.arange(3)*.12+i-.12,vals,color='#222222',s=30,zorder=3)
axs[1].set_xticks([0,1],['vs matched Balanced','vs historical']);axs[1].axhline(0,lw=.8,color='#555555');axs[1].set(title='Paired differences: mean, sample SD',ylabel='Percentage points')
for i,seed in enumerate(c['model_seeds']):
 for arm,off in [('RX_BALANCED',-.17),('RX_VREX',.17)]:
  row=next(x for x in source_obs if x['row_id']==f'{arm}-s{seed}');axs[2].bar(i+off,row['last5_mean_batch_risk_variance'],width=.32,color=colors[arm],label=labels[arm] if i==0 else None)
axs[2].set_xticks([0,1,2],['Seed 01','Seed 02','Seed 03']);axs[2].set(title='Source mechanism: final5 risk variance',ylabel='Mean per-batch population variance');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Complete fixed three-seed comparison | 1,512,000 clean decisions | default jointly trained head',fontsize=13);fig.savefig(a.report/'three_seed_clean_results.png');plt.close(fig)
print(json.dumps(dict(status='PASS',models=9,source_epoch_records=450,metric_records=162,figures=['three_seed_source_curves.png','three_seed_clean_results.png'])))
