"""Render already frozen first-seed aggregates; no data/model/query access."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
r=json.loads((a.report/'results.json').read_text(encoding='utf-8'))
assert r['seed_count']==1 and r['analysis']['metric_records']==54
rows=r['independent_audit']['rows']
assert len(rows)==3 and all(len(x['source_epoch_curve'])==50 for x in rows)
colors={'OLD_GATE_COS50':'#777777','RX_BALANCED':'#2673b8','RX_VREX':'#d46631'}
labels={'OLD_GATE_COS50':'Historical old gate + cosine','RX_BALANCED':'SIG_TIME + RX-balanced CE','RX_VREX':'SIG_TIME + RX-balanced CE + V-REx'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
fig,axs=plt.subplots(2,2,figsize=(12.5,8),constrained_layout=True)
for x in rows:
 c=x['source_epoch_curve'];ep=[v['epoch'] for v in c];arm=x['arm']
 axs[0,0].plot(ep,[v['train_loss_tx_labeled'] for v in c],color=colors[arm],label=labels[arm])
 axs[0,1].plot(ep,[v['val_tx_acc'] for v in c],color=colors[arm])
for arm,x in r['source_qualification']['actual_observations'].items():
 c=x['curves'];assert len(c)==50
 axs[1,0].plot([v['epoch'] for v in c],[v['mean_batch_risk_variance'] for v in c],color=colors[arm],label=labels[arm])
 axs[1,1].plot([v['epoch'] for v in c],[v['worst_RX_CE'] for v in c],color=colors[arm])
for ax,title,yl in zip(axs.flat,['Native labelled CE (raw, unweighted)','Source V health only','Mean per-batch population risk variance','Worst source-RX training CE'],['CE','Accuracy (%)','Risk variance','CE']):
 ax.set(title=title,xlabel='Epoch',ylabel=yl,xlim=(1,50));ax.axvline(33.5,color='#aaaaaa',ls='--',lw=1);ax.grid(alpha=.16)
axs[0,0].set_yscale('log');axs[1,0].set_yscale('log');axs[0,0].legend(fontsize=8,loc='upper right')
fig.suptitle('Frozen first-seed source trajectories | seed 2026092701 | final epoch 50',fontsize=14)
fig.savefig(a.report/'source_training_final50.png');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
s=r['analysis']['summary'];xx=np.arange(3);width=.24
for j,(key,label) in enumerate([('accuracy_mean','Overall'),('worst_RX_mean','Worst RX'),('worst_day_mean','Worst day')]):
 vals=[100*x[key] for x in s]
 bars=axs[0].bar(xx+(j-1)*width,vals,width,label=label,color=['#234c73','#4d88ac','#a1c5dd'][j])
 for b,v in zip(bars,vals):axs[0].text(b.get_x()+b.get_width()/2,v+.25,f'{v:.2f}',ha='center',fontsize=7)
axs[0].set_xticks(xx,['Historical','Balanced','V-REx']);axs[0].set(ylim=(65,88),ylabel='Accuracy (%)',title='Clean default-head predictions');axs[0].axhline(84,color='#d46631',ls='--',lw=1);axs[0].legend(fontsize=8)
pairs=[x for x in r['analysis']['paired'] if x['arm']=='RX_VREX'];ds=[100*x['mean_difference'] for x in pairs]
b=axs[1].bar([0,1],ds,color=['#c55946' if x<0 else '#36867d' for x in ds],width=.55)
for i,v in enumerate(ds):axs[1].text(i,v+(.07 if v>=0 else -.07),f'{v:+.3f} pp',ha='center',va='bottom' if v>=0 else 'top')
axs[1].axhline(0,color='#555555',lw=.8);axs[1].set_xticks([0,1],['vs matched Balanced','vs historical']);axs[1].set(title='Paired first-seed differences',ylabel='Percentage points',ylim=(-1,3.3));axs[1].text(.5,.04,'n = 1; no three-seed estimate yet',transform=axs[1].transAxes,ha='center',fontsize=8)
features=['.t1.pw.weight','.t2.pw.weight','local_projection.weight']
params=r['source_features']['rows'][1]['actual_initial_to_final_parameters']['records']
selected=[next(v for v in params if term in v['parameter']) for term in features]
axs[2].bar(np.arange(3),[x['delta_L2'] for x in selected],color=colors['RX_VREX'])
axs[2].set_xticks(np.arange(3),['Shallow t1','Middle t2','SIG projection']);axs[2].set(title='Actual deployed feature weights changed',ylabel='Initial-to-final weight L2')
for i,v in enumerate(selected):axs[2].text(i,v['delta_L2']+.04,f"{v['changed_elements']}/{v['elements']}",ha='center',fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('First-seed frozen evidence | 168,000 clean decisions/model | default jointly trained head',fontsize=13)
fig.savefig(a.report/'first_seed_clean_results.png');plt.close(fig)
print(json.dumps({'status':'PASS','source_epoch_records':150,'risk_epoch_records':100,'metric_records':54,'figures':['source_training_final50.png','first_seed_clean_results.png'],'target_feedback':False}))
