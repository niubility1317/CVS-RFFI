"""Render only committed aggregate source curves; no training or target input."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=json.loads(a.input.read_text(encoding='utf8'));assert data['source_only'] and data['target_metrics_read'] is False
colors={'OLD_GATE_COS50':'#777777','NORM_CONTROL':'#2477b4','RX_CAGRAD':'#d6652b'};names={'OLD_GATE_COS50':'Historical old gate + cosine','NORM_CONTROL':'SIG_TIME norm control','RX_CAGRAD':'SIG_TIME RX gradient coordination'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150});fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
for row in data['rows']:
 arm=row['arm'];rr=row['source_epoch_curve'];ep=[r['epoch'] for r in rr];kw=dict(color=colors[arm],label=names[arm],lw=1.7);axs[0,0].plot(ep,[r['train_loss_tx_labeled'] for r in rr],**kw);axs[0,1].plot(ep,[r['val_tx_acc'] for r in rr],**kw);uq=[r for r in rr if (r.get('train_pseudo_total') or 0)>0];axs[1,0].plot([r['epoch'] for r in uq],[r['train_pseudo_selected']/r['train_pseudo_total'] for r in uq],**kw);axs[2,1].plot(ep,[r['epoch_time_s'] for r in rr],**kw)
 qq=row['source_gradient_solver_curve']
 if qq:
  ee=[q['epoch'] for q in qq];axs[1,1].plot(ee,[q['mean_norm_ratio'] for q in qq],**kw);axs[2,0].plot(ee,[q['solver_fallbacks']/q['steps'] for q in qq],**kw)
axs[0,0].set(title='Original labelled CE (all recorded epochs)',ylabel='CE',yscale='log');axs[0,1].set(title='Source V: health only; no model selection',ylabel='Accuracy (%)');axs[1,0].set(title='Original U acceptance (starts at epoch 34)',ylabel='Selected / total',ylim=(0,1));axs[1,1].set(title='Solver-proposed gradient norm ratio',ylabel='Norm / original CE norm');axs[2,0].set(title='Solver fallback fraction',ylabel='Fallbacks / steps',ylim=(-.005,.05));axs[2,1].set(title='Observed native epoch duration',ylabel='Seconds')
for ax in axs.flat:ax.set(xlabel='Completed epoch',xlim=(1,50));ax.axvline(33.5,ls='--',lw=1,color='#aaaaaa');ax.grid(alpha=.16)
axs[0,0].legend(fontsize=8);axs[1,1].text(.02,.03,'Control applies the original direction at this norm.',transform=axs[1,1].transAxes,fontsize=8);counts=' | '.join(row['arm']+': '+str(row['source_epoch_count']) for row in data['rows']);fig.suptitle('Source-only recorded prefix: '+counts+'\nFixed final50; default jointly trained head; target performance unknown',fontsize=12);fig.savefig(a.output);plt.close(fig);print(json.dumps(dict(status='PASS',source_epochs=sum(row['source_epoch_count'] for row in data['rows']),new_source_steps=sum(q['steps'] for row in data['rows'] for q in row['source_gradient_solver_curve']),target_metrics_read=False,output=a.output.name)))
