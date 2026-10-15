"""Plot all fixed default-head seeds and all new source training epochs."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'OLD_GATE_COS50':'#858585','SIG_TIME':'#287da1','SINC_UNITS':'#279578'}
STYLES={2026092701:'-',2026092702:'--',2026092703:':'}
def main(root):
 root=Path(root);v=json.loads((root/'results.json').read_text(encoding='utf8'))
 assert v['model_seed_count']==3 and v['metric_records']==162 and v['default_decision_count']==1512000
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,3,figsize=(14.5,4.6),dpi=180)
 for i,s in enumerate(v['summary']):
  y=s['accuracy_mean']*100;ax[0].bar(i,y,color=COLORS[s['arm']],width=.58)
  ax[0].errorbar(i,y,yerr=s['accuracy_sd']*100,color='#333333',capsize=5,fmt='none',lw=1.3)
  ax[0].scatter([i]*3,[x*100 for x in s['seed_accuracy'].values()],color='#222222',s=18,zorder=3)
  ax[0].text(i,77.35,f'{y:.3f}%',ha='center',fontsize=9)
 ax[0].axhline(84,color='#b63f3b',ls='--',lw=1);ax[0].set(xticks=range(3),xticklabels=['Old gate+cos','SIG_TIME','Sinc units'],ylim=(77,86),ylabel='Clean accuracy (%)',title='Mean +/- sample SD; dots are model seeds')
 pair=next(p for p in v['paired'] if p['arm']=='SINC_UNITS' and p['control']=='SIG_TIME')
 values=[x*100 for x in pair['seed_differences']]
 ax[1].bar(range(3),values,color=['#c96851' if x<0 else '#279578' for x in values],width=.58)
 for i,y in enumerate(values):ax[1].annotate(f'{y:+.3f}',(i,y),xytext=(0,6 if y>=0 else -14),textcoords='offset points',ha='center',fontsize=9)
 ax[1].axhline(0,color='#444444',lw=.8);ax[1].axhline(pair['mean_difference']*100,color='#444444',ls='--',lw=1)
 ax[1].set(xticks=range(3),xticklabels=['Seed01','Seed02','Seed03'],ylabel='Paired difference (percentage points)',ylim=(min(values)-.45,max(values)+.45),title='Sinc units - matched SIG_TIME')
 errors=pair['same_ID_errors'];vals=[errors['corrected_errors'],errors['introduced_errors']]
 ax[2].bar([0,1],vals,color=['#279578','#c96851'],width=.58)
 for i,n in enumerate(vals):ax[2].text(i,n+200,f'{n:,}',ha='center')
 ax[2].set(xticks=[0,1],xticklabels=['Corrected','Introduced'],ylabel='Same-ID errors (504,000 paired decisions)',ylim=(0,max(vals)*1.2),title=f"Net correct: {errors['net_correct']:+,}")
 for a in ax:a.grid(axis='y',alpha=.15)
 fig.suptitle('Fixed SIG_TIME | Sinc optimizer units: complete three-seed results',x=.065,y=.985,ha='left',fontsize=14,fontweight='bold')
 fig.text(.065,.015,'Final E50; 168,000 clean decisions/model; original default jointly-trained head; no head fitting. Mean gain is not a new-TX or deployment claim.',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.055,1,.92));fig.savefig(root/'three_seed_clean_results.png');plt.close(fig)
 source=json.loads((root/'full_source_curves.json').read_text(encoding='utf8'))
 assert source['source_epoch_records']==450 and source['source_only'] and not source['U_truth_read']
 fig,ax=plt.subplots(2,2,figsize=(11.8,7.4),dpi=180)
 for row in source['rows']:
  if row['baseline_reused']:continue
  rec=row['records'];assert [r['epoch'] for r in rec]==list(range(1,51))
  seed=int(row['row_id'].rsplit('s',1)[1]);color=COLORS[row['arm']];style=STYLES[seed];x=[r['epoch'] for r in rec];label=row['arm']+' '+str(seed)[-2:]
  ax[0,0].plot(x,[r['train_loss_tx_labeled'] for r in rec],color=color,ls=style,lw=1.5,label=label)
  ax[0,1].plot(x,[r['val_tx_acc'] for r in rec],color=color,ls=style,lw=1.5)
  active=[r for r in rec if r['train_pseudo_total']>0]
  ax[1,0].plot([r['epoch'] for r in active],[100*r['train_pseudo_conf'] for r in active],color=color,ls=style,lw=1.5)
  ax[1,1].plot([r['epoch'] for r in active],[100*r['train_pseudo_selected']/r['train_pseudo_total'] for r in active],color=color,ls=style,lw=1.5)
 for a,title,ylabel in zip(ax.flat,['Native L identity CE','Source V default-head accuracy','Own-EMA weak U mean confidence','Accepted U / processed U'],['Cross-entropy','Accuracy (%)','Confidence (%) - not accuracy','Acceptance (%) - truth hidden']):
  a.set(title=title,ylabel=ylabel,xlabel='Epoch',xlim=(1,50));a.axvline(33.5,color='#888888',ls='--',lw=.8);a.grid(alpha=.15)
 ax[0,0].set_ylim(bottom=0);ax[1,1].set_ylim(bottom=0)
 fig.legend(*ax[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.027),fontsize=9)
 fig.suptitle('Fixed SIG_TIME | all six new 50-epoch source trajectories',x=.075,y=.985,ha='left',fontsize=14,fontweight='bold')
 fig.text(.075,.006,'450 source epoch records preserved including three historical controls; 300 new epochs plotted. Original 33 L + 17 SSL; source V is health-only.',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.115,1,.94));fig.savefig(root/'source_training_three_seeds.png');plt.close(fig)
 print(json.dumps(dict(status='VERIFIED',figures=2,model_seeds=3,metric_records=162,source_epochs_plotted=300,source_epochs_preserved=450)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
