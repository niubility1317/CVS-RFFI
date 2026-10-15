"""Render frozen first-seed default-head results and complete source curves."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'OLD_GATE_COS50':'#858585','SIG_TIME':'#287da1','SINC_UNITS':'#279578'}
def main(root):
 root=Path(root);v=json.loads((root/'results.json').read_text(encoding='utf8'));assert v['model_seed_count']==1 and v['metric_records']==54
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,2,figsize=(11.3,4.5),dpi=180)
 for i,s in enumerate(v['summary']):
  y=s['accuracy_mean']*100;ax[0].bar(i,y,color=COLORS[s['arm']],width=.6);ax[0].text(i,78.4,f'{y:.3f}%',ha='center')
 ax[0].axhline(84,color='#b63f3b',ls='--',lw=1)
 ax[0].set(xticks=range(3),xticklabels=['Old gate + cos','SIG_TIME','Sinc units'],ylim=(78,85),ylabel='Clean accuracy (%)',title='One predefined seed; sample SD is unavailable')
 e=next(x for x in v['paired_error_transitions'] if x['arm']=='SINC_UNITS' and x['control']=='SIG_TIME')
 vals=[e['corrected_errors'],e['introduced_errors']]
 ax[1].bar([0,1],vals,color=['#279578','#c96851'],width=.58)
 for i,n in enumerate(vals):ax[1].text(i,n+70,f'{n:,}',ha='center')
 ax[1].set(xticks=[0,1],xticklabels=['Corrected errors','Introduced errors'],ylabel='Same-ID decisions',ylim=(0,max(vals)*1.22),title='Sinc units versus fixed SIG_TIME')
 for a in ax:a.grid(axis='y',alpha=.15)
 fig.suptitle('Fixed SIG_TIME | numerical Sinc learning did not improve this seed',x=.075,y=.98,ha='left',fontsize=14,fontweight='bold')
 fig.text(.075,.018,'Seed 2026092701; 168,000 clean decisions/model; unchanged default joint head; final E50. Full three-seed goal remains unproven.',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.055,1,.91));fig.savefig(root/'first_seed_clean_results.png');plt.close(fig)
 source=json.loads((root/'full_source_curves.json').read_text(encoding='utf8'));assert source['source_only'] and not source['U_truth_read']
 fig,ax=plt.subplots(2,2,figsize=(11.3,7.0),dpi=180)
 for row in source['rows']:
  if row['baseline_reused']:continue
  records=row['records'];assert [r['epoch'] for r in records]==list(range(1,51));name=row['arm'];color=COLORS[name];x=[r['epoch'] for r in records]
  ax[0,0].plot(x,[r['train_loss_tx_labeled'] for r in records],color=color,lw=1.7,label=name)
  ax[0,1].plot(x,[r['val_tx_acc'] for r in records],color=color,lw=1.7)
  active=[r for r in records if r['train_pseudo_total']>0]
  ax[1,0].plot([r['epoch'] for r in active],[100*r['train_pseudo_conf'] for r in active],color=color,lw=1.7)
  ax[1,1].plot([r['epoch'] for r in active],[100*r['train_pseudo_selected']/r['train_pseudo_total'] for r in active],color=color,lw=1.7)
 for a,title,ylabel in zip(ax.flat,['Native L identity CE','Source V default-head accuracy','Own-EMA weak U mean confidence','Accepted U / processed U'],['Cross-entropy','Accuracy (%) - native percent unit','Confidence (%) - not accuracy','Acceptance (%) - truth hidden']):
  a.set_title(title,loc='left');a.set_ylabel(ylabel);a.set_xlabel('Epoch');a.set_xlim(1,50);a.axvline(33.5,color='#888888',ls='--',lw=.8);a.grid(alpha=.15)
 ax[0,0].set_ylim(bottom=0);ax[0,1].set_ylim(85,100);ax[1,0].set_ylim(97,100);ax[1,1].set_ylim(bottom=0)
 fig.legend(*ax[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.03))
 fig.suptitle('Fixed SIG_TIME | both complete 50-epoch source trajectories',x=.075,y=.985,ha='left',fontsize=14,fontweight='bold')
 fig.text(.075,.008,'All complete epochs shown; original 33 L + 17 SSL. Source V is health-only; U acceptance is not pseudo-label precision.',fontsize=8,color='#555555')
 fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(root/'source_training_final50.png');plt.close(fig)
 print(json.dumps(dict(status='VERIFIED',figures=2,metric_records=54,source_epoch_records=150,new_source_models_plotted=2)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
