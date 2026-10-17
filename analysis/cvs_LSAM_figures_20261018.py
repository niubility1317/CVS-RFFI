"""Scientific plots from completed independent LSAM recount;no IQ/CP/raw truth."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODES=('NOAS_REPLAY','AS_REPLAY','AS_LSAM');LABELS=('No-As replay','Plain As replay','As L-only SAM');COLORS=('#222222','#4575b4','#d95f02')
def run(result,source,output):
 r=json.loads(Path(result).read_text(encoding='utf8'));s=json.loads(Path(source).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 if r['status']!='ANALYZED_VERIFIED' or r['CM_recounts']!=378 or s['status']!='VERIFIED_COMPLETE_SOURCE_LOGS':raise ValueError('Complete independent metadata required')
 views=list(r['summary'][MODES[0]]);names=['Clean','High sub.','Mid sub.','Low sub.','High urban','Mid urban','Low urban'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,ax=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):ax[0].plot(range(7),[100*r['summary'][mode][v]['accuracy'] for v in views],'-o',color=color,label=label,markersize=4)
 ax[0].set_ylim(0,100);ax[0].set_ylabel('Accuracy (%)');ax[0].set_title('Same E30/8L-forward rules; fixed received IQ');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
 for i,key in enumerate(('AS_LSAM__AS_REPLAY','AS_LSAM__NOAS_REPLAY')):ax[1].bar([k+(i-.5)*.32 for k in range(7)],[r['paired'][key][v]['delta_accuracy_pp'] for v in views],width=.31,color=COLORS[i+1],label='SAM minus '+('plain As' if i==0 else 'No-As'))
 ax[1].axhline(0,color='black',linewidth=.7);ax[1].set_ylabel('Accuracy difference (pp)');ax[1].set_title('Controlled SAM and clean net differences');ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.2)
 for a in ax:a.set_xticks(range(7),names,rotation=28,ha='right')
 for ext in ('png','pdf'):fig.savefig(out/('matched_view_results.'+ext))
 plt.close(fig);fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):
  rows=s['rows'][mode]['reference_epoch_curves'];x=[v['cumulative_epoch'] for v in rows]
  for a,k in zip(ax,('total_L','common_CE','extra_CE')):a.plot(x,[v[k]['mean'] for v in rows],'-o',color=color,label=label,markersize=3)
 for a,title in zip(ax,('Recorded unperturbed L objective','Common ordinary-view CE','Second-pair CE*')):a.set_title(title);a.set_xlabel('Cumulative epoch');a.set_ylabel('Recorded mean loss');a.grid(alpha=.2)
 ax[0].legend(fontsize=8);fig.suptitle('*Ordinary for No-As; practical for As. SAM uses perturbed L gradients, not these scalar values.',fontsize=9)
 for ext in ('png','pdf'):fig.savefig(out/('matched_training_curves.'+ext))
 plt.close(fig);return dict(status='FIGURES_WRITTEN',files=4,IQ=0,CP=0,NN=0,updates=0,raw_truth=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.result,a.source,a.output)))
