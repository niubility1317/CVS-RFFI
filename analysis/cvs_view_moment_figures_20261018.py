"""Scientific figures from complete four-row independent aggregates."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODES=('NOAS_MEASURE','AS_MEASURE','NOAS_VIEW','AS_VIEW')
LABELS=('No-As native','Plain As native','No-As view moment','As view moment')
COLORS=('#555555','#2166ac','#8c510a','#d6604d')
def run(result,source,output):
 r=json.loads(Path(result).read_text(encoding='utf8'));s=json.loads(Path(source).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 if r['status']!='ANALYZED_VERIFIED' or r['CM_recounts']!=504 or s['status']!='VERIFIED_COMPLETE_SOURCE_LOGS':raise ValueError('Complete independent four-row results required')
 views=list(r['summary'][MODES[0]]);names=('Clean','High sub.','Mid sub.','Low sub.','High urban','Mid urban','Low urban');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):axes[0].plot(range(7),[100*r['summary'][mode][v]['accuracy'] for v in views],'-o',color=color,label=label,markersize=4)
 axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[0].set_title('E20+10; full source RX; fixed received IQ');axes[0].legend(fontsize=8)
 for i,key in enumerate(('AS_VIEW__AS_MEASURE','NOAS_VIEW__NOAS_MEASURE')):axes[1].bar([k+(i-.5)*.34 for k in range(7)],[r['paired'][key][v]['delta_accuracy_pp'] for v in views],width=.33,color=COLORS[3-i],label='Moment minus native: '+('As' if i==0 else 'No-As'))
 axes[1].plot(range(7),[r['optimizer_As_interaction_pp'][v] for v in views],'k--o',label='Optimizer x As interaction',markersize=3);axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_ylabel('Accuracy difference (pp)');axes[1].set_title('Matched effects; no IQ Ag tested');axes[1].legend(fontsize=8)
 for mode,label,color in zip(MODES,LABELS,COLORS):axes[2].plot(range(3),[r['suburban_gaps_pp'][mode][v] for v in views[1:4]],'-o',label=label,color=color,markersize=4)
 axes[2].axhline(2,color='black',ls='--',linewidth=.8,label='Exploratory 2 pp');axes[2].set_ylabel('Own clean minus suburban (pp)');axes[2].set_title('Absolute clean remains a separate requirement');axes[2].set_xticks(range(3),names[1:4],rotation=28,ha='right');axes[2].legend(fontsize=8)
 for ax in axes:ax.grid(alpha=.2)
 for ax in axes[:2]:ax.set_xticks(range(7),names,rotation=28,ha='right')
 for ext in ('png','pdf'):fig.savefig(out/('matched_view_results.'+ext))
 plt.close(fig);fig,axes=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):
  rows=s['rows'][mode]['reference_epoch_curves'];x=[z['cumulative_epoch'] for z in rows]
  for ax,key in zip(axes,('total_L','common_CE','extra_CE')):ax.plot(x,[z[key]['mean'] for z in rows],'-o',color=color,label=label,markersize=3)
 for ax,title in zip(axes,('Full supervised L objective','Ordinary-pair CE','Second-pair CE*')):ax.set_title(title);ax.set_xlabel('Cumulative epoch');ax.set_ylabel('Recorded mean loss');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('*Ordinary for No-As; practical(raw) for As. Original U objective and cost are reported separately.',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(out/('matched_training_curves.'+ext))
 plt.close(fig);return dict(status='FIGURES_WRITTEN',files=4,IQ=0,CP=0,NN=0,updates=0,raw_truth=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.result,a.source,a.output)))
