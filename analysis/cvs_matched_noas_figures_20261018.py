"""Standard scientific figures from completed matching reports only."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODES=('B00','ORIGINAL','MEASURE','PROJECT')
COLORS=('#222222','#4575b4','#777777','#d95f02')
LABELS=('No-As reference','Plain As concat','As measure only','As projection (closed)')
def run(result,source,as_source,output):
 r=json.loads(Path(result).read_text());b=json.loads(Path(source).read_text());a=json.loads(Path(as_source).read_text());o=Path(output);o.mkdir(parents=True,exist_ok=False)
 if r['status']!='VERIFIED' or r['CM_recounts']!=504 or b['status']!='COMPLETE_SOURCE_LOGS_VERIFIED' or a['status']!='COMPLETE_SOURCE_LOGS_VERIFIED':raise ValueError('Complete independently verified matching reports required')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 views=list(r['summary']['B00']);names=['Clean','High sub.','Mid sub.','Low sub.','High urban','Mid urban','Low urban']
 fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):axes[0].plot(range(7),[100*r['summary'][mode][v]['accuracy'] for v in views],'-o',color=color,label=label,markersize=4)
 axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[0].set_title('Same E30 rules and fixed received IQ');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
 for i,(mode,label,color) in enumerate(zip(MODES[1:],LABELS[1:],COLORS[1:])):
  x=[k+(i-1)*.25 for k in range(7)];y=[r['paired'][mode+'__B00'][v]['delta_accuracy_pp'] for v in views];axes[1].bar(x,y,width=.24,label=label,color=color)
 axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_ylabel('As minus No-As accuracy (pp)');axes[1].set_title('Matched No-As net differences');axes[1].grid(axis='y',alpha=.2);axes[1].legend(fontsize=8)
 for ax in axes:ax.set_xticks(range(7),names,rotation=28,ha='right')
 for ext in ('png','pdf'):fig.savefig(o/('matched_view_results.'+ext))
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):
  rows=(b if mode=='B00' else a)['rows'][mode]['reference_epoch_curves'];x=[v['cumulative_epoch'] for v in rows]
  for ax,k in zip(axes,('total_L','common_CE','extra_CE')):ax.plot(x,[v[k]['mean'] for v in rows],'-o',color=color,label=label,markersize=3)
 for ax,title in zip(axes,['Supervised L objective','Common ordinary-view CE','Second-pair CE*']):ax.set_title(title);ax.set_xlabel('Cumulative epoch');ax.set_ylabel('Actual mean loss');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('*Ordinary IQ for No-As; practical IQ for As. Different-view losses are descriptive.',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(o/('matched_training_curves.'+ext))
 plt.close(fig)
 return dict(status='FIGURES_WRITTEN',files=4,IQ=0,CP=0,NN=0,truth=0,updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--source',required=True);p.add_argument('--as-source',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.result,a.source,a.as_source,a.output)))
