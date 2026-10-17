"""Standalone figures from completed aggregate reports only;zero IQ,CP,NN,truth."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODES=('ORIGINAL','RANDOM','TX_RX_BALANCED');COLORS=('#4575b4','#777777','#d95f02')
LABELS=('Original order','Random repartition','TX x RX repartition')
def run(source,score,output):
 source=json.loads(Path(source).read_text());score=json.loads(Path(score).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
 if source['status']!='COMPLETE_SOURCE_LOGS_VERIFIED' or score['status']!='VERIFIED' or len(source['rows'])!=3 or score['CM_recounts']!=378:raise ValueError('Complete fixed report inputs required')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):
  epochs=source['rows'][mode]['reference_epoch_curves'];x=[v['cumulative_epoch'] for v in epochs]
  for ax,key,title in zip(axes,['total_L','common_CE','extra_CE'],['Full supervised L objective','Ordinary-view CE','Practical-view CE']):ax.plot(x,[v[key]['mean'] for v in epochs],'-o',label=label,color=color,markersize=3);ax.set_title(title);ax.set_xlabel('Cumulative epoch');ax.grid(alpha=.2)
 axes[0].set_ylabel('Actual mean loss');axes[0].legend(fontsize=8)
 for ext in ('png','pdf'):fig.savefig(out/('training_curves.'+ext))
 plt.close(fig)
 views=list(score['summary'][MODES[0]]);names=['Clean','High sub.','Mid sub.','Low sub.','High urban','Mid urban','Low urban'];fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
 for mode,label,color in zip(MODES,LABELS,COLORS):axes[0].plot(range(7),[score['summary'][mode][v]['accuracy']*100 for v in views],'-o',label=label,color=color,markersize=4)
 axes[0].set_xticks(range(7),names,rotation=28,ha='right');axes[0].set_ylabel('Accuracy (%)');axes[0].set_ylim(0,100);axes[0].set_title('All seven fixed received views');axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
 for i,reference in enumerate(('ORIGINAL','RANDOM')):
  values=[score['paired']['TX_RX_BALANCED__'+reference][v]['delta_accuracy_pp'] for v in views];xs=[k+(-.18 if i==0 else .18) for k in range(7)];bars=axes[1].bar(xs,values,width=.34,label='vs '+LABELS[i],color=COLORS[i])
  for x,v in zip(xs,values):axes[1].annotate(f'{v:+.3f}',(x,v),xytext=(0,3 if v>=0 else -3),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=7,rotation=90)
 axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_xticks(range(7),names,rotation=28,ha='right');axes[1].set_ylabel('Repartition accuracy change (pp)');axes[1].set_title('Candidate relative to both fixed controls');axes[1].grid(axis='y',alpha=.2);axes[1].legend(fontsize=8);axes[1].margins(y=.35)
 for ext in ('png','pdf'):fig.savefig(out/('fixed_view_results.'+ext))
 plt.close(fig)
 return dict(status='FIGURES_WRITTEN',files=4,IQ=0,CP=0,NN=0,raw_truth_reads=0,updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--score',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.source,a.score,a.output)))
