"""Scientific plots from existing source logs and frozen scores only."""
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=('Clean','Sub high','Sub mid','Sub low','Urb high','Urb mid','Urb low')
def render_row(parent,data,output,name):
 if len(parent['epoch_metrics'])!=20 or len(data['epoch_metrics'])!=10:raise ValueError('Require complete20+10 curves')
 for row,n in ((parent,4440),(data,2220)):
  if row['step_metrics']['records']!=n or row['step_metrics']['partial']:raise ValueError('Require complete real step logs')
 overall={x['view']:x for x in data['score']['records'] if x['dimension']=='overall'}
 if set(overall)!=set(VIEWS):raise ValueError('Seven frozen scores required')
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(13,8.2),layout='constrained')
 ep=list(range(1,31));source=parent['epoch_metrics']+data['epoch_metrics']
 ax=axes[0,0]
 loss=[r['step_metrics']['epochs'][str(e)]['train/loss']['mean'] for r,nn in ((parent,20),(data,10)) for e in range(1,nn+1)]
 ll=[r['step_metrics']['epochs'][str(e)]['train/loss_labeled']['mean'] for r,nn in ((parent,20),(data,10)) for e in range(1,nn+1)]
 ax.plot(ep,loss,label='Total training objective',color='#2563a6',lw=1.8);ax.plot(ep,ll,label='Native labeled objective',color='#db8b25',lw=1.6)
 ax.set(title='A  All persisted training steps',xlabel='Cumulative epoch',ylabel='Mean loss',xlim=(1,30));ax.legend(frameon=False,fontsize=9)
 ax=axes[0,1]
 for key,label,color in (('train_tx_acc','L training accuracy','#db8b25'),('val_tx_acc','Seen-source V health','#2563a6')):ax.plot(ep,[x[key] for x in source],label=label,color=color,lw=1.8)
 ax.set(title='B  Source curves; fixed final checkpoint',xlabel='Cumulative epoch',ylabel='Accuracy (%)',xlim=(1,30),ylim=(0,100));ax.legend(frameon=False,fontsize=9)
 for ax in axes[0]:
  ax.axvline(13.5,color='0.6',ls=':',lw=1);ax.axvline(20.5,color='#b7524c',ls='--',lw=1.2);ax.text(20.7,.04,'Optimizer reset',transform=ax.get_xaxis_transform(),color='#b7524c',fontsize=8)
 ax=axes[1,0];xx=list(range(7));acc=[100*overall[v]['accuracy'] for v in VIEWS];f1=[100*overall[v]['macro_f1'] for v in VIEWS]
 ax.bar([x-.19 for x in xx],acc,width=.38,label='Accuracy',color='#2563a6');ax.bar([x+.19 for x in xx],f1,width=.38,label='Macro F1',color='#70a58d')
 for i,x in enumerate(acc):ax.text(i-.19,x+.9,f'{x:.2f}',ha='center',fontsize=8)
 ax.set_xticks(xx,LABELS,rotation=25,ha='right');ax.set(title='C  Fixed final30 default-head results',ylabel='Percent',ylim=(0,100));ax.legend(frameon=False,fontsize=9)
 ax=axes[1,1];gap=[100*(overall['clean']['accuracy']-overall[v]['accuracy']) for v in VIEWS[1:4]];ax.bar(range(3),gap,color='#db8b25',width=.6)
 for i,x in enumerate(gap):ax.text(i,x+.6,f'{x:.2f}',ha='center',fontsize=9)
 ax.axhline(2,color='#374151',ls='--',lw=1.1,label='Exploratory gap 2 pp');ax.set_xticks(range(3),LABELS[1:4]);ax.set(title='D  Each suburban gap to this model clean',ylabel='Clean minus suburban (pp)',ylim=(min(-1,min(gap)-3),max(5,max(gap)+5)));ax.legend(frameon=False,fontsize=9)
 for ax in axes.flat:ax.grid(axis='y',color='0.9',linewidth=.6);ax.set_axisbelow(True)
 fig.suptitle(name+' | full source, seed 2026092701 | fixed E20 + 10',fontsize=14)
 stem=output/'curves_and_seven_views';fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
 return dict(status='PLOTS_WRITTEN',row=name,cumulative_epochs=30,cumulative_steps=6660,IQ=0,CP=0,raw_truth=0,NN=0,new_updates=0)


def render_matrix(result, output):
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import pyplot as plt
 import numpy as np
 arms=('B00','B10','B01','B11','FULL_AS');colors=('#6b7280','#2563a6','#279268','#dc8c29','#b7524c')
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(14,8.6),layout='constrained')
 ax=axes[0,0];x=np.arange(7)
 for arm,color in zip(arms,colors):ax.plot(x,[100*result['summary'][arm][v]['accuracy'] for v in VIEWS],marker='o',lw=1.6,label=arm,color=color)
 ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set(title='A  All fixed final30 default-head views',ylabel='Accuracy (%)',ylim=(0,100));ax.legend(frameon=False,fontsize=8,ncols=3)
 ax=axes[0,1]
 for key,label,color in (('B10-B00','Ag without As',colors[1]),('B11-B01','Ag with As',colors[3])):ax.plot(x,[result['comparisons'][key][v]['delta_accuracy_pp'] for v in VIEWS],marker='o',lw=1.6,label=label,color=color)
 ax.plot(x,[result['accuracy_interaction_pp'][v] for v in VIEWS],marker='s',ls='--',lw=1.4,color=colors[4],label='Factorial interaction')
 ax.axhline(0,color='0.3',lw=.8);ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set(title='B  Separate Ag increment and interaction',ylabel='Accuracy difference (pp)',ylim=(-.8,.55));ax.legend(frameon=False,fontsize=9)
 ax=axes[1,0];aa=('B01','B11','FULL_AS')
 for i,arm in enumerate(aa):
  vals=[100*(result['summary'][arm]['clean']['accuracy']-result['summary'][arm][v]['accuracy']) for v in VIEWS[1:4]]
  xx=np.arange(3)+(i-1)*.23;ax.bar(xx,vals,width=.23,label=arm,color=colors[arms.index(arm)])
  for pos,val in zip(xx,vals):ax.text(pos,val+.15,f'{val:.2f}',ha='center',fontsize=8)
 ax.axhline(2,ls='--',color='0.25',label='Exploratory gap 2 pp');ax.set_xticks(range(3),LABELS[1:4]);ax.set(title='C  Practical-trained rows: gap to own clean',ylabel='Clean minus suburban (pp)',ylim=(0,12));ax.legend(frameon=False,fontsize=8,ncols=2)
 ax=axes[1,1];keys=[('B11-B01','clean'),('B11-B01','practical_low_urban'),('B11-FULL_AS','clean'),('B11-FULL_AS','practical_low_urban')]
 p=[result['comparisons'][k][v] for k,v in keys];xx=np.arange(4)
 ax.bar(xx-.18,[v['repairs']/1000 for v in p],width=.36,label='Repairs',color='#279268');ax.bar(xx+.18,[-v['new_errors']/1000 for v in p],width=.36,label='New errors',color='#b7524c')
 for i,v in enumerate(p):ax.text(i,v['repairs']/1000+.4,f"{v['delta_accuracy_pp']:+.3f} pp",ha='center',fontsize=8)
 ax.axhline(0,color='0.3',lw=.8);ax.set_xticks(xx,('vs B01\nClean','vs B01\nUrb low','vs full As\nClean','vs full As\nUrb low'));ax.set(title='D  B11: paired identity repairs and new errors',ylabel='Physical samples (thousands)',ylim=(-12,10));ax.legend(frameon=False,fontsize=9,ncols=2)
 for ax in axes.flat:ax.grid(axis='y',color='0.9',lw=.6);ax.set_axisbelow(True)
 fig.suptitle('Fixed clock / practical factorial | full source | seed 2026092701 | fixed E20 + 10',fontsize=14)
 stem=output/'matched_factorial';fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
 return dict(status='PLOTS_WRITTEN',artifacts=[str(stem.with_suffix(x)) for x in ('.png','.pdf')],IQ_reads=0,CP_loads=0,raw_truth_reads=0,new_updates=0)
