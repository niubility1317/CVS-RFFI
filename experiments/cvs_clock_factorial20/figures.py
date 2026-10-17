"""Plot existing frozen-row metadata only; no model, IQ or truth access."""
import argparse
import json
from pathlib import Path

VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban',
         'practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS = ('Clean','Sub high','Sub mid','Sub low','Urb high','Urb mid','Urb low')


def render_row(data, output, name):
    if len(data['epoch_metrics']) != 20 or data['reference_steps']['records'] != 4440:
        raise ValueError('Plot only complete fixed E20 source curves')
    scores = data.get('scores',data.get('score'))
    overall = {r['view']:r for r in scores['records'] if r['dimension']=='overall'}
    if set(overall) != set(VIEWS):
        raise ValueError('All seven existing frozen-view scores are needed')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,2,figsize=(13,8.2),layout='constrained')
    epoch = list(range(1,21))
    def ref(key):
        return [data['reference_steps']['epochs'][str(e)].get(key,{}).get('mean') for e in epoch]
    ax=axes[0,0]
    for key,label,color in (('total_L','L objective','#2563a6'),
                            ('common_CE','Ordinary CE','#db8b25'),
                            ('extra_CE','Extra-view CE','#29856f')):
        values=ref(key)
        if any(v is not None for v in values):ax.plot(epoch,values,label=label,color=color,lw=1.8)
    ax.axvline(13.5,color='0.5',ls=':',lw=1,label='Original U starts')
    ax.set(title='A  Full labeled training curve',xlabel='Epoch',ylabel='Mean loss')
    ax.legend(frameon=False,fontsize=9)
    ax.set_xlim(1,20)
    ax=axes[0,1]
    for key,label,color in (('train_tx_acc','L training accuracy','#db8b25'),
                            ('val_tx_acc','Seen-source V health','#2563a6')):
        ax.plot([r['epoch'] for r in data['epoch_metrics']],
                [r[key] for r in data['epoch_metrics']],label=label,color=color,lw=1.8)
    ax.axvline(13.5,color='0.5',ls=':',lw=1)
    ax.set(title='B  Source curves; no checkpoint selection',xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,100),xlim=(1,20))
    ax.legend(frameon=False,fontsize=9)
    ax=axes[1,0]
    x=list(range(7));acc=[100*overall[v]['accuracy'] for v in VIEWS]
    f1=[100*overall[v]['macro_f1'] for v in VIEWS]
    ax.bar([i-.19 for i in x],acc,width=.38,label='Accuracy',color='#2563a6')
    ax.bar([i+.19 for i in x],f1,width=.38,label='Macro F1',color='#70a58d')
    for i,value in enumerate(acc):ax.text(i-.19,value+.9,f'{value:.2f}',ha='center',fontsize=8)
    ax.set_xticks(x,LABELS,rotation=25,ha='right')
    ax.set(title='C  Fixed final20 default-head results',ylabel='Percent',ylim=(0,100))
    ax.legend(frameon=False,fontsize=9)
    ax=axes[1,1]
    sub=VIEWS[1:4];gaps=[100*(overall['clean']['accuracy']-overall[v]['accuracy']) for v in sub]
    ax.bar(range(3),gaps,color='#db8b25',width=.6)
    for i,value in enumerate(gaps):ax.text(i,value+.6,f'{value:.2f}',ha='center',fontsize=9)
    ax.axhline(2,color='#374151',ls='--',lw=1.1,label='Exploratory gap 2 pp')
    ax.set_xticks(range(3),LABELS[1:4])
    ax.set(title='D  Each suburban gap to this model clean',ylabel='Clean minus suburban (pp)',ylim=(min(-1,min(gaps)-3),max(5,max(gaps)+5)))
    ax.legend(frameon=False,fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',color='0.9',linewidth=.6);ax.set_axisbelow(True)
    fig.suptitle(name+' | full source, seed 2026092701, scratch E20',fontsize=14)
    stem=output/'curves_and_seven_views'
    fig.savefig(stem.with_suffix('.png'),dpi=180)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    return dict(status='PLOTS_WRITTEN',row=name,epochs=20,optimizer_steps=4440,
                views=list(VIEWS),IQ_reads=0,CP_loads=0,raw_truth_reads=0,
                model_forwards=0,new_updates=0,
                artifacts=[str(stem.with_suffix(s)) for s in ('.png','.pdf')])



def render_matrix(result, output):
 import matplotlib
 matplotlib.use('Agg')
 from matplotlib import pyplot as plt
 import numpy as np
 arms=('B00','B10','B01','B11','FULL_AS_REF');colors=('#6b7280','#2563a6','#279268','#dc8c29','#b7524c')
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(14,8.6),layout='constrained')
 ax=axes[0,0];x=np.arange(7)
 for arm,color in zip(arms,colors):ax.plot(x,[100*result['summary'][arm][v]['accuracy'] for v in VIEWS],marker='o',lw=1.6,label=arm,color=color)
 ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set(title='A  All fixed final20 default-head views',ylabel='Accuracy (%)',ylim=(0,100));ax.legend(frameon=False,fontsize=8,ncols=3)
 ax=axes[0,1]
 for key,label,color in (('B10-B00','Ag without As',colors[1]),('B11-B01','Ag with As',colors[3])):ax.plot(x,[result['comparisons'][key][v]['delta_accuracy_pp'] for v in VIEWS],marker='o',lw=1.6,label=label,color=color)
 ax.plot(x,[result['accuracy_interaction_pp'][v] for v in VIEWS],marker='s',ls='--',lw=1.4,color=colors[4],label='Factorial interaction')
 ax.axhline(0,color='0.3',lw=.8);ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set(title='B  Separate Ag increment and interaction',ylabel='Accuracy difference (pp)',ylim=(-.55,.55));ax.legend(frameon=False,fontsize=9)
 ax=axes[1,0];aa=('B01','B11','FULL_AS_REF')
 for i,arm in enumerate(aa):
  vals=[100*(result['summary'][arm]['clean']['accuracy']-result['summary'][arm][v]['accuracy']) for v in VIEWS[1:4]]
  xx=np.arange(3)+(i-1)*.23;ax.bar(xx,vals,width=.23,label=arm,color=colors[arms.index(arm)])
  for pos,val in zip(xx,vals):ax.text(pos,val+.15,f'{val:.2f}',ha='center',fontsize=8)
 ax.axhline(2,ls='--',color='0.25',label='Exploratory gap 2 pp');ax.set_xticks(range(3),LABELS[1:4]);ax.set(title='C  Practical-trained rows: gap to own clean',ylabel='Clean minus suburban (pp)',ylim=(0,12));ax.legend(frameon=False,fontsize=8,ncols=2)
 ax=axes[1,1];keys=[('B11-B01','clean'),('B11-B01','practical_low_urban'),('B11-FULL_AS_REF','clean'),('B11-FULL_AS_REF','practical_low_urban')]
 p=[result['comparisons'][k][v] for k,v in keys];xx=np.arange(4)
 ax.bar(xx-.18,[v['repairs']/1000 for v in p],width=.36,label='Repairs',color='#279268');ax.bar(xx+.18,[-v['new_errors']/1000 for v in p],width=.36,label='New errors',color='#b7524c')
 for i,v in enumerate(p):ax.text(i,v['repairs']/1000+.4,f"{v['delta_accuracy_pp']:+.3f} pp",ha='center',fontsize=8)
 ax.axhline(0,color='0.3',lw=.8);ax.set_xticks(xx,('vs B01\nClean','vs B01\nUrb low','vs full As\nClean','vs full As\nUrb low'));ax.set(title='D  B11: paired identity repairs and new errors',ylabel='Physical samples (thousands)',ylim=(-12,10));ax.legend(frameon=False,fontsize=9,ncols=2)
 for ax in axes.flat:ax.grid(axis='y',color='0.9',lw=.6);ax.set_axisbelow(True)
 fig.suptitle('Fixed clock / practical factorial | full source | seed 2026092701 | scratch E20',fontsize=14)
 stem=output/'matched_factorial';fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
 return dict(status='PLOTS_WRITTEN',artifacts=[str(stem.with_suffix(x)) for x in ('.png','.pdf')],IQ_reads=0,CP_loads=0,raw_truth_reads=0,new_updates=0)
