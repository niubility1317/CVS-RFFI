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
    arms=('NATIVE','REPEAT','CONCAT')
    if set(result['summary']) != set(arms):
        raise ValueError('The full fixed matrix is required for comparison plots')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    colors=('#7d8b9b','#2563a6','#ce7628')
    fig,axes=plt.subplots(2,2,figsize=(14,8.6),layout='constrained')
    ax=axes[0,0];x=list(range(7))
    for j,arm in enumerate(arms):
        values=[100*result['summary'][arm][v]['accuracy'] for v in VIEWS]
        ax.bar([i+(j-1)*.25 for i in x],values,width=.25,color=colors[j],label=arm)
    ax.set_xticks(x,LABELS,rotation=25,ha='right')
    ax.set(title='A  All seven fixed views',ylabel='Accuracy (%)',ylim=(0,100))
    ax.legend(frameon=False,fontsize=9)
    ax=axes[0,1]
    for offset,key,color,label in ((-.18,'CONCAT-REPEAT',colors[2],'Practical vs matched repeat'),
                                  (.18,'REPEAT-NATIVE',colors[1],'Extra-forward effect')):
        values=[result['comparisons'][key][v]['delta_accuracy_pp'] for v in VIEWS]
        ax.bar([i+offset for i in x],values,width=.36,color=color,label=label)
    ax.axhline(0,color='0.25',lw=.8)
    ax.set_xticks(x,LABELS,rotation=25,ha='right')
    ax.set(title='B  Separate practical and forward effects',ylabel='Paired accuracy difference (pp)')
    ax.margins(y=.22)
    ax.legend(frameon=False,fontsize=9)
    ax=axes[1,0]
    for j,arm in enumerate(arms):
        values=[result['suburban_gap_to_own_clean_pp'][arm][v] for v in VIEWS[1:4]]
        ax.bar([i+(j-1)*.25 for i in range(3)],values,width=.25,color=colors[j],label=arm)
    ax.axhline(2,color='0.25',ls='--',lw=1,label='Exploratory gap 2 pp')
    ax.set_xticks(range(3),LABELS[1:4])
    ax.set(title='C  Gap to each model own clean',ylabel='Clean minus suburban (pp)')
    ax.set_ylim(top=max(result['suburban_gap_to_own_clean_pp'][a][v] for a in arms for v in VIEWS[1:4])+12)
    ax.legend(frameon=False,fontsize=9,ncols=2)
    ax=axes[1,1];paired=result['comparisons']['CONCAT-REPEAT']
    ax.bar([i-.18 for i in x],[paired[v]['repairs']/1000 for v in VIEWS],width=.36,color='#29856f',label='Repaired repeat errors')
    ax.bar([i+.18 for i in x],[-paired[v]['new_errors']/1000 for v in VIEWS],width=.36,color='#b65c5c',label='New errors vs repeat')
    ax.axhline(0,color='0.25',lw=.8)
    ax.set_xticks(x,LABELS,rotation=25,ha='right')
    ax.set(title='D  Practical vs repeat: repaired and new errors',ylabel='Packets (thousands; new errors below zero)')
    ax.margins(y=.18)
    ax.legend(frameon=False,fontsize=9)
    for ax in axes.flat:ax.grid(axis='y',color='0.9',lw=.6);ax.set_axisbelow(True)
    fig.suptitle('Full-source F + CROSS_RX | fixed seed 2026092701 | scratch E20',fontsize=14)
    stem=output/'matched_matrix'
    for suffix in ('.png','.pdf'):fig.savefig(stem.with_suffix(suffix),dpi=180)
    plt.close(fig)
    return dict(status='PLOTS_WRITTEN',rows=list(arms),model_seed_count=1,
                IQ_reads=0,CP_loads=0,raw_truth_reads=0,model_forwards=0,new_updates=0,
                artifacts=[str(stem.with_suffix(s)) for s in ('.png','.pdf')])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('metadata_json');parser.add_argument('output')
    parser.add_argument('--arm',choices=('NATIVE','REPEAT','CONCAT'))
    parser.add_argument('--matrix',action='store_true')
    args=parser.parse_args()
    data=json.loads(Path(args.metadata_json).read_text(encoding='utf-8-sig'))
    if args.matrix:
        print(json.dumps(render_matrix(data.get('result',data),args.output)));return
    if 'rows' in data:
        if args.arm is None:parser.error('--arm is needed for a matrix collection')
        row=data['rows'][args.arm];name=args.arm
    else:row=data;name=args.arm or data.get('row','fixed E20')
    print(json.dumps(render_row(row,args.output,name)))


if __name__=='__main__':main()
