"""Standalone scientific figures from completed, independently recounted source evidence."""
import csv,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'local_artifacts/F_shift_bypass20_20261016/final'

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    a=json.loads((OUT/'analysis_summary.json').read_text(encoding='utf8'))
    with (OUT/'training_curves.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
    with (OUT/'source_strata.csv').open(encoding='utf8',newline='') as f:strata=list(csv.DictReader(f))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'B00':'#306b9c','B10':'#c36935'};views=list(a['scores']['B00']);labels=['clean','high sub','mid sub','low sub','high urban','mid urban','low urban']
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
    panels=[('total_L','Weighted L objective + shared SupCon'),('raw_CE','Raw IQ CE'),('ordinary_CE','Ordinary IQ CE (full / shift bypass)'),('s_slot_CE','Third raw slot CE (both arms raw)'),('raw_KL','Within-raw RDrop KL'),('train_loss_unlabeled','Original hidden-U loss'),('val_tx_acc','Native V accuracy (as logged)'),('lr','Native cosine learning rate')]
    for arm in ('B00','B10'):
        values=[v for v in curves if v['arm']==arm];epochs=[int(v['epoch']) for v in values]
        assert epochs==list(range(1,21))
        for ax,(key,title) in zip(axes.flat,panels):
            ax.plot(epochs,[float(v[key]) for v in values],color=colors[arm],label=arm)
            ax.set(title=title,xlabel='Epoch',xlim=(1,20));ax.axvline(5.5,color='#888888',ls='--',lw=.8);ax.axvline(13.5,color='#888888',ls=':',lw=.8);ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle('F + CROSS_RX / ordinary shift bypass / source RX1 holdout / seed'+str(a['model_seed'])+' / all20 epochs\nDashed: contrast starts; dotted: original U stage starts',fontsize=12)
    fig.savefig(OUT/'complete_training_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4.7),layout='constrained');x=np.arange(7)
    for arm,offset in (('B00',-.18),('B10',.18)):
        values=[100*a['scores'][arm][v]['accuracy'] for v in views];axes[0].bar(x+offset,values,width=.35,color=colors[arm],label=arm)
        for xx,yy in zip(x+offset,values):axes[0].text(xx,yy+.7,f'{yy:.1f}',ha='center',fontsize=8)
        gaps=[a['suburban_gaps'][arm][v]['absolute_gap_pp'] for v in views[1:4]];axes[1].bar(np.arange(3)+offset,gaps,width=.35,color=colors[arm],label=arm)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Default-head source accuracy (%)',title='Fixed final20 / same1260 physical source packets')
    axes[0].tick_params(axis='x',rotation=25);axes[0].legend(frameon=False)
    axes[1].set(xticks=np.arange(3),xticklabels=labels[1:4],ylabel='Absolute accuracy gap to own clean (pp)',title='Source suburban closeness')
    axes[1].axhline(2,color='#666666',ls='--',lw=1,label='Exploratory2pp, not a completed joint goal');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Source qualification: '+('PASS' if a['qualification']['qualified'] else 'FAIL')+' | one seed / no formal target or cross-seed claim',fontsize=12)
    fig.savefig(OUT/'held_source_results.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,dimension,count in zip(axes,('TX','day'),(6,3)):
        numbers=list(range(6)) if dimension=='TX' else [1,2,3]
        array=np.array([[float(next(v['gain_pp'] for v in strata if v['dimension']==dimension and int(v['value'])==n and v['view']==view)) for view in views] for n in numbers])
        limit=max(1.,float(np.abs(array).max()));im=ax.imshow(array,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set(xticks=np.arange(7),xticklabels=labels,yticks=np.arange(count),yticklabels=numbers,title=dimension+' strata: B10 minus B00 (pp)',ylabel=dimension);ax.tick_params(axis='x',rotation=35)
        for i in range(count):
            for j in range(7):ax.text(j,i,f'{array[i,j]:+.1f}',ha='center',va='center',fontsize=8,color='black' if abs(array[i,j])<limit*.7 else 'white')
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle('Every registered source stratum; no best-class or best-day selection',fontsize=12)
    fig.savefig(OUT/'source_strata.png');plt.close(fig)
    print('Three completed-source figures saved')

if __name__=='__main__':
    import sys
    if sys.argv[1:]:raise ValueError('No figure selection overrides')
    main()
