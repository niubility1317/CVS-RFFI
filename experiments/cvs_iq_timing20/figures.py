"""Scientific plots from saved source evidence only; no model or data loading."""
import csv
import json
from .harvest import OUT,ROWS


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (OUT/'training_curves.csv').open(encoding='utf8',newline='') as f:curves=list(csv.DictReader(f))
    with (OUT/'source_strata.csv').open(encoding='utf8',newline='') as f:strata=list(csv.DictReader(f))
    summary=json.loads((OUT/'analysis_summary.json').read_text(encoding='utf8'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'B00':'#356fa0','B10':'#bd6234'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    fields=[('timing/total_L','Weighted labeled objective'),('train_loss_tx_labeled','Raw labeled CE'),('val_tx_acc','Seen-source V accuracy (%)'),('timing/raw_KL','Raw within-IQ RDrop KL'),('train_loss_unlabeled','Original U loss'),('lr','Native cosine learning rate')]
    for row in ROWS:
        arm=row[:3];records=[v for v in curves if v['row']==row];epoch=[int(v['epoch']) for v in records]
        for ax,(key,title) in zip(axes.flat,fields):
            values=[float(v[key]) for v in records]
            ax.plot(epoch,values,label=arm,color=colors[arm]);ax.set(title=title,xlabel='Epoch',xlim=(1,20));ax.axvline(13.5,color='#999999',ls=':',lw=1)
            ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle('F / E20 / source RX1 held out / seed 2026092701\nAll 20 epochs; dotted line: original U stage starts',fontsize=12)
    fig.savefig(OUT/'complete_training_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    arms=['B00','B10'];x=[0,1]
    for arm,pos in zip(arms,x):
        r=summary['results'][arm]
        axes[0].bar(pos,100*r['accuracy'],color=colors[arm],width=.6)
        axes[0].text(pos,100*r['accuracy']+.5,f"{100*r['accuracy']:.2f}%",ha='center')
    axes[0].set(xticks=x,xticklabels=arms,ylabel='Held RX1 accuracy (%)',ylim=(0,100),title='Fixed final20 default head')
    for ax,dimension in zip(axes[1:],('TX','day')):
        records=[r for r in strata if r['dimension']==dimension]
        values=[float(r['gain_pp']) for r in records]
        ax.bar([str(r['value']) for r in records],values,color=['#548f68' if v>=0 else '#bd6234' for v in values]);ax.axhline(0,color='#555555',lw=.8)
        ax.set(title=dimension+' strata',xlabel=dimension,ylabel='B10 minus B00 (pp)');ax.grid(axis='y',alpha=.18)
    q=summary['qualification']
    fig.suptitle('Source qualification: '+('PASS' if q['qualified'] else 'FAIL')+' | one seed / one held RX / no target',fontsize=12)
    fig.savefig(OUT/'held_source_results.png');plt.close(fig)
    print(OUT/'complete_training_curves.png');print(OUT/'held_source_results.png')


if __name__=='__main__':main()
