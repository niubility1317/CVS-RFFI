"""Plot already-scored R17 aggregates; no model, IQ, truth or prediction access."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--summary',type=Path,default=root/'automation_reports/CV-SincNet/'
        '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17/summary.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=json.loads(args.summary.read_text(encoding='utf-8-sig'))
    assert summary['status']=='ANALYZED' and summary['source_decision']['selected']=='unroll_unshared'
    source=summary['target']['unroll_unshared'];control=summary['target']['single_shot_reference']
    assert source['role']=='target' and source['count']==168000 and source['all_predictions_before_truth']
    assert source['original_default_head'] and source['F_seed']==2026092701 and source['head_fit_updates']==0
    views=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
    def record(data,view,dimension='overall',value='all'):
        rows=[r for r in data['records'] if r['view']==view and r['dimension']==dimension and r['value']==value and r['readout']=='restored']
        assert len(rows)==1
        row=rows[0];matrix=np.asarray(row['confusion'])
        assert matrix.sum()==row['count'] and abs(100*matrix.trace()/matrix.sum()-row['accuracy'])<1e-9
        return row
    raw=np.array([record(source,v)['baseline_accuracy'] for v in views])
    restored=np.array([record(source,v)['accuracy'] for v in views])
    threshold=np.array([raw[0]-5.]*5+[70.]);passed=restored[1:]>=threshold
    assert passed.tolist()==[True,False,False,True,False,False]
    txold=np.array([record(control,views[-1],'TX',str(i))['accuracy'] for i in range(6)])
    txnew=np.array([record(source,views[-1],'TX',str(i))['accuracy'] for i in range(6)])
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,9.3),gridspec_kw={'height_ratios':[1.25,1]})
    fig.subplots_adjust(left=.08,right=.98,bottom=.11,top=.88,hspace=.55)
    x=np.arange(7);ax=axes[0]
    ax.bar(x-.18,raw,.35,label='Original received IQ',color='#94a3b8')
    ax.bar(x+.18,restored,.35,label='R17 restored IQ',color='#0f766e')
    ax.scatter(x[1:]+.18,threshold,marker='_',s=600,color='#c2410c',linewidths=2.5,
               label='Required per-scene accuracy',zorder=5)
    for i,value in enumerate(restored):
        near_goal=i>0 and abs(value-threshold[i-1])<3.
        ax.text(i+.18,value-7.5 if near_goal else value+1.5,f'{value:.2f}',ha='center',
                fontsize=9,color='white' if near_goal else '#111827')
    ax.set_xticks(x,['Clean','High\nsuburban','Mid\nsuburban','Low\nsuburban','High\nurban','Mid\nurban','Low\nurban'])
    ax.set_ylim(0,115);ax.set_ylabel('Accuracy (%)');ax.set_title('Formal seven-scene evaluation',loc='left',pad=12)
    ax.legend(loc='upper right',frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1];x=np.arange(6)
    ax.bar(x-.18,txold,.35,label='R16 fixed single-shot control',color='#94a3b8')
    ax.bar(x+.18,txnew,.35,label='R17 restored IQ',color='#0f766e')
    for i,value in enumerate(txnew):ax.text(i+.18,value+2,f'{value:.2f}',ha='center',fontsize=9)
    ax.set_xticks(x,['14-10','14-7','20-15','20-19','6-15','8-20']);ax.set_ylim(0,108)
    ax.set_ylabel('Low urban accuracy (%)');ax.set_xlabel('Transmitter')
    ax.set_title('Every TX improves relative to the fixed control; TX 14-7 remains weak',loc='left',pad=12)
    ax.legend(loc='upper left',frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Recovery helps, but the requested performance targets are not met',fontsize=15,y=.97)
    fig.text(.08,.915,'Frozen original F50 / default head | seed 2026092701 | 168,000 physical records',fontsize=10,color='#334155')
    fig.text(.08,.035,'Goal: low urban >=70%; each other impaired scene >=79.6762% (original clean minus 5 pp).\n'
             'Clean damage: 0.0893 pp. One restoration seed; correlated views. Aggregate evidence does not prove every restored detail.',
             fontsize=9,color='#334155')
    for suffix in ('png','svg'):
        path=args.output/('formal_recovery.'+suffix);fig.savefig(path,dpi=200)
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (args.output/'metrics.json').write_text(json.dumps(dict(run='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17',
        original_received_accuracy=raw.tolist(),restored_accuracy=restored.tolist(),impaired_thresholds=threshold.tolist(),passed=passed.tolist(),
        TX_control_accuracy=txold.tolist(),TX_restored_accuracy=txnew.tolist(),new_predictions=0),indent=2)+'\n',encoding='utf-8')
    print(args.output/'formal_recovery.png')


if __name__=='__main__':main()
