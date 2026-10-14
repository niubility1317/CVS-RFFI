"""Plot saved source scalar curves; no IQ/weights, fitting or target metrics."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'SIG_TIME':'#277DA1','MATCHED_RANDOM':'#E29537','TEMPORAL_RSC':'#279578'}
LABELS={'SIG_TIME':'SIG_TIME (same probe cost)','MATCHED_RANDOM':'Matched random','TEMPORAL_RSC':'Temporal RSC'}

def plot(source,output):
    if not source.get('source_only') or source.get('query_or_new_target_metric_read') is not False or source.get('U_truth_read') is not False:
        raise ValueError('Explicit source-only scalar record required')
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing rendered snapshot')
    rows=source['rows'];max_epoch=max(r['completed_epochs'] for r in rows)
    if len(rows)!=3 or {r['arm'] for r in rows}!=set(COLORS):raise ValueError('Three fixed training arms required')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.0),dpi=180)
    for r in rows:
        name=r['arm'];color=COLORS[name];curve=r['source_curves']
        complete=[c for c in curve if c.get('actual_L_CE_stream_complete')]
        x=[c['epoch'] for c in complete]
        original=[float(c['unmasked_L_CE']) for c in complete]
        gap=[float(c['selected_L_CE']-c['unmasked_L_CE']) for c in complete]
        valid=[(c['epoch'],float(c['val_tx_acc'])) for c in curve if c.get('val_tx_acc') is not None]
        if any(not math.isfinite(v) for v in original+gap+[v for _,v in valid]):raise ValueError('Nonfinite source curve')
        axes[0,0].plot(x,original,color=color,lw=1.8,label=LABELS[name])
        axes[0,1].plot(x,gap,color=color,lw=1.8)
        axes[1,0].plot([e for e,_ in valid],[v for _,v in valid],color=color,lw=1.8)
        active=[c for c in curve if c.get('train_pseudo_total',0)>0]
        axes[1,1].plot([c['epoch'] for c in active],
            [100*c['train_pseudo_selected']/c['train_pseudo_total'] for c in active],color=color,lw=1.8,marker='o',ms=3)
    titles=('Unmasked L identity CE','Mask CE minus unmasked CE','Source V default-head accuracy','Accepted U / processed U')
    ylabels=('Cross-entropy','CE difference (same current model)','Accuracy (%) — native percent unit','Acceptance (%) — U truth hidden')
    for ax,title,ylabel in zip(axes.flat,titles,ylabels):
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold');ax.set_xlabel('Epoch');ax.set_ylabel(ylabel)
        ax.grid(alpha=.16);ax.set_xlim(1,max(2,max_epoch+.4))
        if max_epoch>=34:ax.axvline(33.5,color='#999999',lw=1,ls='--')
    axes[0,0].set_ylim(bottom=0)
    axes[0,1].axhline(0,color='#888888',lw=.7)
    minimum=min(c['val_tx_acc'] for r in rows for c in r['source_curves'] if c.get('val_tx_acc') is not None)
    axes[1,0].set_ylim(min(85,5*math.floor(minimum/5)),100)
    active_ratios=[100*c['train_pseudo_selected']/c['train_pseudo_total'] for r in rows for c in r['source_curves'] if c.get('train_pseudo_total',0)>0]
    axes[1,1].set_ylim(0,max(5,max(active_ratios,default=0)*1.25))
    if not active_ratios:
        axes[1,1].text(.5,.5,'No complete SSL epoch yet\nOriginal SSL starts at E34',ha='center',va='center',transform=axes[1,1].transAxes,color='#666666')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),frameon=False)
    fig.suptitle('Fixed SIG_TIME training mechanisms | source-only snapshot',x=.06,y=.985,ha='left',fontsize=15,fontweight='bold')
    observed=' / '.join(str(r['completed_epochs']) for r in rows)
    fig.text(.06,.944,f'One predefined model seed, three training arms | observed epochs: {observed} | fixed budget: 33 L + 17 SSL = 50',fontsize=9,color='#555555')
    fig.text(.06,.065,'Only complete saved epoch records are plotted. U acceptance is not pseudo-label precision. No target scores or checkpoint selection.',fontsize=8,color='#555555')
    fig.subplots_adjust(left=.09,right=.985,top=.88,bottom=.155,hspace=.43,wspace=.29)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)
    return dict(status='VERIFIED',run_id=source['run_id'],model_seed_count=1,training_arms=3,
        completed_epochs=[r['completed_epochs'] for r in rows],source_only=True,target_metrics_read=False,
        default_head_joint_training=True,U_truth_read=False,coverage_is_not_precision=True,figure=output.name)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=json.loads(args.input.read_text(encoding='utf8'))
    print(json.dumps(plot(value,args.output)))
