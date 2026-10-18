"""Plot saved source/formal aggregates only: no IQ, checkpoints, truth or models.

plot(output_dir, source, formal=None, logs=None) returns created PNG/PDF basenames.
All accuracy inputs are fractions; paired differences are absolute percentage points.
Missing/nonfinite observations remain missing and are never replaced with zero.
"""
import argparse
import json
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('NOAS', 'AS_RD', 'AS_CE', 'AS_DET')
COLORS = {'NOAS':'#69798C', 'AS_RD':'#C98735', 'AS_CE':'#507BAC', 'AS_DET':'#147F75'}
LABELS = {'NOAS':'No As', 'AS_RD':'As + RDrop', 'AS_CE':'As + CE', 'AS_DET':'As + CE, no satellite dropout'}
VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEW_LABELS = ('Clean','High\nsuburban','Mid\nsuburban','Low\nsuburban','High\nurban','Mid\nurban','Low\nurban')
SOURCE_SCENES = ('practical_low_suburban','practical_mid_urban','practical_low_urban')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return np.nan
    return float(value) if math.isfinite(float(value)) else np.nan


def _accuracy(value):
    v = _number(value)
    if math.isfinite(v) and not 0 <= v <= 1:
        raise ValueError('Accuracy/F1 aggregates must be fractions in [0,1]')
    return 100*v


def _style(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8)


def _missing(ax):
    ax.text(.5,.5,'No finite saved values',ha='center',va='center',transform=ax.transAxes,color='#69798C')
    ax.set_axis_off()


def _save(fig, output, name, made, note, synthetic=False):
    title = 'Single seed 2026092701 | E30, matched 6 L forwards | fixed default head'
    if synthetic:
        title = 'SYNTHETIC CHECK ONLY | ' + title
    fig.suptitle(title, fontsize=11, y=.99)
    fig.text(.015, .012, note, fontsize=8, color='#4B5665', va='bottom')
    fig.tight_layout(rect=[0,.075,1,.94])
    for suffix in ('png','pdf'):
        name_out = name+'.'+suffix
        fig.savefig(output/name_out, dpi=180, bbox_inches='tight')
        made.append(name_out)
    plt.close(fig)


def _grouped(ax, labels, records, key='accuracy', annotate=False):
    x = np.arange(len(labels))
    for j, arm in enumerate(ARMS):
        values = [_accuracy(v.get(key)) if v else np.nan for v in records[arm]]
        bars = ax.bar(x+(j-1.5)*.19, values, width=.18, color=COLORS[arm], label=LABELS[arm])
        if annotate:
            for rect, value in zip(bars,values):
                if math.isfinite(value):
                    ax.text(rect.get_x()+rect.get_width()/2,value+.1,f'{value:.3f}',ha='center',fontsize=7)
    ax.set_xticks(x,labels)
    ax.set_ylim(0,100)
    _style(ax,'Accuracy (%)')


def _source_overview(output, source, made, synthetic):
    summary = source.get('summary',{})
    if any(a not in summary for a in ARMS):
        return
    fig, axes = plt.subplots(1,2,figsize=(13.8,4.8),gridspec_kw={'width_ratios':[1,2.1]})
    vals = [_accuracy(summary[a].get('full_V_clean')) for a in ARMS]
    axes[0].bar(np.arange(4), vals, color=[COLORS[a] for a in ARMS],width=.7)
    for i,v in enumerate(vals):
        if math.isfinite(v):
            axes[0].text(i,v+.035,f'{v:.3f}',ha='center',fontsize=8)
    finite = [v for v in vals if math.isfinite(v)]
    if finite:
        axes[0].set_ylim(max(0,min(finite)-.5),min(100,max(finite)+.35))
    else:
        _missing(axes[0])
    axes[0].set_xticks(range(4),['No As','As RD','As CE','As DET'])
    axes[0].set_title('Full source V: clean health (27,000 packets)',fontsize=10)
    _style(axes[0],'Accuracy (%)')
    metrics = source.get('metrics',[])
    rec = {a:[next((m for m in metrics if m.get('arm')==a and m.get('scene')==v and m.get('draw')==-1 and m.get('dimension')=='overall'),None) for v in SOURCE_SCENES] for a in ARMS}
    if not any(r for values in rec.values() for r in values):
        plt.close(fig)
        return
    _grouped(axes[1],['Low suburban','Mid urban','Low urban'],rec)
    axes[1].set_title('Frozen source panel: both fixed channel draws pooled',fontsize=10)
    axes[1].legend(frameon=False,fontsize=8,ncol=2,loc='lower left')
    _save(fig,output,'satellite_dropout_source_overview',made,
          'Source only; 1,800 unique V packets, two observations per scene. Full-V clean values use native logs; no target causality or independent seeds.',synthetic)


def _source_strata(output, source, made, synthetic):
    metrics = source.get('metrics',[])
    chosen = [m for m in metrics if m.get('scene')=='practical_low_urban' and m.get('draw')==-1 and m.get('dimension') in ('state_lock','added_noise_ratio')]
    if not chosen:
        return
    fig, axes = plt.subplots(1,2,figsize=(15,5),gridspec_kw={'width_ratios':[1.6,1]})
    available = False
    for ax,dim,title in zip(axes,('state_lock','added_noise_ratio'),('Physical state / receiver lock','Signal-to-added-noise ratio (dB)')):
        names = list(dict.fromkeys(m['stratum'] for m in chosen if m.get('dimension')==dim))
        if not names:
            ax.text(.5,.5,'No saved nonempty strata',ha='center',transform=ax.transAxes)
            ax.set_axis_off()
            continue
        rec = {a:[next((m for m in chosen if m['arm']==a and m['dimension']==dim and m['stratum']==n),None) for n in names] for a in ARMS}
        count = []
        for i,n in enumerate(names):
            values = [r[i]['count'] for r in rec.values() if r[i]]
            if len(set(values))>1:
                raise ValueError('Paired source stratum counts differ across arms')
            count.append(values[0] if values else 0)
        labels = [n.replace('/','\n')+'\nn='+str(k)+(' *' if k<20 else '') for n,k in zip(names,count)]
        _grouped(ax,labels,rec)
        ax.set_title(title,fontsize=10)
        ax.tick_params(axis='x',labelsize=7)
        available = True
    if not available:
        plt.close(fig)
        return
    axes[0].legend(frameon=False,ncol=2,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.25))
    _save(fig,output,'satellite_dropout_source_lowurban_strata',made,
          'Source low urban only; n counts observations (two draws share physical packets), not independent packets. * n<20: descriptive only. Ratio is not total received SNR.',synthetic)


def _formal_views(output, formal, made, synthetic):
    summary = formal.get('summary',{})
    if any(a not in summary for a in ARMS):
        return
    fig, axes = plt.subplots(1,3,figsize=(16,4.7))
    x = np.arange(7)
    for ax,field,title in zip(axes,('accuracy','macro_f1','worst_RX_accuracy'),('All fixed formal views: accuracy','Macro F1 (six TX classes)','Worst receiver accuracy')):
        finite_values = []
        for arm in ARMS:
            values = [_accuracy(summary[arm].get('views',{}).get(v,{}).get(field)) for v in VIEWS]
            finite_values += [v for v in values if math.isfinite(v)]
            ax.plot(x,values,marker='o',ms=4,lw=1.5,color=COLORS[arm],label=LABELS[arm])
        ax.set_xticks(x,VIEW_LABELS)
        ax.set_ylim(0,100)
        ax.set_title(title,fontsize=10)
        _style(ax,'%' if field=='macro_f1' else 'Accuracy (%)')
        if not finite_values:
            _missing(ax)
    axes[0].legend(frameon=False,fontsize=8,ncol=1,loc='lower left')
    _save(fig,output,'satellite_dropout_formal_views',made,
          'Frozen predictions; unchanged received IQ/channels. CE/dropout changes apply to satellite views; ordinary RDrop remains. One seed; lines only connect views.',synthetic)


def _formal_deltas(output, formal, made, synthetic):
    comparisons = formal.get('comparisons',{})
    keys = ('AS_DET__NOAS','AS_DET__AS_RD','AS_DET__AS_CE')
    if not all(k in comparisons for k in keys):
        return
    fig, axes = plt.subplots(2,2,figsize=(13.4,8.1))
    for ax,key,title in zip(axes.flat,keys,('As DET minus no As','As DET minus As RDrop','As DET minus As CE: dropout attribution')):
        v = [_number(comparisons[key].get(x,{}).get('accuracy_delta_pp')) for x in VIEWS]
        ax.bar(np.arange(7),v,color=['#147F75' if z>=0 else '#B85856' for z in v],width=.65)
        ax.axhline(0,color='#4B5665',lw=.8)
        finite = [z for z in v if math.isfinite(z)]
        scale = max(.1,max((abs(z) for z in finite),default=.1))
        if finite:
            ax.set_ylim(min(0,min(finite))-.18*scale,max(0,max(finite))+.18*scale)
        for i,z in enumerate(v):
            if math.isfinite(z):
                ax.text(i,z+(scale*.025 if z>=0 else -scale*.025),f'{z:+.3f}',ha='center',va='bottom' if z>=0 else 'top',fontsize=8)
        ax.set_xticks(range(7),VIEW_LABELS)
        ax.set_title(title,fontsize=10)
        _style(ax,'Accuracy difference (pp)')
        if not finite:
            _missing(ax)
    ax = axes[1,1]
    vals = [_number(comparisons[k].get('clean',{}).get('accuracy_delta_pp')) for k in keys]
    ax.bar(range(3),vals,color=['#147F75' if v>=0 else '#B85856' for v in vals],width=.55)
    ax.axhline(0,color='#4B5665',lw=.8)
    ax.hlines(-.1,-.4,.4,color='#B05468',linestyle='--',lw=1.4,label='Stage limit: DET - no As = -0.1 pp')
    finite = [v for v in vals if math.isfinite(v)]+[-.1,0]
    span = max(.1,max(finite)-min(finite))
    ax.set_ylim(min(finite)-span*.2,max(finite)+span*.28)
    for i,v in enumerate(vals):
        if math.isfinite(v):
            ax.text(i,v+(.025*span if v>=0 else -.025*span),f'{v:+.4f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
    ax.set_xticks(range(3),['DET - no As','DET - As RD','DET - As CE'])
    if not any(math.isfinite(v) for v in vals):
        _missing(ax)
    ax.set_title('Clean detail: absolute percentage-point cost',fontsize=10)
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    _style(ax,'Accuracy difference (pp)')
    _save(fig,output,'satellite_dropout_formal_paired_deltas',made,
          'All differences use the same physical packets. -0.1 pp is an absolute clean difference; the stage guard applies to the matched no-As comparison only.',synthetic)


def _formal_rx(output, formal, made, synthetic):
    metrics = formal.get('metrics',[])
    selected = ('clean','practical_mid_urban','practical_low_urban')
    rows = [m for m in metrics if m.get('dimension')=='receiver' and m.get('view') in selected]
    if not rows:
        return
    names = sorted(set(str(m['stratum']) for m in rows),key=lambda x:(0,int(x)) if x.isdigit() else (1,x))
    fig, axes = plt.subplots(1,3,figsize=(15.8,4.6))
    for ax,view,title in zip(axes,selected,('Clean','Mid urban','Low urban')):
        for arm in ARMS:
            rec = [next((m for m in rows if m['arm']==arm and m['view']==view and str(m['stratum'])==n),None) for n in names]
            vals = [_accuracy(m.get('accuracy')) if m else np.nan for m in rec]
            ax.plot(range(len(names)),vals,marker='o',lw=1.4,ms=4,label=LABELS[arm],color=COLORS[arm])
        ax.set_xticks(range(len(names)),['RX '+n for n in names])
        ax.set_ylim(0,100)
        ax.set_title(title,fontsize=10)
        _style(ax,'Accuracy (%)')
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    _save(fig,output,'satellite_dropout_formal_RX',made,
          'Formal receiver strata use saved receiver metadata; no receiver is omitted. A single trained seed cannot establish repeatability.',synthetic)


def _logs(output, logs, made, synthetic):
    rows = logs.get('rows',{})
    arm_names = {'NOAS':'B00','AS_RD':'B01','AS_CE':'AS_CE','AS_DET':'AS_DET'}
    if not any(rows.get(a,{}).get('epoch_curves') for a in arm_names.values()):
        return
    fig, axes = plt.subplots(1,2,figsize=(13,4.8))
    any_curve = False
    for arm,name in arm_names.items():
        c = rows.get(name,{}).get('epoch_curves',[])
        if not c:
            continue
        epochs = [_number(r.get('epoch')) for r in c]
        health = [_number(r.get('val_tx_acc')) for r in c]
        if any(math.isfinite(v) and not 0 <= v <= 100 for v in health):
            raise ValueError('Native val_tx_acc must be percent in [0,100]')
        loss = [_number(r.get('total_L')) for r in c]
        if any(math.isfinite(v) for v in health):
            axes[0].plot(epochs,health,color=COLORS[arm],lw=1.3,label=LABELS[arm])
            any_curve = True
        if any(math.isfinite(v) for v in loss):
            axes[1].plot(epochs,loss,color=COLORS[arm],lw=1.3,label=LABELS[arm])
            any_curve = True
    if not any_curve:
        plt.close(fig)
        return
    for ax in axes:
        ax.axvline(20.5,lw=1,ls=':',color='#7B657C')
        ax.set_xlabel('Cumulative epoch')
        ax.set_xticks([1,5,10,15,20,25,30])
        _style(ax)
    axes[0].set_ylabel('Source V accuracy (%)')
    axes[0].set_title('Full-V default-head health',fontsize=10)
    axes[0].set_ylim(0,100)
    axes[0].legend(frameon=False,fontsize=8,loc='lower right')
    axes[1].set_ylabel('Mean supervised L objective per update')
    axes[1].set_title('Actual six-forward L objective (not a success metric)',fontsize=10)
    _save(fig,output,'satellite_dropout_source_training',made,
          'Source training logs only. Dotted line: own E20 to E20+10 optimizer reset. These curves do not establish formal performance or As benefit.',synthetic)


def plot(output_dir, source, formal=None, logs=None):
    """Save source and available frozen formal/log figures; return created basenames."""
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    source = source or {}
    if source.get('model_seed_count',1)!=1 or (formal and formal.get('model_seed_count',1)!=1):
        raise ValueError('This figure module labels single seed 2026092701 only')
    if formal and formal.get('model_seed',2026092701)!=2026092701:
        raise ValueError('Figure label/model seed mismatch')
    synthetic = any(v and v.get('synthetic',False) for v in (source,formal,logs))
    made = []
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42}):
        _source_overview(output,source,made,synthetic)
        _source_strata(output,source,made,synthetic)
        if formal:
            _formal_views(output,formal,made,synthetic)
            _formal_deltas(output,formal,made,synthetic)
            _formal_rx(output,formal,made,synthetic)
        if logs:
            _logs(output,logs,made,synthetic)
    return made


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--formal')
    parser.add_argument('--logs')
    args = parser.parse_args()
    def read(p):
        return json.loads(Path(p).read_text(encoding='utf-8-sig')) if p else None
    print(json.dumps({'created':plot(args.output,read(args.source),read(args.formal),read(args.logs)),
                      'IQ_reads':0,'checkpoint_reads':0,'NN':0,'synthetic':bool(read(args.source).get('synthetic',False))}))
