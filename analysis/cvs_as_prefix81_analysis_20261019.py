"""P81 saved-output analysis only: no model, checkpoint, IQ or truth loading."""
import argparse,csv,gzip,json,math,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

REFERENCE_RUN='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
NEW={'RANDOM30':'RANDOM-s2026092701','PREFIX30':'PREFIX-s2026092701'}
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SOURCE_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
COST_KEYS=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def records(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as stream:
        for line in stream:
            if not line.endswith('\n') or not line.strip():raise ValueError('Incomplete structured record: '+path.name)
            yield json.loads(line)


def csv_write(path,rows):
    fields=sorted({k for row in rows for k in row})
    with Path(path).open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def metric(y,p,rx):
    if not len(y):return dict(count=0,correct=0,accuracy=None,macro_F1=None,worst_RX=None)
    cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
    return dict(count=len(y),correct=int((y==p).sum()),accuracy=float((y==p).mean()),macro_F1=float(f.mean()),worst_RX=float(min((y[rx==r]==p[rx==r]).mean() for r in np.unique(rx))))


def paired(y,new,old,cohort):
    a,b=new==y,old==y;repair=int((a&~b).sum());error=int((~a&b).sum());n=len(y);c=int(cohort.sum())
    return dict(count=n,repaired=repair,new_errors=error,net=repair-error,accuracy_pp=100*(repair-error)/n if n else None,
                fixed_NOAS30_clean_correct_count=c,new_cohort_correct=int((a&cohort).sum()),control_cohort_correct=int((b&cohort).sum()),
                new_cohort_retention=float((a&cohort).sum()/c) if c else None)


def logical_rows(root):
    ref=root.parent/REFERENCE_RUN
    return [('NOAS30',ref/'B00-s2026092701'),('AS30',ref/'B01-s2026092701'),*[(arm,root/row) for arm,row in NEW.items()]]


def source_tables(root,source):
    data={};base=None;metrics=[];effects=[]
    for arm,row in logical_rows(root):
        with np.load(row/'source_qualification/predictions.npz',allow_pickle=False) as packed:
            x={k:packed[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
        if x['ids'].shape!=(1800,) or len(set(x['ids'].tolist()))!=1800 or x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Fixed finite 1800/two-draw source predictions required')
        if base is None:base=x
        if any(not np.array_equal(x[k],base[k]) for k in ('ids','TX','RX','day')):raise ValueError('Source physical panels differ')
        data[arm]=x
    cohort=data['NOAS30']['clean'].argmax(1)==base['TX']
    for scene in ('clean',*SOURCE_SCENES):
        for draw in ((-1,) if scene=='clean' else (-1,0,1)):
            ds=(0,1) if draw==-1 else (draw,);copies=1 if scene=='clean' else len(ds)
            y=np.tile(base['TX'],copies);rx=np.tile(base['RX'],copies);day=np.tile(base['day'],copies);co=np.tile(cohort,copies)
            pred={a:(x['clean'].argmax(1) if scene=='clean' else np.concatenate([x['satellite'][d,SOURCE_SCENES.index(scene)].argmax(1) for d in ds])) for a,x in data.items()}
            groups=[('overall','ALL',np.ones(len(y),bool))]
            for dim,values in [('RX',rx),('day',day),('TX',y)]:groups.extend((dim,str(v),values==v) for v in np.unique(values))
            for dim,label,mask in groups:
                context=dict(scene=scene,draw=draw,dimension=dim,stratum=label)
                for arm,p in pred.items():
                    n=int(co[mask].sum());correct=int(((p==y)&co&mask).sum())
                    metrics.append(dict(arm=arm,**context,**metric(y[mask],p[mask],rx[mask]),fixed_NOAS30_clean_correct_count=n,cohort_correct=correct,cohort_retention=correct/n if n else None))
                for key in source['paired']:
                    a,b=key.split('_minus_');effects.append(dict(comparison=key,**context,**paired(y[mask],pred[a][mask],pred[b][mask],co[mask])))
    return metrics,effects


def bucket(value):
    if value is None or not isinstance(value,(int,float)) or not math.isfinite(value):return 'MISSING'
    return '<3' if value<3 else '3_to_8' if value<8 else '8_to_15' if value<15 else '>=15'


def mask_summary(batches,mode,expected_steps=2220):
    groups={};hist=Counter();plans=Counter();epochs=defaultdict(Counter);steps=0
    def add(key,r):
        g=groups.setdefault(key,dict(packets=0,masked=0,original_power=0.,removed_power=0.,removed_fraction_sum=0.))
        g['packets']+=1;g['masked']+=int(r['enabled']);g['original_power']+=r['original_power'];g['removed_power']+=r['removed_power'];g['removed_fraction_sum']+=r['removed_power_fraction']
    for steps,batch in enumerate(batches,1):
        if batch['step']!=steps or batch['cumulative_step']!=4440+steps or batch['mode']!=mode or batch['packets']!=64 or len(batch['records'])!=64:raise ValueError('Incomplete mask batch sequence')
        epoch=20+(steps-1)//222+1
        for r in batch['records']:
            if r['mode']!=mode or type(r['enabled']) is not bool or r['length']!=61 or r['end']!=r['begin']+61 or not 0<=r['begin']<=195 or not (r['end']<=80 or r['begin']>=160):raise ValueError('Mask/reference geometry differs')
            if mode=='PREFIX' and r['begin']!=0:raise ValueError('Prefix position differs')
            for name in ('original_power','removed_power','removed_power_fraction','output_power'):
                if not isinstance(r[name],(int,float)) or not math.isfinite(r[name]) or r[name]<-1e-12:raise ValueError('Invalid actual mask power: '+name)
            if abs(r['original_power']-r['removed_power']-r['output_power'])>1e-9*max(1.,r['original_power']):raise ValueError('Mask energy arithmetic differs')
            scene=str(r.get('scene','MISSING'));state=str(r.get('state','MISSING'));lock=str(r.get('lock','MISSING'));quality=bucket(r.get('signal_to_added_noise_db'))
            for key in [('overall','ALL'),('scene',scene),('state',state),('lock',lock),('signal_to_added_noise_db',quality),('RX',str(r.get('RX','MISSING'))),('day',str(r.get('day','MISSING'))),('scene_state_addednoise',json.dumps([scene,state,quality]))]:add(key,r)
            plans[r['begin']]+=1
            if r['enabled']:hist[r['begin']]+=1
            epochs[epoch].update(packets=1,masked=int(r['enabled']),original_power=r['original_power'],removed_power=r['removed_power'],removed_fraction_sum=r['removed_power_fraction'])
    if steps!=expected_steps:raise ValueError('Missing final mask updates')
    table=[]
    for (dim,label),g in sorted(groups.items()):
        table.append(dict(mode=mode,dimension=dim,stratum=label,**g,masked_fraction=g['masked']/g['packets'],mean_removed_power_fraction=g['removed_fraction_sum']/g['packets'],energy_weighted_removed_fraction=g['removed_power']/g['original_power'] if g['original_power'] else None))
    return dict(mode=mode,steps=steps,groups=table,begin_histogram={str(k):v for k,v in sorted(hist.items())},planned_begin_histogram={str(k):v for k,v in sorted(plans.items())},epochs=[dict(epoch=k,**v) for k,v in sorted(epochs.items())],quality_definition='Pre-erasure signal_to_added_noise_db; this is not total SNR')


def native_cost(result):
    values={k:result[k] for k in COST_KEYS}
    if any(type(v) is not int or v<0 for v in values.values()):raise ValueError('Actual native NN counters required')
    return dict(**values,native_NN=sum(values.values()),wall_seconds=result.get('wall_seconds'))


def training(root,out):
    epoch_curves=[];step_curves=[];mask={};costs={};missing=[]
    for arm,row in logical_rows(root):
        new=arm in NEW
        parent=(root.parent/REFERENCE_RUN/'B01-s2026092701/stage20') if new else row/'stage20'
        for stage,folder,offset,expected in [('parent20',parent,0,20),('tail10',row/'source',20,10)]:
            path=folder/'epoch_metrics.jsonl'
            if not path.exists():
                if new and stage=='tail10':raise FileNotFoundError(path)
                missing.append(arm+'/'+stage+'/epoch_metrics.jsonl');continue
            values=list(records(path))
            if len(values)!=expected or [v['epoch'] for v in values]!=list(range(1,expected+1)):raise ValueError('Incomplete full epoch telemetry')
            union=set().union(*(v.keys() for v in values))
            for v in values:epoch_curves.append(dict({k:v.get(k) for k in union},arm=arm,stage=stage,cumulative_epoch=offset+v['epoch']))
        if not new:continue
        folder=row/'source';done=read(folder/'completion.json');result=read(folder/'result.json')
        if (done['epoch'],done['optimizer_steps'],result['optimizer_steps'])!=(10,2220,2220) or done['target_access'] or result['target_access'] or result['U_truth_read'] or result['head_fit_updates']:raise ValueError('Fixed legal source E10 tail required')
        refs=list(records(folder/'reference_steps.jsonl.gz'))
        if len(refs)!=2220 or any(r['step']!=i or r['cumulative_step']!=4440+i for i,r in enumerate(refs,1)):raise ValueError('Incomplete original L objective steps')
        union=set().union(*(r.keys() for r in refs))
        step_curves.extend(dict({k:r.get(k) for k in union},arm=arm,cumulative_epoch=20+(i-1)//222+1) for i,r in enumerate(refs,1))
        mask[arm]=mask_summary(records(folder/'mask_steps.jsonl.gz'),NEW[arm].split('-')[0])
        q=read(row/'source_qualification/complete.json');p=read(row/'prediction/complete.json')
        if q['NN']!=12600 or p['actual_NN']!=1176002:raise ValueError('Source and formal forward costs differ')
        costs[arm]=dict(**native_cost(result),new_epochs=10,new_updates=2220,source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],formal_prediction_wall_seconds=p.get('wall_seconds'),source_readout_wall_seconds=q.get('wall_seconds'))
        costs[arm]['new_NN']=costs[arm]['native_NN']+q['NN']+p['actual_NN']
        pc=parent/'result.json';costs[arm]['reused_As20_parent_cost']=native_cost(read(pc)) if pc.exists() else dict(status='NOT_AVAILABLE')
    smoke=read(root/'real_checkpoint_smoke.json')
    if smoke['status']!='PASS' or smoke['NN']!=1540:raise ValueError('Actual shared real-checkpoint smoke cost differs')
    csv_write(out/'training_epoch_curves.csv',epoch_curves);csv_write(out/'training_reference_steps.csv',step_curves)
    with gzip.open(out/'training_reference_steps.jsonl.gz','xt',encoding='utf8') as stream:
        for row in step_curves:stream.write(json.dumps(row,allow_nan=False)+'\n')
    write(out/'training_epoch_curves.json',epoch_curves)
    cost=dict(rows=costs,new_model_epochs=20,new_updates=4440,new_shared_smoke_NN=1540,new_total_NN=1540+sum(c['new_NN'] for c in costs.values()),new_native_NN=sum(c['native_NN'] for c in costs.values()),new_training_process_seconds=sum(c['wall_seconds'] for c in costs.values() if c['wall_seconds'] is not None),wall_is_process_seconds_not_elapsed=True,shared_As20_parent_is_reused_once_not_new_training=True,model_seed_count=1,source_readout_wall_missing=any(c['source_readout_wall_seconds'] is None for c in costs.values()),old_readouts_not_repeated=True,analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0)
    write(out/'costs.json',cost);write(out/'mask_summary.json',mask)
    csv_write(out/'mask_coverage.csv',[dict(arm=a,**g) for a,v in mask.items() for g in v['groups']])
    csv_write(out/'mask_begin_histograms.csv',[dict(arm=a,begin=int(k),planned_count=n,enabled_count=v['begin_histogram'].get(k,0)) for a,v in mask.items() for k,n in v['planned_begin_histogram'].items()])
    return epoch_curves,step_curves,mask,cost,missing


def plots(out,formal,epochs,steps,mask,source_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[];colors={'NOAS30':'#64748b','AS30':'#d97706','RANDOM30':'#16a34a','PREFIX30':'#2563eb'}
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=160);files.append(name+'.'+ext)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,4.6));xx=np.arange(7)
    for arm,views in formal['summary'].items():axes[0].plot(xx,[100*views[v]['accuracy'] for v in VIEWS],marker='.',label=arm,color=colors[arm])
    for key in ('PREFIX30_minus_AS30','RANDOM30_minus_AS30','PREFIX30_minus_RANDOM30'):axes[1].plot(xx,[formal['paired_accuracy_pp'][key][v] for v in VIEWS],marker='.',label=key)
    for ax,title in zip(axes,('Fixed default-head accuracy (%)','Paired absolute accuracy change (pp)')):
        ax.set(xticks=xx,xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'],title=title);ax.tick_params(axis='x',labelrotation=25);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[1].axhline(0,color='grey',lw=.7);save(fig,'formal_all7_paired')
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    for arm in NEW:
        ss=[r for r in steps if r['arm']==arm]
        for ax,key in zip(axes.flat[:3],('common_CE','extra_CE','total_L')):
            ax.plot([r['cumulative_step'] for r in ss],[np.nan if r.get(key) is None else r[key] for r in ss],label=arm,color=colors[arm],lw=.5,alpha=.8);ax.set(title=key+' / every logged update',xlabel='Cumulative optimizer update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for arm in colors:
        ss=[r for r in epochs if r['arm']==arm]
        axes[1,1].plot([r['cumulative_epoch'] for r in ss],[np.nan if r.get('val_tx_acc') is None else r['val_tx_acc'] for r in ss],label=arm,color=colors[arm])
    axes[1,1].set(title='All saved epoch source V val_tx_acc (native units)',xlabel='Cumulative epoch');axes[1,1].axvline(20.5,color='grey',ls='--');axes[1,1].grid(alpha=.2);axes[1,1].legend(fontsize=8);save(fig,'training_full_curves')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6));names=list(NEW);overall=[next(g for g in mask[a]['groups'] if g['dimension']=='overall') for a in names]
    axes[0].bar(names,[g['masked_fraction'] for g in overall],color=[colors[a] for a in names]);axes[0].set(title='Actual fraction of full-As packets erased',ylim=(0,1))
    axes[1].bar(names,[g['mean_removed_power_fraction'] for g in overall],color=[colors[a] for a in names]);axes[1].set(title='Mean removed energy / original energy\n(includes disabled packets)')
    for arm in names:
        h=mask[arm]['begin_histogram'];axes[2].plot([int(k) for k in h],list(h.values()),marker='.',label=arm,color=colors[arm])
    axes[2].set(title='Enabled-mask begin histogram',xlabel='IQ sample index',ylabel='Packet count');axes[2].legend();save(fig,'mask_coverage_energy')
    fig,axes=plt.subplots(1,2,figsize=(14,4.8));rxs=('1','3','4','6','8');xx=np.arange(5)
    for ax,scene in zip(axes,SOURCE_SCENES[1:]):
        for i,arm in enumerate(colors):
            lookup={r['stratum']:r for r in source_metrics if r['arm']==arm and r['scene']==scene and r['draw']==-1 and r['dimension']=='RX'}
            ax.bar(xx+(i-1.5)*.2,[100*lookup[r]['accuracy'] for r in rxs],.2,label=arm,color=colors[arm])
        ax.set(xticks=xx,xticklabels=rxs,title=scene+' / same 1800 physical source panel, 2 channel draws',xlabel='Source RX',ylabel='Accuracy (%)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    save(fig,'source_mid_low_RX');return files


def run(root,out):
    started=time.perf_counter();root=Path(root);out=Path(out)
    source=read(root/'source_matrix_decision.json')
    if source.get('target_scores_read') is not False:raise ValueError('Saved source-first decision required')
    formal=read(root/'automatic_summary.json')
    if formal.get('source_decision')!=source or set(formal['summary'])!={'NOAS30','AS30','RANDOM30','PREFIX30'} or formal.get('formal_feedback') is not False:raise ValueError('Post-score fixed four-row report required')
    for arm,row in logical_rows(root):
        p=read(row/'prediction/complete.json');s=read(row/'scoring/complete.json')
        if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS):raise ValueError('All seven views required for each frozen row')
    out.mkdir(parents=True,exist_ok=False)
    try:
        epochs,steps,mask,cost,missing=training(root,out);sm,sp=source_tables(root,source)
        csv_write(out/'source_metrics_by_draw_RX_day_TX.csv',sm);csv_write(out/'source_paired_by_draw_RX_day_TX.csv',sp)
        csv_write(out/'formal_all7_metrics.csv',[dict(arm=a,view=v,**m) for a,views in formal['summary'].items() for v,m in views.items()])
        csv_write(out/'formal_paired_errors.csv',[dict(comparison=a,view=v,**m) for a,views in formal['paired'].items() for v,m in views.items()])
        write(out/'source_decision.json',source);write(out/'formal_summary.json',formal)
        figures=plots(out,formal,epochs,steps,mask,sm)
        limitations=['Single model seed; two channel draws are not independent model seeds','Source state/quality strata not regenerated; source RX/day/TX strata provided','Formal per-packet physical states unavailable','Matching P43 As20 parent clean-correct prediction cache not supplied','Source readout wall time absent if native artifact does not record it']+missing
        clean=formal['paired_accuracy_pp']['PREFIX30_minus_NOAS30']['clean'];low=formal['paired_accuracy_pp']['PREFIX30_minus_AS30']['practical_low_urban'];mid=formal['paired_accuracy_pp']['PREFIX30_minus_AS30']['practical_mid_urban'];urban=formal['paired_accuracy_pp']['PREFIX30_minus_AS30']['urban_mean']
        text=['# P81 fixed prefix versus random observation erasure','',f"Source criteria PASS: {source['promotion']['passed']}. All seven formal views completed in each row regardless of source outcome.",'',f'PREFIX minus original As: low urban {low:+.6f} pp, mid urban {mid:+.6f} pp, urban mean {urban:+.6f} pp.',f'PREFIX clean minus original NoAs: {clean:+.6f} pp; absolute 0.1 pp stage guard: {clean>=-.1-1e-12}.','Single-seed observations do not establish repeatable dual improvement. No automatic E50, seed expansion, rescue or technical retry.','',f"New training: 20 model-epochs, 4440 updates; actual sample-forwards including readouts/smoke: {cost['new_total_NN']}. Training wall totals are process-seconds, not elapsed time.",'','All epoch records and every reference-step record are retained. Missing epoch fields remain explicit null; figures do not interpolate missing observations.','','Limitations:']+['- '+v for v in limitations]+['','Artifacts:']+['- '+v for v in figures]
        (out/'final_analysis.md').write_text('\n'.join(text)+'\n',encoding='utf8')
        complete=dict(status='ANALYZED',model_seed_count=1,channel_draws=2,source_physical_samples=1800,new_model_rows=2,reused_model_rows=2,new_epochs=20,new_updates=4440,epoch_curve_records=len(epochs),reference_step_records=len(steps),source_metric_records=len(sm),source_paired_records=len(sp),all7_complete=True,NN=0,IQ_reads=0,checkpoint_reads=0,truth_reads=0,reference_reprediction=0,reference_rescoring=0,costs=cost,figures=figures,limitations=limitations,formal_feedback=False,automatic_extension=False,automatic_retry=False,repeatable_dual_gain_proven=False,goal_complete=False,analysis_wall_seconds=time.perf_counter()-started)
        write(out/'complete.json',complete);return complete
    except BaseException as exc:
        write(out/'failure.json',dict(status='ANALYSIS_FAILED',error=str(exc),traceback=traceback.format_exc(),preserve_existing_outputs=True,automatic_retry=False,NN=0,IQ_reads=0,checkpoint_reads=0));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.root,args.output),ensure_ascii=False,allow_nan=False))
