"""Source qualification precedes target-score aggregation; no target-based selection."""
import gzip,json
from pathlib import Path
from . import design as d
from experiments.cvs_urban_identity50.report import overall


def decide(records):
    if set(records)!={'CLEAN','NOMINAL','POSITIVE_J',*d.ARMS}:raise ValueError('All predefined source conditions required')
    for record in records.values():
        if record['epoch']!=50 or not record['source_only'] or record['query_read'] or record['source_count']!=720:
            raise ValueError('Only actual final50 legal L_dev720 may determine expansion')
    clean,nominal=records['CLEAN'],records['NOMINAL'];candidates=[]
    for arm in d.ARMS:
        r=records[arm];delta=dict(clean=r['clean_accuracy']-clean['clean_accuracy'],
            low=r['scene_accuracy']['practical_low_urban']-nominal['scene_accuracy']['practical_low_urban'],
            urban=r['urban_mean']-nominal['urban_mean'],worst_RX=r['worst_urban_RX']-nominal['worst_urban_RX'])
        candidates.append(dict(arm=arm,difference=delta,passed=delta['clean']>=.005 and delta['low']>=.01
            and delta['urban']>=0 and delta['worst_RX']>=-.01))
    control,local=records['BRIDGE_CE'],records['BRIDGE_SIG']
    local_delta=dict(clean=local['clean_accuracy']-control['clean_accuracy'],
        low=local['scene_accuracy']['practical_low_urban']-control['scene_accuracy']['practical_low_urban'])
    return dict(source_only=True,query_read=False,uses_target_metrics=False,source_role='L_dev720',epoch=50,
        rule=d.config(d.rows()[0])['source_expansion_rule'],candidates=candidates,
        family_expandable=any(v['passed'] for v in candidates),
        local_SIG_increment=dict(difference=local_delta,passed=local_delta['low']>=.01 and local_delta['clean']>=-.005),
        seed_count=1,automatic_seed_expansion=False,semantic_family_source_failure_unchanged=True)


def source_decision():
    configs=dict(d.references(),**{r['arm']:d.config(r) for r in d.rows()});records={};fold=None
    for name,c in configs.items():
        source=Path(c['output_root']);inner=d.read(source/'inner_source_contract.json')
        if fold is None:fold=inner
        if inner!=fold:raise ValueError('New and reference actual source physical folds differ')
        records[name]=d.read(source/'source_learning_progress.json')['records'][-1]
    return decide(records)


def collect():
    decision=d.read(d.BASE/'source_family_decision.json')
    if not decision['source_only'] or decision['uses_target_metrics']:raise ValueError('Freeze source decision before target aggregation')
    refs=d.references();configs=dict(refs,**{r['arm']:d.config(r) for r in d.rows()});records=[];costs=[];curves={}
    for name,c in configs.items():
        source=Path(c['output_root']);root=source.parent;result=overall(root)
        result.update(actual_arm=result['arm'],arm=name,actual_row_root=str(root),reused=name in refs)
        records.append(result);done=d.read(source/'completion.json');mechanism=d.read(source/'training_mechanism_complete.json')
        if done['epoch']!=50 or done['optimizer_steps']!=11100 or done['head_fit_updates'] or done['target_access']:
            raise ValueError('Incomplete original default-head budget')
        counts=mechanism['counts'];keys=('clean_L_packets','satellite_L_packets','HQ_L_packets','teacher_L_clean_packets','teacher_HQ_packets')
        main=sum(counts.get(k,0) for k in keys)+2*counts['native_U_packets']
        new=name not in refs
        if new and main!=9031800:raise ValueError('Matched five-view main budget differs')
        cost=dict(arm=name,reused=not new,parent_epochs=0,epochs=50,updates=11100,
            new_training_epochs=50 if new else 0,new_updates=11100 if new else 0,
            original_training_seconds=done['elapsed_seconds'],new_training_seconds=done['elapsed_seconds'] if new else 0.,
            main_forward_packets=main,counts=counts,source_dev_forward_packets=mechanism['source_dev_forward_packets'],
            native_V_forward_packets=52*27000,prediction=d.read(root/'prediction/complete.json'))
        if new:
            calibration=d.read(source/'bridge_calibration.json')
            if (calibration['status']!='SOURCE_BRIDGE_CALIBRATED' or not calibration['source_only'] or calibration['query_read']
                    or calibration['optimizer_updates'] or calibration['EMA_updates'] or calibration['head_fit_updates']
                    or calibration['forward_packets']!=5760 or calibration['warmup_epoch']!=5):
                raise ValueError('Own source-only epoch5 calibration differs')
            cost['calibration']=calibration
            with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:steps=[json.loads(line) for line in stream]
            if len(steps)!=11100 or any(v['step']!=i+1 for i,v in enumerate(steps)):raise ValueError('Incomplete complete-step curve')
            keys=('urban/clean_CE','urban/satellite_CE','urban/HQ_CE','urban/auxiliary_weighted_loss','urban/local_SIG_loss','urban/local_SIG_weighted_loss')
            curves[c['row_id']]=[dict(epoch=epoch,**{key:sum(v['metrics'][key] for v in steps[(epoch-1)*222:epoch*222])/222 for key in keys}) for epoch in range(1,51)]
            clipping=[v['metrics'] for v in steps if v['metrics'].get('train/grad_clip_active',0)]
            cost.update(clipping_active_steps=len(clipping),clipped_steps=sum(v['train/grad_before_clip']>v['train/grad_clip_limit'] for v in clipping))
        costs.append(cost)
    index={r['arm']:r for r in records};pairs=[]
    for treatment,control in (('NOMINAL','CLEAN'),('BRIDGE_CE','POSITIVE_J'),('BRIDGE_CE','NOMINAL'),('BRIDGE_SIG','BRIDGE_CE'),('BRIDGE_SIG','NOMINAL')):
        pairs.append(dict(treatment=treatment,control=control,model_seed=2026092701,
            **{k+'_difference':index[treatment][k]-index[control][k] for k in ('clean','low_urban','mid_urban','urban_mean','urban_macro_f1','worst_urban_RX','suburban_mean')}))
    summary=dict(rows=records,pairs=pairs,dual_gain=[dict(arm=arm,clean_vs_CLEAN=index[arm]['clean']-index['CLEAN']['clean'],
        low_vs_NOMINAL=index[arm]['low_urban']-index['NOMINAL']['low_urban'],low_urban_60=index[arm]['low_urban']>=.6) for arm in d.ARMS],
        source_family_decision=decision,seed_count=1,cross_seed_confirmation='NOT_RUN',target_feedback_forbidden=True,
        comparison_limits='Only BRIDGE_SIG vs BRIDGE_CE matches new forward and CE budgets; archived controls are descriptive. HQ CE and own source strength differ from the old two-view recipe.',
        local_mechanism_attribution='Frozen source descriptor/identity-margin/repair analysis pending; feature loss reduction is not identity recovery')
    d.write(d.BASE/'summary.json',summary);d.write(d.BASE/'resources.json',dict(rows=costs,new_models=2,reused_models=3,
        parent_training_cost=0,operational_smoke=d.read(d.BASE/'source_smoke.json'),head_fit_updates=0))
    d.write(d.BASE/'training_curves.json',curves)
    lines=['# Fixed SIG_TIME HQ bridge pilot','',
        'Source final50 decisions frozen before this report reads target metrics. Two scratch E50 rows, three read-only exact controls; original seven-view IQ/default heads.','',
        '|condition|reused|clean %|low urban %|mid urban %|urban %|urban Macro-F1 %|worst urban RX %|suburban %|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append('|'+r['arm']+'|'+str(r['reused'])+'|'+'|'.join(f"{100*r[k]:.4f}" for k in ('clean','low_urban','mid_urban','urban_mean','urban_macro_f1','worst_urban_RX','suburban_mean'))+'|')
    lines+=['','No cross-seed, convergence or unique-cause claim. Local descriptor/margin/repair attribution remains separate source-only work. All negative differences retained.']
    (d.BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary
