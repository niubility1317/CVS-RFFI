"""Separate dual-gain reporting; source decision is fixed before target reads."""
from pathlib import Path
import gzip,json
from . import design as d
from experiments.cvs_urban_identity50.report import overall


def row_root(c):return Path(c['output_root']).parent


def source_decision():
    records={}
    for row in d.rows():
        c=d.config(row);r=d.read(Path(c['output_root'])/'source_learning_progress.json')['records']
        if r[-1]['epoch']!=50 or not r[-1]['source_only'] or r[-1]['query_read']:raise ValueError('Fixed final50 L_dev source metrics required')
        records[c['arm']]=r[-1]
    base,aug=records['D0'],records['D1'];candidates=[]
    for arm in ('D3','D4'):
        r=records[arm];delta=dict(clean=r['clean_accuracy']-base['clean_accuracy'],
            low=r['scene_accuracy']['practical_low_urban']-aug['scene_accuracy']['practical_low_urban'],
            urban=r['urban_mean']-aug['urban_mean'],worst_RX=r['worst_urban_RX']-aug['worst_urban_RX'])
        passed=delta['clean']>=.005 and delta['low']>=.01 and delta['urban']>=0. and delta['worst_RX']>=-.01
        candidates.append(dict(arm=arm,difference=delta,passed=passed))
    return dict(source_only=True,query_read=False,uses_target_metrics=False,source_role='L_dev720',epoch=50,
        rule=d.config(d.fresh_rows()[0])['source_expansion_rule'],candidates=candidates,
        family_expandable=any(v['passed'] for v in candidates),
        old_eight_position_decision_unchanged=True,seed_count=1,automatic_seed_expansion=False)


def collect():
    records=[];costs=[];curves={};native_keys=('clean_L_packets','satellite_L_packets','teacher_L_clean_packets','native_U_packets')
    old_resources=d.read(d.parent.BASE/'resources.json')['rows'];old_curves=d.read(d.parent.BASE/'training_curves.json')
    for row in d.rows():
        c=d.config(row);root=row_root(c);result=overall(root);actual_arm=result['arm']
        result.update(arm=c['arm'],logical_row_id=row['row_id'],actual_arm=actual_arm,actual_row_root=str(root),reused=c['reused'])
        records.append(result);source=Path(c['output_root']);done=d.read(source/'completion.json');mechanism=d.read(source/'training_mechanism_complete.json')
        if done['epoch']!=50 or done['optimizer_steps']!=11100 or done['head_fit_updates'] or done['target_access']:
            raise ValueError('Incomplete default-head source budget')
        counts=mechanism['counts'];main=sum(counts.get(k,0)*(2 if k=='native_U_packets' else 1) for k in native_keys)
        if main!=(4769400 if c['arm']=='D0' else 6190200):raise ValueError('Declared main forward budget differs')
        cost=dict(arm=c['arm'],reused=c['reused'],parent_epochs=0,epochs=50,updates=11100,
            new_training_epochs=0 if c['reused'] else 50,new_updates=0 if c['reused'] else 11100,
            original_training_seconds=done['elapsed_seconds'],new_training_seconds=0. if c['reused'] else done['elapsed_seconds'],
            main_forward_packets=main,counts={k:counts.get(k,0) for k in native_keys},source_dev_forward_packets=mechanism['source_dev_forward_packets'],
            native_V='preserved complete native validation telemetry and actual args; health-only',
            prediction=d.read(root/'prediction/complete.json'))
        if c['arm'] in ('D1','D2'):
            original=c['reuse_source_config']['row_id'];curves[row['row_id']]=old_curves[original]
            cost['original_resource_record']=next(v for v in old_resources if v['row_id']==original)
        else:
            with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(line) for line in f]
            if len(steps)!=11100:raise ValueError('Incomplete source training curve')
            curve=[]
            for epoch in range(1,51):
                block=steps[(epoch-1)*222:epoch*222]
                keys=('urban/clean_CE','urban/satellite_CE','urban/auxiliary_weighted_loss','urban/semantic_loss','urban/semantic_forward','urban/semantic_reverse')
                curve.append(dict(epoch=epoch,**{k:sum(v['metrics'][k] for v in block)/222 for k in keys}))
            curves[row['row_id']]=curve
            clipping=[s['metrics'] for s in steps if s['metrics'].get('train/grad_clip_active',0)]
            cost.update(clipping_active_steps=len(clipping),clipped_steps=sum(v['train/grad_before_clip']>v['train/grad_clip_limit'] for v in clipping))
        costs.append(cost)
    index={v['arm']:v for v in records};pairs=[]
    for treatment,control in (('D1','D0'),('D2','D1'),('D3','D1'),('D4','D1'),('D4','D3')):
        pairs.append(dict(treatment=treatment,control=control,model_seed=2026092701,
            **{k+'_difference':index[treatment][k]-index[control][k] for k in ('clean','low_urban','mid_urban','urban_mean','urban_macro_f1','worst_urban_RX','suburban_mean')}))
    dual=[dict(arm=arm,clean_vs_D0=index[arm]['clean']-index['D0']['clean'],low_vs_D1=index[arm]['low_urban']-index['D1']['low_urban'],
        low_urban_60=index[arm]['low_urban']>=.6) for arm in ('D2','D3','D4')]
    summary=dict(rows=records,pairs=pairs,dual_gain=dual,source_family_decision=d.read(d.BASE/'source_family_decision.json'),
        seed_count=1,cross_seed_confirmation='NOT_RUN',target_feedback_forbidden=True,no_weighted_dual_gain_score=True)
    d.write(d.BASE/'summary.json',summary);d.write(d.BASE/'training_curves.json',curves)
    diagnostic_source=d.ORIGINAL_BASE/'D0-s2026092701/source'
    diagnostic_done=diagnostic_source/'completion.json'
    diagnostic=d.read(diagnostic_done) if diagnostic_done.is_file() else None
    d.write(d.BASE/'resources.json',dict(rows=costs,new_models=3,reused_models=2,source_calibration=d.read(d.BASE/'semantic_calibration.json'),
        operational_smoke=d.read(d.BASE/'source_smoke.json'),
        failed_original_rows='Original r01 D3/D4 first-step failures retained and reported separately; no completed training budget',
        original_D0=dict(source=str(diagnostic_source),role='Unmatched no-generic-augmentation diagnostic, excluded from five-row comparison',
            epochs=50 if diagnostic else None,updates=diagnostic.get('optimizer_steps') if diagnostic else None,
            seconds=diagnostic.get('elapsed_seconds') if diagnostic else None,complete=diagnostic is not None,
            preserved_untouched=True)))
    lines=['# Fixed SIG_TIME relative identity pilot','',
        'All source decisions frozen before this report reads target scores. D0 preserves native generic augmentation; original r01 no-generic D0 is a separate diagnostic. D1/D2 reuse exact A0/A-J artifacts.','',
        '|arm|reused|clean %|urban %|low urban %|mid urban %|urban Macro-F1 %|worst RX %|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append('|'+r['arm']+'|'+str(r['reused'])+'|'+'|'.join(f"{100*r[k]:.4f}" for k in ('clean','urban_mean','low_urban','mid_urban','urban_macro_f1','worst_urban_RX'))+'|')
    lines+=['','Separate clean-vs-D0 and low-vs-D1 differences are in summary.json; no single-seed superiority or cross-seed claim.']
    (d.BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary
