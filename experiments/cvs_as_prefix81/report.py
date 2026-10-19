"""Fixed source decision, then saved all-seven-view truth-side recount only."""
from pathlib import Path
import numpy as np
from . import design as d

SOURCE_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
URBAN=('practical_high_urban','practical_mid_urban','practical_low_urban')
PAIRS=(('PREFIX30','RANDOM30'),('PREFIX30','AS30'),('RANDOM30','AS30'),
       ('PREFIX30','NOAS30'),('RANDOM30','NOAS30'),('AS30','NOAS30'))


def logical_rows():
    ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
    return [('NOAS30',ref/'B00-s2026092701'),('AS30',ref/'B01-s2026092701'),
            ('RANDOM30',d.BASE/'RANDOM-s2026092701'),('PREFIX30',d.BASE/'PREFIX-s2026092701')]


def stats(y,p,rx):
    y,p,rx=np.asarray(y),np.asarray(p),np.asarray(rx)
    if y.ndim!=1 or not len(y) or p.shape!=y.shape or rx.shape!=y.shape or y.dtype.kind not in 'iu' or p.dtype.kind not in 'iu' or np.any((y<0)|(y>5)|(p<0)|(p>5)):
        raise ValueError('Nonempty aligned six-class decisions required')
    cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
    return dict(count=len(y),correct=int((p==y).sum()),accuracy=float((p==y).mean()),
                macro_F1=float(f.mean()),worst_RX=float(min((p[rx==r]==y[rx==r]).mean() for r in np.unique(rx))))


def delta(a,b,y,cohort=None):
    gooda,goodb=np.asarray(a)==y,np.asarray(b)==y
    repair=int((gooda&~goodb).sum());new=int((~gooda&goodb).sum())
    value=dict(repaired=repair,new_errors=new,net=repair-new,accuracy_pp=100*(repair-new)/len(y),count=len(y))
    if cohort is not None:
        co=np.asarray(cohort,dtype=bool)
        if co.shape!=gooda.shape:raise ValueError('Fixed clean-correct cohort shape differs')
        value['fixed_NOAS30_clean_correct']=dict(count=int(co.sum()),new_correct=int((gooda&co).sum()),
            control_correct=int((goodb&co).sum()),net=int(((gooda.astype(int)-goodb.astype(int))*co).sum()))
    return value


def retention(p,y,cohort):
    n=int(cohort.sum());correct=int(((p==y)&cohort).sum())
    return dict(count=n,correct=correct,retention=correct/n if n else None)


def source_criteria(rows):
    clean=100*(rows['PREFIX30']['full_V_clean']-rows['NOAS30']['full_V_clean'])
    original=100*(rows['PREFIX30']['mid_low_mean']-rows['AS30']['mid_low_mean'])
    position=100*(rows['PREFIX30']['mid_low_mean']-rows['RANDOM30']['mid_low_mean'])
    if not np.isfinite([clean,original,position]).all():raise ValueError('Finite source criteria required')
    checks=dict(fullV_clean_vs_NOAS30_within_0p1pp=bool(clean>=-.1-1e-12),
                midlow_vs_AS30_at_least_0p5pp=bool(original>=.5-1e-12),
                midlow_vs_RANDOM30_at_least_0p25pp=bool(position>=.25-1e-12))
    passed=all(checks.values())
    return dict(passed=passed,source_criteria_pass=passed,checks=checks,fixed_recipe_closed=not passed,
                PREFIX30_minus_NOAS30_fullV_clean_pp=clean,PREFIX30_minus_AS30_midlow_pp=original,
                PREFIX30_minus_RANDOM30_midlow_pp=position,clean_min_pp=-.1,midlow_original_min_pp=.5,
                midlow_random_min_pp=.25,future_budget_source_eligible=passed,
                diagnostic_future_eligibility_only=True,source_failure_skips_formal=False,
                automatic_E50=False,automatic_seed_extension=False,automatic_retry=False)


def source_decision():
    dest=d.BASE/'source_matrix_decision.json'
    if dest.exists():return d.read(dest)
    rows={};data={};expected=d.read(d.SOURCE)['role_ids'];base=None
    for arm,root in logical_rows():
        done=d.read(root/'source/completion.json');sc=d.read(root/'source/source_contract.json')
        frozen=d.read(root/'source/source_frozen.json');q=d.read(root/'source_qualification/complete.json')
        if sc['role_ids']!=expected or done['target_access'] or frozen['target_access'] or q['target_access'] or q.get('formal_scores_read',False) or frozen['selection']!='final_only' or frozen['config']!=done['config'] or done['final_source_metrics']['source_val_count']!=27000:
            raise ValueError('Exact frozen fullsource rows and source-only readouts required')
        with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as packed:
            x={k:packed[k].copy() for k in ('ids','TX','RX','day','clean','satellite')}
        if x['ids'].shape!=(1800,) or len(set(x['ids'].tolist()))!=1800 or any(x[k].shape!=(1800,) for k in ('TX','RX','day')) or x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():
            raise ValueError('Finite same 1800-physical-sample/two-draw source panel required')
        if base is None:base=x
        if any(not np.array_equal(x[k],base[k]) for k in ('ids','TX','RX','day')):raise ValueError('Source panels differ')
        pred={'clean':x['clean'].argmax(-1)}
        pred.update({s:x['satellite'][:,i].argmax(-1).reshape(-1) for i,s in enumerate(SOURCE_SCENES)})
        panel={v:stats(x['TX'] if v=='clean' else np.tile(x['TX'],2),p,x['RX'] if v=='clean' else np.tile(x['RX'],2)) for v,p in pred.items()}
        by_draw={str(draw):{s:stats(x['TX'],x['satellite'][draw,i].argmax(-1),x['RX']) for i,s in enumerate(SOURCE_SCENES)} for draw in range(2)}
        rows[arm]=dict(full_V_clean=done['final_source_metrics']['source_val_accuracy'],
            full_V_RX_accuracy=done['final_source_metrics']['source_val_rx_accuracy'],panel=panel,by_draw=by_draw,
            mid_low_mean=float(np.mean([panel[s]['accuracy'] for s in SOURCE_SCENES[1:]])))
        data[arm]=dict(x=x,pred=pred)
    cohort=data['NOAS30']['pred']['clean']==base['TX'];pairs={};draw_pairs={}
    for arm in rows:
        rows[arm]['fixed_NOAS30_clean_correct_retention']={v:retention(p,base['TX'] if v=='clean' else np.tile(base['TX'],2),cohort if v=='clean' else np.tile(cohort,2)) for v,p in data[arm]['pred'].items()}
        rows[arm]['mid_low_by_draw_accuracy']={str(draw):float(np.mean([rows[arm]['by_draw'][str(draw)][s]['accuracy'] for s in SOURCE_SCENES[1:]])) for draw in range(2)}
    for a,b in PAIRS:
        key=a+'_minus_'+b;pairs[key]={};draw_pairs[key]={}
        for view in data[a]['pred']:
            y=base['TX'] if view=='clean' else np.tile(base['TX'],2);co=cohort if view=='clean' else np.tile(cohort,2)
            pairs[key][view]=dict(delta(data[a]['pred'][view],data[b]['pred'][view],y,co),
                macro_F1_pp=100*(rows[a]['panel'][view]['macro_F1']-rows[b]['panel'][view]['macro_F1']),
                worst_RX_pp=100*(rows[a]['panel'][view]['worst_RX']-rows[b]['panel'][view]['worst_RX']))
        for draw in range(2):
            draw_pairs[key][str(draw)]={s:delta(data[a]['x']['satellite'][draw,i].argmax(-1),data[b]['x']['satellite'][draw,i].argmax(-1),base['TX'],cohort) for i,s in enumerate(SOURCE_SCENES)}
            draw_pairs[key][str(draw)]['mid_low_mean_accuracy_pp']=100*(rows[a]['mid_low_by_draw_accuracy'][str(draw)]-rows[b]['mid_low_by_draw_accuracy'][str(draw)])
    value=dict(status='SOURCE_PREFIX81_READOUT_COMPLETE',rows=rows,paired=pairs,paired_by_draw=draw_pairs,
        fullV_clean_paired_pp={a+'_minus_'+b:100*(rows[a]['full_V_clean']-rows[b]['full_V_clean']) for a,b in PAIRS},
        promotion=source_criteria(rows),panel_draws=2,model_seed_count=1,independent_physical_panel_samples=1800,
        fixed_clean_cohort='NOAS30 clean-correct panel; not an E20-parent cohort',source_high_urban_available=False,
        fullV_F1_or_error_recount_available=False,target_scores_read=False,scientific_failure_skips_formal=False,
        automatic_E50=False,automatic_seed_extension=False,automatic_retry=False,repeatable_gain_proven=False,
        interpretation='Three source criteria fixed before formal-score reads; two channel draws are not model seeds; both new rows still require all seven formal views; no new NoAs row or factorial interaction claim')
    d.write(dest,value);return value


def collect():
    decision_path=d.BASE/'source_matrix_decision.json'
    if not decision_path.exists():raise ValueError('Persist source decision before reading any formal scores')
    decision=d.read(decision_path)
    if decision.get('target_scores_read') is not False:raise ValueError('Source-first decision required')
    result={};states=[];predictions={};score_records={};ids=None
    # All four logical rows finish all seven predictions and independent scoring
    # before this report opens any truth or score file, irrespective of PASS/FAIL.
    for arm,root in logical_rows():
        pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
        if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or sc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit'] or not pc['all_parameters_frozen'] or pc['head_fit_updates'] or not sc.get('truth_last',False):
            raise ValueError('All seven frozen predictions and independent completed scores required')
        states.append(dict(arm=arm,root=str(root),reused=arm in ('NOAS30','AS30')))
    for arm,root in logical_rows():
        with np.load(root/'prediction/predictions.npz',allow_pickle=False) as packed:
            current=packed['ids'].copy();predictions[arm]={v:packed[v].copy() for v in d.VIEWS}
        if ids is None:ids=current
        if ids.shape!=(168000,) or len(set(ids.tolist()))!=168000 or not np.array_equal(current,ids):raise ValueError('Same fixed formal physical IDs required')
        if any(p.shape!=ids.shape or p.dtype.kind not in 'iu' or np.any((p<0)|(p>5)) for p in predictions[arm].values()):raise ValueError('Fixed six-class formal decisions required')
    from experiments.cvs_early_geometry.evaluate import truth_arrays
    y,rx,days=truth_arrays(d.read(d.TRUTH),ids)
    cohort=predictions['NOAS30']['clean']==y
    for arm,root in logical_rows():
        scores=d.read(root/'scoring/scores.json');score_records[arm]=scores['results'];result[arm]={}
        if not scores.get('truth_last',False):raise ValueError('Independent truth-last score artifact required')
        for view in d.VIEWS:
            m=stats(y,predictions[arm][view],rx);records=[r for r in score_records[arm] if r['view']==view]
            overall=next(r for r in records if r['dimension']=='overall');receivers=[r['accuracy'] for r in records if r['dimension']=='receiver']
            if not receivers or not np.allclose([m['accuracy'],m['macro_F1'],m['worst_RX']],[overall['accuracy'],overall['macro_f1'],min(receivers)],atol=1e-12,rtol=0):raise ValueError('Saved independent metrics differ from fixed-decision recount')
            m['fixed_NOAS30_clean_correct']=retention(predictions[arm][view],y,cohort);result[arm][view]=m
        urban=stats(np.tile(y,3),np.concatenate([predictions[arm][v] for v in URBAN]),np.tile(rx,3))
        urban['pooled_macro_F1']=urban['macro_F1'];urban['macro_F1']=float(np.mean([result[arm][v]['macro_F1'] for v in URBAN]))
        result[arm]['urban_mean']=urban
    pairs={};paired_accuracy={}
    for a,b in PAIRS:
        key=a+'_minus_'+b;pairs[key]={}
        for view in (*d.VIEWS,'urban_mean'):
            urban=view=='urban_mean';yy=np.tile(y,3) if urban else y;co=np.tile(cohort,3) if urban else cohort
            pa=np.concatenate([predictions[a][v] for v in URBAN]) if urban else predictions[a][view]
            pb=np.concatenate([predictions[b][v] for v in URBAN]) if urban else predictions[b][view]
            pairs[key][view]=dict(delta(pa,pb,yy,co),macro_F1_pp=100*(result[a][view]['macro_F1']-result[b][view]['macro_F1']),worst_RX_pp=100*(result[a][view]['worst_RX']-result[b][view]['worst_RX']))
        paired_accuracy[key]={v:q['accuracy_pp'] for v,q in pairs[key].items()}
    value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',rows=states,summary=result,paired=pairs,
        paired_accuracy_pp=paired_accuracy,stage_clean_0p1pp=bool(paired_accuracy['PREFIX30_minus_NOAS30']['clean']>=-.1-1e-12),
        source_decision=decision,single_seed=True,model_seed_count=1,repeated_clean_gain_proven=False,
        urban_macro_F1_definition='Arithmetic mean of the three scene macro-F1 values; pooled value also supplied',
        urban_worst_RX_definition='Minimum RX accuracy over the three equally sized concatenated urban views',
        no_new_NoAs_control=True,factorial_interaction_claimed=False,formal_feedback=False,goal_complete=False)
    d.write(d.BASE/'automatic_summary.json',value);return value
