"""Same-seed factorial effects after all registered row scores exist."""
from . import train_design as d


def collect():
    qualification=d.read(d.BASE/'source_qualification.json')
    if qualification['target_scores_read']:raise ValueError('Source qualification must not consume target scores')
    result={};cost={};all_records=[]
    for row in d.rows():
        rid=row['row_id'];c=d.config(row);root=d.BASE/rid
        done=d.read(root/'scoring/complete.json')
        if done.get('status')!='SCORED_COMPLETE' or done.get('views')!=list(d.VIEWS):raise ValueError('All seven same-row scores required')
        records=d.read(root/'scoring/scores.json')['results'];all_records.extend(records)
        overall={r['view']:r for r in records if r['dimension']=='overall'}
        if set(overall)!=set(d.VIEWS):raise ValueError('Complete seven-view matrix required')
        receivers=sorted({r['stratum'] for r in records if r['dimension']=='receiver'})
        urban_rx={rx:sum(next(r['accuracy'] for r in records if r['view']==view and r['dimension']=='receiver' and r['stratum']==rx) for view in d.SCENES[3:])/3 for rx in receivers}
        metrics={v:overall[v]['accuracy'] for v in d.VIEWS}
        metrics.update(urban_mean=sum(overall[v]['accuracy'] for v in d.SCENES[3:])/3,
            urban_macro_f1=sum(overall[v]['macro_f1'] for v in d.SCENES[3:])/3,worst_urban_RX=min(urban_rx.values()))
        result[row['arm']]=dict(metrics=metrics,per_view=overall,urban_RX=urban_rx,
            suburban_gap_pp={v:100*(metrics['clean']-metrics[v]) for v in d.SCENES[:3]})
        source=d.read(root/'source/completion.json');mechanism=d.read(root/'source/training_mechanism_complete.json')
        prediction=d.read(root/'prediction/complete.json')
        cost[row['arm']]=dict(epochs=20,updates=4440,source_seconds=source['elapsed_seconds'],
            measured_student_training_forward_packets=mechanism['student_training_forward_packets'],
            measured_student_eval_forward_packets=mechanism['student_eval_forward_packets'],
            measured_EMA_forward_packets=mechanism['EMA_eval_forward_packets'],
            measured_L_forward_packets=mechanism['L_forward_packets'],
            measured_student_U_forward_packets=mechanism['native_student_U_forward_packets'],
            frozen_source_panel_forward_packets=5040,formal_forward_packets=prediction['actual_forward_packets'],
            formal_wall_seconds=prediction['wall_seconds'],parent_weights_loaded=False)
    names=list(result['B11']['metrics'])
    pairs={f'{a}-{b}':{k:100*(result[a]['metrics'][k]-result[b]['metrics'][k]) for k in names}
           for a,b in (('B10','B00'),('B11','B00'),('B11','R_LEGACY'),('B11','B10'),('B11','B01'))}
    interaction={k:100*(result['B11']['metrics'][k]-result['B10']['metrics'][k]-result['B01']['metrics'][k]+result['B00']['metrics'][k]) for k in names}
    checks=dict(clean_gain_over_both=all(pairs['B11-'+a]['clean']>=.5 for a in ('B00','R_LEGACY')),
        low_urban_over60=result['B11']['metrics']['practical_low_urban']>.60,
        suburban_near_own_clean=all(abs(v)<=2 for v in result['B11']['suburban_gap_pp'].values()),
        suburban_absolute_protection=all(result['B11']['metrics'][v]>=result['B01']['metrics'][v]-.005 for v in d.SCENES[:3]))
    value=dict(status='ANALYZED',run_id=d.RUN,seed_count=1,model_seed=2026092701,rows=result,cost=cost,
        paired_difference_pp=pairs,interaction_pp=interaction,exploration_checks=checks,
        performance_status='SINGLE_SEED_EXPLORATION_TARGETS_MET' if all(checks.values()) else 'EXPLORATION_TARGETS_NOT_MET',
        source_qualification=qualification,truth_last=True,target_feedback=False,goal_completed=False,
        claim='Fixed practical proxy, exposed benchmark; no cross-seed confirmation',records=all_records)
    d.write(d.BASE/'results.json',value);return value
