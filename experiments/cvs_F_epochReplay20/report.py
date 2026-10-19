"""Fixed source choice before formal scoring; one replay row and reused B0/B1."""
import time
from . import design as d

def paths():
    return dict(B0=d.reference('B0'),B1=d.reference('B1'),EPOCH_REPLAY=d.BASE/d.rows()[0]['row_id'])

def source_decision():
    locations=paths()
    values={a:d.read(p/'source_qualification/complete.json') for a,p in locations.items()}
    clean={a:d.read(p/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for a,p in locations.items()}
    midlow=lambda a:sum(values[a]['source_accuracies'][i][j] for i in range(2) for j in (1,2))/4
    gain=100*(midlow('EPOCH_REPLAY')-midlow('B1'))
    net=100*(clean['EPOCH_REPLAY']-clean['B0'])
    passed=gain>=.5 and net>=-.2
    result=dict(status='SOURCE_DECISION_FIXED',timestamp=time.time(),source_qualified=passed,
        midlow_increment_pp=gain,fullVclean_net_pp=net,fullVclean=clean,readouts=values,
        formal_scores_read=False,automatic_extension=False,goal_complete=False,
        decision='E50 source eligible; matched own continuation rules required first' if passed else
        'CLOSE_FIXED_EPOCH_FIRST_AS_REPLAY; no refresh-interval/seed/schedule rescue')
    d.write(d.BASE/'source_matrix_decision.json',result)
    return result

def collect():
    import numpy as np
    from experiments.cvs_early_geometry.evaluate import truth_arrays
    source,locations=d.read(d.BASE/'source_matrix_decision.json'),paths()
    if source['formal_scores_read'] is not False:
        raise ValueError('Fixed source decision must precede scores')
    rows,decisions,ids={},{},None
    for arm,p in locations.items():
        if d.read(p/'scoring/complete.json')['status']!='SCORED_COMPLETE':
            raise ValueError('Complete reused/new seven-view scores required')
        if arm=='EPOCH_REPLAY' and source['timestamp']>=(p/'scoring/complete.json').stat().st_mtime:
            raise ValueError('Source decision did not precede independent scoring')
        data=d.read(p/'scoring/scores.json')['results']
        overall={r['view']:r for r in data if r['dimension']=='overall'}
        if set(overall)!=set(d.VIEWS):
            raise ValueError('All seven views required')
        actual=d.read(p/'source/training_result.json')
        rows[arm]=dict(overall=overall,reused=arm in ('B0','B1'),
            worst_RX={v:min(r['accuracy'] for r in data if r['view']==v and r['dimension']=='receiver') for v in d.VIEWS},
            cost={k:actual[k] for k in ('L_packets','student_train_packets','teacher_packets')},
            own_EMA_eval_packets=actual.get('own_EMA_eval_packets'),
            replay=actual.get('replay'),channel=actual.get('channel'))
        with np.load(p/'prediction/predictions.npz',allow_pickle=False) as f:
            current=f['ids'].copy()
            if ids is not None and not np.array_equal(ids,current):
                raise ValueError('Paired physical IDs differ')
            ids=current;decisions[arm]={v:f[v].copy() for v in d.VIEWS}
    y,_,_=truth_arrays(d.read(d.TRUTH),ids)
    pairs={}
    for ref in ('B0','B1'):
        values={}
        for v in d.VIEWS:
            before,after=decisions[ref][v]==y,decisions['EPOCH_REPLAY'][v]==y
            both,damaged=int((before&after).sum()),int((before&~after).sum())
            values[v]=dict(n=len(y),corrected=int((~before&after).sum()),damaged=damaged,
                both_correct=both,both_wrong=int((~before&~after).sum()),
                net_pp=float(100*(after.mean()-before.mean())),mean_pp=float(100*(after.mean()-before.mean())),
                baseline_correct_retention=None if both+damaged==0 else both/(both+damaged),
                positive_model_seeds=int(after.mean()>before.mean()),model_seed_count=1,sample_SD=None)
        pairs['EPOCH_REPLAY_minus_'+ref]=values
    result=dict(status='ANALYZED',run_id=d.RUN,source_decision=source,rows=rows,paired_errors=pairs,
        suburban_gap_pp={a:{v:100*(rows[a]['overall']['clean']['accuracy']-rows[a]['overall'][v]['accuracy']) for v in d.SCENES[:3]} for a in rows},
        model_seed_count=1,crossseed_confirmed=False,sample_SD=None,goal_complete=False)
    d.write(d.BASE/'analysis.json',result)
    return result
