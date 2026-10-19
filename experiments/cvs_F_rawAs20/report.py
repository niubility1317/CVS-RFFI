"""Source promotion uses no formal scores; final report pairs frozen all7 outputs."""
import time
from . import design as d


def paths():
    return dict(B0=d.reference('B0'), B1=d.reference('B1'),
                **{r['arm']: d.BASE/r['row_id'] for r in d.rows()})


def source_decision():
    locations = paths()
    values = {a: d.read(p/'source_qualification/complete.json') for a, p in locations.items()}
    clean = {a: d.read(p/'source/completion.json')['final_source_metrics']['source_val_accuracy']
             for a, p in locations.items()}
    midlow = lambda arm: sum(values[arm]['source_accuracies'][i][j]
                             for i in range(2) for j in (1, 2))/4
    gain = 100*(midlow('RAW_AS')-midlow('B1'))
    net = {a: 100*(clean['RAW_AS']-clean[a]) for a in ('B0', 'RAW_NOAS')}
    passed = gain >= .5 and min(net.values()) >= -.2
    result = dict(status='SOURCE_DECISION_FIXED', timestamp=time.time(),
        source_qualified=passed, midlow_increment_pp=gain, fullVclean_net_pp=net,
        fullVclean=clean, readouts=values, formal_scores_read=False,
        decision='E50 source eligible; matched continuation rules required first' if passed else
        'CLOSE_FIXED_WHOLE_RAW_RECIPE; no ratio/seed/schedule rescue',
        automatic_extension=False, goal_complete=False)
    d.write(d.BASE/'source_matrix_decision.json', result)
    return result


def collect():
    import numpy as np
    from experiments.cvs_early_geometry.evaluate import truth_arrays
    source, locations = d.read(d.BASE/'source_matrix_decision.json'), paths()
    if source['formal_scores_read'] is not False:
        raise ValueError('Source decision must precede formal score access')
    rows, decisions, ids = {}, {}, None
    for arm, p in locations.items():
        if d.read(p/'scoring/complete.json')['status'] != 'SCORED_COMPLETE':
            raise ValueError('All reused/new row scoring must be complete')
        if arm.startswith('RAW_') and source['timestamp'] >= (p/'scoring/complete.json').stat().st_mtime:
            raise ValueError('Source decision did not precede independent scoring')
        data = d.read(p/'scoring/scores.json')['results']
        overall = {r['view']: r for r in data if r['dimension'] == 'overall'}
        if set(overall) != set(d.VIEWS):
            raise ValueError('Complete seven-view score records required')
        actual = d.read(p/'source/training_result.json')
        rows[arm] = dict(overall=overall, reused=arm in ('B0', 'B1'),
            worst_RX={v: min(r['accuracy'] for r in data if r['view'] == v and r['dimension'] == 'receiver')
                      for v in d.VIEWS},
            cost={k: actual[k] for k in ('L_packets', 'student_train_packets', 'teacher_packets')},
            ordinary_shadow_packets=actual.get('ordinary_shadow_packets', 0),
            As_packets=0 if actual['channel'] is None else actual['channel']['packets'],
            own_EMA_eval_packets=actual.get('own_EMA_eval_packets'),
            own_EMA_train_packets=actual.get('own_EMA_train_packets'))
        with np.load(p/'prediction/predictions.npz', allow_pickle=False) as f:
            current = f['ids'].copy()
            if ids is not None and not np.array_equal(ids, current):
                raise ValueError('Formal paired physical IDs differ')
            ids = current
            decisions[arm] = {v: f[v].copy() for v in d.VIEWS}
    y, _, _ = truth_arrays(d.read(d.TRUTH), ids)
    pairings = {}
    comparisons = (('RAW_AS', 'B0'), ('RAW_AS', 'B1'), ('RAW_AS', 'RAW_NOAS'),
                   ('RAW_NOAS', 'B0'))
    for arm, ref in comparisons:
        values = {}
        for v in d.VIEWS:
            before, after = decisions[ref][v] == y, decisions[arm][v] == y
            both, damaged = int((before & after).sum()), int((before & ~after).sum())
            values[v] = dict(n=len(y), corrected=int((~before & after).sum()), damaged=damaged,
                both_correct=both, both_wrong=int((~before & ~after).sum()),
                net_pp=float(100*(after.mean()-before.mean())),
                mean_pp=float(100*(after.mean()-before.mean())), model_seed=d.SEED,
                baseline_correct_retention=None if both+damaged == 0 else both/(both+damaged),
                positive_model_seeds=int(after.mean() > before.mean()), model_seed_count=1,
                sample_SD=None)
        pairings[arm+'_minus_'+ref] = values
    result = dict(status='ANALYZED', run_id=d.RUN, source_decision=source, rows=rows,
        differences={v: {key+'_pp': value[v]['net_pp'] for key, value in pairings.items()} for v in d.VIEWS},
        paired_errors=pairings,
        suburban_gap_pp={a: {v: 100*(rows[a]['overall']['clean']['accuracy']-rows[a]['overall'][v]['accuracy'])
                               for v in d.SCENES[:3]} for a in rows},
        model_seed_count=1, crossseed_confirmed=False, sample_SD=None, goal_complete=False)
    d.write(d.BASE/'analysis.json', result)
    return result
