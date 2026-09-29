"""Complete support-only OOF and parent-labelled one-shot proxy accounting."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi.d92_branch_orbit_ce import FROZEN_CONFIG
from evaluate_d92_branch_orbit_ce_probe import SCOPE
from summarize_d92_branch_support_probe import (
    COORDS, METRICS, check, read, jsonlines, close, zero_fields, false_fields,
    add, leaves, scalar_tree, expected_cells, check_bind, write_json, finite_tree,
)

ARMS = ('single_ridge','single_ce','orbit_ridge','orbit_ce')
CONTROLS = ('single_ridge','single_ce','orbit_ridge')
PAIRS = {left+'_minus_'+right:(left,right) for left,right in (
    ('orbit_ce','single_ridge'),('orbit_ce','single_ce'),('orbit_ce','orbit_ridge'),
    ('single_ce','single_ridge'),('orbit_ridge','single_ridge'))}
STRATA = dict(by_parent_k=('k',), by_model_seed=('model_seed',), by_support_seed=('support_seed',),
              by_cohort=('cohort',), by_receiver_scene=('receiver','scenario'),
              by_newcount=('new_count',), by_parent_k_newcount=('k','new_count'))


def metrics_from_counts(confusion, sums, counts, classes, old):
    c = len(classes)
    check(len(confusion) == len(sums) == len(counts) == c, 'Class evidence dimensions')
    check(all(len(row) == c and all(type(v) is int and v >= 0 for v in row) for row in confusion), 'Invalid confusion counts')
    check(all(type(v) is int and v > 0 for v in counts) and all(v >= 0 for v in sums), 'Invalid class NLL counts')
    check([sum(row) for row in confusion] == counts, 'Confusion/NLL class count mismatch')
    acc = [confusion[i][i]/counts[i] for i in range(c)]
    nll = [sums[i]/counts[i] for i in range(c)]
    old_acc = [v for name,v in zip(classes,acc) if name in old]
    new_acc = [v for name,v in zip(classes,acc) if name not in old]
    o = sum(old_acc)/len(old_acc) if old_acc else None
    n = sum(new_acc)/len(new_acc) if new_acc else None
    h = (2*o*n/(o+n) if o+n else 0.) if o is not None and n is not None else None
    return dict(accuracy=sum(confusion[i][i] for i in range(c))/sum(counts), macro_accuracy=sum(acc)/c,
                old_accuracy=o,new_accuracy=n,h=h,macro_nll=sum(nll)/c), acc, nll


def verify_class_metrics(result, classes, old, confusion, sums, counts):
    metrics, acc, nll = metrics_from_counts(confusion,sums,counts,classes,old)
    for key,value in metrics.items(): close(result['metrics'][key],value,'Metric mismatch: '+key)
    per_class = result['metrics']['classwise']
    check(len(per_class) == len(classes) and {r['class_id'] for r in per_class} == set(classes), 'Classwise incomplete')
    for row in per_class:
        i = classes.index(row['class_id'])
        check(row['count'] == counts[i], 'Class count mismatch')
        close(row['accuracy'],acc[i],'Class accuracy mismatch'); close(row['nll'],nll[i],'Class NLL mismatch')
    return metrics


def stages(entry, train, c, parent_k, scope):
    check(len(entry['stages']) == 4 and {s['arm'] for s in entry['stages']} == set(ARMS), 'Four fixed arms required')
    factors = 0
    for stage in entry['stages']:
        check(stage['scope'] == scope and stage['parent_k'] == parent_k
              and stage['train_k'] == entry['train_k'] and stage['train_physical_count'] == len(train)
              and len(stage['training_physical_ids']) == len(train) and set(stage['training_physical_ids']) == train
              and stage['all_states_estimated_from_trainfold_only'] is True, 'Stage training boundary mismatch')
        check(stage['train_k']*c == len(train) and stage['ridge_coefficient'] == 1., 'Frozen loss/physical scale mismatch')
        close(stage['physical_loss_mass'],len(train),'Physical loss weight mismatch')
        close(stage['loss_total'],stage['loss_data']+stage['loss_ridge'],'Objective decomposition mismatch')
        check(stage['converged'] is True and stage['iterations'] == stage['optimizer_steps'], 'Unconverged head')
        ce = stage['arm'].endswith('_ce')
        expected = 0 if ce else 1
        if ce:
            verify_ce_steps(stage,len(train))
        else:
            check(stage['optimizer_steps'] == 0 and stage['steps'] == [], 'Ridge invented optimizer steps')
        check(stage['factorization_calls'] == expected, 'Actual factorization count mismatch')
        factors += expected
        for key,value in stage.items():
            if value is not None and (key.endswith('_bytes') or key.endswith('_seconds') or key.endswith('gradient_norm') or key.endswith('equation_residual')):
                check(value >= 0, 'Negative measured resource/residual')
    return factors


def verify_ce_steps(stage,n):
    steps=stage['steps']
    check(len(steps) == stage['optimizer_steps']+1 and [v['iteration'] for v in steps] == list(range(len(steps))), 'CE step trace incomplete')
    tolerance=FROZEN_CONFIG['ce_gradient_rtol']*(1+n**.5)
    close(stage['gradient_tolerance'],tolerance,'Unfrozen CE stationarity tolerance')
    check(type(stage['optimizer_steps']) is int and 0 <= stage['optimizer_steps'] <= FROZEN_CONFIG['ce_max_iterations'] and stage['gradient_norm'] <= tolerance
          and stage['gradient_coordinate'] == 'true_RKHS_parameters', 'CE stationarity not satisfied')
    close(stage['stationarity_residual'],stage['gradient_norm'],'CE stationarity mismatch')
    for step in steps:
        close(step['loss_total'],step['loss_data']+step['loss_ridge'],'CE step objective mismatch')
        check(step['loss_data'] >= 0 and step['loss_ridge'] >= 0 and step['gradient_norm'] >= 0, 'Invalid CE measured loss/gradient')
        check(step['lipschitz_bound'] >= 1 and step['elapsed_seconds'] >= 0, 'Invalid CE numeric schedule')
        close(step['learning_rate'],1/step['lipschitz_bound'],'CE step size mismatch')
        sqrt_l=step['lipschitz_bound']**.5
        close(step['momentum'],(sqrt_l-1)/(sqrt_l+1),'CE momentum mismatch')
        for key in ('lipschitz_bound','learning_rate','momentum'):
            close(step[key],stage[key],'CE schedule changed within fit')
    for key in ('loss_data','loss_ridge','loss_total','gradient_norm'):
        close(steps[-1][key],stage[key],'CE final telemetry differs from final step')


def verify_record(record):
    finite_tree(record)
    classes = record['classes']; old = set(record['old_classes']); c = len(classes); k = record['k']; n = c*k
    check(len(set(classes)) == c and c > 0 and old <= set(classes), 'Invalid registry')
    check(record['parent_k'] == k and record['new_count'] == c-len(old) and record['support_count'] == n, 'Parent/class count mismatch')
    zero_fields(record,('persistent_state_bytes','query_rows_used','source_rows_used'))
    f = 0 if k == 1 else min(k,3)
    check(record['fold_count'] == len(record['folds']) == f, 'Standard folds incomplete')
    if k == 1:
        check(record['oof'] is None and record['paired'] is None and record['oneshot_proxy'] is None
              and record['physical_fold_assignment'] == [] and record['proxy_anchor_count'] == 0
              and record['standard_factorization_count'] == record['factorization_count'] == 0
              and record['optimizer_steps'] == 0
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'True K1 invented holdout')
        return dict(standard={},proxy={},standard_factors=0,proxy_factors=0,held_occurrences=0,optimizer_steps=0,ce_fits=0)
    mapping = {p['physical_id']:p for p in record['physical_fold_assignment']}
    check(len(mapping) == len(record['physical_fold_assignment']) == n, 'Incomplete physical registry')
    by_class = {}
    for cls in classes:
        rows = sorted((p for p in mapping.values() if p['class_id'] == cls),key=lambda p:p['physical_id'])
        check(len(rows) == k and [p['fold'] for p in rows] == [i % f for i in range(k)], 'Physical fold mapping mismatch')
        by_class[cls] = [p['physical_id'] for p in rows]
    check({e['fold'] for e in record['folds']} == set(range(f)), 'Repeated/missing fold')
    standard_factors = 0
    for entry in record['folds']:
        held = {pid for pid,p in mapping.items() if p['fold'] == entry['fold']}; train = set(mapping)-held
        check(len(entry['held_ids']) == len(held) and set(entry['held_ids']) == held
              and len(entry['training_ids']) == len(train) and set(entry['training_ids']) == train
              and entry['train_k'] == len(train)//c and entry['held_k'] == len(held)//c, 'Standard held isolation mismatch')
        standard_factors += stages(entry,train,c,k,'support_oof')
    check(set(record['oof']) == set(ARMS) and set(record['paired']) == set(PAIRS), 'Standard arms/paired incomplete')
    standard = {}
    for arm,result in record['oof'].items():
        rows = result['rows']; lookup = {r['physical_id']:r for r in rows}
        check(len(rows) == n and set(lookup) == set(mapping), 'Standard OOF physical coverage')
        confusion = [[0]*c for _ in classes]; sums = [0.]*c; counts = [0]*c
        for row in rows:
            bound = mapping[row['physical_id']]
            check(row['class_id'] == bound['class_id'] and row['fold'] == bound['fold']
                  and row['predicted_class'] in classes and type(row['correct']) is bool
                  and row['correct'] == (row['class_id'] == row['predicted_class']) and row['nll'] >= 0, 'Standard prediction binding')
            i,j = classes.index(row['class_id']),classes.index(row['predicted_class'])
            confusion[i][j] += 1; sums[i] += row['nll']; counts[i] += 1
        standard[arm] = verify_class_metrics(result,classes,old,confusion,sums,counts)
    for name,(left,right) in PAIRS.items():
        pair = record['paired'][name]; rows = pair['rows']
        check(len(rows) == n and {r['physical_id'] for r in rows} == set(mapping), 'Standard paired coverage')
        a = {r['physical_id']:r['correct'] for r in record['oof'][left]['rows']}
        b = {r['physical_id']:r['correct'] for r in record['oof'][right]['rows']}
        check(all(r['correct_delta'] == int(a[r['physical_id']])-int(b[r['physical_id']]) for r in rows), 'Paired correctness mismatch')
        close(pair['mean_correct_delta'],standard[left]['accuracy']-standard[right]['accuracy'],'Paired mean mismatch')
    proxy = record['oneshot_proxy']
    check(proxy['scope'] == 'SUPPORT_ONESHOT_PROXY_FROM_PARENT_SUPPORT_NOT_FORMAL_K1'
          and proxy['diagnostic'] == 'support_oneshot_proxy' and proxy['parent_split_id'] == record['split_id']
          and proxy['parent_k'] == k and proxy['proxy_train_k'] == 1
          and proxy['trial_count'] == len(proxy['trials']) == record['proxy_anchor_count'] == k
          and proxy['evidence_schema'] == 'anchor_confusion_class_nll_and_exact_physical_mapping_v1', 'Proxy scope/coverage mismatch')
    check({t['trial'] for t in proxy['trials']} == set(range(k)), 'All fixed anchors required')
    proxy_values = {arm:[] for arm in ARMS}; proxy_factors = 0
    for trial in proxy['trials']:
        train = {by_class[cls][trial['trial']] for cls in classes}; held = set(mapping)-train
        check(set(trial['training_ids']) == train and len(trial['training_ids']) == c
              and set(trial['held_ids']) == held and len(trial['held_ids']) == c*(k-1)
              and trial['parent_k'] == k and trial['train_k'] == trial['proxy_train_k'] == 1
              and trial['held_k'] == k-1, 'Proxy anchor/physical isolation mismatch')
        proxy_factors += stages(trial,train,c,k,'support_oneshot_proxy')
        check(set(trial['oof']) == set(ARMS) and set(trial['paired']) == set(PAIRS), 'Proxy arms incomplete')
        for arm,result in trial['oof'].items():
            check('rows' not in result and result['classwise_count'] == [k-1]*c and result['class_order'] == classes and result['record_count'] == len(held), 'Proxy compression/held class count mismatch')
            proxy_values[arm].append(verify_class_metrics(result,classes,old,result['confusion'],result['classwise_nll_sum'],result['classwise_count']))
        for name,(left,right) in PAIRS.items():
            pair = trial['paired'][name]; pcounts=pair['counts']
            check(set(pcounts) == {'both_correct','left_only_correct','right_only_correct','both_wrong'}
                  and all(type(v) is int and v >= 0 for v in pcounts.values())
                  and sum(pcounts.values()) == pair['record_count'] == len(held), 'Proxy paired four-count mismatch')
            close((pcounts['both_correct']+pcounts['left_only_correct'])/len(held),proxy_values[left][-1]['accuracy'],'Left paired count mismatch')
            close((pcounts['both_correct']+pcounts['right_only_correct'])/len(held),proxy_values[right][-1]['accuracy'],'Right paired count mismatch')
            close(trial['paired'][name]['mean_correct_delta'],proxy_values[left][-1]['accuracy']-proxy_values[right][-1]['accuracy'],'Proxy paired mean mismatch')
    parent = {arm:{key:None if values[0][key] is None else sum(v[key] for v in values)/k for key in METRICS}
              for arm,values in proxy_values.items()}
    for arm in ARMS:
        for key in METRICS: close(proxy['parent_mean_metrics'][arm][key],parent[arm][key],'Parent-first proxy aggregation mismatch')
    check(proxy['coverage'] == dict(parent_physical_count=n,training_occurrences=n,held_occurrences=n*(k-1),unique_held_physical_count=n), 'Proxy occurrence accounting mismatch')
    check(proxy['factorization_count'] == proxy_factors and record['standard_factorization_count'] == standard_factors
          and record['factorization_count'] == standard_factors+proxy_factors, 'Actual solve totals mismatch')
    all_stages=[s for e in record['folds']+proxy['trials'] for s in e['stages']]
    optimizer_steps=sum(s['optimizer_steps'] for s in all_stages)
    check(record['optimizer_steps'] == optimizer_steps,'Optimizer step total mismatch')
    return dict(standard=standard,proxy=parent,standard_factors=standard_factors,
                proxy_factors=proxy_factors,held_occurrences=n*(k-1),optimizer_steps=optimizer_steps,
                ce_fits=sum(s['arm'].endswith('_ce') for s in all_stages))


def assessment(by_k, full_matrix):
    output={}
    for diagnostic in ('physical_oof','support_oneshot_proxy'):
        rows=[]
        for k in (5,10,20):
            for control in CONTROLS:
                values={}
                for field in ('h','new_accuracy','old_accuracy'):
                    stat=by_k.get((diagnostic,'new_present',k,'paired','orbit_ce_minus_'+control+'.'+field))
                    values[field]=stat.result()['mean'] if stat else None
                available=full_matrix and all(v is not None for v in values.values())
                passes=(values['h']>0 and values['new_accuracy']>0 and values['old_accuracy']>=-.01) if available else None
                rows.append(dict(parent_k=k,control=control,delta=values,available=available,passes=passes,
                    all_three_strictly_improve=all(v>0 for v in values.values()) if available else None))
        available=all(r['available'] for r in rows)
        output[diagnostic]=dict(comparisons=rows,available=available,passes=all(r['passes'] for r in rows) if available else None)
    return dict(standard=output['physical_oof'],oneshot_proxy=output['support_oneshot_proxy'],automatic_promotion=False,
        rule='Each parent K and each diagnostic separately: orbit_ce beats all three controls in joint new/H, with joint old delta >= -0.01. Strict old/new/H improvement also reported. All-old separate; proxy is not formal K1.')


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec,dict) else spec
    root = Path(run_root or spec['execution']['remote_run_root']); out = Path(output)
    if out.exists(): raise FileExistsError(out)
    rows,cohorts = spec['rows'],spec['probe']['cohorts']
    check(len(rows) == 8 and len({r['row_id'] for r in rows}) == 8 and set(cohorts) == {'rx1','rx3'}, 'Eight lanes required')
    check({(r['cohort'],r['seeds']['model']) for r in rows} == {(c,s) for c in cohorts for s in range(2026092701,2026092705)}, 'Model/cohort coverage mismatch')
    expected_total = sum(len(expected_cells(cohorts[r['cohort']]['matrix'])) for r in rows)
    check(expected_total == spec['probe']['total_episodes'], 'Spec matrix count mismatch')
    launch,complete,state = [read(root/name) for name in ('startup.json','complete.json','state.json')]
    check(launch['spec'] == spec and complete['status'] == 'SUPPORT_PROBE_COMPLETE'
          and complete['completed_rows'] == complete['model_rows'] == 8 and complete['episodes'] == expected_total, 'Incomplete run')
    for value in (launch,complete): false_fields(value,('query_access','source_sample_access'))
    check(complete['commit'] == launch['commit'] and complete['finished'] >= launch['started'], 'Run identity/time mismatch')
    check(set(state) == {r['row_id'] for r in rows} and all(v['status'] == 'SUPPORT_PROBE_COMPLETE' for v in state.values()), 'Incomplete lane state')
    counts = dict(parent_episodes=0,k1_numerical_only=0,standard_oof_parents=0,proxy_parents=0,proxy_anchors=0,
                  standard_factorizations=0,proxy_factorizations=0,actual_factorizations=0,proxy_held_occurrences_per_arm=0,optimizer_steps=0,ce_fits=0)
    overall={}; groups={name:{} for name in STRATA}; costs={}; stage_stats={}
    for row in rows:
        rid=row['row_id']; check(Path(rid).name == rid and '/' not in rid and '\\' not in rid,'Unsafe row ID')
        lane=root/rid/'probe'; co=cohorts[row['cohort']]; expected=expected_cells(co['matrix'])
        marker,startup=[read(lane/name) for name in ('probe_complete.json','startup.json')]
        feature=read(Path(startup['support_features'])/'features_complete.json')
        for value in (marker,startup,feature): check_bind(value,row,co)
        check(feature['status'] == 'BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == len(expected),'Producer completion mismatch')
        false_fields(feature,('query_iq_access','source_data_access','truth_read','adapted_state_inherited'))
        check(marker['status'] == 'SUPPORT_PROBE_COMPLETE' and marker['scope'] == SCOPE
              and marker['algorithm'] == FROZEN_CONFIG and marker['matrix'] == co['matrix']
              and startup['config'] == dict(algorithm=FROZEN_CONFIG,matrix=co['matrix']), 'Frozen algorithm/scope mismatch')
        for value in (marker,startup): zero_fields(value,('query_rows_used','source_rows_used'))
        false_fields(startup,('query_iq_access','truth_read','adapted_state_inherited','cross_row_adapted_state_reuse','checkpoint_loaded','encoder_updated'))
        check(startup['payload_audit'] == marker['payload_audit'] and marker['payload_audit']['frozen_support_cache_loaded'] is True,'Payload mismatch')
        for key,value in leaves(marker['payload_audit']): add(costs,'payload.'+key,value)
        add(costs,'lane.wall_seconds',marker['wall_seconds'])
        seen=set(); split_ids=set(); lane_k1=lane_proxy=lane_factors=lane_standard=lane_steps=0
        for record,small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'),jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None,'Trace/compact count mismatch')
            cell=tuple(record[k] for k in COORDS)
            check(cell in expected and cell not in seen and record['split_id'] not in split_ids,'Duplicate/unexpected parent')
            seen.add(cell);split_ids.add(record['split_id'])
            check(record['config'] == FROZEN_CONFIG and record['scope'] == record['claim_scope'] == SCOPE
                  and set(record['classes']) == set(record['registered_classes']) and record['old_classes'] == sorted(feature['classes']), 'Trace registry/scope mismatch')
            verified=verify_record(record)
            for key in COORDS+('split_id','support_count','fold_count','factorization_count','standard_factorization_count','proxy_anchor_count','optimizer_steps','persistent_state_bytes'):
                check(small[key] == record[key],'Compact metadata mismatch')
            for key in ('numerical','oof','paired','oneshot_proxy'): check(small[key] == scalar_tree(record[key]),'Compact evidence mismatch')
            check(small['completed'] == len(seen) and small['total'] == len(expected) and small['classes'] == len(record['classes']),'Compact progress mismatch')
            zero_fields(small,('query_rows_used','source_rows_used'))
            for key in ('fit_seconds','fit_call_seconds'): close(small[key],record[key],'Fit timing mismatch')
            for key in ('fit_seconds','fit_call_seconds','log_write_seconds','total_seconds','peak_process_rss_bytes'):
                check(small[key] is None or small[key] >= 0,'Negative resource');add(costs,'probe.'+key,small[key])
            k1=record['k'] == 1; anchors=record['proxy_anchor_count']
            counts['parent_episodes']+=1;counts['k1_numerical_only']+=k1;counts['standard_oof_parents']+=not k1
            counts['proxy_parents']+=not k1;counts['proxy_anchors']+=anchors
            counts['standard_factorizations']+=verified['standard_factors'];counts['proxy_factorizations']+=verified['proxy_factors']
            counts['actual_factorizations']+=record['factorization_count'];counts['proxy_held_occurrences_per_arm']+=verified['held_occurrences']
            lane_k1+=k1;lane_proxy+=anchors;lane_factors+=record['factorization_count'];lane_standard+=verified['standard_factors']
            counts['optimizer_steps']+=verified['optimizer_steps'];counts['ce_fits']+=verified['ce_fits']
            lane_steps+=verified['optimizer_steps']
            population='old_only' if record['new_count'] == 0 else 'new_present'
            coordinates=dict(record,model_seed=row['seeds']['model'],cohort=row['cohort'])
            for metric,value in leaves(record['numerical']):
                add(overall,('parent_numerical',population,'numerical',metric),value)
                for name,dimensions in STRATA.items():
                    add(groups[name],('parent_numerical',population,*(coordinates[d] for d in dimensions),'numerical',metric),value)
            for diagnostic,data in [('physical_oof',verified['standard']),('support_oneshot_proxy',verified['proxy'])]:
                metrics=[]
                for arm,values in data.items(): metrics += [('arm',arm+'.'+key,value) for key,value in values.items()]
                for pair,(left,right) in PAIRS.items():
                    if data:
                        metrics += [('paired',pair+'.'+key,None if data[left][key] is None else data[left][key]-data[right][key]) for key in METRICS]
                for kind,metric,value in metrics:
                    add(overall,(diagnostic,population,kind,metric),value)
                    for name,dimensions in STRATA.items():
                        add(groups[name],(diagnostic,population,*(coordinates[d] for d in dimensions),kind,metric),value)
            entries=list(record['folds'])+(record['oneshot_proxy']['trials'] if record['oneshot_proxy'] else [])
            for entry in entries:
                for stage in entry['stages']:
                    for key,value in leaves(stage): add(stage_stats,(stage['scope'],stage['arm'],key),value)
        check(seen == expected and len(seen) == marker['episodes'] == state[rid]['episodes'],'Parent matrix incomplete')
        check(marker['k1_episodes'] == lane_k1 and marker['oof_episodes'] == len(seen)-lane_k1
              and marker['proxy_anchor_count'] == lane_proxy and marker['standard_factorization_count'] == lane_standard
              and marker['factorization_count'] == lane_factors and marker['optimizer_steps'] == lane_steps,'Lane totals mismatch')
    check(counts['parent_episodes'] == expected_total,'Global parent coverage mismatch')
    check(complete['optimizer_steps'] == counts['optimizer_steps'],'Run optimizer total mismatch')
    if expected_total == 4800:
        check(tuple(counts[k] for k in ('k1_numerical_only','standard_oof_parents','proxy_anchors','proxy_held_occurrences_per_arm')) == (1200,3600,42000,7879200),'Full standard/proxy coverage mismatch')
        check(counts['actual_factorizations'] == counts['ce_fits'] == 105600,'Full four-arm solver coverage mismatch')
    counts['episodes'] = counts['parent_episodes']  # Transport compatibility alias; not another population.
    out.mkdir(parents=True,exist_ok=False)
    for name,stats in groups.items():
        fields=['diagnostic','population',*STRATA[name],'kind','metric','count','null_count','sum','mean','min','max']
        with (out/(name+'.csv')).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for key,stat in sorted(stats.items()):writer.writerow(dict(zip(fields[:len(key)],key),**stat.result()))
    summary=dict(status='COMPLETE_SUPPORT_DIAGNOSTIC_VERIFIED',scope=SCOPE,run_id=spec.get('run_id'),
        release_commit=complete['commit'],coverage=counts,run_wall_seconds=complete['finished']-launch['started'],
        statistics=[dict(diagnostic=k[0],population=k[1],kind=k[2],metric=k[3],**v.result()) for k,v in sorted(overall.items())],
        resources={k:v.result() for k,v in sorted(costs.items())},
        fit_stages=[dict(scope=k[0],arm=k[1],metric=k[2],**v.result()) for k,v in sorted(stage_stats.items())],
        candidate_assessment=assessment(groups['by_parent_k'],expected_total == 4800),
        query_rows_used=0,source_rows_used=0,new_source_payload_bytes=0,optimizer_steps=counts['optimizer_steps'],
        automatic_promotion=False,selected_arm=None,
        interpretation=['coverage.episodes aliases parent_episodes; it is not an additional count.',
                       'True K1 is numerical-only; no OOF or proxy performance exists.',
                       'One-shot proxy means use all anchors within each parent first, then equal parent weights; parent K is retained.',
                       'Exact train/held physical mappings and per-anchor confusion/NLL counts remain in full traces; no query scores are used.',
                       'Standard OOF and proxy are separate; repeated held occurrences and support draws are correlated, not independent sample counts.',
                       'CE heads use physical-sum cross entropy and fixed RKHS regularization; full training steps are retained, with source validation unavailable because source inputs are prohibited.',
                       'Macro NLL uses fixed softmax of each arm\'s own scores, without post-hoc calibration. The runtime shared metrics helper labels all arms as uncalibrated_ridge_scores; that inherited text is inaccurate for CE arms, whose scores come from the CE head. Numeric NLL computation is unchanged and NLL is not a screening criterion.',
                       'A/B screens are support development evidence, not formal K1/query generalization or automatic promotion.'])
    write_json(out/'summary.json',summary)
    lines=['# BranchOrbitCE support 诊断','','标准物理 OOF 与 support 内部 1-shot proxy 分别报告；proxy 的 K 表示 parent K，绝不是正式 K1。',
           '',json.dumps(counts,ensure_ascii=False),'','| 诊断 | parent K | 对照 | Δ旧 | Δ新 | ΔH | 通过 |','|---|---:|---|---:|---:|---:|---|']
    for name in ('standard','oneshot_proxy'):
        for row in summary['candidate_assessment'][name]['comparisons']:
            values=['N/A' if row['delta'][m] is None else f"{100*row['delta'][m]:+.3f}" for m in ('old_accuracy','new_accuracy','h')]
            lines.append('| '+' | '.join([name,str(row['parent_k']),row['control'],*values,str(row['passes'])])+' |')
    lines += ['','差值单位为百分点。全部新增类规模、旧类单独任务、模型、support seed、RX×场景和parent K分层见CSV。',
              '真实K1无独立held证据。proxy每parent内先平均全部anchor，不将42000次anchor或重复held次数当独立样本量。','']
    lines += ['- '+line for line in summary['interpretation']]
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--run-root');parser.add_argument('--output',required=True)
    result=summarize(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],coverage=result['coverage'])))


if __name__ == '__main__':main()
