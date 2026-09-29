"""Verify and summarize complete new support interaction traces; no query API."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi.d92_branch_interaction import FROZEN_CONFIG
from summarize_d92_branch_support_probe import (
    SCOPE, COORDS, METRICS, check, read, jsonlines, close, zero_fields, false_fields,
    add, leaves, scalar_tree, expected_cells, check_bind, write_json,
)

ARMS = ('linear', 'energy_control', 'interaction')
PAIRS = {'interaction_minus_linear': ('interaction','linear'),
         'energy_control_minus_linear': ('energy_control','linear'),
         'interaction_minus_energy_control': ('interaction','energy_control')}
STRATA = dict(by_k=('k',), by_model_seed=('model_seed',), by_support_seed=('support_seed',),
    by_cohort=('cohort',), by_receiver_scene=('receiver','scenario'),
    by_newcount=('new_count',), by_k_newcount=('k','new_count'))


def verify_oof(record):
    classes = set(record['classes']); old = set(record['old_classes'])
    c, k = len(classes), record['k']; n = c*k
    check(c == len(record['classes']) and c > 0 and old <= classes, 'Invalid classes')
    check(record['new_count'] == c-len(old) and record['support_count'] == n, 'Class/count mismatch')
    folds = 0 if k == 1 else min(k, 3)
    check(record['fold_count'] == len(record['folds']) == folds
          and record['factorization_count'] == folds*3, 'Fold/factorization mismatch')
    zero_fields(record, ('optimizer_steps','persistent_state_bytes','query_rows_used','source_rows_used'))
    if k == 1:
        check(record['oof'] is None and record['paired'] is None
              and record['physical_fold_assignment'] == []
              and record['heldout_unavailable_reason'] == 'K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT', 'K1 fabricated OOF')
        return
    assignment = record['physical_fold_assignment']
    lookup = {p['physical_id']: p for p in assignment}
    check(len(assignment) == len(lookup) == n, 'Incomplete physical assignment')
    check({p['class_id'] for p in assignment} == classes, 'Assignment class mismatch')
    for cls in classes:
        selected = sorted((p for p in assignment if p['class_id'] == cls), key=lambda p:p['physical_id'])
        check(len(selected) == k and [p['fold'] for p in selected] == [i % folds for i in range(k)], 'Physical fold mismatch')
    check({f['fold'] for f in record['folds']} == set(range(folds)), 'Missing fold')
    for fold in record['folds']:
        held = {pid for pid,p in lookup.items() if p['fold'] == fold['fold']}; training = set(lookup)-held
        check(set(fold['held_ids']) == held and len(fold['held_ids']) == len(held)
              and set(fold['training_ids']) == training and len(fold['training_ids']) == len(training), 'Physical leakage')
        check(fold['train_k'] == len(training)//c and len(fold['stages']) == 3
              and {s['arm'] for s in fold['stages']} == set(ARMS), 'Incomplete three-arm fold')
        for stage in fold['stages']:
            check(stage['factorization_calls'] == 1 and stage['factorization_dim'] == len(training), 'Kernel solve dimension mismatch')
            check(stage['train_physical_count'] == len(training) and stage['train_k'] == fold['train_k']
                  and stage['all_states_estimated_from_trainfold_only'] is True, 'Trainfold contract mismatch')
            zero_fields(stage, ('optimizer_steps',))
            close(stage['loss_total'], stage['loss_data']+stage['loss_ridge'], 'Objective components mismatch')
            for key, value in stage.items():
                if value is not None and (key.endswith('_bytes') or key.endswith('_seconds') or key.startswith('loss_')
                                         or key.endswith('gradient_norm') or key.endswith('equation_residual')):
                    check(value >= 0, 'Negative measured stage value')
    check(set(record['oof']) == set(ARMS), 'Missing OOF arm')
    for result in record['oof'].values():
        rows = result['rows']; by_id = {p['physical_id']:p for p in rows}
        check(len(rows) == len(by_id) == n and set(by_id) == set(lookup), 'Incomplete OOF physical rows')
        for pid, p in by_id.items():
            check(p['class_id'] == lookup[pid]['class_id'] and p['fold'] == lookup[pid]['fold']
                  and p['predicted_class'] in classes and type(p['correct']) is bool
                  and p['correct'] == (p['predicted_class'] == p['class_id']) and p['nll'] >= 0, 'OOF physical row mismatch')
        metrics = result['metrics']; classwise = metrics['classwise']
        check(len(classwise) == c and {p['class_id'] for p in classwise} == classes, 'Classwise coverage mismatch')
        for cls in classwise:
            selected = [p for p in rows if p['class_id'] == cls['class_id']]
            check(cls['count'] == len(selected) == k, 'Class physical count mismatch')
            close(cls['accuracy'], sum(p['correct'] for p in selected)/k, 'Class accuracy mismatch')
            close(cls['nll'], sum(p['nll'] for p in selected)/k, 'Class NLL mismatch')
        accuracy = sum(p['correct'] for p in rows)/n
        close(metrics['accuracy'], accuracy, 'Accuracy mismatch')
        close(metrics['macro_accuracy'], accuracy, 'Balanced macro mismatch')
        close(metrics['macro_nll'], sum(p['nll'] for p in rows)/n, 'NLL mismatch')
        o = sum(p['accuracy'] for p in classwise if p['class_id'] in old)/len(old) if old else None
        new = sum(p['accuracy'] for p in classwise if p['class_id'] not in old)/(c-len(old)) if c > len(old) else None
        h = (2*o*new/(o+new) if o+new else 0.) if o is not None and new is not None else None
        for key, value in (('old_accuracy',o),('new_accuracy',new),('h',h)):
            close(metrics[key], value, key+' mismatch')
    check(set(record['paired']) == set(PAIRS), 'Missing paired comparison')
    for name, (left, right) in PAIRS.items():
        actual = record['paired'][name]; rows = actual['rows']
        left_rows = {p['physical_id']:p for p in record['oof'][left]['rows']}
        right_rows = {p['physical_id']:p for p in record['oof'][right]['rows']}
        check(len(rows) == n and {p['physical_id'] for p in rows} == set(lookup), 'Incomplete paired rows')
        check(all(p['correct_delta'] == int(left_rows[p['physical_id']]['correct'])-int(right_rows[p['physical_id']]['correct'])
                  for p in rows), 'Paired physical delta mismatch')
        close(actual['mean_correct_delta'], sum(p['correct_delta'] for p in rows)/n, 'Paired mean mismatch')


def episode_metrics(record):
    for key, value in (('episodes',1),('k1_episodes',int(record['k'] == 1)),('oof_episodes',int(record['k'] > 1))):
        yield 'coverage', key, value
    for key, value in leaves(record['numerical']): yield 'numerical', key, value
    if record['k'] == 1: return
    for arm, result in record['oof'].items():
        for key in METRICS: yield 'arm', arm+'.'+key, result['metrics'][key]
    for name, (left, right) in PAIRS.items():
        for key in METRICS:
            a,b = (record['oof'][arm]['metrics'][key] for arm in (left,right))
            yield 'paired', name+'.'+key, a-b if a is not None and b is not None else None


def candidate_assessment(overall, by_k, *, full_matrix):
    """Apply the preregistered descriptive rule; this never promotes a method."""
    def mean(stats, key):
        stat = stats.get(key)
        return stat.result()['mean'] if stat else None
    comparisons = {}
    for control in ('linear','energy_control'):
        pair = 'interaction_minus_'+control
        h = mean(overall,('new_present','paired',pair+'.h'))
        new = mean(overall,('new_present','paired',pair+'.new_accuracy'))
        h_by_k = {str(k):mean(by_k,('new_present',k,'paired',pair+'.h')) for k in (5,10,20)}
        new_by_k = {str(k):mean(by_k,('new_present',k,'paired',pair+'.new_accuracy')) for k in (5,10,20)}
        old = {str(k):mean(by_k,('new_present',k,'paired',pair+'.old_accuracy')) for k in (5,10,20)}
        allold = {str(k):mean(by_k,('allold',k,'paired',pair+'.old_accuracy')) for k in (5,10,20)}
        available = full_matrix and all(v is not None for group in (h_by_k,new_by_k,old) for v in group.values())
        h_pass = {k:(v > 0 if v is not None else None) for k,v in h_by_k.items()}
        new_pass = {k:(v > 0 if v is not None else None) for k,v in new_by_k.items()}
        old_pass = {k:(v >= -.01 if v is not None else None) for k,v in old.items()}
        comparisons[control] = dict(mean_h_delta=h,mean_new_accuracy_delta=new,
            h_delta_by_k=h_by_k,new_accuracy_delta_by_k=new_by_k,
            h_improvement_pass_by_k=h_pass,new_improvement_pass_by_k=new_pass,
            old_accuracy_delta_by_k=old,old_guard_pass_by_k=old_pass,
            allold_accuracy_delta_by_k=allold,available=available,
            passes=all(v for group in (h_pass,new_pass,old_pass) for v in group.values()) if available else None)
    available = all(v['available'] for v in comparisons.values())
    return dict(scope='SUPPORT_ONLY_CANDIDATE_SCREEN_NOT_QUERY_GENERALIZATION',
        rule='For each of K5/10/20 separately on joint new-present tasks, mean H and new accuracy strictly improve versus both controls, and old accuracy delta is at least -0.01. Cross-K means are descriptive only.',
        comparisons=comparisons,available=available,
        passes=all(v['passes'] for v in comparisons.values()) if available else None,
        allold_scope='Reported separately; not substituted for joint old-class guard.',
        k1_scope='Numerical diagnostics only; no performance evidence.',automatic_promotion=False)


def summarize(*, spec, run_root=None, output):
    spec = read(spec) if not isinstance(spec, dict) else spec
    root = Path(run_root or spec['execution']['remote_run_root']); out = Path(output)
    if out.exists(): raise FileExistsError(out)
    rows = spec['rows']; cohorts = spec['probe']['cohorts']
    check(len(rows) == 8 and len({r['row_id'] for r in rows}) == 8 and set(cohorts) == {'rx1','rx3'}, 'Eight unique lanes required')
    check({(r['cohort'],r['seeds']['model']) for r in rows} ==
          {(co,seed) for co in cohorts for seed in range(2026092701,2026092705)}, 'Incomplete model/cohort matrix')
    expected_total = sum(len(expected_cells(cohorts[r['cohort']]['matrix'])) for r in rows)
    check(expected_total == spec['probe']['total_episodes'], 'Spec episode count mismatch')
    launch = read(root/'startup.json'); complete = read(root/'complete.json'); state = read(root/'state.json')
    check(launch['spec'] == spec, 'Startup spec binding mismatch')
    false_fields(launch, ('query_access','source_sample_access'))
    check(complete['status'] == 'SUPPORT_PROBE_COMPLETE' and complete['completed_rows'] == complete['model_rows'] == 8
          and complete['episodes'] == expected_total, 'Run incomplete')
    false_fields(complete, ('query_access','source_sample_access','query_performance_claim'))
    zero_fields(complete, ('optimizer_steps',))
    check(complete['commit'] == launch['commit'] and complete['finished'] >= launch['started'], 'Run commit/time mismatch')
    check(set(state) == {r['row_id'] for r in rows} and all(v['status'] == 'SUPPORT_PROBE_COMPLETE' for v in state.values()), 'Incomplete lane state')
    overall = {}; groups = {name:{} for name in STRATA}; costs = {}; stages = {}; lanes = []
    counts = dict(episodes=0,k1_episodes=0,oof_episodes=0,factorizations=0,physical_oof_records_per_arm=0)
    for row in rows:
        rid = row['row_id']; check(Path(rid).name == rid and '/' not in rid and '\\' not in rid, 'Unsafe row ID')
        lane = root/rid/'probe'; co = cohorts[row['cohort']]; expected = expected_cells(co['matrix'])
        marker = read(lane/'probe_complete.json'); startup = read(lane/'startup.json')
        # Only the metadata of the explicitly bound producer is read; no old probe result is opened.
        feature = read(Path(startup['support_features'])/'features_complete.json')
        for record in (marker,startup,feature): check_bind(record,row,co)
        check(feature['status'] == 'BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['split_count'] == len(expected), 'Producer completion mismatch')
        false_fields(feature, ('query_iq_access','source_data_access','truth_read','adapted_state_inherited'))
        check(marker['status'] == 'SUPPORT_PROBE_COMPLETE' and marker['scope'] == SCOPE
              and marker['algorithm'] == FROZEN_CONFIG and marker['matrix'] == co['matrix'], 'New diagnostic contract mismatch')
        check(startup['config'] == dict(algorithm=FROZEN_CONFIG,matrix=co['matrix']), 'Startup algorithm mismatch')
        zero_fields(marker, ('query_rows_used','source_rows_used','optimizer_steps','persistent_state_bytes'))
        zero_fields(startup, ('query_rows_used','source_rows_used','optimizer_steps'))
        false_fields(startup, ('query_iq_access','truth_read','adapted_state_inherited','cross_row_adapted_state_reuse','checkpoint_loaded','encoder_updated'))
        false_fields(marker, ('truth_read',))
        check(startup['payload_audit'] == marker['payload_audit'] and marker['payload_audit']['feature_cache_reused'] is True, 'Payload audit mismatch')
        for key,value in leaves(marker['payload_audit']): add(costs,'payload.'+key,value)
        add(costs,'lane.wall_seconds',marker['wall_seconds'])
        seen = set(); split_ids = set(); k1 = factor = 0
        for record, small in itertools.zip_longest(jsonlines(lane/'fit_trace.jsonl'), jsonlines(lane/'compact.jsonl')):
            check(record is not None and small is not None, 'Trace/compact count mismatch')
            cell = tuple(record[key] for key in COORDS)
            check(cell in expected and cell not in seen and record['split_id'] not in split_ids, 'Duplicate/unexpected episode')
            seen.add(cell); split_ids.add(record['split_id'])
            check(record['config'] == FROZEN_CONFIG and record['scope'] == record['claim_scope'] == SCOPE, 'Trace algorithm/scope mismatch')
            check(set(record['classes']) == set(record['registered_classes']) and record['old_classes'] == sorted(feature['classes']), 'Trace classes mismatch')
            verify_oof(record)
            check(all(small[key] == record[key] for key in COORDS+('split_id','support_count','fold_count','factorization_count',
                  'optimizer_steps','persistent_state_bytes')), 'Compact metadata mismatch')
            check(small['completed'] == len(seen) and small['total'] == len(expected) and small['classes'] == len(record['classes']), 'Compact progress mismatch')
            zero_fields(small, ('query_rows_used','source_rows_used'))
            for key in ('numerical','oof','paired'): check(small[key] == scalar_tree(record[key]), 'Compact diagnostic mismatch')
            for key in ('fit_seconds','fit_call_seconds'): close(small[key],record[key],'Fit timing mismatch')
            for key in ('fit_seconds','fit_call_seconds','log_write_seconds','total_seconds','peak_process_rss_bytes'):
                check(small[key] is None or small[key] >= 0, 'Negative measured resource')
                add(costs,'probe.'+key,small[key])
            population = 'allold' if record['new_count'] == 0 else 'new_present'
            coords = dict(record,model_seed=row['seeds']['model'],cohort=row['cohort'])
            for kind,metric,value in episode_metrics(record):
                add(overall,(population,kind,metric),value)
                for name, dimensions in STRATA.items():
                    add(groups[name],(population,*(coords[d] for d in dimensions),kind,metric),value)
            for fold in record['folds']:
                for stage in fold['stages']:
                    for key,value in leaves(stage): add(stages,(stage['arm'],key),value)
            counts['episodes'] += 1; counts['k1_episodes'] += record['k'] == 1
            counts['oof_episodes'] += record['k'] > 1; counts['factorizations'] += record['factorization_count']
            counts['physical_oof_records_per_arm'] += record['support_count'] if record['k'] > 1 else 0
            k1 += record['k'] == 1; factor += record['factorization_count']
        check(seen == expected and len(seen) == marker['episodes'] == state[rid]['episodes'], 'Lane matrix incomplete')
        check(marker['k1_episodes'] == k1 and marker['oof_episodes'] == len(seen)-k1 and marker['factorization_count'] == factor, 'Lane totals mismatch')
        lanes.append(dict(row_id=rid,cohort=row['cohort'],model_seed=row['seeds']['model'],episodes=len(seen),
            k1_episodes=k1,factorizations=factor,checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id=co['capsule_id']))
    check(counts['episodes'] == expected_total, 'Global count mismatch')
    if expected_total == 4800:
        check((counts['k1_episodes'],counts['oof_episodes'],counts['factorizations']) == (1200,3600,32400), 'Full matrix coverage mismatch')
    out.mkdir(parents=True,exist_ok=False)
    for name, statistics in groups.items():
        fields = ['population',*STRATA[name],'kind','metric','count','null_count','sum','mean','min','max']
        with (out/(name+'.csv')).open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for key,stat in sorted(statistics.items()): writer.writerow(dict(zip(fields[:len(key)],key),**stat.result()))
    stage_records = [dict(arm=key[0],metric=key[1],**stat.result()) for key,stat in sorted(stages.items())]
    with (out/'fit_stage_statistics.csv').open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['arm','metric','count','null_count','sum','mean','min','max'])
        writer.writeheader(); writer.writerows(stage_records)
    interpretation = [
        'Equal-episode descriptive means; repeated and overlapping support draws are correlated. No independent-sample confidence interval or p-value is claimed.',
        'Allold and new-present populations are separate; undefined new/H metrics remain null. Paired deltas compare the same physical OOF rows and episodes.',
        'K1 has numerical diagnostics only; no independent held prediction and no classifier fitting.',
        'All three predetermined kernels are reported. No automatic arm selection, query evaluation, or deployment is performed.',
        'Extraction and checkpoint reuse cost zero new forwards/transfers in this run; cache loading and analytical fitting costs are measured separately.',
        'Process time sums are work totals; RSS values are per-process high-water marks, not aggregate concurrent memory.',
        'Only current run traces/metadata and explicitly bound cache completion metadata were read; no IQ, cache arrays, historical result, or query/truth file was read.',
    ]
    summary = dict(status='COMPLETE_SUPPORT_DIAGNOSTIC_VERIFIED',scope=SCOPE,run_id=spec.get('run_id'),
        run_root=str(root),release_commit=complete['commit'],run_wall_seconds=complete['finished']-launch['started'],
        coverage=counts,lanes=lanes,statistics=[dict(population=key[0],kind=key[1],metric=key[2],**stat.result())
            for key,stat in sorted(overall.items())],resources={key:stat.result() for key,stat in sorted(costs.items())},
        support_candidate_assessment=candidate_assessment(overall,groups['by_k'],full_matrix=expected_total == 4800),
        fit_stage_statistics=stage_records,query_rows_used=0,source_rows_used=0,new_source_payload_bytes=0,
        persistent_classifier_bytes=0,optimizer_steps=0,selected_arm=None,automatic_promotion=False,interpretation=interpretation)
    write_json(out/'summary.json',summary)
    lines = ['# Support-only branch interaction diagnostic','',
        f"Verified {counts['episodes']} episodes: {counts['k1_episodes']} K1 numerical-only, {counts['oof_episodes']} OOF, {counts['factorizations']} factorizations.",'',
        'Descriptive equal-episode means. Deltas are fractions; multiply by 100 for percentage points.','',
        '| Population | Metric | N | Mean |','|---|---|---:|---:|']
    for record in summary['statistics']:
        if record['kind'] in ('arm','paired'):
            mean = 'N/A' if record['mean'] is None else f"{record['mean']:.8g}"
            lines.append(f"| {record['population']} | {record['metric']} | {record['count']} | {mean} |")
    lines += ['']+['- '+line for line in interpretation]+['']
    assessment = summary['support_candidate_assessment']
    lines += ['Support-only candidate screen: '+('PASS' if assessment['passes'] else 'FAIL' if assessment['passes'] is False else 'UNAVAILABLE')+'.',
              assessment['rule'],'This screen does not authorize automatic promotion or establish query performance.','']
    with (out/'report.md').open('x',encoding='utf-8') as stream: stream.write('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True); parser.add_argument('--run-root'); parser.add_argument('--output',required=True)
    result = summarize(**vars(parser.parse_args()))
    print(json.dumps(dict(status=result['status'],coverage=result['coverage']),allow_nan=False))


if __name__ == '__main__': main()
