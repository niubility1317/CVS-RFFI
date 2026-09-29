"""Four-arm support-only orbit and CE probe from frozen four-view branch caches."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi.d92_branch_orbit_ce import FROZEN_CONFIG, probe_branch_orbit_ce
from evaluate_d92_branch_support_probe import (
    read, write, check, scalars, csv_record, peak_rss,
)
from export_d92_branch_orbit_support_features import load_support


SCOPE = 'SUPPORT_OOF_AND_SUPPORT_ONESHOT_PROXY_NOT_QUERY_EVALUATION'
TEXT_STEP_INTERVAL = 25


def compact_proxy(audit, split_id):
    """Add parent-first scalar summaries to core compact anchor evidence."""
    proxy = audit['oneshot_proxy']
    audit['proxy_anchor_count'] = 0 if proxy is None else proxy['trial_count']
    if proxy is None:
        return audit
    proxy.update(parent_split_id=split_id, diagnostic='support_oneshot_proxy',
                 evidence_schema='anchor_confusion_class_nll_and_exact_physical_mapping_v1')
    proxy['parent_mean_metrics'] = {
        arm: {key: (sum(t['oof'][arm]['metrics'][key] for t in proxy['trials'])/len(proxy['trials'])
                    if proxy['trials'][0]['oof'][arm]['metrics'][key] is not None else None)
              for key in ('accuracy','macro_accuracy','old_accuracy','new_accuracy','h','macro_nll')}
        for arm in audit['oof']}
    return audit


ARMS = ('single_ridge','single_ce','orbit_ridge','orbit_ce')


def validate_audit(audit, split):
    """Check complete stage coverage and physical held isolation before logging."""
    support = set(split['support_ids']); k = split['k']; classes = len(split['registered_classes'])
    if k == 1:
        check(not audit['folds'] and audit['oneshot_proxy'] is None and audit['optimizer_steps'] == 0
              and audit['factorization_count'] == 0, 'True K1 has no independent training/holdout diagnostic')
        return [], []
    check(set(audit['oof']) == set(ARMS), 'Incomplete standard arms')
    proxy = audit['oneshot_proxy']
    check(proxy is not None and proxy['parent_k'] == k and proxy['proxy_train_k'] == 1
          and proxy['trial_count'] == len(proxy['trials']) == k, 'Incomplete one-shot anchors')
    groups = [('support_oof', f['fold'], None, f) for f in audit['folds']]
    groups += [('support_oneshot_proxy', None, t['trial'], t) for t in proxy['trials']]
    stages, steps = [], []
    for scope, fold, trial, group in groups:
        train, held = group['training_ids'], group['held_ids']
        check(len(set(train)) == len(train) and len(set(held)) == len(held)
              and not set(train)&set(held) and set(train)|set(held) == support,
              'Physical train/held boundary mismatch')
        check(len(train) == classes*group['train_k'], 'Training physical count mismatch')
        if trial is not None:
            check(group['train_k'] == 1 and len(held) == classes*(k-1)
                  and set(group['oof']) == set(ARMS), 'Invalid one-shot proxy scope')
        check(len(group['stages']) == len(ARMS)
              and {stage['arm'] for stage in group['stages']} == set(ARMS), 'Incomplete stage arms')
        for stage in group['stages']:
            tagged = dict(stage, scope=scope, fold=fold, trial=trial)
            recorded = stage.get('steps', [])
            count = stage['optimizer_steps']
            check(type(count) is int and count >= 0, 'Invalid optimizer steps')
            if stage['arm'].endswith('_ce'):
                check(len(recorded) == count+1 and [s['iteration'] for s in recorded] == list(range(count+1)),
                      'CE trace must include initial iteration zero and every update')
            else:
                check(count == 0 and recorded == [], 'Ridge has no optimizer updates')
            for step in recorded:
                check(isinstance(step,dict), 'Invalid optimizer step')
                steps.append(dict(step,scope=scope,fold=fold,trial=trial,arm=stage['arm'],parent_k=k))
            stages.append(tagged)
    check(sum(s['optimizer_steps'] for s in stages) == audit['optimizer_steps']
          and sum(s['factorization_calls'] for s in stages) == audit['factorization_count'],
          'Core operation totals mismatch')
    return stages, steps


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    check(set(config) == {'algorithm', 'matrix'} and config['algorithm'] == FROZEN_CONFIG,
          'Frozen BranchOrbitCE configuration mismatch')
    started = time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule,
        expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_seed=expected_model_seed,
        config=config)
    load_seconds = time.perf_counter()-started
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(existing_model_file_bytes=producer['model_file_bytes'],
        model_file_bytes_scope='Full training checkpoint package; not loaded by CPU probe; frozen model loaded by support exporter',
        support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'],
        cache_load_seconds=load_seconds, frozen_support_cache_loaded=True,
        feature_extraction_timing=producer['timing'],
        native_view_forward_count=producer['native_view_forward_count'],
        support_physical_observation_count=producer['support_physical_observation_count'],
        native_total_physical_forward_count=producer['native_total_physical_forward_count'],
        smoke_forward_count=producer['smoke_forward_count'],
        new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_already_deployed=None, model_incremental_transfer_bytes=None,
        model_deployment_unknown_reason='Deployment state unknown; probe only reads exported features', checkpoint_loaded=False)
    startup = dict(scope=SCOPE, argv=sys.argv, python=sys.executable, pid=os.getpid(),
        config=config, capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256, model_seed=expected_model_seed,
        provenance=provenance, support_features=str(support_features),
        capsule_manifest=str(Path(capsule)/'manifest.json'), episodes=len(tasks),
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False,
        adapted_state_inherited=False, cross_row_adapted_state_reuse=False,
        checkpoint_loaded=False, encoder_updated=False, payload_audit=payload,
        source_validation=None, source_validation_reason='No source samples or source feature banks accessed',
        learning_rate=None, epoch=None, optimizer_steps=None, device='cpu',
        unavailable_reason='No training has run at startup; actual CE iterations and rates are logged per stage',
        epoch_reason='Deterministic full-batch support optimizer; iteration records, not epochs',
        text_step_interval=TEXT_STEP_INTERVAL,
        stdout_step_policy='CE iteration 0, each multiple of 25, and final iteration; all stages; structured records retain every iteration',
        blas_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    write(out/'startup.json', startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    fields = ['split_id','receiver','scenario','k','new_count','support_seed','completed','total',
        'scope','classes','support_count','fold_count','numerical','oof','paired','fit_seconds',
        'factorization_count','standard_factorization_count','proxy_anchor_count','oneshot_proxy','optimizer_steps','persistent_state_bytes','heldout_unavailable_reason',
        'fit_call_seconds','log_write_seconds','total_seconds','peak_process_rss_bytes',
        'query_rows_used','source_rows_used']
    factorizations = standard_factorizations = proxy_anchors = k1_count = optimizer_steps = 0
    with (out/'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
         (out/'compact.jsonl').open('x', encoding='utf-8') as compact, \
         (out/'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
         (out/'fit_stages.jsonl').open('x', encoding='utf-8') as stages, \
         (out/'fit_stages.csv').open('x', encoding='utf-8', newline='') as stage_csv, \
         (out/'training_steps.jsonl').open('x', encoding='utf-8') as step_stream, \
         (out/'training_steps.csv').open('x', encoding='utf-8', newline='') as step_csv:
        writer = csv.DictWriter(csvfile, fieldnames=fields); writer.writeheader()
        stage_writer = step_writer = None
        for number, (split, positions, labels) in enumerate(tasks, 1):
            begin = time.perf_counter()
            try:
                audit = probe_branch_orbit_ce(**{k: v[positions] for k,v in arrays.items()},
                    support_labels=labels, support_ids=split['support_ids'],
                    classes=split['registered_classes'], old_classes=old)
            except Exception as exc:
                write(out/'technical_failure.json', dict(status='TECHNICAL_FAILURE',
                    split_id=split['split_id'], completed_episodes=number-1,
                    error_type=type(exc).__name__, error=str(exc), core_audit=getattr(exc,'audit',None),
                    query_rows_used=0, source_rows_used=0))
                raise
            elapsed = time.perf_counter()-begin
            k = split['k']; folds = 0 if k == 1 else min(k, 3)
            check(audit['k'] == k and audit['support_count'] == len(positions)
                  and set(audit['classes']) == set(split['registered_classes'])
                  and audit['fold_count'] == folds and len(audit['folds']) == folds
                  and audit['factorization_count'] >= 0 and audit['optimizer_steps'] >= 0
                  and audit['persistent_state_bytes'] == 0, 'BranchOrbitCE core audit mismatch')
            if k == 1:
                check(audit['oof'] is None and audit['paired'] is None and audit['oneshot_proxy'] is None, 'K1 fabricated holdout')
                k1_count += 1
            episode_stages, episode_steps = validate_audit(audit, split)
            optimizer_steps += audit['optimizer_steps']
            audit = compact_proxy(audit, split['split_id'])
            factorizations += audit['factorization_count']
            standard_factorizations += audit['standard_factorization_count']
            proxy_anchors += audit['proxy_anchor_count']
            coords = {key: split[key] for key in ('split_id','receiver','scenario','k','support_seed')}
            coords['new_count'] = len(split['registered_classes'])-len(old)
            record = dict(audit, **{key: value for key,value in coords.items() if key != 'k'},
                scope=SCOPE, registered_classes=split['registered_classes'],
                fit_call_seconds=elapsed, query_rows_used=0, source_rows_used=0)
            writing = time.perf_counter()
            trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush()
            all_stages = episode_stages
            if all_stages and stage_writer is None:
                columns = sorted({key for stage in all_stages for key in scalars(stage)})+['split_id']
                stage_writer = csv.DictWriter(stage_csv, fieldnames=columns); stage_writer.writeheader()
            for stage in all_stages:
                row = dict(scalars(stage), split_id=split['split_id'])
                stages.write(json.dumps(row, allow_nan=False)+'\n')
                stage_writer.writerow(csv_record(row))
                print(json.dumps(dict(event='SUPPORT_FIT_STAGE', **row), allow_nan=False), flush=True)
            if episode_steps and step_writer is None:
                columns = sorted({key for row in episode_steps for key in row})+['split_id']
                step_writer = csv.DictWriter(step_csv, fieldnames=columns); step_writer.writeheader()
            final_iterations = {(s['scope'],s['fold'],s['trial'],s['arm']): s['optimizer_steps'] for s in all_stages}
            for step in episode_steps:
                row = dict(step, split_id=split['split_id'])
                step_stream.write(json.dumps(row, allow_nan=False)+'\n')
                step_writer.writerow(csv_record(row))
                final = final_iterations[(row['scope'],row['fold'],row['trial'],row['arm'])]
                if row['iteration'] % TEXT_STEP_INTERVAL == 0 or row['iteration'] == final:
                    print(json.dumps(dict(event='SUPPORT_OPTIMIZER_STEP', **row), allow_nan=False), flush=True)
            stages.flush(); stage_csv.flush(); step_stream.flush(); step_csv.flush()
            small = dict(coords, completed=number, total=len(tasks), scope=SCOPE, classes=len(audit['classes']),
                **{key: scalars(audit[key]) for key in ('support_count','fold_count','numerical','oof','paired',
                   'fit_seconds','factorization_count','standard_factorization_count','proxy_anchor_count','oneshot_proxy','optimizer_steps','persistent_state_bytes','heldout_unavailable_reason')},
                fit_call_seconds=elapsed, log_write_seconds=time.perf_counter()-writing,
                total_seconds=time.perf_counter()-begin, peak_process_rss_bytes=peak_rss(),
                query_rows_used=0, source_rows_used=0)
            compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
            writer.writerow(csv_record(small)); csvfile.flush()
            print(json.dumps(dict(event='SUPPORT_EPISODE_COMPLETE', **small), allow_nan=False), flush=True)
    marker = dict(status='SUPPORT_PROBE_COMPLETE', scope=SCOPE, capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256, model_seed=expected_model_seed,
        episodes=len(tasks), k1_episodes=k1_count, oof_episodes=len(tasks)-k1_count,
        algorithm=FROZEN_CONFIG, matrix=config['matrix'], query_rows_used=0, source_rows_used=0,
        truth_read=False, factorization_count=factorizations, standard_factorization_count=standard_factorizations,
        proxy_anchor_count=proxy_anchors, optimizer_steps=optimizer_steps,
        persistent_state_bytes=0, payload_audit=payload, wall_seconds=time.perf_counter()-started,
        peak_process_rss_bytes=peak_rss(), peak_process_rss_reason='Linux process high-water RSS; null on unsupported platform')
    write(out/'probe_complete.json', marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features','capsule','output','config','expected-capsule-id','expected-checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config'])
    evaluate(**args)


if __name__ == '__main__': main()
