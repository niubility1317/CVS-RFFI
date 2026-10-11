"""Clean-only frozen checkpoint prediction and independent truth-last scoring."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import design as d

COUNT = 168000
VIEWS = ('clean',)
HELDOUT_RX = ('1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8')
TARGET_DAYS = ('2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23')


def snapshot(row=None):
    rows = [r for r in d.rows() if row is None or r['row_id'] == row]
    if not rows:
        raise ValueError('Unregistered evaluation row')
    configs = [d.config(r) for r in rows]
    for c in configs:
        d.validate(c)
        frozen = d.read(d.BASE / c['row_id'] / 'source_frozen.json')
        if (frozen['status'] != 'SOURCE_FROZEN' or frozen['config'] != c
                or frozen['target_access']):
            raise ValueError('Source freeze differs from preregistered row')
    return configs


def source_provenance(c):
    from .common import verify_parent
    verify_parent(c)
    source = Path(c['output_root'])
    done = d.read(source / 'completion.json')
    if (done['status'] != 'SOURCE_TRAINED' or done['config'] != c
            or done['epoch'] != 40 or done['optimizer_steps'] != 8880
            or done['logged_steps'] != 8880 or done['target_access']):
        raise ValueError('Second-stage checkpoint source provenance differs')
    return source


def checkpoint_provenance(c, checkpoint):
    expected = dict(epoch=40, run_id=c['run_id'], candidate_id=c['row_id'],
                    config=c, parent_checkpoint=c['parent_checkpoint'],
                    checkpoint_selection='final_only')
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError('Second-stage checkpoint identity, parent, or budget differs')
    if 'model' not in checkpoint or 'baseline_args' not in checkpoint:
        raise ValueError('Incomplete identity checkpoint')


def physical_ids():
    import numpy as np
    # Reuse the capsule's already validated observations without generating views.
    manifest = d.read(Path(d.CAPSULE) / 'manifest.json')
    expected = dict(status='VALIDATED_ONCE', count=COUNT, classes=d.CLASSES,
                    source_target_disjoint=True, unique_ids=True,
                    channel='residual/post_sync/noeq')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Validated clean observation contract differs')
    with np.load(Path(d.CAPSULE) / 'index.npz', allow_pickle=False) as ix:
        ids = ix['ids'].copy()
    if ids.shape != (COUNT,) or len(set(ids.tolist())) != COUNT:
        raise ValueError('Clean physical ID coverage differs')
    return ids


def predict(row):
    import numpy as np
    import torch
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    from .common import build_model
    c = snapshot(row)[0]
    source = source_provenance(c)
    checkpoint_path = source / 'final_ssdg.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_provenance(c, checkpoint)
    del checkpoint
    output = d.BASE / row / 'prediction'
    if output.exists():
        raise FileExistsError('Existing prediction must be reconciled, never overwritten')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    started = time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        model = build_model(c, device, checkpoint=checkpoint_path)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 2, 256, device=device))
        if logits.shape != (2, 6) or not torch.isfinite(logits).all():
            raise ValueError('Identity checkpoint no-query forward failed')
        ids = physical_ids()
        array = np.load(Path(d.CAPSULE) / 'clean.npy', mmap_mode='r', allow_pickle=False)
        if array.shape != (COUNT, 2, 256):
            raise ValueError('Complete clean observation shape differs')
        output.mkdir(parents=True, exist_ok=False)
        d.write(output / 'provenance.json', dict(status='VERIFIED', config=c,
            checkpoint=str(checkpoint_path), parent_checkpoint=c['parent_checkpoint'],
            own_E40=True, optimizer_steps=8880, source_roles='EXACT_MATCH',
            truth_read=False, query_fit=False))
        d.write(output / 'resolved_config.json', dict(row_id=row, run_id=d.RUN,
            pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
            evaluation_commit=(d.ROOT / 'release_commit.txt').read_text().strip(),
            views=['clean'], batch_size=256, backend_flags=actual_flags(),
            parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
            hardware=torch.cuda.get_device_name(device), torch_version=torch.__version__))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        infer_started = time.perf_counter()
        predictions = []
        with torch.no_grad():
            for begin in range(0, COUNT, 256):
                values = np.array(array[begin:begin + 256], dtype=np.float32, copy=True)
                logits = model(torch.from_numpy(values).to(device))
                if logits.shape != (len(values), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite or malformed clean classifier output')
                predictions.extend(logits.argmax(1).cpu().tolist())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - infer_started
        np.savez(output / 'predictions.npz', ids=ids, clean=np.asarray(predictions, dtype=np.int64))
        import resource
        d.write(output / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=COUNT,
            views=['clean'], truth_read=False, query_fit=False, inference_seconds=dict(clean=seconds),
            wall_seconds=time.perf_counter() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
        print(json.dumps(dict(row_id=row, status='PREDICTIONS_COMPLETE', count=COUNT,
                             views=['clean'], truth_read=False)), flush=True)


def prediction_preflight(configs):
    import numpy as np
    ids = physical_ids()
    for c in configs:
        output = d.BASE / c['row_id'] / 'prediction'
        done = d.read(output / 'complete.json')
        if (done['status'] != 'PREDICTIONS_COMPLETE' or done['count'] != COUNT
                or done['views'] != ['clean'] or done['truth_read'] or done['query_fit']):
            raise ValueError('Truth remains closed: incomplete clean predictions')
        provenance = d.read(output / 'provenance.json')
        if provenance['config'] != c or provenance['parent_checkpoint'] != c['parent_checkpoint']:
            raise ValueError('Prediction provenance differs')
        with np.load(output / 'predictions.npz', allow_pickle=False) as p:
            if set(p.files) != {'ids', 'clean'} or not np.array_equal(ids, p['ids']):
                raise ValueError('Prediction physical identity differs')
            validate_prediction(p['clean'])
    return ids


def validate_prediction(a):
    if a.shape != (COUNT,) or a.dtype.kind not in 'iu' or a.min() < 0 or a.max() >= 6:
        raise ValueError('Invalid full clean prediction array')


def truth_arrays(truth, ids):
    import numpy as np
    if set(truth) != set(ids.tolist()):
        raise ValueError('Truth physical identities differ')
    y = np.asarray([truth[s]['label'] for s in ids])
    rx = np.asarray([str(truth[s]['receiver']) for s in ids])
    days = np.asarray([str(truth[s]['day']) for s in ids])
    if (y.dtype.kind not in 'iu' or set(y) != set(range(6))
            or set(rx) != set(HELDOUT_RX) or set(days) != set(TARGET_DAYS)):
        raise ValueError('Truth class/receiver/day contract differs')
    for label in range(6):
        for receiver in HELDOUT_RX:
            for day in TARGET_DAYS:
                if int(((y == label) & (rx == receiver) & (days == day)).sum()) != 1000:
                    raise ValueError('Truth registered per-condition count differs')
    return y, rx, days


def metric_rows(config, prediction, ids, y, rx, days):
    import numpy as np
    from comparison_suite.score import metrics
    records, day_records = [], []
    identity = {k: config[k] for k in ('row_id', 'stage', 'arm', 'model_seed')}
    for dimension, strata in [('overall', ['ALL']), ('receiver', sorted(set(rx))),
                              ('transmitter', d.CLASSES), ('day', list(TARGET_DAYS))]:
        for stratum in strata:
            mask = (np.ones(len(ids), bool) if dimension == 'overall' else rx == stratum
                    if dimension == 'receiver' else days == stratum if dimension == 'day'
                    else y == d.CLASSES.index(stratum))
            result = metrics(y[mask], prediction[mask], 6)
            cm = np.bincount(6 * y[mask] + prediction[mask], minlength=36).reshape(6, 6)
            denominator = cm.sum(0) + cm.sum(1)
            f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(6), where=denominator > 0).mean()
            if (result['confusion'] != cm.tolist() or result['query_count'] != int(cm.sum())
                    or abs(result['accuracy'] - float(np.trace(cm) / cm.sum())) > 1e-12
                    or abs(result['macro_f1'] - f1) > 1e-12):
                raise ValueError('Independent metric recount differs')
            (day_records if dimension == 'day' else records).append(
                dict(**identity, view='clean', dimension=dimension, stratum=stratum, **result))
    return records, day_records


def summarize(results):
    import numpy as np
    output = []
    for arm in ('P0', *d.ARMS):
        group = [r for r in results if r['arm'] == arm and r['dimension'] == 'overall']
        if not group:
            continue
        values = [r['accuracy'] for r in group]
        worst = [min(r['accuracy'] for r in results if r['row_id'] == g['row_id']
                     and r['dimension'] == 'receiver') for g in group]
        output.append(dict(arm=arm, view='clean', seed_count=len(group),
            seeds=[r['model_seed'] for r in group],
            seed_accuracy={str(r['model_seed']): r['accuracy'] for r in group},
            accuracy_mean=float(np.mean(values)), accuracy_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            accuracy_min=min(values), macro_f1_mean=float(np.mean([r['macro_f1'] for r in group])),
            worst_rx_mean=float(np.mean(worst)), complete_three_seeds={r['model_seed'] for r in group} == set(d.SEEDS)))
    return output


def paired_results(results):
    import numpy as np
    overall = {(r['arm'], r['model_seed']): r for r in results if r['dimension'] == 'overall'}
    result = []
    for treatment, control in [('C0', 'P0'), ('C1', 'C0'), ('C2', 'C1'), ('C3', 'C2'), ('C3', 'C0'), ('C3', 'P0')]:
        seeds = [s for s in d.SEEDS if (treatment, s) in overall and (control, s) in overall]
        if not seeds:
            continue
        differences = [overall[treatment, s]['accuracy'] - overall[control, s]['accuracy'] for s in seeds]
        result.append(dict(treatment=treatment, control=control, metric='accuracy', view='clean',
            seeds=seeds, seed_count=len(seeds), differences=differences,
            mean_difference=float(np.mean(differences)),
            sd_difference=float(np.std(differences, ddof=1)) if len(seeds) > 1 else None,
            positive_seeds=sum(v > 0 for v in differences), complete_three_seeds=len(seeds) == 3))
    return result


def source_resources(c):
    source = Path(c['output_root'])
    done = d.read(source / 'completion.json')
    resolved = d.read(source / 'resolved_config.json')
    prepared = d.BASE / 'prepared' / ('s' + str(c['model_seed']))
    return dict(parent_checkpoint=c['parent_checkpoint'], parent_epochs=200,
        parent_identity_updates=44400, incremental_epochs=40, incremental_identity_updates=8880,
        cumulative_identity_updates=53280, elapsed_seconds=done.get('elapsed_seconds'),
        hardware=resolved.get('hardware'), parameters=resolved.get('total_parameters'),
        preparation=d.read(prepared / 'preparation_complete.json'),
        parent_cost_ref=str(Path(c['parent_checkpoint']).parent / 'completion.json'),
        measured_training_resources=done.get('resources'),
        training_flops=None, scope='Shared-GPU wall measurements; no inferred speedup')


def score(row=None):
    import numpy as np
    from comparison_suite.score import csvwrite
    configs = snapshot(row)
    # This process alone opens truth, only after all requested-row predictions are fixed.
    ids = prediction_preflight(configs)
    truth = d.read(d.TRUTH)
    y, rx, days = truth_arrays(truth, ids)
    results, day_results, resources, parent_rows = [], [], [], []
    seen_parent = set()
    for c in configs:
        out = d.BASE / c['row_id'] / 'prediction'
        with np.load(out / 'predictions.npz', allow_pickle=False) as p:
            records, day_records = metric_rows(c, p['clean'], ids, y, rx, days)
        results.extend(records)
        day_results.extend(day_records)
        resources.append(dict(row_id=c['row_id'], prediction=d.read(out / 'complete.json'), source=source_resources(c)))
        parent_path = c.get('parent_prediction')
        if parent_path and c['model_seed'] not in seen_parent:
            parent_path = Path(parent_path)
            parent_done = d.read(parent_path.parent / 'complete.json')
            parent_provenance = d.read(parent_path.parent / 'provenance.json')
            if (parent_provenance['status'] != 'VERIFIED'
                    or parent_provenance['checkpoint'] != c['parent_checkpoint']
                    or parent_provenance['config']['model_seed'] != c['model_seed']
                    or parent_provenance['config']['arm'] != 'native'
                    or parent_provenance['truth_read'] or parent_provenance['query_fit']):
                raise ValueError('Parent clean prediction belongs to a different checkpoint')
            if (parent_done['status'] != 'PREDICTIONS_COMPLETE' or parent_done['count'] != COUNT
                    or 'clean' not in parent_done['views'] or parent_done['truth_read'] or parent_done['query_fit']):
                raise ValueError('Existing parent predictions incomplete')
            with np.load(parent_path, allow_pickle=False) as p:
                if not np.array_equal(p['ids'], ids):
                    raise ValueError('Parent clean physical identities differ')
                validate_prediction(p['clean'])
                parent = dict(c, arm='P0', row_id='P0-s' + str(c['model_seed']))
                parent_rows.extend(metric_rows(parent, p['clean'], ids, y, rx, days)[0])
            seen_parent.add(c['model_seed'])
    all_results = results + parent_rows
    root = d.BASE / 'row_scoring' / row if row else d.BASE
    summary, pairs = summarize(all_results), paired_results(all_results)
    d.write(root / 'scores.json', dict(status='SCORED', results=results, target_feedback_forbidden=True))
    csvwrite(root / 'scores.csv', [{k: v for k, v in r.items() if k != 'confusion'} for r in results])
    d.write(root / 'day_scores.json', dict(results=day_results))
    csvwrite(root / 'day_scores.csv', [{k: v for k, v in r.items() if k != 'confusion'} for r in day_results])
    d.write(root / 'parent_scores.json', dict(results=parent_rows, source='Reused fixed clean predictions only'))
    d.write(root / 'summary.json', dict(results=summary, aggregation='ACTUAL_COMPLETED_SEEDS_ONLY'))
    csvwrite(root / 'summary.csv', summary)
    d.write(root / 'paired_results.json', dict(comparisons=pairs, inferential_significance_claim=False))
    d.write(root / 'resources.json', dict(rows=resources, shared_preparation_counted_once_per_seed=True))
    main = next((r for r in summary if r['arm'] == 'C3'), None)
    increment = next((r for r in pairs if r['treatment'] == 'C3' and r['control'] == 'C0'), None)
    d.write(root / 'goal_assessment.json', dict(primary_arm='C3', threshold=.84,
        performance_status='PARTIAL' if main is None or not main['complete_three_seeds'] else 'MET' if main['accuracy_mean'] >= .84 else 'NOT_MET',
        mechanism_increment_status='PARTIAL' if increment is None or not increment['complete_three_seeds'] else 'POSITIVE' if increment['mean_difference'] > 0 else 'NOT_POSITIVE',
        each_seed_must_be_reported=True, scores_must_not_control_training=True))
    lines = ['# Fixed-parent 40-epoch multi-disentanglement: clean test', '',
        'Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.',
        'P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.', '',
        '| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        sd = 'N/A' if r['accuracy_sd'] is None else f"{100*r['accuracy_sd']:.3f}"
        lines.append(f"| {r['arm']} | {r['seed_count']} | {100*r['accuracy_mean']:.3f} | {sd} | {100*r['accuracy_min']:.3f} | {100*r['macro_f1_mean']:.3f} |")
    lines.extend(['', '## Every seed', '', '| Arm | Model seed | Accuracy (%) |', '|---|---:|---:|'])
    for r in all_results:
        if r['dimension'] == 'overall':
            lines.append(f"| {r['arm']} | {r['model_seed']} | {100*r['accuracy']:.3f} |")
    lines.extend(['', '## Paired differences', '', '| Comparison | Seeds | Mean (pp) | Positive seeds |', '|---|---:|---:|---:|'])
    for r in pairs:
        lines.append(f"| {r['treatment']} − {r['control']} | {r['seed_count']} | {100*r['mean_difference']:.3f} | {r['positive_seeds']} |")
    lines.extend(['', 'Primary success requires all three C3 seeds and clean mean ≥84%; no best-arm replacement.',
        'C0−P0 includes the changed optimizer, schedule, sampling, and additional training. C3−C0 estimates the fixed mechanism increment.',
        'Three seed differences are descriptive evidence, not an automatic statistical significance claim.',
        'Parent training, shared preparation, incremental training, and test costs are referenced separately in resources.json.'])
    root.mkdir(parents=True, exist_ok=True)
    (root / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    d.write(root / 'scoring_complete.json', dict(status='SCORED_COMPLETE', models=len(configs), views=1,
        query_count_per_view=COUNT, prediction_count=len(configs) * COUNT,
        metric_records=len(results), day_metric_records=len(day_results), truth_last=True,
        independent_recount='VERIFIED'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['predict', 'score'], required=True)
    parser.add_argument('--row')
    args = parser.parse_args()
    if args.mode == 'predict':
        if not args.row:
            parser.error('predict requires --row')
        predict(args.row)
    else:
        score(args.row)
