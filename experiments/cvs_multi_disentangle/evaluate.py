"""Fixed completed-row snapshot: evaluation only, with no source-selection feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

RUN = '20261009-phase1-multi-disentangle-manysig-m32-r01'
RELEASE = 'cvs_multi_disentangle_20261009_r01'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
ROOT = Path(__file__).resolve().parents[2]
BASE = Path(PROJECT) / 'runs' / RUN
VIEWS_ROOT = Path(PROJECT) / 'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
SCENES = ('practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
VIEWS = ('clean', *SCENES)
COUNT = 168000
MAX_ACTIVE = 4


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def design():
    from experiments.cvs_multi_disentangle import design as d
    from experiments.cvs_phase1_stack.source import require_budget
    d.require_budget = require_budget
    return d


def snapshot(d):
    value = read(BASE / 'source_matrix_frozen.json')
    expected = [d.config(r) for r in d.rows()]
    if (value['status'] != 'ALL_SOURCE_FROZEN' or value['run_id'] != RUN
            or value['rows'] != expected or value['target_access'] is not False
            or value['selection'] != 'ALL_PREREGISTERED_FIXED_CONTROLS'):
        raise ValueError('Incomplete or changed source freeze')
    return expected


def source_provenance(d, c):
    from experiments.cvs_phase1_stack.recover import verify_contract
    source = Path(c['output_root'])
    done = read(source / 'completion.json'); init = read(source / 'initialization.json')
    d.require_budget(done['logged_steps'], done['optimizer_steps'])
    if (done['status'] != 'SOURCE_TRAINED' or done['config'] != c or done['epoch'] != 200
            or done['target_access'] or done['target_evaluated'] or not init['scratch_only']
            or init['ancestors'] or init['checkpoint_sources'] or init['target_contact']
            or init['source_roles'] != 'EXACT_MATCH'):
        raise ValueError('Completed checkpoint provenance mismatch')
    verify_contract(read(source / 'source_contract.json'), read(d.SOURCE))
    execution=read(source/'mechanism_execution.json')
    if execution['status']!='VERIFIED' or execution.get('missing') or done.get('mechanism_execution')!=execution:
        raise ValueError('Required training mechanism did not execute')
    return source


def checkpoint_provenance(d, c, ck):
    from scripts.train_daot_rc4_baseline import resolved_config
    from experiments.cvs_phase1_stack.source import clean
    if (ck['epoch'] != 200 or ck['candidate_id'] != c['row_id'] or ck['run_id'] != c['run_id']
            or ck['checkpoint_selection'] != 'final_only' or ck['args']['baseline_ckpt']
            or ck['args']['teacher_ckpt'] or not ck['args']['from_scratch']):
        raise ValueError('Checkpoint actual initialization or identity differs')
    if clean(resolved_config(SimpleNamespace(**ck['args']))) != read(Path(c['output_root']) / 'resolved_native_args.json'):
        raise ValueError('Actual native checkpoint args differ')


def preflight():
    """Inspect all 32 completed source artifacts on CPU; never open a query view or truth."""
    import torch
    from experiments.cvs_multi_disentangle.runtime import installed
    d = design(); rows = snapshot(d)
    for c in rows:
        source = source_provenance(d, c)
        with installed(c, training=False):
            ck = torch.load(source / 'final_ssdg.pth', map_location='cpu', weights_only=False)
            checkpoint_provenance(d, c, ck)
            del ck
    return dict(status='VERIFIED', models=len(rows), target_read=False,
                rows=[c['row_id'] for c in rows], selection='ALL_COMPLETED_UNTESTED_AT_SNAPSHOT')


def manifest(d):
    value = read(VIEWS_ROOT / 'manifest.json')
    expected = dict(status='VALIDATED_ONCE', count=COUNT, scenes=list(SCENES), classes=d.CLASSES,
                    truth_read=False, channel='residual/post_sync/noeq', source_capsule=d.CAPSULE,
                    clean_ref=d.CAPSULE + '/clean.npy', rows_share_observations=True)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Existing complete sixscene views differ')
    return value


def physical_ids(d):
    import numpy as np
    with np.load(VIEWS_ROOT / 'index.npz', allow_pickle=False) as ix:
        ids = ix['ids'].copy()
    with np.load(Path(d.CAPSULE) / 'index.npz', allow_pickle=False) as ix:
        if len(ids) != COUNT or len(set(ids.tolist())) != COUNT or not np.array_equal(ids, ix['ids']):
            raise ValueError('Complete sixscene physical IDs differ')
    return ids


def predict(rid):
    import numpy as np
    import torch
    from experiments.cvs_multi_disentangle.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    d = design(); rows = snapshot(d)
    matching = [c for c in rows if c['row_id'] == rid]
    if len(matching) != 1:
        raise ValueError('Row is not in fixed completed snapshot')
    c = matching[0]; source = source_provenance(d, c)
    torch.set_num_threads(2); device = torch.device('cuda:0'); started = time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY), installed(c, training=False) as native:
        ck = torch.load(source / 'final_ssdg.pth', map_location=device, weights_only=False)
        checkpoint_provenance(d, c, ck)
        model = native.build_baseline_model(SimpleNamespace(**ck['baseline_args']), device)
        model.load_state_dict(ck['model'], strict=True); model.eval()
        del ck
        with torch.no_grad():
            z = model(torch.zeros(2, 2, 256, device=device))
        if z.shape != (2, 6) or not torch.isfinite(z).all():
            raise ValueError('Actual checkpoint no-query smoke failed')
        m = manifest(d); ids = physical_ids(d)
        out = BASE / rid / 'prediction'; out.mkdir(parents=True, exist_ok=False)
        commit = (ROOT / 'release_commit.txt').read_text().strip()
        write(out / 'provenance.json', dict(status='VERIFIED', config=c,
              checkpoint=str(source / 'final_ssdg.pth'), scratch_sources=[], source_roles='EXACT_MATCH',
              own_E200=True, optimizer_steps=44400, query_fit=False, truth_read=False))
        write(out / 'resolved_config.json', dict(row_id=rid, run_id=RUN, pid=os.getpid(), cwd=os.getcwd(),
              python=sys.executable, evaluation_commit=commit, source_run=c['run_id'], views=list(VIEWS),
              views_root=str(VIEWS_ROOT), backend_flags=actual_flags(), batch_size=256,
              parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
              hardware=torch.cuda.get_device_name(device), torch_version=torch.__version__))
        predictions = {}; timings = {}; torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for view in VIEWS:
                array = np.load(m['clean_ref'] if view == 'clean' else VIEWS_ROOT / (view + '.npy'),
                                mmap_mode='r', allow_pickle=False)
                if array.shape != (COUNT, 2, 256):
                    raise ValueError('Full view shape differs')
                torch.cuda.synchronize(); start = time.perf_counter(); values = []
                for i in range(0, len(ids), 256):
                    a = np.ascontiguousarray(array[i:i + 256], dtype=np.float32)
                    x = torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape).to(device)
                    logits = model(x)
                    if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
                        raise ValueError('Invalid classifier output')
                    values.extend(logits.argmax(1).cpu().tolist())
                torch.cuda.synchronize(); timings[view] = time.perf_counter() - start
                predictions[view] = np.asarray(values, dtype=np.int64)
                print(json.dumps(dict(row_id=rid, view=view, count=len(values), seconds=timings[view], truth_read=False)), flush=True)
        np.savez(out / 'predictions.npz', ids=ids, **predictions)
        import resource
        write(out / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids), views=list(VIEWS),
              truth_read=False, query_fit=False, inference_seconds=timings, wall_seconds=time.perf_counter() - started,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
              peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              memory_measurement='Linux process high-water RSS; CUDA peaks after checkpoint smoke'))


def prediction_preflight(d, rows):
    import numpy as np
    manifest(d); ids = physical_ids(d)
    for c in rows:
        out = BASE / c['row_id'] / 'prediction'; done = read(out / 'complete.json')
        if (done['status'] != 'PREDICTIONS_COMPLETE' or done['count'] != COUNT
                or done['views'] != list(VIEWS) or done['truth_read'] or done['query_fit']):
            raise ValueError('Truth remains closed: incomplete predictions')
        if read(out / 'provenance.json')['config'] != c:
            raise ValueError('Prediction provenance differs from fixed snapshot')
        with np.load(out / 'predictions.npz', allow_pickle=False) as p:
            if set(p.files) != set(VIEWS) | {'ids'} or not np.array_equal(ids, p['ids']):
                raise ValueError('Prediction identity or view coverage mismatch')
            for view in VIEWS:
                a = p[view]
                if a.shape != (COUNT,) or a.dtype.kind not in 'iu' or a.min() < 0 or a.max() >= 6:
                    raise ValueError('Invalid prediction array')
    return ids


def summarize(results):
    import numpy as np
    groups = {}
    for r in results:
        if r['dimension'] == 'overall':
            groups.setdefault((r['stage'], r['arm'], r['view']), []).append(r)
    summary = []
    for (stage, arm, view), group in groups.items():
        item = dict(stage=stage, arm=arm, view=view, seed_count=len(group),
                    seeds=[r['model_seed'] for r in group], row_ids=[r['row_id'] for r in group])
        for metric in ['accuracy', 'macro_f1']:
            values = [r[metric] for r in group]
            item[metric + '_mean'] = float(np.mean(values))
            item[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        worst = [min(r['accuracy'] for r in results if r['row_id'] == row['row_id']
                     and r['view'] == view and r['dimension'] == 'receiver') for row in group]
        item.update(worst_rx_mean=float(np.mean(worst)),
                    worst_rx_sd=float(np.std(worst, ddof=1)) if len(worst) > 1 else None)
        summary.append(item)
    return summary


def score():
    import numpy as np
    from comparison_suite.score import metrics, csvwrite
    d = design(); rows = snapshot(d)
    ids = prediction_preflight(d, rows)  # All 32 x 7 arrays are fixed before opening truth.
    truth = read(d.TRUTH)
    y,rx,days = truth_arrays(truth,ids)
    results = []; day_results = []; resources = []
    for c in rows:
        row = {k: c[k] for k in ['row_id', 'stage', 'arm', 'model_seed']}
        out = BASE / c['row_id'] / 'prediction'
        resources.append(dict(**row, prediction=read(out / 'complete.json'), source=source_resources(c)))
        with np.load(out / 'predictions.npz', allow_pickle=False) as p:
            for view in VIEWS:
                for dimension, strata in [('overall', ['ALL']), ('receiver', sorted(set(rx))), ('transmitter', d.CLASSES), ('day', list(TARGET_DAYS))]:
                    for stratum in strata:
                        mask = np.ones(len(ids), bool) if dimension == 'overall' else rx == stratum if dimension == 'receiver' else days == stratum if dimension == 'day' else y == d.CLASSES.index(stratum)
                        r = metrics(y[mask], p[view][mask], 6)
                        cm = np.bincount(6 * y[mask] + p[view][mask], minlength=36).reshape(6, 6)
                        den = cm.sum(0) + cm.sum(1)
                        f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(6), where=den > 0).mean()
                        if (r['confusion'] != cm.tolist() or r['query_count'] != int(cm.sum())
                                or abs(r['accuracy'] - float(np.trace(cm) / cm.sum())) > 1e-12
                                or abs(r['macro_f1'] - f1) > 1e-12):
                            raise ValueError('Independent metric recount differs')
                        (day_results if dimension=='day' else results).append(dict(**row, view=view, dimension=dimension, stratum=stratum, **r))
    write(BASE / 'scores.json', dict(status='SCORED', results=results, target_feedback_forbidden=True))
    csvwrite(BASE / 'scores.csv', [{k: v for k, v in r.items() if k != 'confusion'} for r in results])
    write(BASE/'day_scores.json',dict(results=day_results))
    csvwrite(BASE/'day_scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in day_results])
    summary = summarize(results)
    write(BASE / 'summary.json', dict(results=summary, aggregation='ACTUAL_COMPLETED_SEEDS_ONLY'))
    csvwrite(BASE / 'summary.csv', summary)
    pairs = paired_results(results)
    write(BASE / 'paired_results.json', dict(comparisons=pairs,
          interpretation='Within-model-seed differences, fixed identical physical queries; four seeds, descriptive evidence only'))
    write(BASE / 'resources.json', dict(rows=resources, measurement='Observed per-process wall times and CUDA peaks under shared GPU load; no isolated speed claim'))
    for c in rows:
        out=BASE/c['row_id']
        rs=[r for r in results if r['row_id']==c['row_id']];ds=[r for r in day_results if r['row_id']==c['row_id']]
        write(out/'scores.json',dict(results=rs));write(out/'day_scores.json',dict(results=ds))
        csvwrite(out/'scores.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rs])
        write(out/'resources.json',next(r for r in resources if r['row_id']==c['row_id']))
    lines = ['# Multi disentanglement: fixed legacy-gating cosine comparison: clean and six full views', '',
             'All 32 controls were preregistered and trained from scratch. Every view contains 168000 physical queries.',
             'All predictions preceded truth access. The source selection controller does not consume these results.',
             'Each arm has four paired model seeds. Fixed E200 checkpoints; no target feedback.', '',
             '| Stage | Arm | View | Seeds | Accuracy (%) | SD (pp) | Macro-F1 (%) | SD (pp) |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        sd = lambda key: 'N/A' if r[key] is None else f'{100*r[key]:.3f}'
        lines.append(f'| {r["stage"]} | {r["arm"]} | {r["view"]} | {r["seed_count"]} | {100*r["accuracy_mean"]:.3f} | {sd("accuracy_sd")} | {100*r["macro_f1_mean"]:.3f} | {sd("macro_f1_sd")} |')
    lines += ['', '## Paired accuracy differences', '',
              '| Comparison | View | Mean (pp) | SD (pp) | Positive seeds |',
              '|---|---|---:|---:|---:|']
    for r in pairs:
        if r['metric'] == 'accuracy':
            lines.append(f"| {r['treatment']} − {r['control']} | {r['view']} | {100*r['mean_difference']:.3f} | {100*r['sd_difference']:.3f} | {r['positive_seeds']}/4 |")
    lines += ['', 'Per-row/view/RX/TX confusion matrices and metrics: scores.json/csv. Resources and timings: resources.json.',
              'Phase2, K, adaptation and new-class metrics: N/A. This is an exposed benchmark; no claim of a new blind test.']
    (BASE / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(BASE / 'scoring_complete.json', dict(status='SCORED_COMPLETE', models=len(rows), views=7,
          query_count_per_view=COUNT, prediction_count=len(rows) * 7 * COUNT,
          metric_records=len(results), day_metric_records=len(day_results), truth_last=True, independent_recount='VERIFIED'))


def paired_results(results):
    import statistics
    from experiments.cvs_multi_disentangle.design import SEEDS
    from experiments.cvs_multi_disentangle.design import ARMS
    contrasts = [(a,'legacy_cosine') for a in ARMS if a!='legacy_cosine'] + [('multi','fixed_interventions'),('multi','unified'),('multi','linear'),('multi','temporal'),('multi','receiver'),('multi_interaction','multi')]
    lookup = {(r['arm'],r['model_seed'],r['view']):r for r in results if r['dimension']=='overall'}
    output = []
    for a,b in contrasts:
        for view in (*VIEWS,'six_view_mean'):
            for metric in ('accuracy','macro_f1'):
                differences = [statistics.mean(lookup[a,s,v][metric]-lookup[b,s,v][metric] for v in SCENES) if view=='six_view_mean' else lookup[a,s,view][metric]-lookup[b,s,view][metric] for s in SEEDS]
                output.append(dict(treatment=a,control=b,view=view,metric=metric,seeds=list(SEEDS),
                    differences=differences,mean_difference=statistics.mean(differences),
                    sd_difference=statistics.stdev(differences),positive_seeds=sum(x>0 for x in differences)))
    return output



HELDOUT_RX={0:'1-1',2:'14-7',5:'2-1',7:'20-1',9:'7-14',10:'7-7',11:'8-8'}
TARGET_DAYS=('2021_03_01','2021_03_08','2021_03_15','2021_03_23')


def source_resources(c):
    source=Path(c['output_root']);done=read(source/'completion.json');resolved=read(source/'resolved_config.json')
    epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text().splitlines()]
    return dict(elapsed_seconds=done['elapsed_seconds'],hardware=resolved['hardware'],
        parameters=resolved['total_parameters'],last_epoch_resource_metrics={k:v for k,v in epochs[-1].items() if k.startswith('multi_')},
        auxiliary_cost=read(source/'auxiliary_cost.json'),mechanism_execution=read(source/'mechanism_execution.json'),
        peak_process_rss_bytes=None,training_flops=None,transmission_bytes=0,
        scope='Source training only; no satellite adaptation/transfer; RSS/FLOPs not measured')


def truth_arrays(truth,ids):
    import numpy as np
    if set(truth)!=set(ids.tolist()):raise ValueError('Truth physical IDs differ')
    y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    days=np.asarray([str(truth[s]['day']) for s in ids])
    if y.dtype.kind not in 'iu' or set(y)!=set(range(6)) or set(rx)!=set(HELDOUT_RX.values()) or set(days)!=set(TARGET_DAYS):
        raise ValueError('Truth label/physical RX/day coverage differs')
    if len(ids)!=COUNT:raise ValueError('Truth count differs')
    for label in range(6):
        for receiver in HELDOUT_RX.values():
            for day in TARGET_DAYS:
                if int(((y==label)&(rx==receiver)&(days==day)).sum())!=1000:
                    raise ValueError('Registered label/RX/day coverage differs')
    return y,rx,days


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['predict', 'score'], required=True)
    parser.add_argument('--row'); args = parser.parse_args()
    if args.mode == 'predict': predict(args.row)
    else: score()
