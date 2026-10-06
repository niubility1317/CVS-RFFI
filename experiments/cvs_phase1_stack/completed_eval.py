"""Fixed completed-row snapshot: evaluation only, with no source-selection feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

RUN = '20261006-phase1-reference-completed-sixscene-manysig-m32-r01'
RELEASE = 'cvs_reference_completed_sixscene_20261006_r01'
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
    sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
    from experiments.cvs_phase1_stack import dispatch as d
    if d.ROOT != ROOT:
        raise ValueError('Wrong scientific release import')
    return d


def snapshot(d):
    frozen = read(ROOT / 'frozen_rows.json')
    expected = dict(run_id=RUN, status='FROZEN_COMPLETED_ROWS',
                    selection='ALL_COMPLETED_AT_SNAPSHOT', target_feedback_forbidden=True)
    if any(frozen.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected completed-row snapshot')
    rows = frozen['rows']
    if len(rows) != 32 or len({c['row_id'] for c in rows}) != 32:
        raise ValueError('Expected exactly 32 distinct frozen rows')
    for c in rows:
        d.validate(c)
        if c['stage'] not in {'r2', 'r3'}:
            raise ValueError('Snapshot extends beyond the authorized completed stages')
        if read(Path(d.BASE) / 'configs' / ('source-' + c['row_id'] + '.json')) != c:
            raise ValueError('Original source config differs from fixed snapshot')
    return rows


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
    from experiments.cvs_phase1_stack.runtime import installed
    d = design(); rows = snapshot(d)
    for c in rows:
        source = source_provenance(d, c)
        with installed(c):
            ck = torch.load(source / 'final_ssdg.pth', map_location='cpu', weights_only=False)
            checkpoint_provenance(d, c, ck)
            del ck
    return dict(status='VERIFIED', models=len(rows), target_read=False,
                rows=[c['row_id'] for c in rows], selection='ALL_COMPLETED_AT_SNAPSHOT')


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
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    d = design(); rows = snapshot(d)
    matching = [c for c in rows if c['row_id'] == rid]
    if len(matching) != 1:
        raise ValueError('Row is not in fixed completed snapshot')
    c = matching[0]; source = source_provenance(d, c)
    torch.set_num_threads(2); device = torch.device('cuda:0'); started = time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY), installed(c) as native:
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
    y = np.asarray([truth[s]['label'] for s in ids]); rx = np.asarray([str(truth[s]['receiver']) for s in ids])
    if set(y) != set(range(6)) or len(set(rx)) != 7:
        raise ValueError('Truth population differs')
    results = []; resources = []
    for c in rows:
        row = {k: c[k] for k in ['row_id', 'stage', 'arm', 'model_seed']}
        out = BASE / c['row_id'] / 'prediction'
        resources.append(dict(**row, **read(out / 'complete.json')))
        with np.load(out / 'predictions.npz', allow_pickle=False) as p:
            for view in VIEWS:
                for dimension, strata in [('overall', ['ALL']), ('receiver', sorted(set(rx))), ('transmitter', d.CLASSES)]:
                    for stratum in strata:
                        mask = np.ones(len(ids), bool) if dimension == 'overall' else rx == stratum if dimension == 'receiver' else y == d.CLASSES.index(stratum)
                        r = metrics(y[mask], p[view][mask], 6)
                        cm = np.bincount(6 * y[mask] + p[view][mask], minlength=36).reshape(6, 6)
                        den = cm.sum(0) + cm.sum(1)
                        f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(6), where=den > 0).mean()
                        if (r['confusion'] != cm.tolist() or r['query_count'] != int(cm.sum())
                                or abs(r['accuracy'] - float(np.trace(cm) / cm.sum())) > 1e-12
                                or abs(r['macro_f1'] - f1) > 1e-12):
                            raise ValueError('Independent metric recount differs')
                        results.append(dict(**row, view=view, dimension=dimension, stratum=stratum, **r))
    write(BASE / 'scores.json', dict(status='SCORED', results=results, target_feedback_forbidden=True))
    csvwrite(BASE / 'scores.csv', [{k: v for k, v in r.items() if k != 'confusion'} for r in results])
    summary = summarize(results)
    write(BASE / 'summary.json', dict(results=summary, aggregation='ACTUAL_COMPLETED_SEEDS_ONLY'))
    csvwrite(BASE / 'summary.csv', summary)
    write(BASE / 'resources.json', dict(rows=resources, measurement='Inference only; overlapping execution with source training'))
    lines = ['# Completed reference_response models: clean and six full views', '',
             'All 32 models were fixed by completion time, not target metrics. Every view contains 168000 physical queries.',
             'All predictions preceded truth access. The source selection controller does not consume these results.',
             'Unequal seed counts describe this completion snapshot, not a complete stage comparison. Single-seed SD is N/A.', '',
             '| Stage | Arm | View | Seeds | Accuracy (%) | SD (pp) | Macro-F1 (%) | SD (pp) |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        sd = lambda key: 'N/A' if r[key] is None else f'{100*r[key]:.3f}'
        lines.append(f'| {r["stage"]} | {r["arm"]} | {r["view"]} | {r["seed_count"]} | {100*r["accuracy_mean"]:.3f} | {sd("accuracy_sd")} | {100*r["macro_f1_mean"]:.3f} | {sd("macro_f1_sd")} |')
    lines += ['', 'Per-row/view/RX/TX confusion matrices and metrics: scores.json/csv. Resources and timings: resources.json.',
              'Phase2, K, adaptation and new-class metrics: N/A. This is an exposed benchmark; no claim of a new blind test.']
    (BASE / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(BASE / 'scoring_complete.json', dict(status='SCORED_COMPLETE', models=len(rows), views=7,
          query_count_per_view=COUNT, prediction_count=len(rows) * 7 * COUNT,
          metric_records=len(results), truth_last=True, independent_recount='VERIFIED'))


def slot(active, available_gpu, occupancy):
    if len(active) >= MAX_ACTIVE:
        return None
    gpu = available_gpu(active)
    if gpu is None:
        return None
    cap = occupancy(active)[gpu]
    return gpu if len(cap['pids']) < 2 and cap['free_mb'] >= 12000 else None


def dispatch():
    d = design(); rows = snapshot(d)
    BASE.mkdir(parents=True, exist_ok=False)
    try:
        logs = Path(PROJECT) / 'logs' / RUN; logs.mkdir(parents=True, exist_ok=False)
        from experiments.cvs_phase1_stack.capacity16 import proc
        write(BASE / 'dispatcher.json', dict(**proc(os.getpid()), run_id=RUN, release=RELEASE,
              commit=(ROOT / 'release_commit.txt').read_text().strip(), max_active=MAX_ACTIVE, per_gpu_limit=2))
        write(BASE / 'frozen_rows.json', read(ROOT / 'frozen_rows.json'))
        # Publisher already performs CPU preflight. Each worker rechecks its own source and actual checkpoint.
        sys.path.insert(0, str(ROOT / 'experiments/adv3b02_xuc/code'))
        from scripts.dispatch_xuc_full import available_gpu, occupancy
        pending = list(rows); active = {}; receipts = []; failures = []
        while pending or active:
            for rid, job in list(active.items()):
                code = job['process'].poll()
                if code is not None:
                    done = BASE / rid / 'prediction' / 'complete.json'
                    if code or not done.exists() or read(done).get('status') != 'PREDICTIONS_COMPLETE':
                        failures.append(dict(row_id=rid, exit_code=code, error='Prediction worker failed or incomplete'))
                    del active[rid]
            while pending and not failures:
                gpu = slot(active, available_gpu, occupancy)
                if gpu is None:
                    break
                c = pending.pop(0); rid = c['row_id']
                cmd = [sys.executable, '-u', '-m', 'experiments.cvs_phase1_stack.completed_eval', '--mode', 'predict', '--row', rid]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'code'),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                log = logs / (rid + '.log')
                with log.open('x') as handle:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[rid] = dict(process=child, gpu=gpu)
                receipts.append(dict(row_id=rid, pid=child.pid, gpu=gpu, argv=cmd, cwd=str(ROOT), log=str(log)))
                write(BASE / 'launch.json', dict(rows=receipts))
            write(BASE / 'state.json', dict(status='PREDICTING', active=list(active), pending=[c['row_id'] for c in pending],
                  failures=failures, updated_at=time.time(), max_active=MAX_ACTIVE, per_gpu_limit=2))
            if failures and not active:
                raise RuntimeError('Prediction failures preserved; no retry ' + repr(failures))
            if pending or active:
                time.sleep(10)
        subprocess.run([sys.executable, '-m', 'experiments.cvs_phase1_stack.completed_eval', '--mode', 'score'], cwd=ROOT, check=True)
        if read(BASE / 'scoring_complete.json')['status'] != 'SCORED_COMPLETE':
            raise ValueError('Scoring completion missing')
        write(BASE / 'completion.json', dict(status='ANALYZED', rows=len(rows), views=list(VIEWS),
              independent_recount='VERIFIED', target_feedback_forbidden=True))
    except Exception as error:
        write(BASE / 'failure.json', dict(status='FAILED', error=repr(error), no_retry=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['dispatch', 'predict', 'score'], required=True)
    parser.add_argument('--row'); args = parser.parse_args()
    if args.mode == 'dispatch':
        dispatch()
    elif args.mode == 'predict':
        predict(args.row)
    else:
        score()
