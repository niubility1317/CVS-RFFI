"""User-requested early test of fourteen fixed E200 models; never controls training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path('/home/szu2070436088/2510044040/CV-SincNet')
WORKER = PROJECT / 'releases/cvs_reference_repair_20261008_r01'
WORKER_COMMIT = 'ae783c81cd4949f93a80dbd2e80c4b2568eb99ea'
RUN = '20261009-evaluation-reference-repair-completed14-manysig-m14-r01'
BASE = PROJECT / 'runs' / RUN
ROW_IDS = ('bank_cosine_domain-s2026092701', 'bank_cosine_domain-s2026092702', 'bank_cosine_mixstyle-s2026092701', 'bank_cosine_mixstyle-s2026092702', 'ce_cosine-s2026092702', 'ce_cosine-s2026092703', 'leo_cosine-s2026092702', 'leo_cosine-s2026092703', 'pseudo_bank_constant-s2026092702', 'pseudo_bank_cosine-s2026092701', 'pseudo_bank_cosine-s2026092702', 'pseudo_batch_constant-s2026092702', 'pseudo_batch_constant-s2026092703', 'pseudo_batch_cosine-s2026092702')
SELECTION = "FIXED_COMPLETED_AT_20261009_092722_EXCLUDING_PRIOR_FIVE_NO_METRIC_SELECTION"


def validate_freeze(value, rows):
    if (value.get('run_id') != RUN or value.get('selection') != SELECTION
            or value.get('rows') != rows or value.get('target_access') is not False
            or [r['row_id'] for r in rows] != list(ROW_IDS)):
        raise ValueError('Changed fourteen-row source freeze')
    return rows


def setup():
    if (WORKER / 'release_commit.txt').read_text().strip() != WORKER_COMMIT:
        raise ValueError('Immutable training release differs')
    sys.path[:0] = [str(WORKER), str(WORKER / 'code')]
    from experiments.cvs_phase1_repair import evaluate as e
    d = e.design()
    lookup = {r['row_id']: r for r in d.rows()}
    rows = [d.config(lookup[rid]) for rid in ROW_IDS]
    e.BASE = BASE
    e.RUN = RUN
    e.ROOT = Path(__file__).resolve().parent
    e.snapshot = lambda unused: validate_freeze(e.read(BASE / 'source_matrix_frozen.json'), rows)
    return e, d, rows


def score():
    import numpy as np
    e, d, rows = setup()
    e.snapshot(d)
    ids = e.prediction_preflight(d, rows)  # All 98 predictions must exist before truth.
    from comparison_suite.score import metrics, csvwrite
    truth = e.read(d.TRUTH)
    y = np.asarray([truth[s]['label'] for s in ids])
    rx = np.asarray([str(truth[s]['receiver']) for s in ids])
    if set(y) != set(range(6)) or len(set(rx)) != 7:
        raise ValueError('Truth population differs')
    results = []
    resources = []
    for c in rows:
        row = {k: c[k] for k in ('row_id', 'stage', 'arm', 'model_seed')}
        out = BASE / c['row_id'] / 'prediction'
        resources.append(dict(**row, **e.read(out / 'complete.json')))
        with np.load(out / 'predictions.npz', allow_pickle=False) as p:
            for view in e.VIEWS:
                for dimension, strata in [('overall', ['ALL']), ('receiver', sorted(set(rx))), ('transmitter', d.CLASSES)]:
                    for stratum in strata:
                        mask = np.ones(len(ids), bool) if dimension == 'overall' else rx == stratum if dimension == 'receiver' else y == d.CLASSES.index(stratum)
                        m = metrics(y[mask], p[view][mask], 6)
                        cm = np.bincount(6*y[mask]+p[view][mask], minlength=36).reshape(6, 6)
                        den = cm.sum(0)+cm.sum(1)
                        f1 = np.divide(2*np.diag(cm), den, out=np.zeros(6), where=den > 0).mean()
                        if m['confusion'] != cm.tolist() or m['query_count'] != int(cm.sum()) or abs(m['accuracy']-np.trace(cm)/cm.sum()) > 1e-12 or abs(m['macro_f1']-f1) > 1e-12:
                            raise ValueError('Independent metric recount differs')
                        results.append(dict(**row, view=view, dimension=dimension, stratum=stratum, **m))
    summary = e.summarize(results)
    e.write(BASE / 'scores.json', dict(status='SCORED', results=results, target_feedback_forbidden=True))
    e.write(BASE / 'summary.json', dict(results=summary, seed_count_by_arm={a: len({c['model_seed'] for c in rows if c['arm'] == a}) for a in {c['arm'] for c in rows}}, interpretation='Newly completed subset; combine with prior five only on matching seeds; no four-seed claim'))
    e.write(BASE / 'resources.json', dict(rows=resources, measurement='Inference concurrent with unrelated source training'))
    csvwrite(BASE / 'scores.csv', [{k:v for k,v in r.items() if k != 'confusion'} for r in results])
    csvwrite(BASE / 'summary.csv', summary)
    e.write(BASE / 'scoring_complete.json', dict(status='SCORED_COMPLETE', models=len(rows), views=7,
        query_count_per_view=e.COUNT, prediction_count=len(rows)*7*e.COUNT,
        metric_records=len(results), truth_last=True, independent_recount='VERIFIED'))


def dispatch():
    e, d, rows = setup()
    BASE.mkdir(parents=True, exist_ok=False)
    try:
        for c in rows:
            e.source_provenance(d, c)
        e.write(BASE / 'source_matrix_frozen.json', dict(run_id=RUN, selection=SELECTION,
            rows=rows, target_access=False, frozen_at=time.time(), source_worker_commit=WORKER_COMMIT))
        e.write(BASE / 'preflight.json', e.preflight())
        e.manifest(d)
        # One inference worker only. No new training; existing training jobs remain untouched.
        for c in rows:
            gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
            free, gpu = max((int(line.split(',')[1]), int(line.split(',')[0])) for line in gpu_rows.splitlines())
            if free < 12000:
                raise RuntimeError('Insufficient memory for one inference worker; no retry')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--mode', 'predict', '--row', c['row_id']]
            logpath = BASE / (c['row_id']+'.log')
            with logpath.open('x') as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                e.write(BASE / 'active.json', dict(row_id=c['row_id'], gpu=gpu, pid=child.pid, argv=cmd, log=str(logpath)))
                if child.wait() != 0:
                    raise RuntimeError('Prediction failed for '+c['row_id'])
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--mode', 'score'], check=True)
        e.write(BASE / 'completion.json', dict(status='ANALYZED', rows=list(ROW_IDS), views=list(e.VIEWS), target_feedback_forbidden=True))
    except Exception as exc:
        e.write(BASE / 'failure.json', dict(status='FAILED', error=repr(exc), no_retry=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dispatch', 'predict', 'score'], required=True)
    parser.add_argument('--row', choices=ROW_IDS)
    args = parser.parse_args()
    if args.mode == 'dispatch': dispatch()
    elif args.mode == 'predict':
        e, _, _ = setup()
        e.predict(args.row)
    else: score()
