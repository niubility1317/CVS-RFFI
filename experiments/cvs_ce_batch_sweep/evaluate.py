"""User-requested early test of five fixed E200 models; never controls training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design as newdesign
PROJECT,WORKER,WORKER_COMMIT,RUN,BASE=newdesign.PROJECT,newdesign.WORKER,newdesign.WORKER_COMMIT,newdesign.RUN,newdesign.BASE
ROW_IDS=tuple(newdesign.config(b)['row_id'] for b in newdesign.BATCHES)
SELECTION='ALL_FIXED_CE_BATCHES_E200_FINAL_ONLY'


def validate_freeze(value, rows):
    if (value.get('run_id') != RUN or value.get('selection') != SELECTION
            or value.get('rows') != rows or value.get('target_access') is not False
            or [r['row_id'] for r in rows] != list(ROW_IDS)):
        raise ValueError('Changed five-row source freeze')
    return rows


def setup():
    if (WORKER / 'release_commit.txt').read_text().strip() != WORKER_COMMIT:
        raise ValueError('Immutable training release differs')
    sys.path[:0] = [str(WORKER), str(WORKER / 'code')]
    from experiments.cvs_phase1_repair import evaluate as e
    d = e.design()
    rows = [newdesign.config(b) for b in newdesign.BATCHES]
    d.require_budget=newdesign.require_budget
    original=e.source_provenance
    def provenance(d,c):
        done=e.read(Path(c['output_root'])/'completion.json')
        if done['logged_steps']!=200*c['steps_per_epoch'] or done['optimizer_steps']!=200*c['steps_per_epoch']:raise ValueError('Wrong row budget')
        return original(d,c)
    e.source_provenance=provenance
    e.BASE = BASE
    e.RUN = RUN
    e.ROOT = Path(__file__).resolve().parent
    e.snapshot = lambda unused: validate_freeze(e.read(BASE / 'source_matrix_frozen.json'), rows)
    return e, d, rows


def score():
    import numpy as np
    e, d, rows = setup()
    e.snapshot(d)
    ids = e.prediction_preflight(d, rows)  # All 35 predictions must exist before truth.
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
    e.write(BASE / 'summary.json', dict(results=summary, seed_count=1, interpretation='Fixed completed subset; no four-seed claim'))
    e.write(BASE / 'resources.json', dict(rows=resources, measurement='Inference concurrent with unrelated source training'))
    csvwrite(BASE / 'scores.csv', [{k:v for k,v in r.items() if k != 'confusion'} for r in results])
    csvwrite(BASE / 'summary.csv', summary)
    e.write(BASE / 'scoring_complete.json', dict(status='SCORED_COMPLETE', models=4, views=7,
        query_count_per_view=e.COUNT, prediction_count=4*7*e.COUNT,
        metric_records=len(results), truth_last=True, independent_recount='VERIFIED'))


def dispatch():
    e,d,rows=setup()
    BASE.mkdir(parents=True,exist_ok=False)
    try:
        active=[]
        free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        available=[int(line.split(',')[0]) for line in free.splitlines() if int(line.split(',')[1])<100]
        if len(available)<4:raise RuntimeError('Need four idle GPUs; no interference with existing training')
        for c,gpu in zip(rows,available[:4]):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            cfg=BASE/'configs'/(c['row_id']+'.json');e.write(cfg,c)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve().parent/'source.py'),'--config',str(cfg)]
            log=BASE/(c['row_id']+'.train.log')
            with log.open('x') as f:child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
            active.append((child,c));e.write(BASE/('launch_'+c['row_id']+'.json'),dict(pid=child.pid,gpu=gpu,argv=cmd,log=str(log)))
        errors=[]
        for child,c in active:
            if child.wait()!=0:errors.append(c['row_id'])
        if errors:raise RuntimeError('Failed source rows; retain all artifacts; no retuning/retry: '+repr(errors))
        for c in rows:e.source_provenance(d,c)
        e.write(BASE/'source_matrix_frozen.json',dict(run_id=RUN,selection=SELECTION,rows=rows,target_access=False,frozen_at=time.time(),source_worker_commit=WORKER_COMMIT))
        e.write(BASE/'preflight.json',e.preflight());e.manifest(d)
        for c in rows:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(available[0]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--mode','predict','--row',c['row_id']]
            with (BASE/(c['row_id']+'.predict.log')).open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--mode','score'],check=True)
        e.write(BASE/'completion.json',dict(status='ANALYZED',rows=list(ROW_IDS),views=list(e.VIEWS),target_feedback_forbidden=True))
    except Exception as exc:
        e.write(BASE/'failure.json',dict(status='FAILED',error=repr(exc),no_retry=True));raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dispatch', 'predict', 'score'], required=True)
    parser.add_argument('--row', choices=ROW_IDS)
    args = parser.parse_args()
    if args.mode == 'dispatch': dispatch()
    elif args.mode == 'predict':
        e, _, _ = setup()
        e.predict(args.row)
        c=next(newdesign.config(b) for b in newdesign.BATCHES if newdesign.config(b)['row_id']==args.row)
        p=BASE/args.row/'prediction/provenance.json';v=e.read(p);v['optimizer_steps']=200*c['steps_per_epoch'];e.write(p,v)
    else: score()
